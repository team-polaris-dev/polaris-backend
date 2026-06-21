"""GraphRAG 하이브리드 검색 오케스트레이터.

matcher → route(질문종류 1회 판정) → 구조화 실행 or text2cypher → traverse → assemble.
정적 패턴이 hit 0이면 seed별 fallback subgraph 호출.

질문종류 판정은 graphrag.router.classify 가 단일 진입점이다(흩어진 키워드 게이트를 통합).
route.plan 이 있으면(chain/community/multi_anchor) 구조화 실행기로, 없으면 관계/모드 질문이라
text2cypher → PPR 폴백으로 흐른다.
"""
from __future__ import annotations

import time
from typing import Iterable

from config.graphrag import PPR_ENABLED, TEXT2CYPHER_ENABLED
from config.relations import metric_for_query
from graphrag.cypher_executor import _anchor_code, map_results, rank_results
from graphrag.matcher import match
from graphrag.router import Route, classify
from graphrag.schema import GraphHit, GraphSearchOutput, Seed
from graphrag.structured_executor import execute as execute_structured
from graphrag.text2cypher import run_relationship_query
from graphrag.traverse import expand, expand_ppr, fallback_for


def _structured_abstain_output(
    query: str,
    seeds: list[Seed],
    structured_plan,
    errors: list[str],
    started: float,
    reason: str,
) -> GraphSearchOutput:
    anchor_hits: list[GraphHit] = []
    seen: set[str] = set()
    for seed in seeds:
        if seed.get("label") != "organization":
            continue
        sid = str(seed.get("id") or seed.get("key_value") or seed.get("name") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        anchor_hits.append({
            "id": sid,
            "label": "organization",
            "name": str(seed.get("name") or ""),
            "attrs": {"structured_role": "anchor", "abstained": True},
            "score": float(seed.get("score") or 1.0),
            "seed_origin": "structured",
        })
        break

    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "graph_hits": anchor_hits,
        "graph_seeds": [dict(s) for s in seeds],
        "graph_meta": {
            "mode": "structured",
            "structured": {
                "mode": "structured",
                "plan": structured_plan.to_dict() if structured_plan else None,
                "answer_edges": [],
                "abstained": True,
                "abstain_reason": reason,
            },
            "patterns_run": ["structured_plan", "abstain"],
            "n_seeds": len(seeds),
            "n_hits": len(anchor_hits),
            "fallback_used": False,
            "latency_ms": round(elapsed, 1),
            "errors": errors,
        },
    }


def _try_text2cypher(
    query: str, seeds: list[Seed], errors: list[str], started: float
) -> GraphSearchOutput | None:
    """공식 Text2CypherRetriever 관계검색 우선 시도(플래그 on, organization 앵커 존재 시).

    생성·실행 어느 단계든 None/예외면 None 을 돌려 호출자가 기존 planner 경로로
    폴백한다(회귀 0). 성공 시 구조화 조기반환과 동일한 모양으로 패키징한다.
    엔진은 라이브러리(run_relationship_query); 도메인 레이어(map_results)가 행→hit·근거를 얹는다.
    """
    if not TEXT2CYPHER_ENABLED:
        return None
    org_seeds = [s for s in seeds if s.get("label") == "organization"]
    if not org_seeds:
        return None
    anchor_dicts = [dict(s) for s in org_seeds]
    anchor_codes = [c for c in (_anchor_code(a) for a in anchor_dicts) if c]
    if not anchor_codes:
        return None
    try:
        produced = run_relationship_query(query, anchor_codes)
    except Exception as e:
        errors.append(f"text2cypher: {e}")
        return None
    if not produced:
        return None
    rows, cypher = produced
    result = map_results(rows, cypher, anchor_dicts, query)
    if not result:
        return None
    hits, structured_meta = result
    # 재무 지표 랭킹은 그래프(text2cypher)가 아니라 MariaDB 결정적 SQL 이 담당하는 value-add
    # 레이어 — 랭킹 의도가 있으면 후보를 줄세워 hits/meta 에 주석. 의도 없으면 no-op.
    hits, structured_meta = rank_results(hits, structured_meta, query, anchor_codes)
    elapsed = (time.perf_counter() - started) * 1000.0
    structured_meta["latency_ms"] = round(elapsed, 1)
    structured_meta["n_seeds"] = len(seeds)
    structured_meta["errors"] = errors
    return {
        "graph_hits": hits,
        "graph_seeds": [dict(s) for s in seeds],
        "graph_meta": structured_meta,
    }


def _run_structured(
    plan, seeds: list[Seed], query: str, errors: list[str], started: float
) -> GraphSearchOutput:
    """구조화 plan 실행 → 성공 시 canonical 출력, 실패 시 abstain 출력으로 패키징.

    보존 결정적 kind 와 다중홉 체인이 공유한다(조기반환 모양 동일).
    """
    try:
        structured = execute_structured(plan, seeds, query)
    except Exception as e:
        errors.append(f"structured: {e}")
        structured = None
    if structured:
        hits, structured_meta = structured
        elapsed = (time.perf_counter() - started) * 1000.0
        structured_meta["latency_ms"] = round(elapsed, 1)
        structured_meta["n_seeds"] = len(seeds)
        structured_meta["errors"] = errors
        return {
            "graph_hits": hits,
            "graph_seeds": [dict(s) for s in seeds],
            "graph_meta": structured_meta,
        }
    return _structured_abstain_output(
        query,
        seeds,
        plan,
        errors,
        started,
        "structured plan found no candidate that passed relation, evidence, and metric gates",
    )


def search(
    query: str,
    upstream_seeds: Iterable[str] | None = None,
    route: Route | None = None,
) -> GraphSearchOutput:
    """단일 호출 검색.

    query: reconstructed_query 평문
    upstream_seeds: 앞단(Gemini)이 동봉한 식별자 리스트 (옵션)
    route: 상위(node)가 이미 판정한 질문종류. None 이면 여기서 1회 분류(global 경로·직접 호출자).
    """
    started = time.perf_counter()
    errors: list[str] = []

    try:
        seeds: list[Seed] = match(query, upstream_seeds=upstream_seeds or [])
    except Exception as e:
        errors.append(f"matcher: {e}")
        seeds = []

    hits: list[GraphHit] = []
    patterns_run: list[str] = []
    fallback_used = False

    if seeds:
        # 질문종류 1회 판정(이미 받았으면 재사용). route.plan 이 있으면 구조화 실행 —
        # 보존 결정적 kind(community/multi_anchor)와 다중홉 체인(cutline)을 한 디스패치로 모은다.
        # 체인을 text2cypher *앞*에 두는 이유: "수혜의 수혜"처럼 영향이 2단계로 전파되는 질문을
        # text2cypher 의 단일 홉 Cypher 가 가로채면 2번째 홉이 결과에 안 담겨 "자료 부족"이 된다.
        if route is None:
            route = classify(query, has_metric=metric_for_query(query) is not None)
        if route.plan is not None:
            return _run_structured(route.plan, seeds, query, errors, started)

        # route.plan 없음 = 관계/모드 질문 → 공식 Text2CypherRetriever 관계검색(+ SQL 랭킹
        # 후처리). 실패/abstain 이면 None → 폴백.
        t2c = _try_text2cypher(query, seeds, errors, started)
        if t2c is not None:
            return t2c

        # PPR 우선(시드 관련성 멀티홉). 실패·빈 결과면 패턴 확장으로 폴백.
        if PPR_ENABLED:
            try:
                hits, patterns_run = expand_ppr(seeds)
            except Exception as e:
                errors.append(f"ppr: {e}")
                hits = []
        if not hits:
            try:
                hits, patterns_run = expand(seeds)
            except Exception as e:
                errors.append(f"traverse: {e}")
                hits = []

        if not hits:
            # 정적 hit 0 → seed별 fallback (모든 seed 합쳐서)
            fallback_used = True
            for sd in seeds:
                try:
                    hits.extend(fallback_for(sd))
                    patterns_run.append(f"fallback({sd['id']})")
                except Exception as e:
                    errors.append(f"fallback({sd['id']}): {e}")

    elapsed = (time.perf_counter() - started) * 1000.0

    out: GraphSearchOutput = {
        "graph_hits": hits,
        "graph_seeds": [dict(s) for s in seeds],
        "graph_meta": {
            "latency_ms": round(elapsed, 1),
            "patterns_run": patterns_run,
            "n_seeds": len(seeds),
            "n_hits": len(hits),
            "fallback_used": fallback_used,
            "errors": errors,
        },
    }
    return out
