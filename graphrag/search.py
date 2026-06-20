"""GraphRAG 하이브리드 검색 오케스트레이터.

matcher → traverse → assemble.
정적 패턴이 hit 0이면 seed별 fallback subgraph 호출.
"""
from __future__ import annotations

import time
from typing import Iterable

from config.graphrag import PPR_ENABLED
from config.relations import FOCUS_KEYWORD_GROUPS as _FOCUS_KEYWORDS
from graphrag.llm_planner import plan as plan_structured_llm
from graphrag.matcher import match
from graphrag.planner import plan as plan_structured
from graphrag.schema import GraphHit, GraphSearchOutput, Seed
from graphrag.structured_executor import execute as execute_structured
from graphrag.traverse import expand, expand_ppr, fallback_for

# 결정적 플래너가 확정한 멀티/단일 앵커 branch-rank 와 그룹 군집 멤버 랭킹은 LLM 플래너로
# 덮어쓰지 않고 그대로 쓴다 — 그룹 범위 탐지는 결정적이라 LLM 재분류가 불필요·해로움.
DETERMINISTIC_PRIORITY_KINDS = {
    "multi_anchor_branch_rank",
    "single_anchor_branch_rank",
    "community_member_rank",
    "two_hop_list",
}


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


def _relation_focus(query: str) -> set[str]:
    """질문에서 어떤 관계를 앞세울지 결정. 매칭 없으면 빈 set(=전체)."""
    focus: set[str] = set()
    for keywords, rels in _FOCUS_KEYWORDS:
        if any(k in query for k in keywords):
            focus.update(rels)
    return focus


def _apply_focus(hits: list[GraphHit], focus: set[str]) -> list[GraphHit]:
    """관계 hit을 focus 유형으로 스코프. 비관계 hit(노드/속성)은 보존.
    과필터로 관계가 0이 되면 원본 유지(빈 망 방지)."""
    if not focus:
        return hits
    kept = [
        h for h in hits
        if h.get("label") != "relationship"
        or h.get("attrs", {}).get("rel_type") in focus
    ]
    if any(h.get("label") == "relationship" for h in kept):
        return kept
    return hits


def search(query: str, upstream_seeds: Iterable[str] | None = None) -> GraphSearchOutput:
    """단일 호출 검색.

    query: reconstructed_query 평문
    upstream_seeds: 앞단(Gemini)이 동봉한 식별자 리스트 (옵션)
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
        deterministic_plan = plan_structured(query)
        structured_plan = deterministic_plan
        if not deterministic_plan or deterministic_plan.kind not in DETERMINISTIC_PRIORITY_KINDS:
            try:
                structured_plan = plan_structured_llm(query)
            except Exception as e:
                errors.append(f"llm_planner: {e}")
            if structured_plan is None:
                structured_plan = deterministic_plan
        if structured_plan:
            try:
                structured = execute_structured(structured_plan, seeds, query)
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
                structured_plan,
                errors,
                started,
                "structured plan found no candidate that passed relation, evidence, and metric gates",
            )

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

    # 질문 키워드로 관계 스코프 (주주→지분망 / 공급망→공급망). 질문 무관 덤프 방지.
    focus = _relation_focus(query)
    if focus:
        before = len(hits)
        hits = _apply_focus(hits, focus)
        if len(hits) != before:
            patterns_run.append("focus(" + "+".join(sorted(focus)) + ")")

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
