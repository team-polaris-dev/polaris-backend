"""LangGraph 진입점. AgentState → search → 신·구 키 동시 emit (셰임).

Issue #17 후속에서 nodes/rag.py의 합성기가 제거되고 gen이 graph_facts를
직접 포매팅하므로, legacy 키(graph_facts/paths/provenance) 셰임 유지가 필수.
"""
from __future__ import annotations

import logging

from config.entities import normalize_corp_name as _norm
from tool.graph_client import neo4j_driver
from graphrag import planner
from graphrag.matcher import match
from graphrag.router import classify
from graphrag.schema import adapt_to_legacy
from graphrag.search import search
from graphrag.global_search import global_search
from graphrag.traverse import fallback_for

log = logging.getLogger(__name__)

_preflight_done = False


def _last_human_text(state: dict) -> str:
    """state.messages 에서 마지막 사용자(human) 메시지 본문을 반환. 없으면 ''."""
    for msg in reversed(state.get("messages") or []):
        if getattr(msg, "type", "") == "human":
            return str(msg.content)
    return ""


def _preflight() -> None:
    """1회성 DB 사전 점검. entity_fulltext 인덱스 없으면 경고만(검색은 빈 결과로 degrade)."""
    global _preflight_done
    if _preflight_done:
        return
    _preflight_done = True
    try:
        with neo4j_driver.session() as s:
            row = s.run(
                "SHOW INDEXES YIELD name WHERE name = 'entity_fulltext' "
                "RETURN count(*) AS c"
            ).single()
        if not row or row["c"] == 0:
            log.warning(
                "graphrag: entity_fulltext index missing — "
                "run `python -m pipeline_scripts.graph.setup_fulltext_index`"
            )
    except Exception as e:
        log.warning("graphrag preflight skipped (Neo4j unreachable?): %s", e)


def _anchor_corp_codes(graph_seeds: list[dict]) -> list[str]:
    """로컬 시드에서 corp_code 앵커만 추출(순서·중복제거). DRIFT 군집 선택용.

    Seed.key_type=='corp_code' 인 시드의 key_value 가 실제 corp_code(schema.Seed).
    """
    codes: list[str] = []
    seen: set[str] = set()
    for sd in graph_seeds or []:
        if sd.get("key_type") != "corp_code":
            continue
        code = sd.get("key_value")
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _assemble_local(out: dict) -> dict:
    """search() 출력 → Local Search 결과 dict(신규 키 + legacy 셰임 키).

    Issue #17 이후 result_check/gen 이 legacy 키(graph_facts/paths/provenance)를 직접
    보므로 셰임을 함께 emit 한다. ctx 경로와 global-앵커(DRIFT) 경로가 공유한다.
    """
    legacy = adapt_to_legacy(out["graph_hits"])
    return {
        # 신규
        "graph_hits": out["graph_hits"],
        "graph_seeds": out["graph_seeds"],
        "graph_meta": out["graph_meta"],
        # 셰임 (Issue #17 result_check/gen이 이 키를 직접 봄)
        "graph_facts": legacy["facts"],
        "graph_paths": legacy["paths"],
        "graph_provenance": legacy["provenance"],
        # 패널 엣지별 출처 — graph_paths 와 행 정렬(serialize.build_graph 가 i 로 읽음).
        "graph_path_sources": legacy["path_sources"],
        "graph_path_chunks": legacy["path_chunks"],
    }


def _attach_communities(result: dict, query: str, out: dict) -> None:
    """DRIFT: 로컬 앵커가 속한 군집의 map-reduce 부분답을 result 에 결합(in-place).

    out["graph_seeds"] 의 corp_code 앵커가 속한 군집만 map-reduce 한다. 앵커가 없으면
    (엔티티 미해소) 아무것도 붙이지 않는다 — 노이즈 0. ctx·global-앵커 경로 공유.
    """
    anchors = _anchor_corp_codes(out["graph_seeds"])
    if not anchors:
        return
    community_results = global_search(query, anchor_corp_codes=anchors)
    if community_results:
        print(f"🌐 [GraphRAG/DRIFT] 앵커 군집 {len(community_results)}개 결합")
        result["community_results"] = community_results


def _safe_match(query: str, upstream: list[str]) -> list[dict]:
    """match() 를 fail-closed 로 감싼다(silent 선처리용). 실패 시 []."""
    try:
        return match(query, upstream_seeds=upstream or [])
    except Exception as e:
        log.warning("graph silent match failed: %s", e)
        return []


def _silent_output(seeds: list[dict]) -> dict:
    """순수 속성 질문 → 그래프 침묵. org 앵커가 해소되면 단일 노드 스텁만, 미해소면 빈 {}.

    스텁은 graph_facts 길이를 1 로 만들어 ctx 게이트(empty_sources)는 통과시키되, 망 엣지가
    없어 패널은 자동으로 닫힌다(serialize.build_graph). rdb/vec 가 실제 답을 낸다.
    """
    anchor = next((s for s in (seeds or []) if s.get("label") == "organization"), None)
    if not anchor:
        return {}
    hit = {
        "id": str(anchor.get("id") or anchor.get("key_value") or anchor.get("name") or ""),
        "label": "organization",
        "name": str(anchor.get("name") or ""),
        "attrs": {"silent": True},
        "score": float(anchor.get("score") or 1.0),
        "seed_origin": "silent",
    }
    out = {
        "graph_hits": [hit],
        "graph_seeds": [dict(s) for s in seeds],
        "graph_meta": {
            "mode": "silent",
            "patterns_run": ["silent"],
            "n_seeds": len(seeds),
            "n_hits": 1,
            "fallback_used": False,
            "errors": [],
        },
    }
    return _assemble_local(out)


# 재구성(LLM)이 표준어 치환·축소로 지워버리면 안 되는 구조화 의도. 원문이 이 kind 로
# 잡히는데 재구성 후 같은 kind 를 잃으면, 재구성이 의도를 깬 것이라 원문을 쓴다.
#  - community_member_rank: 그룹 범위어(계열사 등)를 특정 관계어(종속회사)로 좁히는 축소.
#  - multi_anchor_rank: 공통 앵커 cue(동시에/둘 다)를 재구성이 지우면 교집합 의도가
#    사라져 단일 앵커 관계답으로 뒤집히는 축소.
_RECONSTRUCT_PRESERVE_KINDS = {"community_member_rank", "multi_anchor_rank"}


def effective_query(state: dict) -> str:
    """의도 보존 질의. 보통은 재구성 질의지만, 재구성이 구조화 의도
    (_RECONSTRUCT_PRESERVE_KINDS)를 표준어 치환·축소로 지운 경우엔 원문을 쓴다. 판정은
    결정적 플래너가 한다 — 원문은 해당 kind 인데 재구성은 아니면 재구성이 의도를 깬 것.
    대명사 해소는 reconstructed_seeds(upstream)가 매처에 따로 전달돼 보존되므로 원문을
    써도 앵커는 유지된다.

    graph 노드(여기)와 gen 노드(render.generate_report_node)가 공유한다 — 그래프
    시각화와 답변 본문이 같은 질의를 봐 의도 축소가 한쪽만 새지 않도록.
    """
    reconstructed = state.get("reconstructed_query") or ""
    original = _last_human_text(state)
    if not original or original == reconstructed:
        return reconstructed
    orig_plan = planner.plan(original)
    if orig_plan and orig_plan.kind in _RECONSTRUCT_PRESERVE_KINDS:
        recon_plan = planner.plan(reconstructed)
        if not recon_plan or recon_plan.kind != orig_plan.kind:
            return original
    return reconstructed


def _first_org_seed(graph_seeds: list[dict]) -> dict | None:
    return next((s for s in (graph_seeds or []) if s.get("label") == "organization"), None)


def _explicit_company_mention(query: str, graph_seeds: list[dict]) -> bool:
    """사용자가 회사를 *명시 호명*했는가 — 해소된 org 시드 이름이 질의에 그대로 포함되는가.

    matcher._select 의 'strong' 판정과 같은 규칙(정규화 이름이 질의의 부분문자열). 업종어가
    회사명에 우연히 퍼지매칭된 경우엔 이름이 질의에 안 들어 있어 False → MACRO 가 유지된다.
    명시 호명이면 매크로 cue 가 있어도 그 회사 관계망(+DRIFT)을 보여준다.
    """
    qn = _norm(query)
    if not qn:
        return False
    for sd in graph_seeds or []:
        if sd.get("label") != "organization":
            continue
        nm = _norm(sd.get("name") or "")
        if nm and nm in qn:
            return True
    return False


def _is_rankless(out: dict) -> bool:
    """줄세울 결과가 없는 상태인가 — 구조화 abstain 또는 앵커 스텁만 남은 경우."""
    meta = out.get("graph_meta") or {}
    if (meta.get("structured") or {}).get("abstained"):
        return True
    hits = out.get("graph_hits") or []
    non_anchor = [h for h in hits if (h.get("attrs") or {}).get("structured_role") != "anchor"]
    return not non_anchor


def _with_fallback_network(out: dict, seed: dict) -> dict:
    """비어 있는 결과를 시드 depth-2 관계망(fallback_for)으로 채워 재조립한다.

    기존 hit(앵커 스텁 등)와 id 로 dedup. 결정성은 fallback_for 내부 _postprocess 가 보장.
    """
    existing = list(out.get("graph_hits") or [])
    seen = {h.get("id") for h in existing if h.get("id")}
    try:
        extra = fallback_for(seed)
    except Exception as e:
        log.warning("graph degrade fallback_for failed: %s", e)
        extra = []
    for h in extra:
        hid = h.get("id")
        if hid and hid in seen:
            continue
        if hid:
            seen.add(hid)
        existing.append(h)
    merged = dict(out)
    merged["graph_hits"] = existing
    meta = dict(out.get("graph_meta") or {})
    meta["n_hits"] = len(existing)
    meta["fallback_used"] = True
    merged["graph_meta"] = meta
    return _assemble_local(merged)


def graph_search_node(state: dict) -> dict:
    """GraphRAG 단일 진입점. 질문마다 모드(역할)를 골라 무엇을 할지/말지 결정한다.

    - global(매크로/업계): 로컬 search() 로 앵커 해소를 시도해 잡히면 대칭 DRIFT(로컬 사실
      + 앵커 군집), 없으면 순수 매크로 map-reduce(community_results). (현행 보존)
    - 그 외(ctx/local): silent(순수 속성)→침묵, macro(앵커없는 매크로)→커뮤니티,
      relation_only(줄세울 지표 없음)→시드 관계망 + 'rankable=False' 신호로 degrade,
      relation_rank/relation_explore·구조화 kind→search() 가 이미 실행한 결과를 조립.
    질문종류는 graphrag.router.classify(결정적 프리필터 우선, 애매하면 LLM 1회)가 한 번 판정해
    search 와 공유한다 — 분류는 한 곳, 소비는 두 곳. 별도 플로우 노드 없이 이 노드 안에서 분기한다.
    """
    _preflight()

    if state.get("intent") == "global":
        # global 은 ctx 를 건너뛰므로 reconstructed_query 가 비어 원문으로 폴백.
        query = state.get("reconstructed_query") or _last_human_text(state)
        out = search(query)
        anchors = _anchor_corp_codes(out["graph_seeds"])
        # 업종어가 회사명에 퍼지매칭돼 corp_code 앵커가 잡혀도(예: "반도체"→한미반도체),
        # 사용자가 회사를 명시 호명하지 않았으면 DRIFT 가 아니라 순수 매크로로 본다 —
        # 비-global 경로(아래 MACRO 분기)와 같은 explicit-signal-over-noise 규칙.
        # 명시 호명이라도 그 앵커가 어느 군집에도 없어 DRIFT 가 비면 순수 매크로로 폴백한다
        # (community_results 가 비면 result_check 가 답을 막으므로 빈 답변 방지).
        if anchors and _explicit_company_mention(query, out["graph_seeds"]):
            result = _assemble_local(out)
            _attach_communities(result, query, out)
            if result.get("community_results"):
                print(f"🌐 [GraphRAG/DRIFT-global] 앵커 {len(anchors)}개 + 로컬 사실 결합")
                return result
        results = global_search(query)
        print(f"🌐 [GraphRAG/Global] 커뮤니티 {len(results)}개 선택")
        return {"community_results": results}

    query = effective_query(state)
    upstream = state.get("reconstructed_seeds") or []
    has_metric = planner._metric_id(query) is not None

    # 질문종류 1회 판정(router). search 도 같은 route 를 받아 분류를 두 번 하지 않는다.
    route = classify(query, has_metric=has_metric)

    # SILENT 선처리: PPR 펼치기 전에 차단(억지 관계망 방지). 앵커는 match() 로 가볍게 해소.
    if route.type == "silent":
        return _silent_output(_safe_match(query, upstream))

    out = search(query, upstream_seeds=upstream, route=route)
    explicit = _explicit_company_mention(query, out["graph_seeds"])

    # 매크로 질문(업계 큰그림) → 커뮤니티 map-reduce. 업종어가 회사명에 퍼지매칭돼 corp_code
    # 앵커가 잡혀도, 사용자가 회사를 명시 호명하지 않았으면 매크로로 본다(MACRO 사문화 차단).
    # 명시 호명(explicit)이면 아래로 흘러 그 회사 관계망 + DRIFT 를 보존한다.
    if route.type == "macro" and not explicit:
        results = global_search(query)
        print(f"🌐 [GraphRAG/Global] 커뮤니티 {len(results)}개 선택")
        return {"community_results": results}

    result = _assemble_local(out)

    # RELATION_ONLY: 줄세울 노드 지표가 없는 관계질문 → 억지 1위 금지.
    # 시드 관계망은 (PPR 로 이미 있거나) fallback_for 로 채우고, 신호는 graph_meta 에만 둔다.
    if route.type == "relation_only":
        if _is_rankless(out):
            seed = _first_org_seed(out["graph_seeds"])
            if seed:
                result = _with_fallback_network(out, seed)
        result["graph_meta"]["rankable"] = False
        result["graph_meta"].setdefault("degrade_reason", "순위 매길 관계 없음; 시드 관계망 표시")

    _attach_communities(result, query, out)
    return result
