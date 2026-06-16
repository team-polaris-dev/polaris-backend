"""GraphRAG 하이브리드 검색 오케스트레이터.

matcher → traverse → assemble.
정적 패턴이 hit 0이면 seed별 fallback subgraph 호출.
"""
from __future__ import annotations

import time
from typing import Iterable

from config.graphrag import PPR_ENABLED
from graphrag.matcher import match
from graphrag.schema import GraphHit, GraphSearchOutput, Seed
from graphrag.traverse import expand, expand_ppr, fallback_for


# 질문 키워드 → 앞세울 관계 유형. 질문이 "주주"면 지분망만, "공급망"이면 공급망만
# 보여주도록 관계 hit을 스코프한다 (질문 무시하고 동네 전체 덤프하던 문제 해결).
_FOCUS_KEYWORDS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("주주", "지분", "대주주", "주식", "소유", "오너", "지배구조", "지배", "자회사",
      "계열", "모회사", "종속"),
     ("IS_MAJOR_SHAREHOLDER_OF", "IS_SUBSIDIARY_OF", "INVESTS_IN")),
    (("공급", "납품", "매입", "매출처", "고객", "공급망", "협력사", "벤더", "거래처",
      "수혜", "이득", "낙수", "납품처"),  # "수혜주" = 거래 상대(공급사)가 수혜
     ("SUPPLIES_TO",)),
    (("투자",), ("INVESTS_IN",)),
    (("특수관계", "계열거래"), ("RELATED_PARTY",)),
    (("제품", "생산", "만드", "품목"), ("PRODUCES",)),
    (("기술", "공정"), ("USES_TECH",)),
    (("임원", "대표", "경영진", "이사", "CEO"), ("EXECUTIVE_OF",)),
]


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
