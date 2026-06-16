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
