"""Graph retriever — Neo4j 엔티티/관계 기반 청크 회수 + 사실카드.

두 모드:
1) entity-anchor: 질의에서 회사/사람 엔티티 추출 → 시작노드에서 1~N홉 traversal
   → 도달한 (:Chunk) 노드 반환 (관계 근거 청크 우선)
2) fact-card: 정형 엣지(HAS_METRIC, EXECUTIVE_OF, IS_MAJOR_SHAREHOLDER_OF,
   INVESTS_IN, IS_SUBSIDIARY_OF, SUPPLIES_TO)를 직접 답변 카드로 반환.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import os

from ..neo4j_client import neo4j_driver

# 커뮤니티 산출물 (db/graph/build_communities.py 결과).
# 이식 패키지 self-contained — `langgraph_app/catalogs/` 에 동봉.
# 외부 갱신 경로를 쓰려면 POLARIS_COMMUNITIES_PATH 환경변수로 오버라이드.
COMMUNITIES_PATH = Path(
    os.getenv("POLARIS_COMMUNITIES_PATH")
    or Path(__file__).resolve().parent.parent / "langgraph_app" / "catalogs" / "communities_summary.json"
)


@dataclass
class GraphHit:
    chunk_id: str | None = None
    score: float = 0.0
    path: list[str] = field(default_factory=list)
    fact_card: dict[str, Any] | None = None


class GraphRetriever:
    def __init__(self, top_n: int = 50, max_hops: int = 3):
        self.top_n = top_n
        self.max_hops = max_hops
        self.driver = neo4j_driver()
        self.org_dict = self._load_org_dict()

    # 한국어 음차 ↔ 영문 약자 (자주 쓰이는 것만)
    _ALIAS_RULES: list[tuple[str, str]] = [
        ("에스케이", "SK"),
        ("엘지", "LG"),
        ("케이티", "KT"),
        ("에스에프에이", "SFA"),
        ("디비", "DB"),
        ("씨에스", "CS"),
        ("엘비", "LB"),
        ("에이에스이", "ASE"),
    ]

    def _load_org_dict(self) -> dict[str, str]:
        """이름 → corp_code (Organization 노드). 별칭/음차/접미사 모두 등록."""
        out: dict[str, str] = {}
        suffix_re = re.compile(r"(주식회사|㈜|\(주\)|주\)|\(유\)|유한회사)")
        with self.driver.session() as s:
            for rec in s.run(
                "MATCH (o:Organization) WHERE o.corp_code IS NOT NULL "
                "RETURN o.corp_code AS code, o.name AS name"
            ):
                code = rec["code"]
                name = rec["name"] or ""
                if not code:
                    continue
                variants = {name}
                stripped = suffix_re.sub("", name).strip()
                variants.add(stripped)
                # 음차 ↔ 영문 약자 양방향
                for ko, en in self._ALIAS_RULES:
                    for v in list(variants):
                        if ko in v:
                            variants.add(v.replace(ko, en))
                        if en in v:
                            variants.add(v.replace(en, ko))
                for v in variants:
                    v = v.strip()
                    if v and v not in out:
                        out[v] = code
        return out

    def detect_entities(self, query: str) -> list[str]:
        """질의에서 corp_code 리스트 추출 (가장 긴 이름부터 매칭)."""
        found: list[str] = []
        names = sorted(self.org_dict.keys(), key=len, reverse=True)
        text = query
        for n in names:
            if n and n in text:
                code = self.org_dict[n]
                if code not in found:
                    found.append(code)
                text = text.replace(n, " ")
        return found

    # ── 1) 엔티티 앵커 → 청크 회수 ──────────────────────────
    def anchor_chunks(self, corp_codes: list[str]) -> list[GraphHit]:
        if not corp_codes:
            return []
        cypher = """
        UNWIND $codes AS code
        MATCH (o:Organization {corp_code: code})
        OPTIONAL MATCH (o)-[r]-(c:Chunk)
        WITH o, c, r LIMIT $limit
        RETURN o.corp_code AS code, c.chunk_id AS chunk_id, type(r) AS rel
        """
        hits: list[GraphHit] = []
        with self.driver.session() as s:
            recs = s.run(cypher, codes=corp_codes, limit=self.top_n * 3).data()
        for i, rec in enumerate(recs):
            cid = rec.get("chunk_id")
            if not cid:
                continue
            hits.append(
                GraphHit(
                    chunk_id=cid,
                    score=1.0 / (1 + i),
                    path=[rec.get("code", ""), rec.get("rel") or ""],
                )
            )
        return hits[: self.top_n]

    # ── 2) 사실 카드 (정형 엣지 직접 조회) ───────────────
    def fact_cards(self, corp_codes: list[str]) -> list[GraphHit]:
        """corp_code 시작 정형 엣지를 카드로. 정답 직답 후보."""
        if not corp_codes:
            return []
        cards: list[GraphHit] = []
        with self.driver.session() as s:
            # 임원
            recs = s.run(
                """
                MATCH (p:Person)-[r:EXECUTIVE_OF]->(o:Organization)
                WHERE o.corp_code IN $codes AND r.valid_to IS NULL
                RETURN o.corp_code AS code, o.name AS org, p.name AS person,
                       r.ofcps AS pos LIMIT 50
                """,
                codes=corp_codes,
            ).data()
            for r in recs:
                cards.append(
                    GraphHit(
                        score=0.9,
                        path=[r["code"], "EXECUTIVE_OF", r["person"] or ""],
                        fact_card={"type": "executive", **r},
                    )
                )
            # 주요주주 ('계'·합계행 노이즈 제외)
            recs = s.run(
                """
                MATCH (x)-[r:IS_MAJOR_SHAREHOLDER_OF]->(o:Organization)
                WHERE o.corp_code IN $codes AND r.valid_to IS NULL
                  AND NOT coalesce(x.name, '') IN ['계', '소계', '합계', '-', '']
                RETURN o.corp_code AS code, o.name AS org,
                       coalesce(x.name, x.corp_code) AS holder,
                       r.qota_rt AS qota
                ORDER BY r.qota_rt DESC LIMIT 30
                """,
                codes=corp_codes,
            ).data()
            for r in recs:
                cards.append(
                    GraphHit(
                        score=0.9,
                        path=[r["holder"] or "", "IS_MAJOR_SHAREHOLDER_OF", r["code"]],
                        fact_card={"type": "shareholder", **r},
                    )
                )
            # 종속회사 (대규모 모회사 대응 — LIMIT 1500)
            recs = s.run(
                """
                MATCH (sub:Organization)-[r:IS_SUBSIDIARY_OF]->(o:Organization)
                WHERE o.corp_code IN $codes
                RETURN o.corp_code AS code, o.name AS parent,
                       sub.corp_code AS sub_code, sub.name AS sub_name LIMIT 1500
                """,
                codes=corp_codes,
            ).data()
            for r in recs:
                cards.append(
                    GraphHit(
                        score=0.9,
                        path=[r["code"], "IS_SUBSIDIARY_OF", r["sub_code"] or ""],
                        fact_card={"type": "subsidiary", **r},
                    )
                )
            # 출자 — [FIX] LIMIT 50 → 1500 (가나다순 첫번째 정확 회수)
            recs = s.run(
                """
                MATCH (o:Organization)-[r:INVESTS_IN]->(t)
                WHERE o.corp_code IN $codes AND r.valid_to IS NULL
                RETURN o.corp_code AS code, o.name AS investor,
                       coalesce(t.name, t.corp_code) AS investee LIMIT 1500
                """,
                codes=corp_codes,
            ).data()
            for r in recs:
                cards.append(
                    GraphHit(
                        score=0.9,
                        path=[r["code"], "INVESTS_IN", r["investee"] or ""],
                        fact_card={"type": "investment", **r},
                    )
                )
            # 공급망 (양방향 분리)
            recs = s.run(
                """
                MATCH (s:Organization)-[r:SUPPLIES_TO]->(b:Organization)
                WHERE s.corp_code IN $codes OR b.corp_code IN $codes
                RETURN s.corp_code AS s_code, s.name AS supplier,
                       b.corp_code AS b_code, b.name AS buyer LIMIT 80
                """,
                codes=corp_codes,
            ).data()
            for r in recs:
                cards.append(
                    GraphHit(
                        score=0.85,
                        path=[r["s_code"] or "", "SUPPLIES_TO", r["b_code"] or ""],
                        fact_card={"type": "supplies_to", **r},
                    )
                )
            # 핵심 IFRS 계정 — 사업보고서(11011=연간 확정치)만, 연결(CFS) 우선, 최신연도 우선.
            # [FIX] 분기/반기 보고서(11013/11012/11014) 혼입으로 2026 1분기값이 먼저 와
            #       연간값을 가리던 버그 해결 — 연간 확정치만 직답 후보로.
            recs = s.run(
                """
                MATCH (o:Organization)-[:HAS_METRIC]->(m:FinMetric)
                WHERE o.corp_code IN $codes
                  AND m.account_id IN [
                    'ifrs-full_Revenue','ifrs-full_Assets',
                    'ifrs-full_Liabilities','ifrs-full_Equity',
                    'ifrs-full_ProfitLoss'
                  ]
                  AND m.reprt_code = '11011'
                  AND m.bsns_year >= 2023
                RETURN o.corp_code AS code, o.name AS org, m.bsns_year AS year,
                       m.account_id AS account, m.value AS value,
                       m.fs_div AS fs_div, m.reprt_code AS reprt
                ORDER BY m.bsns_year DESC,
                         CASE WHEN m.fs_div = 'CFS' THEN 0 ELSE 1 END
                """,
                codes=corp_codes,
            ).data()
            for r in recs:
                cards.append(
                    GraphHit(
                        score=0.95,
                        path=[r["code"], "HAS_METRIC", r["account"] or ""],
                        fact_card={"type": "fin_metric", **r},
                    )
                )
            # 그 외 재무지표 (참고용, 한도 내)
            recs = s.run(
                """
                MATCH (o:Organization)-[:HAS_METRIC]->(m:FinMetric)
                WHERE o.corp_code IN $codes
                  AND NOT m.account_id IN [
                    'ifrs-full_Revenue','ifrs-full_Assets',
                    'ifrs-full_Liabilities','ifrs-full_Equity',
                    'ifrs-full_ProfitLoss'
                  ]
                RETURN o.corp_code AS code, o.name AS org, m.bsns_year AS year,
                       m.account_id AS account, m.value AS value,
                       m.fs_div AS fs_div, m.reprt_code AS reprt
                ORDER BY m.bsns_year DESC LIMIT 600
                """,
                codes=corp_codes,
            ).data()
            for r in recs:
                cards.append(
                    GraphHit(
                        score=0.75,
                        path=[r["code"], "HAS_METRIC", r["account"] or ""],
                        fact_card={"type": "fin_metric", **r},
                    )
                )
        return cards

    def aggregate_counts(self, corp_codes: list[str]) -> list[GraphHit]:
        """집계 전용 카드 — LIMIT 없이 카운트만."""
        if not corp_codes:
            return []
        hits: list[GraphHit] = []
        with self.driver.session() as s:
            recs = s.run(
                """
                UNWIND $codes AS code
                MATCH (o:Organization {corp_code: code})
                OPTIONAL MATCH (sub:Organization)-[:IS_SUBSIDIARY_OF]->(o)
                OPTIONAL MATCH (p:Person)-[re:EXECUTIVE_OF]->(o)
                WHERE re.valid_to IS NULL
                OPTIONAL MATCH (h)-[rh:IS_MAJOR_SHAREHOLDER_OF]->(o)
                WHERE rh.valid_to IS NULL
                RETURN o.corp_code AS code, o.name AS org,
                       count(DISTINCT sub) AS sub_cnt,
                       count(DISTINCT p) AS exec_cnt,
                       count(DISTINCT h) AS holder_cnt
                """,
                codes=corp_codes,
            ).data()
        for r in recs:
            hits.append(
                GraphHit(
                    score=0.95,
                    path=[r["code"], "AGG"],
                    fact_card={"type": "agg_count", **r},
                )
            )
        return hits

    # ── 3) 커뮤니티 카드 (global/sensemaking 질의) ────────
    def _load_communities(self) -> list[dict[str, Any]]:
        if not hasattr(self, "_communities"):
            try:
                self._communities = json.loads(COMMUNITIES_PATH.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                self._communities = []
        return self._communities

    def community_cards(self, query: str) -> list[GraphHit]:
        """클러스터 전체 조망 카드 — Microsoft GraphRAG global search의 결정론 축소판.

        매칭: (a) 멤버명(접미사 제거·음차 변환)이 질의에 등장, 또는
              (b) 질의 토큰이 클러스터 멤버 2개 이상의 접두(그룹 어간: 삼성·SK 등).
        매칭 0건이면 전체 클러스터 개요 카드 반환(질의가 global 타입일 때 호출 전제).
        """
        clusters = self._load_communities()
        if not clusters:
            return []
        suffix_re = re.compile(r"(주식회사|㈜|\(주\)|주\)|\(유\)|유한회사)")

        def strip_name(n: str) -> str:
            s = suffix_re.sub("", n or "").strip()
            for ko, en in self._ALIAS_RULES:
                s = s.replace(ko, en)
            return s

        q_tokens = [t for t in re.findall(r"[가-힣A-Za-z0-9]+", query) if len(t) >= 2]
        matched: list[dict[str, Any]] = []
        for cl in clusters:
            names = [strip_name(m["name"]) for m in cl.get("anchor_members", [])]
            hit = any(n and n in query for n in names) or any(
                sum(1 for n in names if n.startswith(t)) >= 2 for t in q_tokens
            )
            if hit:
                matched.append(cl)
        targets = matched or clusters

        hits: list[GraphHit] = []
        for cl in targets:
            dist = cl.get("edge_type_dist", {})
            top_edge = max(dist, key=dist.get) if dist else None
            hits.append(GraphHit(
                score=0.95,
                path=[f"cluster_{cl['cluster_id']}", "COMMUNITY"],
                fact_card={
                    "type": "community",
                    "cluster_id": cl["cluster_id"],
                    "size": cl.get("size"),
                    "anchors": [m["name"] for m in cl.get("anchor_members", [])],
                    "edge_type_dist": dist,
                    "top_edge_type": top_edge,
                    "n_clusters_total": len(clusters),
                    "matched": bool(matched),
                },
            ))
        return hits

    def search(self, query: str) -> list[GraphHit]:
        codes = self.detect_entities(query)
        if not codes:
            return []
        # 청크 앵커 + 사실 카드 + 집계 카드. 사실카드는 정답 정보라 cap 우회.
        anchor = self.anchor_chunks(codes)
        facts = self.fact_cards(codes)
        aggs = self.aggregate_counts(codes)
        return (anchor[: self.top_n] + aggs + facts)

    def close(self) -> None:
        self.driver.close()
