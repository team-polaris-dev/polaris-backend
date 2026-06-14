"""Louvain 커뮤니티 검출 배치 (A-6) — 글로벌 질의 기반 (GraphRAG Global Search 흉내).

지분·종속·공통임원 엣지를 무방향 가중 그래프로 투영 → networkx Louvain →
Organization.cluster_id 적재 + 클러스터 통계 요약(JSON).

설계 (03_neo4j.md §1 Organization cluster_id† 참조):
  - 노드: corp_code 보유 Organization (needs_er 노드는 노이즈라 제외)
  - 엣지 가중: IS_SUBSIDIARY_OF 3.0 / IS_MAJOR_SHAREHOLDER_OF 2.0 /
              INVESTS_IN 1.0 / INTERLOCKING_DIRECTORATE 1.0 (confidence=low 감안)
  - valid_to 마감 엣지 제외 (현재 사실만)
  - 크기 2 미만 군집은 cluster_id 미부여 (singleton 노이즈 방지)
  - seed=42 결정론. 멱등: 재실행 시 cluster_id 전체 재계산·덮어쓰기.

요약 JSON: 클러스터별 size·대표 멤버(차수 상위)·엣지타입 분포 →
LLM 글로벌 요약의 입력(요약 생성은 별도 — 키 없는 환경 고려해 통계만).

사용: cd db && uv run python graph/build_communities.py [--dry-run]
의존: networkx (uv add networkx)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from db import neo4j_driver  # noqa: E402

OUT_PATH = HERE / "communities_summary.json"

EDGE_WEIGHTS = {
    "IS_SUBSIDIARY_OF": 3.0,
    "IS_MAJOR_SHAREHOLDER_OF": 2.0,
    "INVESTS_IN": 1.0,
    "INTERLOCKING_DIRECTORATE": 1.0,
}


def fetch_projection(s) -> tuple[nx.Graph, dict[str, str]]:
    g = nx.Graph()
    names: dict[str, str] = {}
    for rel, w in EDGE_WEIGHTS.items():
        rows = s.run(
            f"MATCH (a:Organization)-[r:{rel}]-(b:Organization) "
            "WHERE a.corp_code IS NOT NULL AND b.corp_code IS NOT NULL "
            "  AND a.corp_code < b.corp_code AND r.valid_to IS NULL "
            "RETURN a.corp_code AS ac, a.name AS an, "
            "       b.corp_code AS bc, b.name AS bn"
        ).data()
        for rec in rows:
            names.setdefault(rec["ac"], rec["an"] or rec["ac"])
            names.setdefault(rec["bc"], rec["bn"] or rec["bc"])
            if g.has_edge(rec["ac"], rec["bc"]):
                g[rec["ac"]][rec["bc"]]["weight"] += w
                g[rec["ac"]][rec["bc"]]["rels"].append(rel)
            else:
                g.add_edge(rec["ac"], rec["bc"], weight=w, rels=[rel])
    return g, names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="적재 없이 군집 통계만")
    args = ap.parse_args()

    drv = neo4j_driver()
    with drv.session() as s:
        g, names = fetch_projection(s)
        print(f"[projection] 노드 {g.number_of_nodes()} / 엣지 {g.number_of_edges()}")

        comms = nx.community.louvain_communities(g, weight="weight", seed=42)
        comms = sorted(comms, key=len, reverse=True)
        sized = [c for c in comms if len(c) >= 2]
        print(f"[louvain] 군집 {len(comms)}개 (크기 2 이상 {len(sized)}개)")

        summary: list[dict] = []
        assign: dict[str, int] = {}
        for cid, members in enumerate(sized):
            for m in members:
                assign[m] = cid
            deg = sorted(members, key=lambda x: -g.degree(x, weight="weight"))
            rel_counter: Counter[str] = Counter()
            for a in members:
                for b in g.neighbors(a):
                    if b in members and a < b:
                        rel_counter.update(g[a][b]["rels"])
            summary.append({
                "cluster_id": cid,
                "size": len(members),
                "anchor_members": [{"corp_code": m, "name": names.get(m, m)}
                                   for m in deg[:8]],
                "edge_type_dist": dict(rel_counter),
            })
            if cid < 5:
                tops = ", ".join(names.get(m, m) for m in deg[:5])
                print(f"  cluster {cid}: {len(members)}사 — {tops} …")

        if not args.dry_run:
            # 전체 초기화 후 재부여 (멱등 재계산)
            s.run("MATCH (o:Organization) WHERE o.cluster_id IS NOT NULL "
                  "REMOVE o.cluster_id")
            by_cid: dict[int, list[str]] = defaultdict(list)
            for code, cid in assign.items():
                by_cid[cid].append(code)
            for cid, codes in by_cid.items():
                s.run(
                    "UNWIND $codes AS cc MATCH (o:Organization {corp_code: cc}) "
                    "SET o.cluster_id = $cid",
                    codes=codes, cid=cid,
                )
            n = s.run("MATCH (o:Organization) WHERE o.cluster_id IS NOT NULL "
                      "RETURN count(o) AS n").single()["n"]
            print(f"[load] cluster_id 부여: {n}사")

    drv.close()
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"요약 저장: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
