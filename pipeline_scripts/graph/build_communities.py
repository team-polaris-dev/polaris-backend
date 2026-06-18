"""Louvain 커뮤니티 검출 배치 (A-6) — 글로벌 질의 기반 (GraphRAG Global Search 흉내).

지분·종속·공통임원 엣지를 무방향 가중 그래프로 투영 → networkx Louvain →
Organization.cluster_id 적재 + 클러스터 통계 요약(JSON).

설계 (03_neo4j.md §1 Organization cluster_id† 참조):
  - 노드: corp_code 보유 Organization (needs_er 노드는 노이즈라 제외)
  - 엣지 가중: IS_SUBSIDIARY_OF 3.0 / IS_MAJOR_SHAREHOLDER_OF 2.0 /
              INVESTS_IN 1.0 / SUPPLIES_TO 1.0 / INTERLOCKING_DIRECTORATE 1.0
              (SUPPLIES_TO·INTERLOCKING 은 추출 엣지라 confidence=low 감안해 1.0)
  - valid_to 마감 엣지 제외 + QC 비활성(qc_disabled_at) 엣지 제외 (현재 사실만)
  - 크기 2 미만 군집은 cluster_id 미부여 (singleton 노이즈 방지)
  - seed=42 결정론. 멱등: 재실행 시 cluster_id 전체 재계산·덮어쓰기.

요약 JSON: 클러스터별 size·대표 멤버(차수 상위)·엣지타입 분포 →
LLM 글로벌 요약의 입력.

GraphRAG Global Search 확장(2026-06): 클러스터별로 멤버사·관계 분포를 근거로
config.llm.llm 에 3~4문장 한국어 요약을 생성시키고, 클러스터당 Community 노드를
Neo4j 에 MERGE 한다(멱등). 질의측(graphrag/global_search.py)이 이 Community 노드를
읽어 매크로/업계 질문에 답한다. LLM 키 없는 환경/실패 시 멤버명+엣지분포 기반의
결정론 폴백 요약을 저장한다(빌드 중단 금지). --no-summary 로 통계만(구 동작).

사용: cd db && uv run python graph/build_communities.py [--dry-run] [--no-summary]
   또는 repo 루트에서: python -m pipeline_scripts.graph.build_communities
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
# db 가 repo 루트를 sys.path 에 넣은 뒤라야 config 임포트가 보장된다(순서 중요).
from config.llm import llm  # noqa: E402
from config.relations import REL_LABELS  # noqa: E402

OUT_PATH = HERE / "communities_summary.json"

EDGE_WEIGHTS = {
    "IS_SUBSIDIARY_OF": 3.0,
    "IS_MAJOR_SHAREHOLDER_OF": 2.0,
    "INVESTS_IN": 1.0,
    "SUPPLIES_TO": 1.0,          # 공급망 군집을 잡으려면 필수. 추출 엣지라 confidence 낮아 1.0.
    "INTERLOCKING_DIRECTORATE": 1.0,
}


def fetch_projection(s) -> tuple[nx.Graph, dict[str, str]]:
    g = nx.Graph()
    names: dict[str, str] = {}
    for rel, w in EDGE_WEIGHTS.items():
        rows = s.run(
            f"MATCH (a:Organization)-[r:{rel}]-(b:Organization) "
            "WHERE a.corp_code IS NOT NULL AND b.corp_code IS NOT NULL "
            "  AND a.corp_code < b.corp_code "
            "  AND r.valid_to IS NULL AND r.qc_disabled_at IS NULL "
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


def _edge_dist_ko(edge_dist: dict[str, int]) -> str:
    """엣지타입 분포 dict → '대주주 23건, 투자 32건 …' 한글 요약 문자열."""
    parts = []
    for rel, cnt in sorted(edge_dist.items(), key=lambda kv: -kv[1]):
        parts.append(f"{REL_LABELS.get(rel, rel)} {cnt}건")
    return ", ".join(parts)


def _fallback_summary(member_names: list[str], edge_dist: dict[str, int]) -> str:
    """LLM 없이(또는 실패 시) 멤버명+엣지분포로 만드는 결정론 폴백 요약."""
    head = ", ".join(member_names[:5])
    tail = " 등" if len(member_names) > 5 else ""
    dist = _edge_dist_ko(edge_dist)
    return (
        f"{head}{tail}으로 구성된 {len(member_names)}개 기업 군집입니다. "
        f"이들은 {dist} 관계로 연결되어 있습니다."
    )


def build_summary(member_names: list[str], edge_dist: dict[str, int]) -> str:
    """클러스터 멤버사·관계 분포 → LLM 3~4문장 한국어 요약.

    LLM 호출이 실패하면(키 없음/타임아웃 등) 결정론 폴백 요약을 반환한다.
    절대 예외를 위로 던지지 않는다 — 빌드 전체가 죽지 않도록.
    """
    members_txt = ", ".join(member_names)
    dist_txt = _edge_dist_ko(edge_dist)
    prompt = (
        "당신은 한국 기업 지배구조·계열 관계 분석가입니다.\n"
        "아래는 그래프 커뮤니티 검출로 묶인 한 기업 군집의 구성원과 그들 사이의 관계 분포입니다.\n"
        "이 군집이 무엇으로 묶였는지(어느 그룹·계열·밸류체인인지, 무엇이 이들을 연결하는지)를\n"
        "3~4문장의 자연스러운 한국어로 요약하세요. 군집에 없는 사실을 지어내지 말고,\n"
        "주어진 구성원·관계 분포만 근거로 설명하세요. 인사말·머리말 없이 요약문만 출력하세요.\n\n"
        f"[구성원]\n{members_txt}\n\n"
        f"[관계 분포]\n{dist_txt}\n\n"
        "요약:"
    )
    try:
        text = llm.invoke(prompt).content
        text = (text or "").strip()
        if not text:
            raise ValueError("빈 요약")
        return text
    except Exception as e:
        print(f"  [warn] LLM 요약 실패 → 폴백 사용: {e}")
        return _fallback_summary(member_names, edge_dist)


def upsert_communities(s, summary: list[dict], names: dict[str, str],
                       by_cid: dict[int, list[str]], do_summary: bool) -> int:
    """클러스터별 Community 노드를 Neo4j 에 MERGE(멱등). 생성된 노드 수 반환.

    각 Community: {cluster_id, summary, size, members[corp_code], member_names[], edge_dist}.
    do_summary=False 면 결정론 폴백 요약만 채운다(LLM 미호출).
    """
    n = 0
    for entry in summary:
        cid = entry["cluster_id"]
        codes = by_cid.get(cid, [])
        member_names = [names.get(c, c) for c in codes]
        edge_dist = entry.get("edge_type_dist", {})
        if do_summary:
            text = build_summary(member_names, edge_dist)
        else:
            text = _fallback_summary(member_names, edge_dist)
        # 대표 멤버(차수 상위) 이름 — global_search 가 name 으로 쓰기 좋게 함께 저장.
        anchor_names = [a["name"] for a in entry.get("anchor_members", [])]
        s.run(
            "MERGE (c:Community {cluster_id: $cid}) "
            "SET c.summary = $summary, c.size = $size, "
            "    c.members = $members, c.member_names = $member_names, "
            "    c.anchor_names = $anchor_names, "
            "    c.edge_dist = $edge_dist",
            cid=cid,
            summary=text,
            size=entry["size"],
            members=codes,
            member_names=member_names,
            anchor_names=anchor_names,
            # Neo4j 노드 속성은 중첩 dict 불가 → JSON 문자열로 보관.
            edge_dist=json.dumps(edge_dist, ensure_ascii=False),
        )
        n += 1
        if cid < 5:
            print(f"  [community {cid}] {text[:60]}…")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="적재 없이 군집 통계만")
    ap.add_argument("--no-summary", action="store_true",
                    help="LLM 요약/Community 노드 생성 생략(통계만, 구 동작)")
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

        by_cid: dict[int, list[str]] = defaultdict(list)
        for code, cid in assign.items():
            by_cid[cid].append(code)

        if not args.dry_run:
            # 전체 초기화 후 재부여 (멱등 재계산)
            s.run("MATCH (o:Organization) WHERE o.cluster_id IS NOT NULL "
                  "REMOVE o.cluster_id")
            for cid, codes in by_cid.items():
                s.run(
                    "UNWIND $codes AS cc MATCH (o:Organization {corp_code: cc}) "
                    "SET o.cluster_id = $cid",
                    codes=codes, cid=cid,
                )
            n = s.run("MATCH (o:Organization) WHERE o.cluster_id IS NOT NULL "
                      "RETURN count(o) AS n").single()["n"]
            print(f"[load] cluster_id 부여: {n}사")

            # Community 노드 생성/갱신 (LLM 요약 포함, --no-summary 면 폴백만).
            if args.no_summary:
                print("[community] --no-summary: LLM 요약 생략, 폴백 요약만 적재")
            cn = upsert_communities(s, summary, names, by_cid,
                                    do_summary=not args.no_summary)
            print(f"[community] Community 노드 {cn}개 MERGE")

    drv.close()
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"요약 저장: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
