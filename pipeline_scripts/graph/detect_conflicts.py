"""그래프 모순 검출 배치 (읽기 전용) — 검토 큐 출력.

검출 3종:
  1) 양방향 SUPPLIES_TO — A→B 와 B→A 가 동시에 존재 (방향 노이즈, QC ⑥ 위반)
  2) self-loop — 시작=도착 같은 Organization (QC ⑦ 위반)
  3) 원장-그래프 방향 충돌 — extraction_provenance 에는 A→B 인데 그래프에는
     B→A 만 존재 (q095 실측 사례: 06-05 그래프 정리분이 불변 원장에 잔존.
     원장 직답 금지 — 정리 반영분은 그래프가 SSOT 라는 규약의 모니터링).

출력: 콘솔 요약 + conflicts_queue.json (검토 큐).
이 결과는 score 4축의 consistency 입력으로도 사용 가능
(충돌 엣지 = consistency 0.5, langgraph_app/graph/score.py 참조).

사용:
  cd db && uv run python graph/detect_conflicts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from db import mariadb_conn, neo4j_driver  # noqa: E402

OUT_PATH = HERE / "conflicts_queue.json"


def graph_supply_pairs(s) -> set[tuple[str, str]]:
    """그래프의 SUPPLIES_TO (from_id, to_id). id = corp_code 우선, 없으면 er_name."""
    pairs: set[tuple[str, str]] = set()
    for r in s.run(
        "MATCH (a:Organization)-[:SUPPLIES_TO]->(b:Organization) "
        "RETURN coalesce(a.corp_code, a.er_name) AS f, "
        "       coalesce(b.corp_code, b.er_name) AS t"
    ):
        if r["f"] and r["t"]:
            pairs.add((r["f"], r["t"]))
    return pairs


def main() -> int:
    queue: list[dict] = []
    d = neo4j_driver()
    with d.session() as s:
        # 1) 양방향 SUPPLIES_TO
        rows = s.run(
            "MATCH (a:Organization)-[r1:SUPPLIES_TO]->(b:Organization) "
            "MATCH (b)-[r2:SUPPLIES_TO]->(a) "
            "WHERE elementId(a) < elementId(b) "
            "RETURN a.name AS a, b.name AS b, "
            "       coalesce(a.corp_code, a.er_name) AS a_id, "
            "       coalesce(b.corp_code, b.er_name) AS b_id, "
            "       r1.chunk_id AS fwd_chunk, r2.chunk_id AS rev_chunk, "
            "       r1.source AS fwd_src, r2.source AS rev_src"
        ).data()
        for r in rows:
            queue.append({"kind": "bidirectional_supplies", **r,
                          "조치": "본문(chunk) 확인 후 한 방향 삭제 — event 보강(source=event:*)이 있으면 그쪽이 우선"})
        print(f"[1] 양방향 SUPPLIES_TO: {len(rows)}건")

        # 2) self-loop (모든 관계타입)
        rows = s.run(
            "MATCH (a:Organization)-[r]->(a) "
            "RETURN a.name AS org, type(r) AS rel, r.chunk_id AS chunk_id"
        ).data()
        for r in rows:
            queue.append({"kind": "self_loop", **r, "조치": "삭제 (QC ⑦)"})
        print(f"[2] self-loop: {len(rows)}건")

        # 3) 원장-그래프 방향 충돌
        gpairs = graph_supply_pairs(s)
    d.close()

    conn = mariadb_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT subject_id, predicate, object_id, chunk_id, rcept_no "
            "FROM extraction_provenance WHERE predicate='SUPPLIES_TO'"
        )
        prov = cur.fetchall()
    finally:
        conn.close()

    n_dir = 0
    for row in prov:
        # DictCursor 가 아닐 수 있어 양쪽 지원
        if isinstance(row, dict):
            subj, obj = row["subject_id"], row["object_id"]
            chunk, rcept = row["chunk_id"], row["rcept_no"]
        else:
            subj, _, obj, chunk, rcept = row
        fwd = (subj, obj)
        rev = (obj, subj)
        if fwd not in gpairs and rev in gpairs:
            n_dir += 1
            queue.append({
                "kind": "ledger_graph_direction_conflict",
                "ledger": f"{subj} -> {obj}", "graph": f"{obj} -> {subj}",
                "chunk_id": chunk, "rcept_no": rcept,
                "조치": "그래프가 SSOT(정리 반영분) — 원장 직답 금지 확인. 본문 재확인 시 chunk_id 사용",
            })
    print(f"[3] 원장-그래프 방향 충돌: {n_dir}건 (원장 SUPPLIES_TO {len(prov)}건 중)")

    OUT_PATH.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n검토 큐 {len(queue)}건 → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
