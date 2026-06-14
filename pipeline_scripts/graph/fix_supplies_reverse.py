"""SUPPLIES_TO 역방향(양방향) 노이즈 정리 — dry-run 우선 워크플로.

Default (dry-run):
  uv run python graph/fix_supplies_reverse.py

Apply:
  uv run python graph/fix_supplies_reverse.py --apply

검토된 좁은 룰만 처리 (NOISE_TODO·03_neo4j.md §7 QC⑥ 연장):
- RULE A: 양방향 쌍에서 한쪽이 이벤트 공시 정형 보강(source='event:*' 또는
  revenue_exposure_pct 보유)이면 그 방향이 진실 — 반대 방향 삭제.
  양쪽 다 이벤트 보강이면 충돌로 보고만 하고 건드리지 않음.
- RULE B: 검토된 장비 제조사(TRUSTED_SUPPLIER_CORPS)가 낀 양방향 쌍은
  제조사→상대 방향이 진실(장비사의 고객이 OSAT/IDM) — 상대→제조사 삭제.
- 그 외 양방향 쌍·단방향 인바운드는 보고만(수동 검토 대상).

extraction_provenance 원장은 불변 이력이라 건드리지 않는다(2026-06-05 전례,
GRAPH_VALUE_REPORT §3-3 — 정리 반영분은 그래프가 SSOT).
결과는 graph/supplies_reverse_cleanup.json 에 기록.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

# Windows 콘솔(cp949)에서도 한글·em-dash 깨지지 않게 stdout UTF-8 강제
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from db import neo4j_driver  # noqa: E402

OUT_PATH = HERE / "supplies_reverse_cleanup.json"

# corp_code -> 사유. 이 회사가 낀 양방향 쌍은 "이 회사 → 상대" 방향만 진실.
TRUSTED_SUPPLIER_CORPS = {
    "00161383": "한미반도체 — 후공정 장비 제조사. OSAT/IDM/메모리사는 전부 고객(매출처)이지 공급사가 아님",
}


def is_event_backed(rec: dict[str, Any], prefix: str) -> bool:
    src = rec.get(f"{prefix}_src") or ""
    pct = rec.get(f"{prefix}_pct")
    return src.startswith("event:") or pct is not None


def fetch_bidirectional(session) -> list[dict[str, Any]]:
    return session.run(
        """
        MATCH (a:Organization)-[r1:SUPPLIES_TO]->(b:Organization)
        MATCH (b)-[r2:SUPPLIES_TO]->(a)
        WHERE elementId(a) < elementId(b)
        RETURN a.name AS a_name, a.corp_code AS a_code,
               b.name AS b_name, b.corp_code AS b_code,
               elementId(r1) AS ab_id, r1.source AS ab_src,
               r1.revenue_exposure_pct AS ab_pct, r1.extracted_by AS ab_by,
               r1.chunk_id AS ab_chunk,
               elementId(r2) AS ba_id, r2.source AS ba_src,
               r2.revenue_exposure_pct AS ba_pct, r2.extracted_by AS ba_by,
               r2.chunk_id AS ba_chunk
        ORDER BY a_name, b_name
        """
    ).data()


def fetch_self_loops(session) -> list[dict[str, Any]]:
    return session.run(
        "MATCH (a:Organization)-[r:SUPPLIES_TO]->(a) "
        "RETURN a.name AS name, a.corp_code AS code, elementId(r) AS rel_id, "
        "       r.extracted_by AS by, r.chunk_id AS chunk_id"
    ).data()


def fetch_inbound_to_trusted(session) -> list[dict[str, Any]]:
    """검토 제조사로 들어오는 인바운드 전체 (양방향 여부 표시)."""
    return session.run(
        """
        MATCH (x:Organization)-[r:SUPPLIES_TO]->(t:Organization)
        WHERE t.corp_code IN $codes
        OPTIONAL MATCH (t)-[rev:SUPPLIES_TO]->(x)
        RETURN t.corp_code AS trusted_code, t.name AS trusted_name,
               x.name AS counter_name, x.corp_code AS counter_code,
               elementId(r) AS rel_id, r.source AS src,
               r.revenue_exposure_pct AS pct, r.extracted_by AS by,
               r.chunk_id AS chunk_id,
               rev IS NOT NULL AS has_reverse
        ORDER BY counter_name
        """,
        codes=list(TRUSTED_SUPPLIER_CORPS),
    ).data()


def build_plan(session) -> dict[str, Any]:
    bidir = fetch_bidirectional(session)
    deletions: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    review_only: list[dict[str, Any]] = []

    for rec in bidir:
        ab_event = is_event_backed(rec, "ab")
        ba_event = is_event_backed(rec, "ba")
        pair = f"{rec['a_name']} <-> {rec['b_name']}"
        if ab_event and ba_event:
            conflicts.append({"pair": pair, **rec, "reason": "both_event_backed"})
            continue
        if ab_event:
            deletions.append({"rel_id": rec["ba_id"], "pair": pair,
                              "delete_dir": f"{rec['b_name']} -> {rec['a_name']}",
                              "keep_dir": f"{rec['a_name']} -> {rec['b_name']}",
                              "rule": "A_event_backed", "deleted_by": rec["ba_by"],
                              "deleted_chunk": rec["ba_chunk"]})
            continue
        if ba_event:
            deletions.append({"rel_id": rec["ab_id"], "pair": pair,
                              "delete_dir": f"{rec['a_name']} -> {rec['b_name']}",
                              "keep_dir": f"{rec['b_name']} -> {rec['a_name']}",
                              "rule": "A_event_backed", "deleted_by": rec["ab_by"],
                              "deleted_chunk": rec["ab_chunk"]})
            continue
        if rec["a_code"] in TRUSTED_SUPPLIER_CORPS:
            deletions.append({"rel_id": rec["ba_id"], "pair": pair,
                              "delete_dir": f"{rec['b_name']} -> {rec['a_name']}",
                              "keep_dir": f"{rec['a_name']} -> {rec['b_name']}",
                              "rule": "B_trusted_supplier", "deleted_by": rec["ba_by"],
                              "deleted_chunk": rec["ba_chunk"]})
            continue
        if rec["b_code"] in TRUSTED_SUPPLIER_CORPS:
            deletions.append({"rel_id": rec["ab_id"], "pair": pair,
                              "delete_dir": f"{rec['a_name']} -> {rec['b_name']}",
                              "keep_dir": f"{rec['b_name']} -> {rec['a_name']}",
                              "rule": "B_trusted_supplier", "deleted_by": rec["ab_by"],
                              "deleted_chunk": rec["ab_chunk"]})
            continue
        review_only.append({"pair": pair, **rec, "reason": "no_rule_manual_review"})

    return {
        "bidirectional_total": len(bidir),
        "deletions": deletions,
        "conflicts": conflicts,
        "review_only": review_only,
        "self_loops": fetch_self_loops(session),
        "inbound_to_trusted": fetch_inbound_to_trusted(session),
    }


def apply_plan(session, plan: dict[str, Any]) -> dict[str, int]:
    deleted = 0
    for d in plan["deletions"]:
        result = session.run(
            "MATCH ()-[r:SUPPLIES_TO]->() WHERE elementId(r)=$rid "
            "WITH r, count(r) AS c DELETE r RETURN c",
            rid=d["rel_id"]).single()
        deleted += int(result["c"] if result else 0)
    loop_deleted = 0
    for sl in plan["self_loops"]:
        result = session.run(
            "MATCH ()-[r:SUPPLIES_TO]->() WHERE elementId(r)=$rid "
            "WITH r, count(r) AS c DELETE r RETURN c",
            rid=sl["rel_id"]).single()
        loop_deleted += int(result["c"] if result else 0)
    # RULE C: 검토 제조사(순수 공급사)로 들어오는 모든 SUPPLIES_TO는 방향노이즈.
    # 03_neo4j.md §269 "소부장·공급사 → 대기업이 정방향". 한미반도체는 후공정 장비사라
    # OSAT/IDM/메모리사가 전부 고객 — inbound 엣지는 ER 미병합 변형(Micron vs Micron
    # Technology)으로 양방향 쿼리에 안 잡힌 standalone 역엣지까지 포함해 전부 제거.
    inbound_deleted = 0
    for ib in plan["inbound_to_trusted"]:
        result = session.run(
            "MATCH ()-[r:SUPPLIES_TO]->() WHERE elementId(r)=$rid "
            "WITH r, count(r) AS c DELETE r RETURN c",
            rid=ib["rel_id"]).single()
        inbound_deleted += int(result["c"] if result else 0)
    return {"reverse_deleted": deleted, "self_loop_deleted": loop_deleted,
            "inbound_to_trusted_deleted": inbound_deleted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="apply deletions")
    args = parser.parse_args()

    driver = neo4j_driver()
    try:
        with driver.session() as session:
            plan = build_plan(session)
            out: dict[str, Any] = {
                "mode": "apply" if args.apply else "dry_run",
                "ts": datetime.now(timezone.utc).isoformat(),
                "plan": plan,
            }
            if args.apply:
                out["applied"] = apply_plan(session, plan)
                out["post_check"] = {
                    "bidirectional_remaining": len(fetch_bidirectional(session)),
                    "self_loops_remaining": len(fetch_self_loops(session)),
                }
    finally:
        driver.close()

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "mode": out["mode"],
        "bidirectional_total": plan["bidirectional_total"],
        "delete_planned": len(plan["deletions"]),
        "conflicts": len(plan["conflicts"]),
        "review_only": len(plan["review_only"]),
        "self_loops": len(plan["self_loops"]),
        "inbound_to_trusted": len(plan["inbound_to_trusted"]),
        **out.get("applied", {}),
        **out.get("post_check", {}),
        "detail_file": str(OUT_PATH),
    }, ensure_ascii=False, indent=2))
    if not args.apply:
        print("\n[dry-run] 삭제 예정 목록:")
        for d in plan["deletions"]:
            print(f"  DEL {d['delete_dir']}  (rule={d['rule']}, keep={d['keep_dir']})")
        for c in plan["conflicts"]:
            print(f"  CONFLICT {c['pair']} — 양쪽 다 이벤트 보강, 수동 검토")
        for r in plan["review_only"][:20]:
            print(f"  REVIEW {r['pair']} — 룰 없음, 수동 검토")


if __name__ == "__main__":
    main()
