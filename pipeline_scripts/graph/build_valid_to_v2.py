"""정형 엣지 시점 마감(valid_to) v2 — SSOT(dart_raw_index) 디프 방식. 03_neo4j.md §2-1 †.

v1 의 결함(2026-06-11 dry-run 실측): 엣지의 rcept_no 는 MERGE last-write-wins 라
"이 사실이 실린 최신 공시"가 아님 → 회사별 max(rcept_no) 비교는 과잉 마감
(INVESTS_IN 75% 오마감 위험). v2 는 원본 JSON 으로 보고서별 사실집합을 복원해 디프:

  1. dart_raw_index 의 exctvSttus(임원)·hyslrSttus(주주)·otrCprInvstmntSttus(출자)
     body_json 을 행 단위 rcept_no 로 그룹 → 보고서별 완전한 스냅샷 (MERGE 오염 없음)
  2. 회사·엔드포인트별 최신 rcept_no 의 집합 = "현재 사실"
  3. Neo4j 엣지 중 (rcept_no < 최신) AND (상대 키가 최신 집합에 없음) → valid_to 부여
     키: 임원 = person_id(corp|nm|birth_ym) 재계산(로더와 동일) / 주주·출자 = normalize_corp_name

안전장치:
  - 최신 집합이 빈 회사·엔드포인트 → 스킵
  - 최신 집합 크기 < 전체 합집합의 50% → 자동 마감 대신 검토 큐(valid_to_review.json)
  - 엣지 rcept_no == 최신 rcept → 무조건 유지 (정규화 미스매치 방어)
  - dry-run 기본, --apply 명시 시만 write. 멱등(valid_to IS NULL 만). 되돌리기: REMOVE r.valid_to

사용:
  cd db && uv run python graph/build_valid_to_v2.py            # dry-run
  cd db && uv run python graph/build_valid_to_v2.py --apply    # 적용
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from db import mariadb_conn, neo4j_driver, normalize_corp_name, person_id  # noqa: E402

REVIEW_PATH = HERE / "valid_to_review.json"

HOLDER_ARTIFACTS = {"계", "소계", "합계", "보통주합계", "우선주합계", "총계",
                    "합 계", "소 계", "기타", "기타주주", "소액주주"}


# ── 1) 원본 로드: (corp, endpoint) → {rcept_no: [row...]} ─────────────
def load_raw(conn, endpoint: str) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT corp_code, body_json FROM dart_raw_index WHERE endpoint=%s",
            (endpoint,),
        )
        for corp, body in cur.fetchall():
            try:
                obj = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                continue
            for row in obj.get("list") or []:
                if not isinstance(row, dict):
                    continue
                rc = (row.get("rcept_no") or "").strip()
                if rc:
                    out[corp][rc].append(row)
    return out


# ── 2) 보고서별 키 집합 ───────────────────────────────────────────────
def exec_keys(corp: str, rows: list[dict]) -> set[str]:
    ks = set()
    for r in rows:
        nm = (r.get("nm") or "").strip()
        if nm:
            ks.add(person_id(corp, nm, (r.get("birth_ym") or "").strip()))
    return ks


def holder_keys(_corp: str, rows: list[dict]) -> set[str]:
    ks = set()
    for r in rows:
        nm = (r.get("nm") or "").strip()
        if nm and nm not in HOLDER_ARTIFACTS and len(nm) >= 2:
            ks.add(normalize_corp_name(nm))
    return ks


def invest_keys(_corp: str, rows: list[dict]) -> set[str]:
    ks = set()
    for r in rows:
        nm = (r.get("inv_prm") or "").strip()
        if nm and nm not in HOLDER_ARTIFACTS and len(nm) >= 2:
            ks.add(normalize_corp_name(nm))
    return ks


# ── 3) 관계별 그래프 엣지 조회 + 키 추출 ─────────────────────────────
# (endpoint, rel, key_fn, fetch_cypher) — fetch 는 corp 당 엣지의 (eid, rcept, key재료)
SPECS = [
    ("exctvSttus", "EXECUTIVE_OF", exec_keys,
     "MATCH (p:Person)-[r:EXECUTIVE_OF]->(o:Organization {corp_code:$cc}) "
     "RETURN elementId(r) AS eid, r.rcept_no AS rc, p.person_id AS k"),
    ("hyslrSttus", "IS_MAJOR_SHAREHOLDER_OF", holder_keys,
     "MATCH (x)-[r:IS_MAJOR_SHAREHOLDER_OF]->(o:Organization {corp_code:$cc}) "
     "RETURN elementId(r) AS eid, r.rcept_no AS rc, "
     "       coalesce(x.name, x.er_name, x.corp_code) AS k"),
    ("otrCprInvstmntSttus", "INVESTS_IN", invest_keys,
     "MATCH (o:Organization {corp_code:$cc})-[r:INVESTS_IN]->(t) "
     "RETURN elementId(r) AS eid, r.rcept_no AS rc, "
     "       coalesce(t.name, t.er_name, t.corp_code) AS k"),
]


def edge_key(rel: str, raw_k: str) -> str:
    """그래프 측 키 정규화 — 임원은 person_id 그대로, 나머지는 회사명 정규화."""
    if rel == "EXECUTIVE_OF":
        return raw_k or ""
    return normalize_corp_name(raw_k or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제 valid_to write")
    args = ap.parse_args()
    dry = not args.apply

    conn = mariadb_conn()
    drv = neo4j_driver()
    review: list[dict] = []
    grand_total = 0

    with drv.session() as s:
        for endpoint, rel, key_fn, fetch_cy in SPECS:
            raw = load_raw(conn, endpoint)
            n_close = n_keep = n_skip_corp = 0
            for corp, by_rcept in raw.items():
                latest_rc = max(by_rcept)
                latest_set = key_fn(corp, by_rcept[latest_rc])
                union_set = set()
                for rows in by_rcept.values():
                    union_set |= key_fn(corp, rows)
                # 안전장치 1·2
                if not latest_set:
                    n_skip_corp += 1
                    continue
                if union_set and len(latest_set) < 0.5 * len(union_set):
                    review.append({
                        "endpoint": endpoint, "corp_code": corp,
                        "latest_rcept": latest_rc,
                        "latest_n": len(latest_set), "union_n": len(union_set),
                        "사유": "최신 집합이 합집합의 50% 미만 — 부분 보고서 의심, 자동마감 보류",
                    })
                    n_skip_corp += 1
                    continue

                edges = s.run(fetch_cy, cc=corp).data()
                to_close: list[str] = []
                for e in edges:
                    rc = e.get("rc") or ""
                    k = edge_key(rel, e.get("k") or "")
                    if rc == latest_rc or (k and k in latest_set):
                        n_keep += 1
                    elif rc and rc < latest_rc:
                        to_close.append(e["eid"])
                    else:
                        n_keep += 1  # rc 가 최신보다 새롭거나 없음 — 보수적으로 유지
                n_close += len(to_close)
                if to_close and not dry:
                    s.run(
                        f"MATCH ()-[r:{rel}]->() "
                        "WHERE elementId(r) IN $eids AND r.valid_to IS NULL "
                        "SET r.valid_to = substring($rc, 0, 8), "
                        "    r.valid_to_by = 'build_valid_to_v2:ssot_diff'",
                        eids=to_close, rc=latest_rc,
                    )
            grand_total += n_close
            tag = "대상" if dry else "마감"
            print(f"[{rel:26s}] {tag} {n_close}건 / 유지 {n_keep}건 / 스킵회사 {n_skip_corp}")

        # 표적 검증 — q018: SK실트론(00138020) 대표이사 유효분
        rows = s.run(
            "MATCH (p:Person)-[r:EXECUTIVE_OF]->(o:Organization {corp_code:'00138020'}) "
            "WHERE r.ofcps CONTAINS '대표이사' "
            "RETURN p.name AS nm, r.rcept_no AS rc, r.valid_to AS vto"
        ).data()
        print("\n[표적검증 q018] SK실트론 대표이사 엣지:")
        for r in rows:
            print(f"  {r['nm']}  rcept={r['rc']}  valid_to={r['vto']}")

        if not dry:
            ceo = s.run(
                "MATCH (p:Person)-[r:EXECUTIVE_OF]->(o:Organization) "
                "WHERE r.valid_to IS NULL AND r.ofcps CONTAINS '대표이사' "
                "WITH o, count(DISTINCT p) AS c WHERE c > 1 "
                "RETURN count(o) AS multi"
            ).single()["multi"]
            print(f"\n[검증] 유효 대표이사 2명 이상인 회사: {ceo} (각자대표 제외하면 0 수렴 기대)")

    conn.close()
    drv.close()
    if review:
        REVIEW_PATH.write_text(json.dumps(review, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\n검토 큐 {len(review)}건 → {REVIEW_PATH}")
    print(f"\n합계 {grand_total}건 ({'dry-run' if dry else '적용'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
