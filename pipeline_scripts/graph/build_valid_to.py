"""정형 엣지 시점 마감(valid_to) 배치 — 03_neo4j.md §2-1 †.

문제: 연도별 보고서를 합산 적재해 전임 임원·매각 지분 엣지가 영구 잔존
(SQL ablation q018 'SK실트론 대표=이용욱(전임)' 오답의 원인).

룰: 회사별·관계타입별 최신 rcept_no 를 기준으로, 그보다 오래된 rcept_no 를 가진
엣지(= 최신 공시에서 사라진 사실)에 valid_to = 최신 rcept_no 의 YYYYMMDD 부여.

멱등: valid_to IS NULL 인 엣지만 갱신. 되돌리기: REMOVE r.valid_to.
질의 규약: 현재 사실 = WHERE r.valid_to IS NULL / 이력 = 필터 없이.

────────────────────────────────────────────────────────────────────────
⚠ 2026-06-11 dry-run 실측: 현 데이터에는 적용 금지 (DO NOT --apply).
  룰의 전제 = "회사별 최신 rcept_no 보고서가 전체 사실을 재기재" 인데, 실데이터에서
  한 회사가 사업보고서·반기보고서·출자현황(otrCprInvstmntSttus) 등 **여러 보고서
  유형**을 서로 다른 rcept_no 로 적재한다. 예) SK(주) INVESTS_IN:
    2024-08:18건 / 2025-08:104건 / 2026-03:3건  → 최신(3건)이 부분목록.
  그 결과 dry-run 에서 INVESTS_IN 1,222건 중 918건(75%)이 만료 처리됨 → 적용 시
  affiliates_fin 멀티홉(핵심 가치)이 붕괴. EXECUTIVE_OF 399·주주 70 도 같은 위험.
  엣지는 (subject,object) 당 1개뿐(중복 0)이라 "동일 트리플 재적재" 안전 마감도 0건.

  올바른 해법(데이터/로더 작업, B 범주): 정형 엣지에 reprt_code(+bsns_year)를
  스탬프 → 같은 회사의 동일 reprt_code('11011' 연간) 보고서 간에서만 최신 비교.
  그 전까지 본 배치의 write 는 봉인. 질의 측 `WHERE r.valid_to IS NULL` 필터는
  valid_to 가 0건이면 전부 통과(무해 no-op)라 그대로 둬도 안전하다.
────────────────────────────────────────────────────────────────────────

사용:
  cd db && uv run python graph/build_valid_to.py            # dry-run(기본, 안전)
  cd db && uv run python graph/build_valid_to.py --apply    # 실제 write (현재 봉인 — 위 경고)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from db import neo4j_driver  # noqa: E402


# (관계타입, 기준 Org 방향) — 기준 Org = 해당 공시를 제출한 회사
#   EXECUTIVE_OF / IS_MAJOR_SHAREHOLDER_OF: 공시 제출사 = 도착(target) Org
#   INVESTS_IN: 공시 제출사(otrCprInvstmntSttus) = 출발(source) Org
RULES = (
    ("EXECUTIVE_OF", "target"),
    ("IS_MAJOR_SHAREHOLDER_OF", "target"),
    ("INVESTS_IN", "source"),
)


def _cypher(rel: str, anchor: str, dry: bool) -> str:
    org_pat = (
        f"()-[r:{rel}]->(o:Organization)" if anchor == "target"
        else f"(o:Organization)-[r:{rel}]->()"
    )
    head = (
        f"MATCH {org_pat} "
        "WHERE r.rcept_no IS NOT NULL AND o.corp_code IS NOT NULL "
        "WITH o, max(r.rcept_no) AS latest "
        f"MATCH {org_pat} "
        "WHERE r.rcept_no IS NOT NULL AND r.rcept_no < latest "
        "  AND r.valid_to IS NULL "
    )
    if dry:
        return head + "RETURN count(r) AS n"
    return head + (
        "SET r.valid_to = substring(latest, 0, 8), "
        "    r.valid_to_by = 'build_valid_to:superseded' "
        "RETURN count(r) AS n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    # 기본은 dry-run(안전). 실제 write 는 --apply 명시 강제 (위 ⚠ 경고 참조).
    ap.add_argument("--apply", action="store_true",
                    help="실제 valid_to write (현재 데이터에선 봉인 — docstring ⚠)")
    args = ap.parse_args()
    dry = not args.apply
    if not dry:
        print("⚠ --apply 모드: docstring 경고 확인 — 현 데이터(다중 보고서유형)에선 "
              "과잉 마감 위험. 그래도 진행합니다.\n")

    d = neo4j_driver()
    total = 0
    with d.session() as s:
        for rel, anchor in RULES:
            n = s.run(_cypher(rel, anchor, dry)).single()["n"]
            total += n
            tag = "대상" if dry else "마감"
            print(f"[{rel:26s}] {tag} {n}건")
        # 검증: 회사별 현재 유효 대표이사 수 분포 (마감 후 1명 수렴 기대)
        if not dry:
            rows = s.run(
                "MATCH (p:Person)-[r:EXECUTIVE_OF]->(o:Organization) "
                "WHERE r.valid_to IS NULL AND r.ofcps CONTAINS '대표이사' "
                "RETURN o.name AS org, count(DISTINCT p) AS ceos "
                "ORDER BY ceos DESC LIMIT 10"
            ).data()
            print("\n[검증] 유효(valid_to IS NULL) 대표이사 수 상위 10사:")
            for r in rows:
                print(f"  {r['org']}: {r['ceos']}명")
    d.close()
    print(f"\n합계 {total}건 ({'dry-run' if dry else '적용'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
