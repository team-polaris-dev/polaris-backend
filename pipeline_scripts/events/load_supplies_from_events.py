"""이벤트 단일판매계약 → SUPPLIES_TO `revenue_exposure_pct` 보강 (안전 정확매칭).

dart_raw_index(endpoint='event:단일판매공급계약') 의 계약상대를, 그래프 노드를 보유한
국내사로 **접미사 제거 후 정확일치**한 건만 매칭(자회사 오매칭 방지 — '삼성전자판매' 제외).
방향: 공시주체=공급사, 계약상대=수요사 → SUPPLIES_TO(공급사 → 수요사).
같은 (공급사,수요사) 쌍에 여러 계약이면 매출대비% 최대값을 대표로.

설계 SSOT: 03_neo4j.md §SUPPLIES_TO 보강(계약상대 명시·매칭 건만 엣지화).
가드: self-loop 금지 / 방향 고정 / 멱등(MERGE+SET). source='event:단일판매공급계약'.

실행: cd db && uv run python events/load_supplies_from_events.py
      cd db && uv run python events/load_supplies_from_events.py --dry  (미리보기)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "graph"))
from db import mariadb_conn, neo4j_driver  # noqa: E402

ENDPOINT = "event:단일판매공급계약"

# 그래프 노드 보유 국내사만 — canon(접미사·괄호·영문 제거) 정확일치 키
KNOWN: dict[str, str] = {
    "삼성전자": "00126380",
    "하이닉스": "00164779",
    "삼성디스플레이": "00912006",
}


def canon(name: str) -> str:
    """괄호내용·회사접미사·영문/숫자/공백 제거 → 한글 핵심명."""
    s = re.sub(r"\(.*?\)", "", name or "")
    s = re.sub(r"주식회사|㈜|\(주\)", "", s)
    s = re.sub(r"[A-Za-z0-9.,·\s]", "", s)
    return s.strip()


def to_float(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r"-?\d[\d,]*\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def buyer_cogs(cur, buyer_cc: str, year: int) -> tuple[float, int] | None:
    """수요사 COGS(연간·연결). 계약연도 없으면 전년 fallback. (값, 사용연도)."""
    for y in (year, year - 1):
        cur.execute(
            "SELECT value FROM fin_metric WHERE corp_code=%s "
            "AND account_id='ifrs-full_CostOfSales' AND bsns_year=%s "
            "AND reprt_code='11011' AND fs_div='CFS'",
            (buyer_cc, y),
        )
        row = cur.fetchone()
        if row and row[0]:
            return float(row[0]), y
    return None


def collect() -> dict[tuple[str, str], dict]:
    """(공급사 corp_code, 수요사 corp_code) → 대표 계약(최대 매출대비%) + 양방향 COGS 비중."""
    conn = mariadb_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT corp_code, rcept_no, body_json FROM dart_raw_index WHERE endpoint=%s",
        (ENDPOINT,),
    )
    rows = cur.fetchall()
    best: dict[tuple[str, str], dict] = {}
    for supplier_cc, rcept_no, body in rows:
        try:
            cells = json.loads(body).get("cells", {})
        except Exception:
            continue
        buyer_cc = KNOWN.get(canon(cells.get("계약상대") or ""))
        if not buyer_cc or buyer_cc == supplier_cc:  # 미매칭·self-loop 제외
            continue
        pct = to_float(cells.get("매출액대비"))
        amt = to_float(cells.get("계약금액"))
        key = (supplier_cc, buyer_cc)
        cur_best = best.get(key)
        if cur_best is None or (pct or 0) > (cur_best["pct"] or 0):
            best[key] = {"pct": pct, "amt": amt, "rcept_no": rcept_no}
    # Bloomberg 2단계: 계약금액 ÷ 수요사 COGS = 수요사 원가 중 이 계약 비중
    for (sup, buy), v in best.items():
        v["cogs_share_pct"] = None
        v["cogs_year"] = None
        if v["amt"]:
            year = int(v["rcept_no"][:4])
            cogs = buyer_cogs(cur, buy, year)
            if cogs:
                v["cogs_share_pct"] = round(v["amt"] / cogs[0] * 100, 2)
                v["cogs_year"] = cogs[1]
    cur.close()
    conn.close()
    return best


MERGE = """
MATCH (s:Organization {corp_code:$sup}), (b:Organization {corp_code:$buy})
MERGE (s)-[r:SUPPLIES_TO]->(b)
SET r.revenue_exposure_pct = $pct,
    r.contract_amount      = $amt,
    r.cogs_share_pct       = $cogs_share_pct,
    r.cogs_year            = $cogs_year,
    r.source               = 'event:단일판매공급계약',
    r.rcept_no             = coalesce(r.rcept_no, $rcept_no)
RETURN s.name AS sup, b.name AS buy
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="미리보기(쓰기 안 함)")
    args = ap.parse_args()

    pairs = collect()
    print(f"정확매칭 쌍 {len(pairs)}건")
    for (sup, buy), v in sorted(pairs.items(), key=lambda x: -(x[1]["pct"] or 0)):
        print(f"  {sup} → {buy}  매출대비={v['pct']}%  "
              f"수요사COGS중={v['cogs_share_pct']}%({v['cogs_year']})  {v['rcept_no']}")
    if args.dry:
        print("[dry] 그래프 쓰기 생략")
        return 0

    d = neo4j_driver()
    wrote = 0
    with d.session() as s:
        for (sup, buy), v in pairs.items():
            rec = s.run(MERGE, sup=sup, buy=buy, pct=v["pct"],
                        amt=v["amt"], cogs_share_pct=v["cogs_share_pct"],
                        cogs_year=v["cogs_year"], rcept_no=v["rcept_no"]).single()
            if rec:
                wrote += 1
                print(f"  ✓ {rec['sup']} → {rec['buy']} ({v['pct']}%)")
            else:
                print(f"  ! 노드 없음: {sup} → {buy} (그래프에 corp_code 미존재)")
    d.close()
    print(f"\nSUPPLIES_TO 보강 {wrote}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
