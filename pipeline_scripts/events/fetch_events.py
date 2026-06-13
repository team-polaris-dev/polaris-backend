"""POLARIS — 이벤트 공시(단일판매ㆍ공급계약) 수집·파싱.

흐름: document_index 의 '단일판매ㆍ공급계약체결' rcept_no → DART document.xml 원문(zip)
      → <td> 셀 양식 파싱(계약상대·계약금액·최근매출액·매출액대비) → dart_raw_index upsert
      (endpoint='event:단일판매공급계약').

설계 SSOT: docs/DBdocs/01_mariadb.md §2.1 (신규 테이블 없이 dart_raw_index 재사용),
           03_neo4j.md §SUPPLIES_TO 보강(계약상대 명시 건만 엣지화 — 별도 단계).

재사용(막 만들지 않음): fetch_dart.py 의 throttle/차단방지(DOC_INTERVAL·BLOCK_COOLDOWN),
                       graph/db.py 의 mariadb_conn.
멱등·resumable: 이미 적재된 rcept_no(hash8) 는 skip. 청킹·임베딩 안 함(양식 정형 직파싱).

실행: cd db && uv run python events/fetch_events.py --limit 5   (테스트)
      cd db && uv run python events/fetch_events.py             (전체 211건)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
DB_DIR = HERE.parent
sys.path.insert(0, str(DB_DIR))             # fetch_dart 재사용
sys.path.insert(0, str(DB_DIR / "graph"))   # db.mariadb_conn 재사용

from fetch_dart import (  # noqa: E402
    API_KEY, BASE, BLOCK_COOLDOWN, DOC_INTERVAL, throttle,
)
from db import mariadb_conn  # noqa: E402

# ── 이벤트 타입 레지스트리 ─────────────────────────────────────────
# mode='contract' = 단일판매계약 정밀 파싱 / mode='rows' = 라벨 뒤 4셀 배열(best-effort,
# 당해·직전·증감액·증감비율 등 컬럼 순서 케이스 다양 → _cells 보관으로 후속 정밀화 가능)
EVENT_TYPES: dict[str, dict] = {
    "단일판매공급계약": {"doc_like": "%단일판매%공급계약%", "mode": "contract"},
    "영업잠정실적": {"doc_like": "%영업%잠정%실적%", "mode": "rows",
                  "labels": ("매출액", "영업이익", "당기순이익")},
    "매출손익30변경": {"doc_like": "%30%변경%", "mode": "rows",
                   "labels": ("매출액", "영업이익", "당기순이익", "자본총계")},
}

# 계약상대 익명화 판별: 회사 접미사가 있으면 '명시'로 간주(거친 휴리스틱, 명시율 실측용)
NAME_SUFFIX = ("(주)", "㈜", "주식회사", "Co", "Ltd", "Inc", "Corp", "LLC", "GmbH")
ANON_HINT = ("지역", "선주", "고객사", "거래처", "발주처", "비공개", "영업비밀")


def sha1_8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _extract_cells(xml: str) -> list[str]:
    body = re.sub(r"<style.*?</style>", " ", xml, flags=re.S)
    raw = re.findall(r"<td[^>]*>(.*?)</td>", body, flags=re.S)
    cells = [re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", c)).strip() for c in raw]
    return [c for c in cells if c]


def parse_cells(xml: str, etype: str = "단일판매공급계약") -> dict:
    """이벤트 양식 <td> 셀 파싱.

    contract 모드(단일판매계약): 라벨 다음 1셀 정밀 추출.
      [기재정정] 공시는 본문 앞에 '정정사항' 표가 붙음 → 본문 양식 시작
      라벨('공급계약 구분')부터 슬라이스해 정정표 스킵.
    rows 모드(잠정실적·30%변경): 라벨 뒤 4셀 배열(당해·직전·증감액·증감비율
      등 — 컬럼 순서가 케이스별로 달라 best-effort, _cells 보관으로 후속 정밀화).
    """
    cells = _extract_cells(xml)
    cfg = EVENT_TYPES[etype]

    if cfg["mode"] == "contract":
        for i, c in enumerate(cells):
            if "공급계약 구분" in c:
                cells = cells[i:]
                break

        def after(keyword: str) -> str | None:
            for i, c in enumerate(cells):
                if keyword in c and i + 1 < len(cells):
                    return cells[i + 1]
            return None

        return {
            "계약상대": after("계약상대"),
            "계약금액": after("계약금액"),
            "최근매출액": after("최근매출액"),
            "매출액대비": after("매출액대비"),
            "계약명": after("체결계약명") or after("계약명"),
            "계약시작일": after("시작일"),
            "계약종료일": after("종료일"),
            "_cells": cells,
            "_cell_count": len(cells),
        }

    # rows 모드
    out: dict = {}
    for label in cfg["labels"]:
        for i, c in enumerate(cells):
            if c == label or c.startswith(label):
                out[label] = cells[i + 1 : i + 5]
                break
        else:
            out[label] = None
    out["_cells"] = cells
    out["_cell_count"] = len(cells)
    return out


def is_named(counterparty: str | None) -> bool:
    if not counterparty or counterparty in ("-", ""):
        return False
    if any(h in counterparty for h in ANON_HINT):
        return False
    return any(s in counterparty for s in NAME_SUFFIX)


def fetch_doc_xml(client: httpx.Client, rno: str) -> str | None:
    """document.xml(zip) → XML 텍스트. 차단의심 시 fetch_dart 정책대로 쿨다운."""
    for attempt in range(5):
        throttle(DOC_INTERVAL)
        try:
            r = client.get(f"{BASE}/document.xml",
                           params={"crtfc_key": API_KEY, "rcept_no": rno}, timeout=60)
            r.raise_for_status()
            raw = r.content
            if raw[:2] != b"PK":
                txt = r.text
                if "020" in txt or "021" in txt:
                    print(f"    … 한도 대기 60s @ {rno}", flush=True)
                    time.sleep(60)
                    continue
                print(f"    ! {rno} zip 아님(데이터 없음/오류)", flush=True)
                return None
            z = zipfile.ZipFile(io.BytesIO(raw))
            return z.read(z.namelist()[0]).decode("utf-8", errors="replace")
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
            print(f"    ! {rno} 차단 의심({type(e).__name__}) {BLOCK_COOLDOWN}s 대기", flush=True)
            time.sleep(BLOCK_COOLDOWN)
        except Exception as e:  # noqa: BLE001
            print(f"    ! {rno} 재시도 {attempt+1}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return None


UPSERT = (
    "INSERT INTO dart_raw_index "
    "(corp_code, endpoint, hash8, rcept_no, body_json, status, collected_at) "
    "VALUES (%s,%s,%s,%s,%s,%s,%s) "
    "ON DUPLICATE KEY UPDATE rcept_no=VALUES(rcept_no), body_json=VALUES(body_json), "
    "status=VALUES(status), collected_at=VALUES(collected_at)"
)


def run_type(etype: str, limit: int, force: bool) -> None:
    cfg = EVENT_TYPES[etype]
    endpoint = f"event:{etype}"
    conn = mariadb_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT rcept_no, corp_code FROM document_index "
        "WHERE doc_type LIKE %s AND corp_code IS NOT NULL ORDER BY rcept_no",
        (cfg["doc_like"],),
    )
    targets = cur.fetchall()
    if force:
        done: set[str] = set()
    else:
        cur.execute("SELECT hash8 FROM dart_raw_index WHERE endpoint=%s", (endpoint,))
        done = {r[0] for r in cur.fetchall()}
    todo = [(rno, cc) for rno, cc in targets if sha1_8(rno) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{etype}] 대상 {len(targets)}건 / 미수집 {len(todo)}건", flush=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) polaris-collector/0.1"
    ok = named = 0
    with httpx.Client(headers={"User-Agent": ua}, follow_redirects=True) as client:
        for i, (rno, cc) in enumerate(todo, 1):
            xml = fetch_doc_xml(client, rno)
            if xml is None:
                continue
            cells = parse_cells(xml, etype)
            cells["xml_len"] = len(xml)
            body = json.dumps({"rcept_no": rno, "cells": cells}, ensure_ascii=False)
            cur.execute(UPSERT, (cc, endpoint, sha1_8(rno), rno, body, "ok", now))
            conn.commit()
            ok += 1
            if cfg["mode"] == "contract":
                cp = cells.get("계약상대")
                if is_named(cp):
                    named += 1
                print(f"  [{i}/{len(todo)}] {rno} 상대={cp!r} "
                      f"매출대비={cells.get('매출액대비')!r}", flush=True)
            else:
                first = cfg["labels"][0]
                print(f"  [{i}/{len(todo)}] {rno} {first}={cells.get(first)!r}",
                      flush=True)
    if cfg["mode"] == "contract" and ok:
        print(f"[{etype}] 적재 {ok}건 / 계약상대 명시 {named}건 ({100*named/ok:.0f}%)",
              flush=True)
    else:
        print(f"[{etype}] 적재 {ok}건", flush=True)
    cur.close()
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", default="단일판매공급계약",
                    choices=[*EVENT_TYPES, "all"], help="이벤트 타입(all=전부)")
    ap.add_argument("--limit", type=int, default=0, help="테스트용 N건만")
    ap.add_argument("--force", action="store_true",
                    help="이미 적재된 건도 재수집·재파싱(파서 보정 반영용)")
    args = ap.parse_args()
    if not API_KEY:
        print("DART_API_KEY 없음 (.env 확인)")
        return 1
    types = list(EVENT_TYPES) if args.type == "all" else [args.type]
    for t in types:
        run_type(t, args.limit, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
