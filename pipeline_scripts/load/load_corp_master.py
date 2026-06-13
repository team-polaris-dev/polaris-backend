"""DART 전체 기업코드 → corp_master 적재 (멱등 upsert).

관리자 콘솔의 "회사 검색"이 보유/미보유와 무관하게 전체 상장사에서 검색할 수
있도록, DART corpCode.xml(전체 공시대상 회사)을 받아 **상장사만**(stock_code
보유, ~2,800사) corp_master 테이블에 넣는다. 비상장 포함 전체는 10만+ 건이라
검색 노이즈가 심해 제외 — 필요해지면 --include-unlisted 로 확장.

사용 (pipeline_scripts 에서):
    uv run python load/load_corp_master.py
재실행 안전: ON DUPLICATE KEY UPDATE.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import mariadb_conn  # noqa: E402

CORPCODE_URL = "https://opendart.fss.or.kr/api/corpCode.xml"

# COLLATE 명시 금지 — 기존 테이블(document_index 등)과 JOIN 시 collation 불일치
# (1267 Illegal mix) 가 나므로 DB 기본 콜레이션을 그대로 상속한다.
SCHEMA = """
CREATE TABLE IF NOT EXISTS corp_master (
  corp_code   CHAR(8)      NOT NULL PRIMARY KEY,
  corp_name   VARCHAR(255) NOT NULL,
  stock_code  CHAR(6)      NULL,
  modify_date CHAR(8)      NULL,
  INDEX idx_corp_master_name (corp_name)
) CHARACTER SET utf8mb4
"""


def dart_api_key() -> str:
    key = os.getenv("DART_API_KEY", "")
    if not key:
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("DART_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise SystemExit("DART_API_KEY 없음 (.env 또는 환경변수)")
    return key


def fetch_corp_list(include_unlisted: bool) -> list[tuple[str, str, str | None, str | None]]:
    r = httpx.get(CORPCODE_URL, params={"crtfc_key": dart_api_key()}, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        xml_bytes = zf.read(zf.namelist()[0])
    root = ET.fromstring(xml_bytes)
    rows = []
    for el in root.iter("list"):
        code = (el.findtext("corp_code") or "").strip()
        name = (el.findtext("corp_name") or "").strip()
        stock = (el.findtext("stock_code") or "").strip() or None
        mdate = (el.findtext("modify_date") or "").strip() or None
        if not code or not name:
            continue
        if not include_unlisted and not stock:
            continue
        rows.append((code, name, stock, mdate))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-unlisted", action="store_true",
                    help="비상장 포함 전체 적재 (10만+ 건)")
    args = ap.parse_args()

    rows = fetch_corp_list(args.include_unlisted)
    print(f"DART corpCode 파싱: {len(rows)}건 "
          f"({'전체' if args.include_unlisted else '상장사만'})")

    conn = mariadb_conn()
    try:
        cur = conn.cursor()
        cur.execute(SCHEMA)
        sql = ("INSERT INTO corp_master (corp_code, corp_name, stock_code, modify_date) "
               "VALUES (%s,%s,%s,%s) "
               "ON DUPLICATE KEY UPDATE corp_name=VALUES(corp_name), "
               "stock_code=VALUES(stock_code), modify_date=VALUES(modify_date)")
        for i in range(0, len(rows), 500):
            cur.executemany(sql, rows[i:i + 500])
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM corp_master")
        print(f"corp_master 총 {cur.fetchone()[0]}건 적재 완료")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
