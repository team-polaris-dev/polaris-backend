"""DART 원본 정형 데이터(dart_raw_index) 직접 조회 — 접근 B.

fin_metric(재무수치)·document_index(공시메타) 외에, dart_raw_index 의 body_json(원본
DART API 응답)에만 있고 아직 검색에 안 쓰던 정형 데이터를 질의 시점에 파싱해 쓴다.
별도 적재 테이블을 만들지 않는다(접근 B). 앵커 회사(corp_code)로 한정해 최신 스냅샷만
취하고, 컨텍스트 폭주를 막기 위해 cap 을 둔다.

현재 노출: 타법인 출자현황(지분 관계) / 최대주주 현황 / 재무비율(주요지표).
모든 함수는 DB 미가용·파싱 실패 시 [] 로 degrade(파이프라인 보호).
"""
from __future__ import annotations

import json

from tool.rdb_client import mariadb_conn

# 회사당 노출 상한 — 답변 컨텍스트가 한 종류로 뒤덮이지 않게 한다.
_INVEST_CAP = 12
_SHAREHOLDER_CAP = 12
_INDICATOR_CAP = 18


def _to_float(s) -> float:
    """'12,345.6'·'(8,923)'(음수)·'-' 같은 DART 문자열을 정렬용 float 로. 실패 시 0."""
    t = str(s or "").strip().replace(",", "")
    if not t or t == "-":
        return 0.0
    neg = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        v = float(t)
        return -v if neg else v
    except ValueError:
        return 0.0


def _fetch_raw(corp_codes: list[str], endpoint: str) -> list[dict]:
    """dart_raw_index 에서 corp_codes×endpoint 행(최신 수집순)을 가져온다."""
    codes = [c for c in dict.fromkeys(corp_codes) if c]
    if not codes:
        return []
    ph = ", ".join(["%s"] * len(codes))
    # status 컬럼 수집 성공 표식은 'ok'(문자열) — DART API 의 '000'은 body_json 내부 값이다.
    sql = (
        "SELECT corp_code, rcept_no, body_json, collected_at "
        "FROM dart_raw_index "
        f"WHERE endpoint = %s AND status = 'ok' AND corp_code IN ({ph}) "
        "ORDER BY collected_at DESC"
    )
    try:
        with mariadb_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, (endpoint, *codes))
            return list(cur.fetchall())
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ [dart_extra] {endpoint} 조회 실패 → []: {e!r}")
        return []


def _iter_items(rows: list[dict]):
    """(row, item) 제너레이터 — 최신 스냅샷부터. body_json.list 의 각 항목을 흘린다.

    오염 방지: 항목 corp_code 가 행 corp_code 와 다르면 건너뛴다.
    """
    for r in rows:
        cc = str(r.get("corp_code") or "")
        try:
            data = json.loads(r.get("body_json") or "{}")
        except Exception:
            continue
        for it in data.get("list") or []:
            row_cc = str(it.get("corp_code") or "").strip()
            if row_cc and cc and row_cc != cc:
                continue
            yield r, it


def fetch_other_corp_investments(corp_codes: list[str], cap: int = _INVEST_CAP) -> list[dict]:
    """타법인 출자현황 — 'A사가 B사 지분 x%를 장부가 y원 보유'. 지분율 내림차순.

    반환: corp_code, rcept_no, target(피출자사), qota_rt(기말 지분율), book_amount(기말 장부가), purpose.
    회사+피출자사 기준 최신 1건만(연도 스냅샷 dedup).
    """
    rows = _fetch_raw(corp_codes, "otrCprInvstmntSttus")
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r, it in _iter_items(rows):
        cc = str(r.get("corp_code") or "")
        target = str(it.get("inv_prm") or "").strip()
        if not target or target == "-":
            continue
        key = (cc, target)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "corp_code": cc,
            "rcept_no": str(r.get("rcept_no") or ""),
            "target": target,
            "qota_rt": str(it.get("trmend_blce_qota_rt") or "").strip(),
            "book_amount": str(it.get("trmend_blce_acntbk_amount") or "").strip(),
            "purpose": str(it.get("invstmnt_purps") or "").strip(),
        })
    out.sort(key=lambda d: _to_float(d["qota_rt"]), reverse=True)
    return out[:cap]


def fetch_major_shareholders(corp_codes: list[str], cap: int = _SHAREHOLDER_CAP) -> list[dict]:
    """최대주주 현황 — 보유자·관계·지분율. 지분율 내림차순.

    반환: corp_code, rcept_no, holder(성명/법인), relate(관계), qota_rt(기말 지분율).
    회사+보유자 기준 최신 1건만.
    """
    rows = _fetch_raw(corp_codes, "hyslrSttus")
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r, it in _iter_items(rows):
        cc = str(r.get("corp_code") or "")
        holder = str(it.get("nm") or "").strip()
        if not holder or holder == "-":
            continue
        key = (cc, holder)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "corp_code": cc,
            "rcept_no": str(r.get("rcept_no") or ""),
            "holder": holder,
            "relate": str(it.get("relate") or "").strip(),
            "qota_rt": str(it.get("trmend_posesn_stock_qota_rt") or "").strip(),
        })
    out.sort(key=lambda d: _to_float(d["qota_rt"]), reverse=True)
    return out[:cap]


def fetch_financial_indicators(corp_codes: list[str], cap: int = _INDICATOR_CAP) -> list[dict]:
    """재무비율(주요지표) — 부채비율·수익성·성장성 등. 회사별 최신 회계연도만.

    반환: corp_code, rcept_no, bsns_year, name(지표명), value(값), category(지표 분류).
    회사별로 가장 최신 스냅샷의 bsns_year 에 한정하고 지표코드 기준 dedup.
    """
    rows = _fetch_raw(corp_codes, "fnlttSinglIndx")
    corp_year: dict[str, str] = {}      # 회사별 최신 회계연도 잠금(최신 스냅샷 우선)
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for r, it in _iter_items(rows):
        cc = str(r.get("corp_code") or "")
        byr = str(it.get("bsns_year") or "").strip()
        corp_year.setdefault(cc, byr)
        if byr != corp_year[cc]:
            continue
        name = str(it.get("idx_nm") or "").strip()
        value = str(it.get("idx_val") or "").strip()
        if not name or not value:
            continue
        code = str(it.get("idx_code") or "").strip() or name
        key = (cc, code)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "corp_code": cc,
            "rcept_no": str(r.get("rcept_no") or ""),
            "bsns_year": byr,
            "name": name,
            "value": value,
            "category": str(it.get("idx_cl_nm") or "").strip(),
        })
    return out[:cap]
