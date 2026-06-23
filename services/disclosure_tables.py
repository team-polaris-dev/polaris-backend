"""공시 한 건(rcept_no)의 표를 섹션별·표별 구조(헤더/행)로 복원한다.

DART pont AI 식 '보고서 표 → 엑셀' 기능의 데이터 소스. chunk_index 의 table_nl 청크
(공시 표를 '헤더: a | b\n행값 | k=v; …' 자연어로 인코딩한 것)를 파싱해 프론트가 표로
렌더하고 .xlsx 로 추출할 수 있는 JSON 으로 돌려준다.

한계(v1): (1) 표 소제목(가./나./다.)은 청킹 때 버려 섹션 단위 '표 N' 으로만 표시한다.
(2) chunk_id 가 해시라 섹션 내부 표 순서는 원문 순서와 다를 수 있다. 둘 다 청킹을
다시 돌려야 개선되므로 별도 작업으로 둔다.
"""
from __future__ import annotations

import re

from tool.rdb_client import mariadb_conn

_HEADER_RE = re.compile(r"^헤더:\s*(.+)$")
_UNIT_RE = re.compile(r"^\(단위\s*[:：]")
_ROMAN = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}
_ROMAN_RE = re.compile(r"^\s*([IVX]+)\.")


def _row_to_cells(line: str, cols: list[str]) -> list[str]:
    """데이터 줄 → 헤더 순서에 맞춘 셀 리스트.

    형식: '맨앞값 | 키=값; 키=값'  또는  '키=값; …'  또는  '맨앞값'(단일 라벨).
    맨앞값은 첫 열, 나머지는 '키=값'을 헤더명으로 매칭한다(serialize._parse_table_row 와 동일 규칙).
    """
    bare: str | None = None
    rest = line
    if " | " in line:
        bare, rest = line.split(" | ", 1)
    elif "=" not in line:
        bare, rest = line, ""  # 단일 라벨 행(중간 소제목 등)
    pairs: dict[str, str] = {}
    for kv in rest.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            pairs[k.strip()] = v.strip()
    cells: list[str] = []
    for idx, h in enumerate(cols):
        if idx == 0 and bare is not None:
            cells.append(bare.strip())
        else:
            cells.append(pairs.get(h, ""))
    return cells


def _parse_table_text(text: str) -> dict | None:
    """table_nl embedding_text → {unit, columns, rows}. 헤더/행이 없으면 None."""
    unit = ""
    cols: list[str] | None = None
    rows: list[list[str]] = []
    for raw in (text or "").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if cols is None and line.startswith("[") and line.endswith("]"):
            continue  # [회사 · 보고서 · 섹션] 머리말
        if cols is None and _UNIT_RE.match(line):
            unit = line
            continue
        hm = _HEADER_RE.match(line)
        if hm:
            cols = [c.strip() for c in hm.group(1).split("|") if c.strip()]
            continue
        if cols is not None:
            row = _row_to_cells(line, cols)
            if any(c for c in row):
                rows.append(row)
    if not cols or not rows:
        return None
    return {"unit": unit, "columns": cols, "rows": rows}


def _section_sort_key(sp: str) -> tuple:
    """섹션을 보고서 원문 순서(로마숫자 장 → 절 번호)로 정렬하기 위한 키."""
    m = _ROMAN_RE.match(sp or "")
    roman = _ROMAN.get(m.group(1), 99) if m else 99
    sub = 0
    if ">" in (sp or ""):
        m2 = re.search(r"(\d+)\s*\.", sp.split(">", 1)[1])
        if m2:
            sub = int(m2.group(1))
    return (roman, sub, sp or "")


def fetch_disclosure_tables(rcept_no: str) -> dict:
    """공시 한 건의 표를 섹션별로 묶어 반환.

    반환: {rcept_no, corp_name, title, date, sections:[{section_path, tables:[{caption,unit,columns,rows}]}]}
    """
    rcept_no = re.sub(r"[^0-9]", "", rcept_no or "")  # 14자리 숫자만(주입 방어)
    if not rcept_no:
        return {"rcept_no": "", "corp_name": "", "title": "", "date": "", "sections": []}

    with mariadb_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT corp_name, title, date FROM document_index WHERE rcept_no=%s",
            (rcept_no,),
        )
        meta = cur.fetchone() or {}
        cur.execute(
            "SELECT chunk_id, section_path, embedding_text FROM chunk_index "
            "WHERE rcept_no=%s AND chunk_type='table_nl' "
            "ORDER BY section_path, chunk_id",
            (rcept_no,),
        )
        chunks = cur.fetchall()

    # 섹션별 그룹 → 각 표 파싱
    by_section: dict[str, list[dict]] = {}
    for ch in chunks:
        sp = str(ch.get("section_path") or "(기타)")
        parsed = _parse_table_text(str(ch.get("embedding_text") or ""))
        if not parsed:
            continue
        by_section.setdefault(sp, []).append(parsed)

    sections = []
    for sp in sorted(by_section, key=_section_sort_key):
        tables = []
        for i, t in enumerate(by_section[sp], 1):
            tables.append({
                "caption": f"표 {i}",  # v1: 소제목 미저장 → 섹션 내 순번
                "unit": t["unit"],
                "columns": t["columns"],
                "rows": t["rows"],
            })
        sections.append({"section_path": sp, "tables": tables})

    return {
        "rcept_no": rcept_no,
        "corp_name": str(meta.get("corp_name") or ""),
        "title": str(meta.get("title") or ""),
        "date": str(meta.get("date") or ""),
        "sections": sections,
    }
