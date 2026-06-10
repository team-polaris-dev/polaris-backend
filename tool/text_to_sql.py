# tool/text_to_sql.py — 자연어 질문을 MariaDB SELECT 로 변환
from __future__ import annotations

import re

from config.llm import llm
from tool.rdb_client import get_schema_prompt

_SYSTEM = """\
당신은 한국 반도체 기업 GraphRAG 'POLARIS'의 MariaDB Text-to-SQL 전문가다.
사용자 질문을 MariaDB(MySQL 호환) SELECT 쿼리 하나로 변환한다.
규칙:
- 반드시 단일 SELECT 문만 출력한다. INSERT/UPDATE/DELETE/DDL 금지.
- 설명·주석·코드펜스 없이 SQL 한 문장만 출력한다.
- 아래 스키마에 없는 테이블·컬럼은 절대 지어내지 않는다.
- 회사 식별(필터)은 corp_code(8자리)로 하되, 답에 회사가 보여야 하면 corp_name 을 SELECT 한다.
- 사람이 읽을 결과를 만든다 — 코드·ID보다 이름·제목 같은 의미 있는 컬럼을 우선 선택한다.
- 결과가 많을 수 있으면 LIMIT 을 붙인다."""


def _build_prompt(question: str, read_run_id: str | None, error_feedback: str | None) -> str:
    parts = [_SYSTEM, "## 스키마\n" + get_schema_prompt()]
    parts.append("## 질문\n" + question)
    if error_feedback:
        parts.append("## 직전 SQL 실행 오류 (반드시 수정)\n" + error_feedback)
    parts.append("## 출력 (SELECT 한 문장):")
    return "\n\n".join(parts)


def _extract_sql(text: str) -> str:
    """LLM 응답에서 SQL 한 문장만 추출 (코드펜스/설명/꼬리 제거)."""
    fence = re.search(r"```(?:sql)?\s*(.+?)```", text, re.IGNORECASE | re.DOTALL)
    if fence:
        text = fence.group(1)
    m = re.search(r"(?is)\b(SELECT|WITH)\b.*", text)  # CTE(WITH) 도 시작점으로 인정
    sql = (m.group(0) if m else text).strip()
    if ";" in sql:  # 첫 세미콜론 뒤 설명 제거
        sql = sql.split(";", 1)[0]
    sql = re.split(r"\n\s*\n", sql, maxsplit=1)[0]  # 빈 줄 뒤 설명 제거
    return sql.strip()


def generate_sql(
    question: str,
    read_run_id: str | None = None,
    error_feedback: str | None = None,
) -> str:
    """질문 → SQL 문자열. error_feedback 가 있으면 직전 오류를 반영해 재생성."""
    prompt = _build_prompt(question, read_run_id, error_feedback)
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", resp)
    return _extract_sql(str(content))
