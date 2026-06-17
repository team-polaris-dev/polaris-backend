# nodes/rag.py
from __future__ import annotations

from core.state import AgentState
from config.llm import llm
import re
from pydantic import BaseModel, Field
from core.state import AgentState, UnifiedResult
from graphrag import graph_search_node as _graphrag_search_node
from tool.rdb_client import execute_sql_query, get_schema_prompt
from tool.vector_store import search_vector_db
from tool.graph_client import execute_cypher_query, GraphQueryError


def _last_human_text(state: AgentState) -> str:
    """reconstructed_query 가 없을 때 마지막 사용자 메시지를 폴백으로 쓴다."""
    for msg in reversed(state.get("messages", []) or []):
        content = getattr(msg, "content", None)
        if content:
            return str(content)
    return ""


def _truncate(text: str, limit: int = 1000) -> str:
    """청크 본문이 너무 길 때 컨텍스트 폭주 방지."""
    return text if len(text) <= limit else text[:limit] + "…"


# ---------------------------------------------------------------- NL → SQL 생성
# (구 tool/text_to_sql.py 에서 이동: 노드가 LLM 프롬프트 구성을 직접 소유)

class SQLQuery(BaseModel):
    """변환된 MariaDB SELECT 쿼리."""

    sql: str = Field(description="실행할 단일 SELECT(또는 WITH) 문. 코드펜스·설명 없이 SQL만.")


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


def _build_sql_prompt(question: str, error_feedback: str | None) -> str:
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


def generate_sql(question: str, error_feedback: str | None = None) -> str:
    """질문 → SQL 문자열. error_feedback 가 있으면 직전 오류를 반영해 재생성.

    구조화 출력(`with_structured_output`)을 지원하는 LLM이면 이를 우선 사용하고,
    FakeLLM 등 미지원 모델이거나 실패 시 텍스트 응답을 정규식으로 파싱한다.
    """
    prompt = _build_sql_prompt(question, error_feedback)
    if hasattr(llm, "with_structured_output"):
        try:
            result = llm.with_structured_output(SQLQuery).invoke(prompt)
            return _extract_sql(result.sql)
        except Exception:
            pass
    resp = llm.invoke(prompt)
    content = getattr(resp, "content", resp)
    return _extract_sql(str(content))


def _rdb_row_to_unified(row: dict, sql: str) -> UnifiedResult:
    """SELECT 결과 한 행 → UnifiedResult.

    질의마다 컬럼 구성이 달라 corp_code/corp_name/rcept_no 를 식별 정보로
    분리하고 나머지 컬럼을 value 에 담는다(없으면 행 전체를 value 로).
    """
    rest = {k: v for k, v in row.items() if k not in ("corp_code", "corp_name", "rcept_no")}
    return {
        "type": "rdb_row",
        "code": str(row.get("corp_code", "")),
        "name": str(row.get("corp_name", "")),
        "value": rest or row,
        "extra": {"sql": sql},
        "source": str(row.get("rcept_no", "")),
    }


def rdb_search_node(state: AgentState):
    """RDB 검색: NL→SQL→실행 → UnifiedResult 리스트.

    LLM/DB 호출이 예외를 던지거나(Ollama 다운 등) 결과가 없으면
    rdb_results=[] 를 반환해 파이프라인을 보호한다.
    SQL 실행 실패 시 에러를 LLM에 되먹여 1회 재시도한다.
    """
    question = state.get("reconstructed_query") or _last_human_text(state)
    try:
        sql = generate_sql(question)
        result = execute_sql_query(sql)
        if not result["ok"]:
            sql = generate_sql(question, error_feedback=result["error"])
            result = execute_sql_query(sql)
        if not result["ok"] or not result["rows"]:
            return {"rdb_results": []}
        return {"rdb_results": [_rdb_row_to_unified(row, result["sql"]) for row in result["rows"]]}
    except Exception:
        return {"rdb_results": []}


def _chunk_to_unified(row: dict) -> UnifiedResult:
    """Vector 검색 청크 결과 → UnifiedResult."""
    name = row.get("corp_name") or row.get("title") or row.get("section_path") or ""
    return {
        "type": "vec_chunk",
        "code": str(row.get("corp_code", "")),
        "name": str(name),
        "value": _truncate(str(row.get("text") or "")),
        "extra": {
            "score": row.get("score"),
            "year": row.get("year"),
            "doc_type": row.get("doc_type"),
            "section_path": row.get("section_path"),
            "rcept_no": row.get("rcept_no"),
            "corp_name": row.get("corp_name"),
            "title": row.get("title"),
        },
        "source": str(row.get("chunk_id", "")),
    }



def vector_search_node(state: AgentState):
    """Vector 검색: 하이브리드 검색(Dense+BM25+rerank) → UnifiedResult 리스트.

    실패하거나 결과가 없으면 vec_results=[] 를 반환해 파이프라인을 보호한다.
    """
    question = state.get("reconstructed_query") or _last_human_text(state)
    try:
        rows = search_vector_db(question)
        if not rows:
            return {"vec_results": []}
        return {"vec_results": [_chunk_to_unified(row) for row in rows]}
    except Exception:
        return {"vec_results": []}


#======================================================================
# Graph RAG 노드 — graphrag 패키지(self-contained) 위임
#======================================================================
def graph_search_node(state: AgentState):
    """LangGraph 노드 진입점. 본체는 graphrag 패키지가 담당.

    I/O 계약(core/state.py:AgentState):
      입력  reconstructed_query: str
      출력  graph_facts: List[UnifiedResult]
            graph_paths: List[List[str]]
            graph_provenance: List[str]   # rcept_no
    Neo4j/Anthropic 키 등 외부 의존이 끊기면 graphrag 가 빈 결과로 degrade.
    """
    return _graphrag_search_node(state)

