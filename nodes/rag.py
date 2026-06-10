# nodes/rag.py
from core.state import AgentState
from tool.rdb_client import execute_sql_query
from tool.text_to_sql import generate_sql
from tool.vector_store import search_vector_db

def supervisor_node(state: AgentState):
    return {"search_plan": ["RDB"]}


def _last_human_text(state: AgentState) -> str:
    """reconstructed_query 가 없을 때 마지막 사용자 메시지를 폴백으로 쓴다."""
    for msg in reversed(state.get("messages", []) or []):
        content = getattr(msg, "content", None)
        if content:
            return str(content)
    return ""


def _truncate(text: str, limit: int = 500) -> str:
    """행에 LONGTEXT(본문)가 있을 때 컨텍스트 폭주 방지."""
    return text if len(text) <= limit else text[:limit] + "…"


def _format_result(question: str, sql: str, result: dict) -> str:
    """검색 결과를 합성 노드가 읽을 텍스트로 포맷."""
    head = f"[RDB 검색]\n질문: {question}\nSQL: {sql}"
    if not result["ok"]:
        return f"{head}\n결과: 검색 실패 ({result['error']})"
    rows = result["rows"]
    if not rows:
        return f"{head}\n결과: 해당 데이터 없음"
    lines = [_truncate(str(r)) for r in rows[:20]]
    return f"{head}\n결과 {len(rows)}건:\n" + "\n".join(lines)


def rdb_search_node(state: AgentState):
    """RDB 검색: NL→SQL→실행. 실패 시 에러를 LLM에 되먹여 1회 재시도.

    LLM/DB 호출이 예외를 던져도(예: Ollama 다운) 노드가 죽지 않게 graceful 처리한다.
    """
    question = state.get("reconstructed_query") or _last_human_text(state)

    try:
        sql = generate_sql(question)
        result = execute_sql_query(sql)
        if not result["ok"]:
            sql = generate_sql(question, error_feedback=result["error"])
            result = execute_sql_query(sql)
        text = _format_result(question, result["sql"], result)
    except Exception as e:
        text = f"[RDB 검색]\n질문: {question}\n결과: 검색 중 오류 ({e})"

    return {"search_results": [text]}

def vector_search_node(state: AgentState):
    question = state.get("reconstructed_query") or _last_human_text(state)
    try:
        rows = search_vector_db(question)
        if not rows:
            text = f"[Vector 검색]\n질문: {question}\n결과: 해당 데이터 없음"
        else:
            lines = []
            for i, row in enumerate(rows[:10], 1):
                corp = row.get("corp_name") or row.get("corp_code") or "회사 미상"
                year = row.get("year") or "연도 미상"
                title = row.get("title") or row.get("doc_type") or row.get("section_path") or "문서"
                score = row.get("score")
                score_text = f"{float(score):.4f}" if isinstance(score, (int, float)) else str(score or "")
                body = _truncate(str(row.get("text") or ""), 700)
                lines.append(
                    f"{i}. {corp} / {year} / {title} / score={score_text}\n"
                    f"chunk_id={row.get('chunk_id')}\n{body}"
                )
            text = f"[Vector 검색]\n질문: {question}\n결과 {len(rows)}건:\n" + "\n\n".join(lines)
    except Exception as e:
        text = f"[Vector 검색]\n질문: {question}\n결과: 검색 중 오류 ({e})"
    return {"search_results": [text]}

def graph_search_node(state: AgentState):
    return {"search_results": ["Graph 검색 결과"]}

def synthesizer_node(state: AgentState):
    combined = "\n".join(state["search_results"])
    return {"synthesized_info": combined}

def reflection_node(state: AgentState):
    return {"is_sufficient": True}

def generate_report_node(state: AgentState):
    info = state.get("synthesized_info") or "\n".join(state.get("search_results", []) or [])
    if not info:
        info = "검색 결과가 없습니다."
    return {"final_draft": info}
