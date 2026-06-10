# nodes/rag.py
from core.state import AgentState, UnifiedResult
from tool.rdb_client import execute_sql_query
from tool.text_to_sql import generate_sql
from tool.vector_store import search_vector_db


def supervisor_node(state: AgentState):
    # (예시) 검색 계획 수립
    return {"search_plan": ["RDB", "Vector", "Graph"]}


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


def graph_search_node(state: AgentState):
    # 분리된 키인 graph_facts 사용 + 통일된 dict 규격 적용
    return {
        "graph_facts": [
            {
                "type": "subsidiary",
                "code": "CORP002",
                "name": "자회사A",
                "value": "지분율 100%",
                "extra": {"relation": "owns"},
                "source": "rcept_no_333"
            }
        ],
        "graph_paths": [["CORP001", "owns", "CORP002"]],   # 멀티홉 경로 mock
        "graph_provenance": ["rcept_no_333"]               # 근거 rcept_no
    }


def synthesizer_node(state: AgentState):
    # 1. 3곳의 검색 결과를 모두 가져옵니다 (데이터가 없을 수도 있으니 .get() 사용)
    rdb = state.get("rdb_results", [])
    vec = state.get("vec_results", [])
    graph = state.get("graph_facts", [])

    # 2. 모든 결과를 하나의 리스트로 합칩니다
    all_results = rdb + vec + graph

    # 3. 통일된 dict 키(name, value, source 등)를 활용해 프롬프트에 넣을 텍스트로 변환합니다
    formatted_texts = []
    for item in all_results:
        text = f"[{item['type']}] {item['name']}: {item['value']} (출처: {item['source']})"
        formatted_texts.append(text)

    combined = "\n".join(formatted_texts)

    return {"synthesized_info": combined}


def reflection_node(state: AgentState):
    return {"is_sufficient": True}


def generate_report_node(state: AgentState):
    info = state.get("synthesized_info") or "검색 결과가 없습니다."
    return {"final_draft": info}
