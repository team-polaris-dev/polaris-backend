# nodes/rag.py
from __future__ import annotations

import json
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from core.state import AgentState
# config.llm 의 llm/json_llm 은 이미 생성된 ApimakerLLM 인스턴스다(호출 X).
# json_llm 은 JSON 강제 모드 — 라우터/리플렉션처럼 JSON 응답이 필요한 곳에 쓴다.
from config.llm import llm, json_llm
import re
from pydantic import BaseModel, Field
from core.state import AgentState, UnifiedResult
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
    """Vector 검색 청크 결과 → UnifiedResult.

    rcept_no/title 을 extra 에 보존해 프론트가 '원본 문서' 메타(제목·공시번호)를
    표시하고 document_index 로 추가 조회(요약·공시일)할 수 있게 한다.
    """
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
            "rcept_no": str(row.get("rcept_no", "")),
            "title": str(row.get("title", "")),
            "corp_name": str(row.get("corp_name", "")),
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

CYPHER_PROMPT = """당신은 한국 DART 공시 KG 전문가다.
다음 자연어 질문을 Neo4j Cypher로 변환하라.
스키마:
- (:Organization {{corp_code,name}})
- (:FilingDocument {{rcept_no}})
- (:Organization)-[:IS_SUBSIDIARY_OF]->(:Organization)
- (:Person)-[:EXECUTIVE_OF]->(:Organization)
- (:Organization)-[:IS_MAJOR_SHAREHOLDER_OF {{qota_rt}}]->(:Organization)
반드시 RETURN에 path와 rcept_no를 포함하라.
질문: {q}
Cypher:"""


def _looks_like_cypher(q: str) -> bool:
    """LLM이 생성한 쿼리가 최소한의 Cypher 문법(MATCH/RETURN)을 갖추었는지 검사."""
    if not q or len(q) < 10:
        return False
    up = q.upper()
    return "MATCH" in up and "RETURN" in up


def _row_to_unified(row: dict) -> UnifiedResult:
    """그래프DB row → UnifiedResult(type/code/name/value/extra/source) 변환."""
    nodes = [x for x in row["path"] if "corp_code" in x]
    tail = nodes[-1] if nodes else {"corp_code": "", "name": ""}
    rels = [x.get("rel") for x in row["path"] if "rel" in x]

    return {
        "type": "graph_path",
        "code": tail["corp_code"],
        "name": tail["name"],
        "value": " -> ".join(rels) if rels else "",
        "extra": {"hops": len(rels)},
        "source": row.get("rcept_no", ""),
    }



def graph_search_node(state: AgentState):
    """자연어 질문 → Cypher → 그래프DB → graph_facts/paths/provenance.

    현재 ``execute_cypher_query``는 목업. 실제 Neo4j 드라이버 연결과
    쿼리 템플릿 라우팅(___test/hybrid_rag 자산 이식)은 후속 이슈로 분리.
    """
    q = state.get("reconstructed_query") or ""
    empty = {"graph_facts": [], "graph_paths": [], "graph_provenance": []}

    try:
        resp = llm.invoke(CYPHER_PROMPT.format(q=q))
        cypher = getattr(resp, "content", str(resp)).strip().strip("`")

        if not _looks_like_cypher(cypher):
            raise GraphQueryError(f"LLM produced non-cypher: {cypher!r}")

        rows = execute_cypher_query(cypher)
    except Exception as e:   # GraphQueryError 포함
        print(f"[graph_search_node] degraded: {e}")
        return empty

    facts: list[UnifiedResult] = []
    paths: list[list[str]] = []
    prov: list[str] = []
    for row in rows:
        facts.append(_row_to_unified(row))
        paths.append([x.get("name") or x.get("rel", "") for x in row["path"]])
        if row.get("rcept_no"):
            prov.append(row["rcept_no"])

    return {"graph_facts": facts, "graph_paths": paths, "graph_provenance": prov}

# 그래프 관계 타입 → 한글 라벨 (LLM 가독성용; core.serialize 와 동일 키)
_REL_LABELS: dict[str, str] = {
    "IS_SUBSIDIARY_OF": "자회사",
    "EXECUTIVE_OF": "임원",
    "IS_MAJOR_SHAREHOLDER_OF": "대주주",
    "SUPPLIES_TO": "공급",
    "ACQUIRES": "인수",
    "INVESTS": "투자",
}


def _humanize_rel(rel: str) -> str:
    return _REL_LABELS.get(rel, (rel or "").replace("_", " ").strip() or "관계")


def _format_graph_path(path: list[str]) -> str:
    """graph_paths 한 항목([노드, 관계, 노드, …])을 사람이 읽는 경로 문자열로.

    예: ['삼성전자','IS_SUBSIDIARY_OF','반도체에피'] → '삼성전자 -[자회사]-> 반도체에피'.
    graph_facts 는 꼬리 노드와 관계 체인만 남겨 머리/중간 노드가 유실되므로,
    전체 노드 시퀀스를 가진 graph_paths 를 직접 풀어 쓴다.
    """
    parts: list[str] = []
    for idx, token in enumerate(path):
        if not token:
            continue
        if idx % 2 == 1:  # 홀수 인덱스 = 관계
            parts.append(f"-[{_humanize_rel(token)}]->")
        else:             # 짝수 인덱스 = 노드(기업/인물)
            parts.append(str(token))
    return " ".join(parts)


def synthesizer_node(state: AgentState):
    """
    Syn: 3개의 DB(RDB, Vector, Graph)에서 검색된 결과를 하나의 텍스트로 병합합니다.
    """
    print("🧩 [Syn Node] 검색 결과 취합 중...")

    rdb_results = state.get("rdb_results", [])
    vec_results = state.get("vec_results", [])
    graph_facts = state.get("graph_facts", [])
    graph_paths = state.get("graph_paths", [])

    synthesized_text = ""

    if rdb_results:
        synthesized_text += "### [정형 데이터 (RDB)]\n"
        for res in rdb_results:
            # 새로운 dict 규격에 맞춰 문자열 조립
            synthesized_text += f"- 항목: {res.get('name')}, 값: {res.get('value')} (출처: {res.get('source')})\n"

    if vec_results:
        synthesized_text += "\n### [비정형 문서 (Vector)]\n"
        for res in vec_results:
            synthesized_text += f"- 내용: {res.get('value')} (출처: {res.get('source')})\n"

    if graph_paths or graph_facts:
        synthesized_text += "\n### [관계망 데이터 (Graph)]\n"
        # graph_paths 와 graph_facts 는 행 단위로 정렬됨(graph_search_node 에서 매 행 append).
        # 전체 경로(머리→관계→꼬리)를 풀어 쓰고, 근거 공시(rcept_no)는 fact.source 에서 가져온다.
        if graph_paths:
            for i, path in enumerate(graph_paths):
                readable = _format_graph_path(path)
                if not readable:
                    continue
                source = ""
                if i < len(graph_facts):
                    source = str(graph_facts[i].get("source") or "")
                line = f"- 관계 경로: {readable}"
                if source:
                    line += f" (근거 공시: {source})"
                synthesized_text += line + "\n"
        else:
            # 경로가 비어 있고 fact 만 있는 예외적 경우의 폴백
            for res in graph_facts:
                synthesized_text += (
                    f"- 관계: {res.get('name')} - {res.get('value')} (출처: {res.get('source')})\n"
                )

    # 검색된 결과가 전혀 없을 경우에 대한 방어 로직
    if not synthesized_text.strip():
        synthesized_text = "검색된 데이터가 없습니다."

    return {"synthesized_info": synthesized_text}



json_parser = JsonOutputParser()

# 1-2. 문자열 기반 프롬프트 템플릿 작성
# (SystemMessage와 HumanMessage로 나누지 않고 하나의 텍스트로 합칩니다)
reflection_prompt = PromptTemplate(
    template="""당신은 공시 분석 데이터 검증(Reflection) 전문가입니다.
사용자의 질문에 답하기 위해 검색된 [취합된 정보]가 충분한지 엄격하게 평가하세요.

[평가 기준]
1. 질문에서 요구하는 핵심 수치, 기업명, 사실관계가 취합된 정보에 존재하는가?
2. 정보가 부족해서 사용자가 원하는 형태의 답변을 생성할 수 없는가?

설명이나 마크다운 없이 반드시 아래 JSON 형식으로만 응답하세요:
{{"is_sufficient": true 또는 false, "reason": "충분한 이유 또는 누락된 정보 설명"}}

[질문]
{query}

[취합된 정보]
{info}

결과 JSON:""",
    input_variables=["query", "info"]
)

# ApimakerLLM 은 Runnable 이 아니라 LCEL 파이프(|)를 쓸 수 없다.
# 프롬프트 포맷 → json_llm.invoke → JsonOutputParser 로 직접 연결한다.
def _run_reflection(query: str, info: str) -> dict:
    prompt_text = reflection_prompt.format(query=query, info=info)
    return json_parser.invoke(json_llm.invoke(prompt_text))


# ==========================================
# 2. 개선된 Reflection 노드
# ==========================================
def reflection_node(state: dict): # AgentState 대신 dict 타입 힌트 (환경에 맞게 수정)
    """
    Reflect: 취합된 정보가 사용자의 질문에 답변하기에 충분한지 자체 검증합니다.
    """
    query = state.get("reconstructed_query", "")
    info = state.get("synthesized_info", "")
    current_retry = state.get("retry_count", 0)
    
    print("🔍 [Reflect Node] 데이터 충분성 자체 검증 중...")
    
    try:
        # 프롬프트 완성 → json_llm 호출 → JSON 딕셔너리로 파싱
        parsed_response = _run_reflection(query, info)

        is_sufficient = parsed_response.get("is_sufficient", True)
        reason = parsed_response.get("reason", "검증 완료")
        
    except Exception as e:
        # LLM이 JSON을 완전히 망가뜨렸을 경우를 대비한 폴백(안전망)
        print(f"⚠️ [Reflect Node] 체인 실행/파싱 실패. 에러: {e}")
        print("강제로 검증 통과(True) 처리합니다.")
        is_sufficient = True
        reason = "파싱 오류"
        
    print(f"   -> 검증 결과: {'통과✅' if is_sufficient else '불충분❌'} (사유: {reason})")
    
    # ---------------- 아래부터는 기존 로직과 완전히 동일합니다 ----------------

    # 테스트 위해 임시로 카운트를 0으로 잡은 로직 (원본 유지)
    if not is_sufficient and current_retry >= 1: 
        print("🛑 [Reflect Node] 최대 재시도 횟수(2회)에 도달했습니다.(지금은 개발단계임으로 바로 통과) 누락된 정보가 있더라도 최종 답변 생성을 강제 진행합니다.")
        return {
            "is_sufficient": True, # 강제로 True를 주어 gen 노드로 보내기
            "retry_count": current_retry + 1
        }
    
    # 정상적으로 충분한 경우
    if is_sufficient:
        return {
            "is_sufficient": True,
            "retry_count": current_retry # 통과 시 카운트 유지
        }
    
    # 💡 정보가 불충분할 경우 내부 피드백 생성
    feedback_message = SystemMessage(
        content=(
            f"[자체 검증 시스템 알림] 이전 검색 결과가 질문을 해결하기에 불충분했습니다. "
            f"누락된 원인: {reason} "
            f"이 피드백을 반영하여, 다른 키워드를 사용하거나 검색 범위를 넓히는 방향으로 질문을 다시 재구성하세요."
        )
    )
    
    return {
        "is_sufficient": False,
        "messages": [feedback_message],  # 이 메시지가 추가되어 Ctx 노드로 돌아갑니다.
        "retry_count": current_retry + 1
    }

