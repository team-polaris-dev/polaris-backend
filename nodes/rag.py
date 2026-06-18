# nodes/rag.py
from __future__ import annotations

from core.state import AgentState, UnifiedResult
from graphrag import graph_search_node as _graphrag_search_node
from tool.rdb_client import (
    fetch_financial_card,
    fetch_documents_by_rcept,
    fetch_recent_documents,
    parse_year,
    resolve_corp_codes_from_text,
)
from tool.vector_store import search_vector_db


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


# ---------------------------------------------------------------- 결정론 RDB
# GraphDB 가 확정한 앵커(corp_code / rcept_no)를 받아 고정 SQL 템플릿만 실행한다.
# LLM Text-to-SQL 을 제거해 "같은 질문 = 같은 결과"(결정성)를 보장한다.
# (앵커 수집 → 재무카드 + 공시 메타 템플릿 호출 → UnifiedResult)

_CORP_CODE_CAP = 5    # 앵커 회사 상한(다회사 비교 대비)
_RCEPT_CAP = 20       # 인용 공시 상한


def _is_corp_code(s: str) -> bool:
    """8자리 숫자 corp_code 형태인지(그래프 id 에 섞인 er_name·rel id 배제)."""
    return bool(s) and len(s) == 8 and s.isdigit()


def _collect_corp_codes(state: AgentState, question: str) -> list[str]:
    """앵커 corp_code 수집: graph_seeds → graph_facts → 질문 자가해소 순(순서보존 dedup).

    GraphDB 가 시드를 잡았으면 그걸 1순위로, 못 잡았으면 질문에서 직접 회사명을
    결정론으로 해소한다(재무질문이 그래프 빈 결과에도 동작하도록).
    """
    codes: list[str] = []

    def add(c: str | None) -> None:
        if c and _is_corp_code(c) and c not in codes:
            codes.append(c)

    for seed in state.get("graph_seeds") or []:
        if isinstance(seed, dict) and seed.get("key_type") == "corp_code":
            add(str(seed.get("key_value") or ""))

    for fact in state.get("graph_facts") or []:
        if not isinstance(fact, dict):
            continue
        add(str(fact.get("code") or ""))
        extra = fact.get("extra") or {}
        add(str(extra.get("from_id") or ""))
        add(str(extra.get("to_id") or ""))

    for c in resolve_corp_codes_from_text(question):
        add(c)

    return codes[:_CORP_CODE_CAP]


def _collect_rcept_nos(state: AgentState) -> list[str]:
    """앵커 rcept_no 수집: graph_provenance(정제된 근거) + graph_facts.source."""
    nos: list[str] = []

    def add(r: str | None) -> None:
        if r and r not in nos:
            nos.append(r)

    for r in state.get("graph_provenance") or []:
        add(str(r))
    for fact in state.get("graph_facts") or []:
        if isinstance(fact, dict):
            add(str(fact.get("source") or ""))

    return [r for r in nos if r][:_RCEPT_CAP]


def _fin_to_unified(row: dict) -> UnifiedResult:
    """재무카드 행 → UnifiedResult(render._fmt_rdb 가 기대하는 rdb_row 규격).

    value 에 account_id/value/bsns_year 를 담아 gen 이 기업·연도로 묶어 렌더한다.
    source 는 rcept_no — 원본 공시 링크 앵커.
    """
    return {
        "type": "rdb_row",
        "code": str(row.get("corp_code", "")),
        "name": str(row.get("corp_name") or ""),
        "value": {
            "account_id": row.get("account_id"),
            "value": row.get("value"),
            "bsns_year": row.get("bsns_year"),
            "unit": row.get("unit"),
            "fs_div": row.get("fs_div"),
        },
        "extra": {"kind": "financial"},
        "source": str(row.get("rcept_no") or ""),
    }


def _doc_to_unified(row: dict) -> UnifiedResult:
    """공시 메타 행 → UnifiedResult. render._fmt_rdb 가 별도 문서 섹션으로 렌더."""
    return {
        "type": "rdb_doc",
        "code": str(row.get("corp_code", "")),
        "name": str(row.get("corp_name") or ""),
        "value": {
            "title": row.get("title"),
            "doc_type": row.get("doc_type"),
            "date": str(row.get("date") or ""),
            "summary_short": row.get("summary_short"),
        },
        "extra": {"kind": "document"},
        "source": str(row.get("rcept_no") or ""),
    }


def _rdb_abstain() -> UnifiedResult:
    """정형 데이터가 없을 때 내보내는 '표식 행'(SQL 결과 아님, 직접 생성).

    이 백엔드는 세 검색 소스(rdb/vec/graph) 중 하나라도 비면 답변이 막히는 게이트
    (router.result_check)를 갖는다. 공급망·보유기술처럼 RDB 에 정형 데이터가 없는
    질문은 빈 결과([])를 내면 답변 전체가 차단되고, 그걸 피하려 LLM 이 chunk_index 를
    LIKE 로 긁어 노이즈를 만든다(원인). 그래서 '진짜로 정형 데이터가 없을 때'는
    빈 리스트 대신 이 표식 1행을 내보내 게이트는 통과시키되, gen 노드가 "정형 데이터
    없음"으로 정직하게 렌더링하게 한다(render.py 규칙 #3). SQL 검증/실행을 거치지 않고
    직접 만든 dict 라 SQL 안전성 경로와 무관하며, serialize(프론트 패널)는 rdb_results 를
    읽지 않아 UI 로 새어나가지 않는다.
    """
    return {
        "type": "rdb_note",
        "code": "",
        "name": "정형 데이터",
        "value": "이 질문에 해당하는 정형(공시 메타·재무수치) 데이터가 없습니다.",
        "extra": {},
        "source": "",
    }


def rdb_search_node(state: AgentState):
    """RDB 검색(결정론): GraphDB 앵커(corp_code/rcept_no) → 고정 SQL 템플릿 → UnifiedResult.

    LLM Text-to-SQL 을 제거했다 — 같은 앵커는 항상 같은 SQL/결과를 낸다(비결정성 해소).
      1) 회사(corp_code) 앵커가 있으면 재무카드(연결·연간, 최신연도 또는 질문 연도) 조회.
      2) 그래프가 인용한 rcept_no 가 있으면 그 공시 메타를, 없으면 회사 최근 공시를 보강.
    정형 데이터가 전혀 없으면 빈 리스트 대신 '표식 행'(_rdb_abstain)을 내 result_check
    게이트를 통과시키되 gen 이 '정형 데이터 없음'으로 정직하게 렌더하게 한다.
    """
    question = state.get("reconstructed_query") or _last_human_text(state)
    try:
        corp_codes = _collect_corp_codes(state, question)
        rcept_nos = _collect_rcept_nos(state)
        year = parse_year(question)

        results: list[UnifiedResult] = []

        if corp_codes:
            results.extend(_fin_to_unified(r) for r in fetch_financial_card(corp_codes, year))

        # 그래프가 인용한 공시(rcept_no) 우선, 없으면 회사 최근 공시로 맥락 보강.
        if rcept_nos:
            results.extend(_doc_to_unified(r) for r in fetch_documents_by_rcept(rcept_nos))
        elif corp_codes:
            results.extend(_doc_to_unified(r) for r in fetch_recent_documents(corp_codes))

        if not results:
            return {"rdb_results": [_rdb_abstain()]}
        return {"rdb_results": results}
    except Exception as e:
        print(f"⚠️ [rdb_search_node] 예외 → abstain: {e!r}")
        return {"rdb_results": [_rdb_abstain()]}


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
            # 정상 0건 — 검색은 동작했고 관련 청크가 없었다(관련도 하한/메타필터).
            # 아래 예외 경로(검색 자체 실패)와 구분해 로깅한다. 콜드스타트 0건은
            # vector_store 의 콜드빌드 로그로 따로 식별된다.
            print("ℹ️ [vector_search_node] 관련 청크 0건(정상 degrade)")
            return {"vec_results": []}
        return {"vec_results": [_chunk_to_unified(row) for row in rows]}
    except Exception as e:
        # 예외로 인한 0건 — Qdrant/Ollama/DB 장애 등. 정상 0건과 절대 같게 보지 말 것.
        print(f"⚠️ [vector_search_node] 검색 예외 → 0건 degrade: {e!r}")
        return {"vec_results": []}


#======================================================================
# Graph RAG 노드 — graphrag 패키지(self-contained) 위임
#======================================================================
def graph_search_node(state: AgentState):
    """LangGraph 노드 진입점. 본체는 graphrag 패키지가 담당(Local/Global 통합).

    graphrag.node.graph_search_node 가 intent 로 분기한다:
      입력  reconstructed_query: str, intent: str
      출력  (local)  graph_facts / graph_paths / graph_provenance
            (global) community_results: List[UnifiedResult]  # type="community"
    Neo4j/Anthropic 키 등 외부 의존이 끊기면 graphrag 가 빈 결과로 degrade.
    """
    print("*"*50)
    print("김구현의 귀여운 데이터가 주입됩니다.")
    print("*"*50)
    return _graphrag_search_node(state)

