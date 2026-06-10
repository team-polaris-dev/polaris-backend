# nodes/rag.py
from core.state import AgentState, UnifiedResult
from tool.graph_client import execute_cypher_query, GraphQueryError
from config.llm import llm 

def supervisor_node(state: AgentState):
    # (예시) 검색 계획 수립
    return {"search_plan": ["RDB", "Vector", "Graph"]}

def rdb_search_node(state: AgentState):
    # 분리된 키인 rdb_results 사용 + 통일된 dict 규격 적용
    return {
        "rdb_results": [
            {
                "type": "rdb_row",
                "code": "CORP001",
                "name": "매출액",
                "value": "100억",
                "extra": {"currency": "KRW"},
                "source": "rcept_no_111"
            }
        ]
    }

def vector_search_node(state: AgentState):
    # 분리된 키인 vec_results 사용 + 통일된 dict 규격 적용
    return {
        "vec_results": [
            {
                "type": "vec_chunk",
                "code": "CORP001",
                "name": "사업 개요",
                "value": "당사는 AI 솔루션을 개발하는...",
                "extra": {"similarity_score": 0.92},
                "source": "chunk_id_222"
            }
        ]
    }

#======================================================================
# Graph rag 함수 -
#======================================================================
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
#======================================================================
# Graph rag 구현 함수
#======================================================================
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
    return {"final_draft": "초안 리포트"}