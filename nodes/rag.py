# nodes/rag.py
from core.state import AgentState

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
    return {"final_draft": "초안 리포트"}