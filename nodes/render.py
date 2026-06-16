from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import AgentState
from config.llm import llm
from core.serialize import _humanize_rel


def _truncate(text: str, limit: int = 1000) -> str:
    """청크 본문이 너무 길 때 컨텍스트 폭주 방지."""
    return text if len(text) <= limit else text[:limit] + "…"


# 사람이 읽을 도메인 사실이 아닌(랭킹·쿼리·내부 식별) 필드는 프롬프트에서 제외.
# - score: 넣으면 LLM 이 "연관성 0.87" 같은 의미를 창작함
# - sql: 답변에 쿼리문이 새거나 LLM 이 SQL 을 지어냄
# - chunk_id/rcept_no: 출처는 별도(source/graph_provenance)로 이미 넘김
_EXTRA_SKIP = {"score", "sql", "chunk_id", "rcept_no", "stage", "hops"}


def _format_kv(data: dict) -> str:
    """dict → '키=값, 키=값' 요약 문자열. 내부/랭킹 키와 빈 값은 제외."""
    if not isinstance(data, dict):
        return ""
    pairs = []
    for k, v in data.items():
        if k in _EXTRA_SKIP or v in (None, ""):
            continue
        text = " ".join(str(v).split())
        if text:
            pairs.append(f"{k}={_truncate(text, 120)}")
    return ", ".join(pairs)


def generate_report_node(state: AgentState):
    """
    Gen (Generate & Render):
    AgentState 의 검색 결과(rdb_results/vec_results/graph_facts)를 직접 참조해
    포매팅하고, 사용자 선호도(user_preferences)에 맞춰 최종 분석 보고서를 생성합니다.
    """
    # 1. State의 검색 결과(rdb/vec/graph)만을 분석 기초 자료 텍스트로 직접 포매팅한다.
    #    gen 은 오직 AgentState 의 검색 결과만 근거로 설명한다. 이전 대화 내용이나
    #    LLM 사전 지식으로 보강하지 않는다(result_check 가 세 소스 모두 결과가 있을
    #    때만 gen 으로 보내므로 draft_info 는 비어 있지 않다).
    rdb_results = state.get("rdb_results", []) or []
    vec_results = state.get("vec_results", []) or []
    graph_facts = state.get("graph_facts", []) or []
    graph_paths = state.get("graph_paths", []) or []
    graph_provenance = state.get("graph_provenance", []) or []

    parts: list[str] = []

    if rdb_results:
        parts.append("### [정형 데이터 (RDB)]")
        for res in rdb_results:
            # value 가 컬럼 dict 면 '컬럼=값'으로 평탄화, 스칼라면 그대로 둔다.
            value = res.get("value")
            value_text = _format_kv(value) if isinstance(value, dict) else str(value)
            name = res.get("name") or "(회사 미상)"
            parts.append(f"- {name}: {value_text} (출처: {res.get('source')})")

    if vec_results:
        parts.append("\n### [비정형 문서 (Vector)]")
        for res in vec_results:
            # 본문 앞에 출처 맥락(회사·연도·문서종류·절 위치)을 헤더로 붙인다.
            extra = res.get("extra") or {}
            ctx = [str(x) for x in (
                res.get("name"),
                extra.get("year"),
                extra.get("doc_type"),
                extra.get("section_path"),
            ) if x not in (None, "")]
            header = f"[{' · '.join(ctx)}] " if ctx else ""
            parts.append(
                f"- {header}{_truncate(str(res.get('value')))} (출처: {res.get('source')})"
            )

    if graph_facts:
        parts.append("\n### [관계망 데이터 (Graph)]")
        for res in graph_facts:
            # 관계 종류(type)·대상(name)·값(value)에 더해, extra 의 상세
            # (지분율 qota_rt·직책 pos·연도 year 등)까지 함께 넘긴다.
            rel = _humanize_rel(str(res.get("type") or ""))
            line = f"- 관계({rel}): {res.get('name')} - {res.get('value')}"
            detail = _format_kv(res.get("extra") or {})
            if detail:
                line += f" [{detail}]"
            line += f" (출처: {res.get('source')})"
            parts.append(line)

        # 방향성 있는 경로(graph_paths): [시작노드, 관계, 끝노드] → "A →(관계)→ B"
        directed = [p for p in graph_paths if isinstance(p, (list, tuple)) and len(p) == 3]
        if directed:
            parts.append("\n#### [관계 방향 (A → B)]")
            for src, rel, dst in directed:
                parts.append(f"- {src} →({_humanize_rel(str(rel))})→ {dst}")

        # 근거 공시 접수번호(rcept_no)
        if graph_provenance:
            parts.append(f"\n(관계망 근거 공시: {', '.join(map(str, graph_provenance))})")

    draft_info = "\n".join(parts).strip()
    prefs = state.get("user_preferences", {})

    # 기본 선호도 설정 (Store에 값이 없을 경우를 대비)
    tone = prefs.get("tone", "전문적이고 신뢰감 있는")

    # 사용자 질문 — 이게 빠지면 LLM 이 질문과 무관한 일반 기업소개 보고서를 짓는다.
    question = state.get("reconstructed_query") or ""
    if not question:
        for msg in reversed(state.get("messages", []) or []):
            if getattr(msg, "type", "") == "human":
                question = str(msg.content)
                break

    # 2. 분석 기초 자료 기반 최종 답변 생성을 위한 시스템 프롬프트
    system_prompt = f"""당신은 기업 공시 및 재무 분석을 도와주는 전문적이고 친절한 AI 어시스턴트입니다.
[분석 기초 자료]만을 근거로 [사용자 질문]에 직접 답하는 최종 답변을 작성하세요.

[사용자 맞춤 설정]
- 말투 및 톤: {tone}

[작성 규칙]
1. 답변의 중심은 반드시 [사용자 질문]에 대한 직접적인 답이어야 합니다.
   질문과 무관한 일반 기업 소개·연혁·홍보성 서사로 흐르지 마세요.
2. [분석 기초 자료]에 있는 사실·수치·관계만 사용하세요. 자료에 없는 내용
   (사전 학습 지식, 추정·재계산한 수치)을 추가하지 마세요.
3. 자료가 질문의 일부를 못 다루면 그 부분은 "자료에 없다"고 솔직하게 밝히세요.
4. [분석 기초 자료] 중 질문과 관련된 핵심 정보(수치, 사실관계, 기업명, 관계)는
   누락·왜곡하지 마세요. 특히 [관계망 데이터 (Graph)]의 'A → B' 관계는
   방향(누가 누구에게)을 정확히 유지하세요.
5. 지정된 '말투 및 톤'에 맞춰 자연스럽게 구성하고,
   가독성을 위해 마크다운을 적극 활용하세요.
6. "알겠습니다", "요청하신 보고서입니다" 같은 서론 없이 곧바로 최종 결과물만 출력하세요.
"""

    human_prompt = f"[사용자 질문]\n{question}\n\n[분석 기초 자료]\n{draft_info}"

    # 3. LLM 호출하여 렌더링 진행
    messages_to_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]
    
    print(f"📝 [Gen Node] 최종 보고서 작성 중... (톤: {tone})")
    # 커스텀 LLM(LLM 기반)은 문자열을 반환하므로 AIMessage 로 감싸 state 에 넣는다.
    response_text = llm.invoke(messages_to_llm).content

    # 4. 최종 변환된 메시지를 State의 messages 배열에 추가하고,
    # final_draft 키에도 명시적으로 저장하여 상태 업데이트
    return {
        "messages": [AIMessage(content=response_text)],
        "final_draft": response_text,
    }