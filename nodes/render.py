from __future__ import annotations

from collections import defaultdict
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import AgentState
from config.llm import llm
from core.serialize import _humanize_rel, _ACCOUNT_KR, _fmt_krw
from graphrag import effective_query


def _truncate(text: str, limit: int = 1000) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


# ──────────────────────────────────────────────────
# 섹션별 포매터
# ──────────────────────────────────────────────────

def _fmt_rdb(rdb_results: list[dict]) -> str:
    """rdb_row → (기업·연도) 단위로 묶어 한글 라벨 + 조/억 단위 변환.

    SQL이 동일 결과를 두 번 반환하는 경우도 dedup 처리한다.
    """
    # (corp_name, year) → {account_id: amount}
    groups: dict[tuple[str, Any], dict[str, Any]] = defaultdict(dict)
    for r in rdb_results:
        if r.get("type") != "rdb_row":
            continue
        val = r.get("value") or {}
        if not isinstance(val, dict):
            continue
        aid   = str(val.get("account_id") or "")
        amt   = val.get("value")
        year  = val.get("bsns_year")
        corp  = r.get("name") or "(회사 미상)"
        if aid:
            groups[(corp, year)][aid] = amt

    # 공시 메타(rdb_doc) — 그래프가 인용한 원본 공시/회사 최근 공시 목록.
    docs: list[str] = []
    seen_docs: set[tuple] = set()
    for r in rdb_results:
        if r.get("type") != "rdb_doc":
            continue
        val = r.get("value") or {}
        if not isinstance(val, dict):
            continue
        corp = r.get("name") or "(회사 미상)"
        title = str(val.get("title") or val.get("doc_type") or "").strip()
        date = str(val.get("date") or "").strip()
        rcept = str(r.get("source") or "").strip()
        if not title:
            continue
        key = (corp, title, date)
        if key in seen_docs:
            continue
        seen_docs.add(key)
        meta = " · ".join(x for x in (corp, date) if x)
        src = f" (근거: {rcept})" if rcept else ""
        docs.append(f"  - [{meta}] {title}{src}")

    if not groups and not docs:
        return ""

    lines: list[str] = []
    if groups:
        lines.append("### [정형 데이터 — 재무수치]")
        for (corp, year), metrics in sorted(groups.items(), key=lambda x: -(x[0][1] or 0)):
            lines.append(f"\n**{corp} ({year}년 기준)**")
            for aid, amt in metrics.items():
                label = _ACCOUNT_KR.get(aid, aid)
                lines.append(f"  - {label}: {_fmt_krw(amt)}")
    if docs:
        if lines:
            lines.append("")
        lines.append("### [정형 데이터 — 공시 문서]")
        lines.extend(docs[:15])
    return "\n".join(lines)


def _fmt_graph(
    graph_facts: list[dict],
    graph_paths: list,
    graph_provenance: list[str],
) -> str:
    """graph_facts를 타입별로 분리해 의미 있는 텍스트로 변환.

    fin_metric  → 재무수치 (한글 라벨 + 단위)
    shareholder / subsidiary / investment → 방향 관계 (from → to, 지분율)
    person      → 임원 목록 (이름 + 직책)
    organization→ 생략 (단순 노드, 관계 없이 나오면 정보가 없음)
    """
    fin: dict[tuple[Any, str], Any] = {}   # (year, account_id) → amount
    rels: list[str] = []
    seen_rels: set[tuple] = set()
    execs: list[str] = []
    seen_execs: set[str] = set()

    for r in graph_facts:
        t = r.get("type") or ""
        extra = r.get("extra") or {}

        # ── 재무지표 노드 ──────────────────────────────
        if t == "fin_metric":
            aid  = str(extra.get("account_id") or r.get("name") or "")
            amt  = extra.get("value") if "value" in extra else r.get("value")
            year = extra.get("bsns_year")
            if aid and amt is not None:
                fin[(year, aid)] = amt

        # ── 관계 엣지 (주주/자회사/투자) ────────────────
        elif t in ("shareholder", "subsidiary", "investment",
           "supply", "related_party", "interlocking_directorate"):
            rel_type  = str(extra.get("rel_type") or t)
            from_name = str(extra.get("from_name") or "")
            to_name   = str(extra.get("to_name") or "")
            pct       = extra.get("qota_rt") or r.get("value")
            if not from_name or not to_name:
                continue
            key = (from_name, to_name, rel_type)
            if key in seen_rels:
                continue
            seen_rels.add(key)
            lbl = _humanize_rel(rel_type)
            line = f"  - {from_name} →({lbl})→ {to_name}"
            try:
                fv = float(pct)
                if fv > 0:
                    line += f" [지분 {fv:.2f}%]"
            except (TypeError, ValueError):
                pass
            rels.append(line)

        # ── 임원 ────────────────────────────────────────
        elif t == "person":
            name = r.get("name") or ""
            pos  = extra.get("pos") or ""
            if name and name not in seen_execs:
                seen_execs.add(name)
                execs.append(f"  - {name}{f' ({pos})' if pos else ''}")

        # organization 은 단순 노드 — 관계 없이 나오면 정보가 없으므로 생략

    parts: list[str] = []

    if fin:
        parts.append("### [관계망 — 재무지표 (Neo4j)]")
        # 연도별로 묶어 정렬
        by_year: dict[Any, dict[str, Any]] = defaultdict(dict)
        for (year, aid), amt in fin.items():
            by_year[year][aid] = amt
        for year in sorted(by_year.keys(), key=lambda y: -(y or 0)):
            parts.append(f"\n**{year}년 주요 재무지표:**")
            for aid, amt in by_year[year].items():
                label = _ACCOUNT_KR.get(aid, aid)
                parts.append(f"  - {label}: {_fmt_krw(amt)}")

    if rels:
        if parts:
            parts.append("")
        parts.append("### [관계망 — 기업 관계]")
        parts.extend(rels)

    if execs:
        if parts:
            parts.append("")
        parts.append(f"### [관계망 — 임원] (상위 {min(len(execs), 10)}명)")
        parts.extend(execs[:10])

    if graph_provenance:
        parts.append(f"\n(관계망 근거 공시: {', '.join(map(str, graph_provenance[:5]))})")

    return "\n".join(parts)


def _fmt_vec(vec_results: list[dict]) -> str:
    """Vector 청크 → 출처 헤더 + 본문 요약."""
    if not vec_results:
        return ""
    lines = ["### [비정형 문서 — 공시 원문]"]
    for res in vec_results:
        extra = res.get("extra") or {}
        ctx = [str(x) for x in (
            res.get("name"),
            extra.get("year"),
            extra.get("doc_type"),
            extra.get("section_path"),
        ) if x not in (None, "")]
        header = f"[{' · '.join(ctx)}] " if ctx else ""
        lines.append(f"- {header}{_truncate(str(res.get('value') or ''), 800)}")
    return "\n".join(lines)


def _fmt_community(community_results: list[dict]) -> str:
    """GraphRAG Global Search 결과(군집별 LLM 요약) → 매크로/업계 종합 텍스트.

    intent="global" 경로에서만 채워진다. 다른 소스가 비어 있어도 이 블록만으로
    gen 이 업계 전체 구조를 종합할 수 있다.
    """
    if not community_results:
        return ""
    lines = ["### [업계/그룹 종합 — Community (GraphRAG Global)]"]
    for res in community_results:
        extra = res.get("extra") or {}
        size = extra.get("size")
        size_txt = f" ({size}사)" if size else ""
        name = res.get("name") or "(군집 미상)"
        lines.append(f"\n**{name}{size_txt}**")
        lines.append(str(res.get("value") or "").strip())
    return "\n".join(lines)


# ──────────────────────────────────────────────────
# GEN 노드
# ──────────────────────────────────────────────────

def generate_report_node(state: AgentState):
    """Gen (Generate & Render):
    rdb_results / vec_results / graph_facts / community_results 를 각 타입에 맞게
    포매팅해 LLM 에 넘기고 최종 분석 보고서를 생성한다.
    """
    rdb_results       = state.get("rdb_results", []) or []
    vec_results       = state.get("vec_results", []) or []
    graph_facts       = state.get("graph_facts", []) or []
    graph_paths       = state.get("graph_paths", []) or []
    graph_provenance  = state.get("graph_provenance", []) or []
    community_results = state.get("community_results", []) or []

    sections: list[str] = []

    community_text = _fmt_community(community_results)
    rdb_text       = _fmt_rdb(rdb_results)
    vec_text       = _fmt_vec(vec_results)
    graph_text     = _fmt_graph(graph_facts, graph_paths, graph_provenance)

    # 매크로/업계(global) 답변에선 군집 요약이 본론. 로컬 답변에선 비어 있어
    # 자연스럽게 정형/관계망/원문 순으로 떨어진다.
    if community_text:
        sections.append(community_text)
    if rdb_text:
        sections.append(rdb_text)
    if graph_text:
        sections.append(graph_text)
    if vec_text:
        sections.append(vec_text)

    draft_info = "\n\n".join(sections).strip()

    prefs    = state.get("user_preferences", {})
    tone     = prefs.get("tone", "전문적이고 신뢰감 있는")
    # 그래프 노드와 동일한 결정적 질의 — 재구성이 그룹 범위어(계열사)를 특정 관계어
    # (종속회사)로 좁히면 원문을 써, 본문이 그래프(1위=그룹 전체 최댓값)와 어긋나지 않게.
    question = effective_query(state)
    if not question:
        for msg in reversed(state.get("messages", []) or []):
            if getattr(msg, "type", "") == "human":
                question = str(msg.content)
                break

    system_prompt = f"""당신은 기업 공시 및 재무 분석 전문 AI 어시스턴트입니다.
[분석 기초 자료]를 바탕으로 [사용자 질문]에 대한 심층 분석 답변을 작성하세요.

[사용자 맞춤 설정]
- 말투 및 톤: {tone}

[작성 지침]
1. **분석 중심으로 작성하세요.**
   수치를 단순 나열하지 말고 그 의미와 시사점을 해석하세요.
   자료 내에서 도출 가능한 지표(영업이익률·부채비율·증감 등)는 직접 계산해 언급하세요.

2. **기업 관계(종속회사·주주·투자 관계)는 별자리 그래프로 시각화되어 사용자에게 제공됩니다.**
   관계를 단순 나열하지 말고, 해당 구조의 사업적 의미·맥락을 설명하세요.
   (예: "A가 B의 자회사" → "A를 통해 XXX 사업을 수직계열화하여 원가 경쟁력을 확보")

3. **재무지표는 차트로 별도 시각화되어 제공됩니다.**
   핵심 수치를 인용하되, 수치보다는 재무 건전성·수익 구조·특이사항 해석에 집중하세요.
   수치는 조원/억원 단위로 자연스럽게 표현하고, 원시 IFRS 코드나 원화 단위 숫자는 사용하지 마세요.

4. **[업계/그룹 종합] 블록이 있으면** 그 군집 요약을 토대로 업계·계열 구조의 전반을 설명하세요.
   특정 회사 단일 분석이 아니라 구성·관계 분포의 의미를 짚어 주세요.

5. [분석 기초 자료]에 없는 내용(사전 학습 지식·추정 수치)은 사용하지 마세요.
   자료로 다루지 못한 부분은 솔직하게 밝히세요.

6. 마크다운(##헤더·**볼드**·표)으로 가독성을 높이세요.

7. "알겠습니다" 같은 서론 없이 바로 시작하세요.
"""

    human_prompt = f"[사용자 질문]\n{question}\n\n[분석 기초 자료]\n{draft_info}"

    messages_to_llm = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    print(f"📝 [Gen Node] 최종 보고서 작성 중... (톤: {tone})")
    response_text = llm.invoke(messages_to_llm).content

    return {
        "messages": [AIMessage(content=response_text)],
        "final_draft": response_text,
    }
