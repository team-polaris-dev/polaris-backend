# core/digest.py — 검색 근거(문서)들을 LLM 으로 하나의 통합 정리본으로 합친다.
"""우측 패널의 'AI 통합 근거 정리'를 만든다.

serialize.build_documents 가 이미 정리한 documents(중복제거·표 변환·원문 정리 완료)를
입력으로, 별도 LLM 호출로 흩어진 근거를 주제별 한 편의 글로 통합한다. 분석(메인 답변,
nodes/render.py)과 달리 '근거 자료 정리'에 집중하고, 원문 사실만 쓰며 각 항목에 출처를
표기한다. 실패하면 빈 문자열을 돌려 패널이 정리본 없이도 동작하게 한다.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from config.llm import llm

# 문서별 본문은 길 수 있어 정리 입력에선 앞부분만 사용(통합·요약이 목적).
_PER_DOC_LIMIT = 1800

_SYSTEM_PROMPT = """당신은 기업 공시 근거에서 데이터만 뽑아 나열하는 추출기입니다.
아래 [검색된 공시 근거]에 명시된 데이터를 그대로 나열하세요. 글이 아니라 데이터 시트처럼.

[규칙]
1. 완결된 문장을 쓰지 마세요. 서술어·연결어·군더더기('~입니다', '~했습니다', '한편', '또한', '그리고', '~로 나타났다' 등) 금지.
2. 형식은 `항목: 값` 또는 `대상 — 관계 — 대상 (수치)` 또는 마크다운 표. 한 줄 = 데이터 한 건.
   예) `매출액: 333조원 (2025)`  /  `삼성디스플레이㈜ — 종속회사 — 지분 84.8%`  /  `과징금: 8.41억원 (2023.08, 개인정보보호위)`
3. 자료에 명시된 값만. 추측·해석·평가·전망·분석·설명 절대 금지(분석은 메인 답변이 따로 합니다).
4. 수치·기업명·고유명사·날짜·단위는 원문 그대로. 같은 데이터가 여러 출처에 나오면 한 번만.
5. 종류가 같은 데이터가 여러 건이면 마크다운 표로 묶으세요. 다른 주제는 짧은 소제목(**굵게**) 한 줄로 구분 가능.
6. 출처(보고서명)는 쓰지 마세요 — 출처는 화면에서 따로 표시됩니다.
7. 서론·맺음말·설명 문장 없이 데이터 항목부터 바로. 한국어.
"""


def _source_label(doc: dict) -> str:
    corp = str(doc.get("corp_name") or "").strip()
    label = str(doc.get("title") or doc.get("doc_type") or "문서").strip()
    sec = str(doc.get("section_path") or "").strip()
    parts = [p for p in (corp, label) if p]
    head = " ".join(parts) if parts else "문서"
    return f"{head} · {sec}" if sec else head


def build_evidence_digest(question: str, documents: list[dict]) -> str:
    """documents → LLM 통합 근거 정리본(마크다운). 문서 없거나 실패하면 ''."""
    if not documents:
        return ""

    blocks: list[str] = []
    for d in documents:
        text = str(d.get("text") or d.get("summary") or "").strip()
        if not text:
            continue
        blocks.append(f"[출처: {_source_label(d)}]\n{text[:_PER_DOC_LIMIT]}")
    if not blocks:
        return ""

    sources = "\n\n".join(blocks)
    human = f"[사용자 질문]\n{question or '(질문 없음)'}\n\n[검색된 공시 근거]\n{sources}"

    try:
        result = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)]
        )
        return str(getattr(result, "content", "") or "").strip()
    except Exception as e:  # LLM 실패해도 패널은 원문 카드로 동작해야 한다
        print(f"⚠️ 통합 근거 정리본 생성 실패(무시): {e}")
        return ""
