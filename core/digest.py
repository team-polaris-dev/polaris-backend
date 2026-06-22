# core/digest.py — 검색 근거(공시 원문)들을 LLM 으로 읽기 좋게 '정리'해 준다.
"""우측 패널의 'AI가 정리한 원문'을 만든다.

serialize.build_documents 가 모은 documents(공시 원문 청크)를 입력으로, 별도 LLM 호출로
원문을 읽기 좋게 다듬는다. 요약(내용 압축)이 아니라 '형식 교정'이 목적이다 — 깨진
줄바꿈·분리된 숫자·HTML 엔티티를 복원하고, 섹션 소제목으로 구조화하며, 핵심 수치·항목은
표/굵게로 강조한다. 원문에 없는 내용·해석은 더하지 않는다. 분석(메인 답변, nodes/render.py)
과는 별개이며, 실패하면 빈 문자열을 돌려 패널이 정리본 없이도 동작하게 한다.
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from config.llm import llm

# 문서별 본문은 길 수 있어 입력 한도를 둔다(원문 정리가 목적이라 넉넉히).
_PER_DOC_LIMIT = 2600

_SYSTEM_PROMPT = """당신은 기업 공시 원문을 읽기 좋게 정리하는 편집기입니다.
아래 [정리할 공시 원문]을 보고서별로, 사용자가 읽기 쉽도록 다듬어 주세요. '요약'이 아니라 '정리'입니다.

[규칙]
0. 사용자 질문에 답하지 마세요. '~에 대한 내용은 없습니다', '직접 확인되는 문장은 없습니다', '관련 자료가 없습니다' 같은 평가·판단·부재(없음) 언급을 절대 쓰지 마세요. 원문에 실제로 있는 사실만, 있는 그대로 정리하면 됩니다.
1. 원문 내용을 보존하세요. 문장·서술을 임의로 줄이거나 빼지 말고, 원문에 없는 내용·해석·평가·전망을 더하지 마세요.
2. 형식만 교정하세요:
   - 문장 중간에 끊긴 줄바꿈을 잇고, 줄로 분리된 숫자(예: "총 / 47개의")를 원래 문장에 붙이세요.
   - HTML 엔티티(&#x27; 등)와 깨진 기호를 올바른 문자로 복원하세요.
   - 붙어버린 문장('…있습니다.미주에는')은 사이를 띄워 주세요.
3. **보고서별로 정리하세요.** 입력은 '===== [보고서] 회사 · 보고서명 (날짜) ====='로 구분돼 있습니다.
   각 보고서의 제목 줄은 '=====' 와 '=====' 사이의 텍스트를 **글자 그대로 복사**해 '## ' 뒤에 붙이세요
   (예: '===== [보고서] 삼성전자 · 분기보고서 (2024.09) =====' → '## 삼성전자 · 분기보고서 (2024.09)').
   제목을 바꾸거나 줄이지 말고, 그 아래에는 해당 블록 내용만 정리하세요.
   서로 다른 보고서의 내용을 한 곳에 섞지 마세요. 보고서가 등장한 순서를 유지하세요.
4. 한 보고서 안에서 섹션이 여러 개면 섹션명을 **굵은 소제목**으로 나누세요(원문의 섹션명 활용).
5. 핵심 수치·항목(매출·이익·지분율·법인 수 등)은 마크다운 **표**나 **굵게**로 눈에 띄게 정리하세요. 단 수치·기업명·고유명사·날짜·단위는 원문 그대로.
6. 같은 데이터·문장이 한 보고서 안에서 중복되면 한 번만 쓰세요.
7. '=====', '[보고서]', '[출처:]' 같은 입력 구분 기호 자체는 출력에 쓰지 마세요(제목으로 바꿔 표현).
8. 서론·맺음말 없이 정리된 본문부터 바로. 한국어.
"""


def _report_key(doc: dict) -> str:
    """보고서 단위 그룹 키 — 같은 공시(rcept_no)면 한 보고서. 없으면 회사+보고서명."""
    rcept = str(doc.get("rcept_no") or "").strip()
    if rcept:
        return f"rcept:{rcept}"
    corp = str(doc.get("corp_name") or "").strip()
    title = str(doc.get("title") or doc.get("doc_type") or "").strip()
    return f"name:{corp}|{title}"


def _report_header(doc: dict) -> str:
    """보고서 입력 헤더 — '회사 · 보고서명 (날짜)'."""
    corp = str(doc.get("corp_name") or "").strip()
    title = str(doc.get("title") or doc.get("doc_type") or "문서").strip()
    date = str(doc.get("date") or "").strip()
    head = " · ".join(p for p in (corp, title) if p) or "문서"
    return f"{head} ({date})" if date else head


def _dart_url(rcept_no: str) -> str:
    """14자리 접수번호 → DART 공식 공시 뷰어 URL(프론트 dartUrl 과 동일)."""
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"


_HEADING_RE = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")


def _match_report(heading: str, groups: list[dict]) -> dict | None:
    """제목 텍스트 → 보고서 그룹. 정확 일치 우선, 안 되면 공백 제거 후 포함 관계로."""
    for g in groups:
        if g["header"].strip() == heading:
            return g
    norm = re.sub(r"\s+", "", heading)
    for g in groups:
        gh = re.sub(r"\s+", "", g["header"])
        if gh and (gh in norm or norm in gh):
            return g
    return None


def _linkify_report_headings(text: str, groups: list[dict]) -> str:
    """LLM 이 낸 '## 회사 · 보고서명 (날짜)' 제목 줄을 그 보고서의 DART 원문 링크로 바꾼다.

    URL 은 LLM 에 맡기지 않고 rcept_no 로 결정적으로 만든다. 제목이 매칭되지 않으면
    그대로 둔다(링크 없이도 본문은 정상). 이미 링크([..])인 줄은 건너뛴다.
    """
    linkable = [g for g in groups if g.get("rcept_no")]
    if not linkable:
        return text
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if not m:
            continue
        heading = m.group(2).strip()
        if heading.startswith("["):
            continue
        g = _match_report(heading, linkable)
        if g:
            lines[i] = f"{m.group(1)} [{heading}]({_dart_url(g['rcept_no'])})"
    return "\n".join(lines)


def build_evidence_digest(question: str, documents: list[dict]) -> str:
    """documents(공시 원문) → LLM 으로 보고서별로 읽기 좋게 정리한 본문(마크다운). 문서 없거나 실패하면 ''."""
    if not documents:
        return ""

    # 보고서 단위로 묶는다 — 등장 순서 유지. 재무 요약 카드(rdb)는 우측 패널의
    # '재무지표 차트'(막대그래프)로 따로 보여주므로 정리 원문에선 제외한다.
    from collections import OrderedDict

    groups: "OrderedDict[str, dict]" = OrderedDict()
    for d in documents:
        if d.get("source_kind") == "rdb":
            continue
        text = str(d.get("text") or d.get("summary") or "").strip()
        if not text:
            continue
        text = text[:_PER_DOC_LIMIT]
        key = _report_key(d)
        g = groups.get(key)
        if g is None:
            g = {
                "header": _report_header(d),
                "rcept_no": str(d.get("rcept_no") or "").strip(),
                "sections": [],
            }
            groups[key] = g
        sec = str(d.get("section_path") or "").strip()
        g["sections"].append((sec or "본문", text))

    if not groups:
        return ""

    # 보고서별로 ===== 구분선을 넣어 경계를 명확히 한다.
    parts: list[str] = []
    for g in groups.values():
        lines = [f"===== [보고서] {g['header']} ====="]
        for sec, text in g["sections"]:
            lines.append(f"[{sec}]\n{text}")
        parts.append("\n\n".join(lines))

    # 질문은 일부러 넣지 않는다 — 넣으면 LLM 이 '질문에 답이 되나'를 판단해
    # "관련 내용 없음" 같은 평가 문장을 만들어버린다. 정리는 질문과 무관하게 원문만 다듬는다.
    body = "\n\n".join(parts)
    human = f"[정리할 공시 원문]\n다음은 보고서별로 구분된 원문입니다. 각 ===== 블록의 경계를 지켜 보고서별로 정리하세요.\n\n{body}"

    try:
        result = llm.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)]
        )
        out = str(getattr(result, "content", "") or "").strip()
        # 보고서 제목 줄에 DART 원문 링크를 결정적으로 붙인다.
        return _linkify_report_headings(out, list(groups.values())) if out else ""
    except Exception as e:  # LLM 실패해도 패널은 원문 카드로 동작해야 한다
        print(f"⚠️ 원문 정리본 생성 실패(무시): {e}")
        return ""
