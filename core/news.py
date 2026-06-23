# core/news.py — 뉴스 + DART 공시 청크를 LLM 으로 연관 분석한다(digest.py 의 쌍둥이).
"""뉴스 분석 탭의 2단계: 수집된 최근 뉴스를 우리 DART 데이터(공시)와 연결해 기업별로 분석한다.

설계(2026-06-22-news-analysis-tab-design.md §2-결정3, §7):
- 질문에서 추출한 기업(들) 각각에 대해:
  1) tool/news_client.fetch_company_news 로 최근 뉴스 N건 수집
  2) 그 기업의 뉴스 텍스트로 search_vector_db(corp_codes=[code]) 1회 → 관련 DART 청크 2~3건
  3) 뉴스 + DART 청크를 함께 LLM 1회에 넣어 "최근 동향 요약 + 공시 근거 연결" 생성
- digest.py 처럼 JSON only 강제, 실패 시 빈 결과로 격리(챗봇 본체·기존 패널 무영향).

companies_from_query 는 매핑을 인자로 주입받는 **순수 함수**라 단위 테스트 가능하다(§5).
build_news_analysis 는 가짜 뉴스/청크 주입으로도 검증 가능하게 fetch/search 를 인자로 받는다.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from config.llm import json_llm
from tool.news_client import NewsItem, fetch_company_news

# 분석 기업 상한 — 4사 이상 언급 시 등장순 상위 N개만(API·LLM 비용 통제, §5).
_MAX_COMPANIES = int(os.getenv("NEWS_MAX_COMPANIES", "3"))
# 기업당 표시 기사 수(최근 N일 윈도우 내). 메타데이터는 네이버 API 가 공짜로 줘서 늘려도 싸다.
_PER_COMPANY = int(os.getenv("NEWS_PER_COMPANY", "12"))
# 기업별 DART 청크 수 — 공시 연관 근거용. 기본 3(균형: 입력↓로 속도). 더 자세히는 4~5.
_DART_CHUNKS = int(os.getenv("NEWS_DART_CHUNKS", "3"))
# LLM 입력 절단 — 본문/청크 길이 한도(컨텍스트 폭주 방지).
_BODY_INPUT_LIMIT = 600
_CHUNK_INPUT_LIMIT = 400

_VALID_SENTIMENT = {"positive", "negative", "neutral"}
# 카드뉴스 이벤트 유형 — 프론트가 이 값으로 lucide 아이콘을 매핑한다. 밖이면 "기타"로 강제.
_VALID_EVENT_TYPE = {"실적", "증설", "HBM·신제품", "인사", "리스크", "기타"}
# 기업당 카드 상한 — 프롬프트로도 막지만 코드에서 한 번 더 잘라 출력 폭주를 방지.
_MAX_CARDS_PER_COMPANY = 3
_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL | re.IGNORECASE)


def _parse_llm_json(raw: str) -> dict:
    """LLM 출력에서 JSON 객체를 추출한다. 코드펜스·앞뒤 잡설을 제거하고,
    잘린 JSON은 마지막 완성된 `}` 위치까지 잘라 재시도한다.
    """
    # 코드펜스 제거
    fence = _FENCE_RE.search(raw)
    if fence:
        raw = fence.group(1)
    # 바깥 { ... } 추출
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("JSON 구조 없음")
    candidate = raw[start : end + 1]
    # 1차 시도
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # 2차 시도: 마지막 ] } 로 잘린 경우 — companies 배열만이라도 살린다
    # 잘린 companies 배열을 강제로 닫아 파싱 시도
    for suffix in ["]}", "]}}"]:
        try:
            return json.loads(candidate + suffix)
        except json.JSONDecodeError:
            pass
    # 3차: companies 배열 안에서 완성된 객체만 수집
    m = re.search(r'"companies"\s*:\s*\[', candidate)
    if m:
        arr_start = m.end() - 1  # '[' 위치
        depth = 0
        objs: list[str] = []
        buf: list[str] = []
        in_str = False
        esc = False
        for i, ch in enumerate(candidate[arr_start:], arr_start):
            if esc:
                buf.append(ch); esc = False; continue
            if ch == "\\" and in_str:
                buf.append(ch); esc = True; continue
            if ch == '"':
                in_str = not in_str
            if not in_str:
                if ch == "{":
                    if depth == 0:
                        buf = ["{"]
                    else:
                        buf.append(ch)
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    buf.append(ch)
                    if depth == 0:
                        try:
                            json.loads("".join(buf))
                            objs.append("".join(buf))
                        except json.JSONDecodeError:
                            pass
                        buf = []
                elif ch == "]" and depth == 0:
                    break
                else:
                    buf.append(ch)
            else:
                buf.append(ch)
        if objs:
            return {"companies": [json.loads(o) for o in objs]}
    raise ValueError(f"JSON 복구 실패: {candidate[:200]}")


class NewsArticleCard(TypedDict):
    title: str
    url: str
    press: str
    pub_date: str
    sentiment: str          # 기사별 논조
    evidence: list[str]     # 연결된 DART 근거 라벨


class NewsTopicCard(TypedDict):
    headline: str           # LLM — 정제된 한 줄 제목(6~14자)
    desc: str               # LLM — 두 줄 정도 설명
    event_type: str         # 실적 | 증설 | HBM·신제품 | 인사 | 리스크 | 기타 (아이콘 매핑용)
    sentiment: str          # positive | negative | neutral — 카드 컬러용


class NewsAnalysisCompany(TypedDict):
    corp_name: str
    corp_code: str
    summary: str            # LLM — 최근 동향 2~3문장(마크다운)
    relevance: str          # LLM — DART 공시와의 연관 설명(마크다운)
    sentiment: str          # positive | negative | neutral — 전반 논조
    cards: list[NewsTopicCard]  # LLM — 카드뉴스용 핵심 토픽(0~3장)
    articles: list[NewsArticleCard]


# ----------------------------------------------------------------- 대상 기업 결정
def companies_from_query(
    query: str,
    name_to_code: dict[str, str],
    *,
    cap: int = _MAX_COMPANIES,
) -> list[tuple[str, str]]:
    """질문에 '언급된' 기업을 등장순으로 (corp_name, corp_code) 리스트로 돌려준다(상한 cap).

    extract_filter_signals 의 mentioned 추출 규칙(긴 이름 우선·부분문자열 후보 제외)과
    동일하되, 뉴스는 비교/소유 구분 없이 **언급된 기업 전부**를 cap 까지 대상으로 삼는다(§5).
    매핑을 인자로 주입받는 순수 함수 — 단위 테스트 가능.
    """
    if not query:
        return []
    # 대소문자 무시 매칭 — 사용자가 "sk하이닉스"처럼 소문자로 쳐도 매핑의 "SK하이닉스"와
    # 매칭되게 한다(영문 약칭 SK/LG/HBM 등은 입력 표기가 들쭉날쭉). 매칭은 lower 로 하되
    # 반환·정렬 기준은 매핑의 원래 표기를 유지한다.
    q_lower = query.lower()
    # 긴 이름 우선(예: "SK하이닉스" > "SK") — 질문 내 등장 위치로 정렬해 등장순 보존.
    candidates = sorted(
        (name for name in name_to_code if name and name.lower() in q_lower), key=len, reverse=True
    )
    mentioned: list[str] = []
    for name in candidates:
        # 이미 채택된 이름의 부분문자열인 후보는 제외(중복 회사 방지)
        if any(name in kept for kept in mentioned):
            continue
        mentioned.append(name)
    # 질문에서 처음 등장하는 위치 순으로 정렬(사용자가 먼저 말한 기업 우선)
    mentioned.sort(key=lambda n: q_lower.find(n.lower()))

    out: list[tuple[str, str]] = []
    seen_codes: set[str] = set()
    for name in mentioned:
        code = name_to_code.get(name, "")
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        out.append((name, code))
        if len(out) >= cap:
            break
    return out


# ----------------------------------------------------------------- LLM 프롬프트
_SYSTEM_PROMPT = """당신은 반도체·기업 분석가입니다. 아래 [기업별 최근 뉴스]와 [관련 공시 발췌(우리 DB)]를 읽고,
기업별로 "최근 동향 상세 분석 + 공시 근거 연결"을 만드세요.

[규칙]
0. 공시 발췌에 **실제로 있는** 사실만 근거로 연결하세요. '관련 자료 없음', '확인되지 않습니다' 같은 부재·평가 문장을 쓰지 마세요.
1. summary: 그 기업의 최근 뉴스에서 **가장 중요한 핵심 동향만** 골라 **3~5개**의 불릿(`- `)으로 **간결하게** 정리하세요(마크다운). 각 불릿은 **1문장**(필요시 짧게 2문장)으로, 핵심 사실·수치만 담고 배경 설명은 최소화하세요. 같은 사건을 다룬 기사들은 한 불릿으로 묶고, 서로 다른 주제는 별도 불릿으로 분리하세요. 사소한 기사는 과감히 생략하세요(전부 담으려 하지 말 것). 뉴스에 없는 내용은 절대 지어내지 마세요.
2. relevance: 뉴스 내용이 우리 공시 발췌와 연결되는 지점을 1~2문장으로 설명(마크다운). 연관이 약하면 빈 문자열("")로 두세요 — 억지 연결 금지.
3. sentiment: 그 기업 뉴스의 전반 논조를 "positive" | "negative" | "neutral" 중 하나로.
4. cards: 그 기업의 최근 뉴스에서 가장 중요한 핵심 토픽 **2~3개**를 카드뉴스용으로 뽑으세요(요약 불릿과 별개로, 한눈에 보는 용도).
   - headline: 정제된 한 줄 제목(6~14자, 예: "시총 역전", "HBM 증설"). 기사 제목을 그대로 베끼지 말고 핵심만 압축.
   - desc: 두 줄 정도의 짧은 설명(한 문장, 40자 내외).
   - event_type: 다음 중 하나로만 — "실적" | "증설" | "HBM·신제품" | "인사" | "리스크" | "기타".
   - sentiment: 그 토픽의 논조 "positive" | "negative" | "neutral".
   - 뉴스에 없는 내용은 만들지 마세요. 토픽이 적으면 1개만, 없으면 빈 배열 []로 두세요.
5. articles: 각 기사를 입력에 매겨진 **번호(index)** 로 참조해 {"index": 번호, "sentiment": 기사별 논조, "evidence": [공시 라벨...]} 로만 출력하세요.
   - title/url/press/pub_date 는 **출력하지 마세요**(번호로 매칭하므로 불필요 — 출력을 짧게).
   - evidence 는 입력 [관련 공시 발췌]에 제공된 라벨만 인용하세요(지어내지 말 것). 연결이 없으면 빈 배열 [].
6. 설명·서론 없이 아래 JSON 객체 하나만 출력하세요.

[출력 JSON 형식]
{
  "companies": [
    {
      "corp_name": "삼성전자",
      "summary": "- HBM ...\\n- 파운드리 ...",
      "relevance": "...",
      "sentiment": "neutral",
      "cards": [
        {"headline": "HBM 증설", "desc": "AI 수요 대응 평택 라인 증설", "event_type": "증설", "sentiment": "positive"}
      ],
      "articles": [
        {"index": 1, "sentiment": "neutral", "evidence": ["사업보고서 §II"]}
      ]
    }
  ]
}
"""


def _coerce_sentiment(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in _VALID_SENTIMENT else "neutral"


def _coerce_event_type(value: Any) -> str:
    v = str(value or "").strip()
    return v if v in _VALID_EVENT_TYPE else "기타"


def _coerce_cards(value: Any) -> list[NewsTopicCard]:
    """LLM 의 cards 배열을 검증해 안전한 카드 리스트로. headline 없는 항목은 버리고 상한까지 자른다."""
    out: list[NewsTopicCard] = []
    if not isinstance(value, list):
        return out
    for c in value:
        if not isinstance(c, dict):
            continue
        headline = str(c.get("headline") or "").strip()
        if not headline:
            continue  # 제목 없는 카드는 무의미 — 버린다
        out.append(
            {
                "headline": headline[:24],
                "desc": str(c.get("desc") or "").strip()[:80],
                "event_type": _coerce_event_type(c.get("event_type")),
                "sentiment": _coerce_sentiment(c.get("sentiment")),
            }
        )
        if len(out) >= _MAX_CARDS_PER_COMPANY:
            break
    return out


def _dart_label(chunk: dict) -> str:
    """공시 청크 → 사람이 읽는 근거 라벨(예: '삼성전자 · 사업보고서 (2023.12) §II')."""
    extra = chunk.get("extra") or {}
    name = str(chunk.get("name") or extra.get("corp_name") or "").strip()
    section = str(extra.get("section_path") or "").strip()
    # name 에 이미 '회사 · 보고서명 (날짜)' 형태가 들어오는 경우가 많다.
    label = name or "공시"
    if section:
        label = f"{label} {section}" if section not in label else label
    return label[:80]


def _build_company_input(corp_name: str, news: list[NewsItem], chunks: list[dict]) -> str:
    """기업 한 곳의 LLM 입력 블록을 만든다(뉴스 목록 + 공시 발췌)."""
    lines = [f"[기업] {corp_name}", "[최근 뉴스] (각 기사를 맨 앞 번호로 참조하세요)"]
    for i, item in enumerate(news, 1):
        body = (item.get("body") or item.get("description") or "").strip()[:_BODY_INPUT_LIMIT]
        head = " ".join(p for p in (item.get("pub_date", ""), item.get("press", "")) if p)
        lines.append(f'{i}. ({head}) "{item.get("title", "")}"')
        if body:
            lines.append(f"   본문: {body}")
    lines.append("[관련 공시 발췌 (우리 DB)]")
    if chunks:
        for c in chunks:
            text = str(c.get("value") or (c.get("extra") or {}).get("text") or "").strip()
            if isinstance(c.get("value"), dict):
                text = str(c["value"].get("embedding_text") or c["value"].get("text") or "").strip()
            lines.append(f"- {_dart_label(c)}: {text[:_CHUNK_INPUT_LIMIT]}")
    else:
        lines.append("- (관련 공시 없음)")
    return "\n".join(lines)


def _fallback_company(
    corp_name: str, corp_code: str, news: list[NewsItem]
) -> NewsAnalysisCompany:
    """LLM 실패 시에도 뉴스 카드는 보여주는 최소 결과(분석 텍스트만 비움)."""
    return {
        "corp_name": corp_name,
        "corp_code": corp_code,
        "summary": "",
        "relevance": "",
        "sentiment": "neutral",
        "cards": [],
        "articles": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "press": item.get("press", ""),
                "pub_date": item.get("pub_date", ""),
                "sentiment": "neutral",
                "evidence": [],
            }
            for item in news
        ],
    }


def _merge_llm_company(
    corp_name: str,
    corp_code: str,
    news: list[NewsItem],
    llm_obj: dict,
) -> NewsAnalysisCompany:
    """LLM 결과를 입력 뉴스와 **번호(index)** 로 병합. title/url/press/pub_date 는 항상 입력값.

    카드는 입력 뉴스 전부(표시할 N건)로 만들고, LLM 이 index 로 짚어준 기사에만 sentiment/
    evidence 를 채운다. LLM 이 일부만 짚었거나 비워도 나머지 기사는 neutral 로 그대로 표시된다
    (그래서 "더 많은 뉴스"를 다 보여줄 수 있고, LLM 출력이 짧아 잘림에도 강하다).
    """
    by_index: dict[int, dict] = {}
    for art in llm_obj.get("articles") or []:
        try:
            by_index[int(art.get("index"))] = art
        except (TypeError, ValueError):
            continue
    cards: list[NewsArticleCard] = []
    for i, item in enumerate(news, 1):
        art = by_index.get(i, {})
        ev = art.get("evidence") or []
        cards.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "press": item.get("press", ""),
                "pub_date": item.get("pub_date", ""),
                "sentiment": _coerce_sentiment(art.get("sentiment")),
                "evidence": [str(e).strip() for e in ev if str(e).strip()][:4],
            }
        )
    return {
        "corp_name": corp_name,
        "corp_code": corp_code,
        "summary": str(llm_obj.get("summary") or "").strip(),
        "relevance": str(llm_obj.get("relevance") or "").strip(),
        "sentiment": _coerce_sentiment(llm_obj.get("sentiment")),
        "cards": _coerce_cards(llm_obj.get("cards")),
        "articles": cards,
    }


# ----------------------------------------------------------------- 수집 / 카드 / 분석 (점진 렌더링 분리)
def collect_company_news(
    companies: list[tuple[str, str]],
    *,
    fetch_news: Callable[..., list[NewsItem]] = fetch_company_news,
    body_k: int | None = None,
) -> list[dict]:
    """기업 목록 → 기업별 최근 뉴스만 수집(벡터검색·LLM 없음). 점진 렌더링 1단계의 빠른 경로.

    반환: [{"corp_name", "corp_code", "news": [NewsItem...]}] — 뉴스 0건 기업 제외, 등장순.
    회사별 fetch 는 병렬(기업 수 ≤ cap). 실패는 건너뛴다(빈 결과 degrade).

    body_k: 본문 fetch 상위 건수. None=fetch_news 기본값(2-arg 호출 — 주입 fake 도 안전).
    0=메타데이터만(카드 1단계용 — 본문 미수집이라 빠르고, 캐시해도 본문 영속화 없음, §불변식).
    """
    if not companies:
        return []

    def _one(corp_name: str, corp_code: str) -> dict | None:
        try:
            news = (
                fetch_news(corp_name, _PER_COMPANY)
                if body_k is None
                else fetch_news(corp_name, _PER_COMPANY, analyze_k=body_k)
            )
        except Exception as e:
            print(f"⚠️ [news] {corp_name} 뉴스 수집 실패(건너뜀): {e!r}")
            return None
        if not news:
            return None
        return {"corp_name": corp_name, "corp_code": corp_code, "news": news}

    if len(companies) == 1:
        r = _one(*companies[0])
        return [r] if r else []
    with ThreadPoolExecutor(max_workers=len(companies)) as pool:
        futures = {pool.submit(_one, name, code): (name, code) for name, code in companies}
        ordered: dict[str, dict] = {}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                ordered[r["corp_name"]] = r
    return [ordered[n] for n, _ in companies if n in ordered]


def cards_only(collected: list[dict]) -> list[NewsAnalysisCompany]:
    """수집된 뉴스 → 분석 전 '카드만'(summary/relevance 비움). 점진 렌더링 1단계 응답.

    LLM 을 돌리기 전에 뉴스 목록부터 즉시 보여주기 위함 — 기사 메타(title/url/press/date)만 채운다.
    """
    return [_fallback_company(c["corp_name"], c["corp_code"], c["news"]) for c in collected]


def analyze_collected(
    collected: list[dict],
    *,
    vector_search: Callable[..., list[dict]] | None = None,
) -> list[NewsAnalysisCompany]:
    """이미 수집된 뉴스(collected) → 기업별 공시 청크 검색 + LLM 분석 카드. 점진 렌더링 2단계.

    1단계가 캐시해 둔 원본을 그대로 받아 **네이버 재fetch 없이** 분석만 한다. 전체 실패는 빈 리스트.
    기업별 LLM 은 독립 호출(병렬 X — 순차). 벡터검색 1회 + LLM CLI 서브프로세스 1회가 기업당 메모리를
    적잖이 쓰는데, 병렬로 cap(3)개를 동시에 돌리면 로컬 PC 가용 메모리를 넘겨 LLM 호출이 한꺼번에
    OOM(MemoryError/VirtualAlloc 실패)으로 죽는 게 실측으로 확인됐다(기업이 다 나와도 summary 가 전부
    빈 채로 degrade). 기업 수와 무관하게 출력 잘림은 없으니(이미 기업별 독립 호출), 안정성을 위해
    순차로 바꾼다 — 느려지는 대신 분석이 통째로 비는 일이 없다.
    """
    if not collected:
        return []
    # 검색 함수는 지연 import — core/news 가 vector_store 무거운 의존을 강제 로드하지 않게.
    if vector_search is None:
        from tool.vector_store import search_vector_db as vector_search  # type: ignore

    def _one(c: dict) -> NewsAnalysisCompany:
        news = c["news"]
        news_text = " ".join(
            (item.get("title", "") + " " + (item.get("description") or "")) for item in news
        )[:500]
        try:
            chunks = vector_search(news_text, top_k=_DART_CHUNKS, corp_codes=[c["corp_code"]]) or []
        except Exception as e:
            print(f"⚠️ [news] {c['corp_name']} 공시 청크 검색 실패(연관 없이 진행): {e!r}")
            chunks = []
        human = (
            "다음 기업의 최근 뉴스와 관련 공시를 읽고 기업별 분석 JSON 을 만드세요.\n\n"
            + _build_company_input(c["corp_name"], news, chunks)
        )
        try:
            result = json_llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human)])
            raw = str(getattr(result, "content", "") or "").strip()
            data = _parse_llm_json(raw)
            llm_list = data.get("companies") if isinstance(data, dict) else None
            obj = llm_list[0] if isinstance(llm_list, list) and llm_list else None
            if obj:
                return _merge_llm_company(c["corp_name"], c["corp_code"], news, obj)
        except Exception as e:
            print(f"⚠️ [news] {c['corp_name']} LLM 분석 실패(뉴스 카드만 degrade): {e!r}")
        return _fallback_company(c["corp_name"], c["corp_code"], news)

    return [_one(c) for c in collected]


def build_news_analysis(
    companies: list[tuple[str, str]],
    query: str = "",
    *,
    fetch_news: Callable[[str, int], list[NewsItem]] = fetch_company_news,
    vector_search: Callable[..., list[dict]] | None = None,
) -> list[NewsAnalysisCompany]:
    """기업 목록 → 분석 카드(수집 + 벡터 + LLM 한 번에). 점진 렌더링을 안 쓰는 경로/캐시 미스 폴백.

    companies: (corp_name, corp_code) 리스트(이미 cap 적용됨).
    fetch_news/vector_search 는 테스트에서 가짜로 주입 가능(기본은 실서비스 함수).
    """
    collected = collect_company_news(companies, fetch_news=fetch_news)
    return analyze_collected(collected, vector_search=vector_search)
