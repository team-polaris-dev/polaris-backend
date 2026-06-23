# tool/news_client.py — 네이버 검색 API + 인링크 본문 fetch (순수 데이터 수집, LLM 없음)
"""뉴스 분석 탭의 1단계: 기업명으로 최근 뉴스를 수집한다.

네이버 검색 API(`openapi.naver.com/v1/search/news.json`)는 제목 + ~100자 description 만
주므로 분석 근거로 부족하다 → API 가 주는 인링크(`n.news.naver.com`)로 **본문을 따로
fetch** 한다. 인링크는 HTML 구조가 일정해 파싱이 안정적이다(상위 CLAUDE.md §collect/fetch).

설계 결정(2026-06-22-news-analysis-tab-design.md §2-결정2, §6):
- 인링크 필터: `link` 이 `n.news.naver.com` 패턴인 것만 통과. 아웃링크는 폐기.
- graceful degrade: 본문 fetch 실패 시 제목+description 만으로 진행(0건 처리 안 함).
- 키 없거나 NEWS_ENABLED!=true 면 빈 리스트(기능 자체 OFF — §0, §9-5).
- 본문 fetch 는 순차(N≤5 라 병렬 불필요, YAGNI). sync 함수라 FastAPI 스레드풀에서 돈다.

이 모듈은 LLM 을 호출하지 않는다(연관 분석은 core/news.py 담당). API 키만 있으면 단독 실행 가능.
"""
from __future__ import annotations

import html
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict

import httpx

# C:\DART\news_fetcher.py 의 본문 셀렉터를 재활용(인링크 외 폴백용). trafilatura 우선.
_BODY_SELECTORS = [
    "article",
    "#dic_area",            # 네이버 뉴스 본문(현행 인링크 구조)
    "#articleBodyContents",
    "#newsct_article",
    "#articeBody",
    ".article_view",
    ".newsct_article",
    "#article-view-content-div",
    ".article_body",
    "#article_content",
]

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}

_NAVER_NEWS_API = "https://openapi.naver.com/v1/search/news.json"
# 인링크 판정 — 모바일/PC 양쪽 도메인 모두 CP 제휴 인링크다.
_INLINK_RE = re.compile(r"https?://n\.news\.naver\.com/", re.IGNORECASE)
# 인링크 URL 의 oid(언론사 코드) — best-effort 언론사 식별용.
# URL 형식: https://n.news.naver.com/mnews/article/{oid}/{aid}
# → /article/ 뒤의 숫자를 잡는다.
_OID_RE = re.compile(r"/article/(\d+)/")
_TAG_RE = re.compile(r"<[^>]+>")

# 본문 입력 한도 — LLM 컨텍스트 폭주 방지(분석 입력으로만 쓰고 화면엔 안 띄움, §9-7).
_BODY_LIMIT = int(os.getenv("NEWS_BODY_LIMIT", "1200"))
# 본문 HTTP fetch 타임아웃 — 느린 언론사는 빨리 포기하고 description 으로 degrade(체감 속도↑).
_FETCH_TIMEOUT = float(os.getenv("NEWS_FETCH_TIMEOUT", "3.0"))
# 네이버 검색 API 호출 타임아웃 — 1회 호출이라 본문 fetch 보다 여유.
_API_TIMEOUT = float(os.getenv("NEWS_API_TIMEOUT", "6.0"))
# 최근 며칠치 뉴스만 — "3일치" 윈도우(pub_date 기준). 0 이면 날짜 필터 끔.
_LOOKBACK_DAYS = int(os.getenv("NEWS_LOOKBACK_DAYS", "3"))
# 본문까지 fetch 해 '깊게' 분석할 상위 기사 수 — 나머지는 제목+description 만(속도 고정).
# 표시는 n 건 다 하되, 느린 본문 HTTP 는 이 수만큼만 → 기사 수를 늘려도 지연이 안 늘어난다.
# 기본 3(균형) — LLM 입력을 줄여 속도를 잡는다. 더 자세히는 5+, 더 빠르게는 2.
_ANALYZE_BODY_K = int(os.getenv("NEWS_ANALYZE_BODY_K", "3"))
# 날짜 윈도우가 너무 비면(조용한 주) 최신 이 개수까지는 보장(빈 탭 방지).
_MIN_ITEMS = int(os.getenv("NEWS_MIN_ITEMS", "3"))


class NewsItem(TypedDict):
    title: str           # 네이버 API title (HTML 태그 제거)
    url: str             # n.news.naver.com 인링크
    description: str     # 네이버 API description (~100자)
    body: str            # 인링크 본문 (fetch 성공 시, 실패 시 "")
    press: str           # 언론사명 (URL oid 또는 본문에서, best-effort)
    pub_date: str        # YYYY-MM-DD


def news_enabled() -> bool:
    """기능 토글 — NEWS_ENABLED=true 이고 네이버 키가 둘 다 있어야 켜진다(§9-5).

    기본 OFF. 키가 없거나 토글이 false 면 fetch_company_news 가 빈 리스트를 돌려준다.
    발표 환경에서만 .env 로 켠다.
    """
    if os.getenv("NEWS_ENABLED", "").strip().lower() != "true":
        return False
    return bool(_client_id()) and bool(_client_secret())


def _client_id() -> str:
    return (os.getenv("NAVER_CLIENT_ID") or "").strip()


def _client_secret() -> str:
    return (os.getenv("NAVER_CLIENT_SECRET") or "").strip()


def _strip_tags(text: str) -> str:
    """네이버 API title/description 의 <b> 강조 태그·HTML 엔티티 제거."""
    return html.unescape(_TAG_RE.sub("", text or "")).strip()


def _press_from_oid(url: str) -> str:
    """인링크 URL 의 oid → 언론사명. 매핑에 없으면 빈 문자열(best-effort)."""
    m = _OID_RE.search(url or "")
    if not m:
        return ""
    return _OID_PRESS.get(m.group(1), "")


# 상위 CLAUDE.md §5-2 화이트리스트 언론사 oid (인링크 샘플에서 확인된 것만).
# 추측 하드코딩 금지 원칙이라 best-effort 표시용으로만 쓰고, 없으면 빈 문자열로 둔다.
_OID_PRESS = {
    "001": "연합뉴스",
    "421": "뉴스1",
    "003": "뉴시스",
    "015": "한국경제",
    "009": "매일경제",
    "011": "서울경제",
    "008": "머니투데이",
    "018": "이데일리",
    "030": "전자신문",
    "029": "디지털타임스",
    "092": "ZDNet Korea",
}


def _normalize_pub_date(pub_date: str) -> str:
    """네이버 API 의 RFC1123 pubDate(예: 'Mon, 21 Jun 2026 09:00:00 +0900') → YYYY-MM-DD."""
    if not pub_date:
        return ""
    try:
        from email.utils import parsedate_to_datetime

        dt = parsedate_to_datetime(pub_date)
        return dt.strftime("%Y-%m-%d") if dt else ""
    except (TypeError, ValueError):
        return ""


def _cutoff_date(days: int) -> str:
    """오늘 포함 최근 `days` 일의 시작 날짜(YYYY-MM-DD). days=3 → 오늘과 이틀 전까지.

    days<=0 이면 빈 문자열(날짜 필터 비활성) — 모든 기사 통과.
    """
    if days <= 0:
        return ""
    from datetime import date, timedelta

    return (date.today() - timedelta(days=days - 1)).isoformat()


def _within_lookback(pub_date: str, cutoff: str) -> bool:
    """pub_date(YYYY-MM-DD) 가 cutoff 이상이면 True(ISO 날짜는 사전식 비교가 곧 시간 비교).

    cutoff 가 빈 값이면 항상 통과. pub_date 가 빈 값(파싱 실패)이면 보수적으로 통과(드롭 안 함).
    """
    if not cutoff or not pub_date:
        return True
    return pub_date >= cutoff


def _extract_body(url: str) -> str:
    """인링크 본문 추출 — trafilatura 우선, 실패 시 bs4 셀렉터 폴백. 실패하면 ""(graceful)."""
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=_FETCH_TIMEOUT, follow_redirects=True)
        resp.raise_for_status()
        raw_html = resp.text
    except httpx.HTTPError:
        return ""

    # 1순위: trafilatura (일반화된 본문 추출 — 상위 CLAUDE.md §fetch 근거)
    try:
        import trafilatura

        extracted = trafilatura.extract(raw_html, include_comments=False, include_tables=False)
        if extracted and len(extracted.strip()) >= 100:
            return extracted.strip()[:_BODY_LIMIT]
    except Exception:
        pass

    # 2순위: bs4 셀렉터 폴백 (인링크 구조 셀렉터)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        for sel in _BODY_SELECTORS:
            tag = soup.select_one(sel)
            if tag:
                body = tag.get_text(separator="\n", strip=True)
                if len(body) > 100:
                    return body[:_BODY_LIMIT]
    except Exception:
        pass

    return ""


def _meta_item(it: dict, body: str = "") -> NewsItem:
    """네이버 API 원소 → NewsItem(본문은 인자로). 본문 없이도 카드는 완성된다(빠른 표시)."""
    link = (it.get("link") or "").strip()
    return {
        "title": _strip_tags(it.get("title", "")),
        "url": link,
        "description": _strip_tags(it.get("description", "")),
        "body": body,
        "press": _press_from_oid(link),
        "pub_date": _normalize_pub_date(it.get("pubDate", "")),
    }


def fetch_company_news(
    corp_name: str, n: int = 12, *, analyze_k: int | None = None
) -> list[NewsItem]:
    """기업명으로 최근 N일(NEWS_LOOKBACK_DAYS) 인링크 뉴스를 최대 n건 수집. 비활성/실패 시 빈 리스트.

    속도/비용 분리가 핵심:
      - **표시**는 n 건 전부(메타데이터는 네이버 API 1회로 공짜 — title/press/date/link).
      - **본문 fetch**(느린 HTTP)는 상위 analyze_k 건만 → 기사 수를 늘려도 지연이 고정된다.
      - 나머지 기사는 본문 없이 제목+description 만(LLM 이 sentiment 판단엔 충분).

    1) 네이버 검색 API(sort=date, display 크게) → 인링크만 통과 → 최근 N일 윈도우로 필터
    2) 상위 analyze_k 건만 본문 병렬 fetch(실패는 description 으로 degrade)
    """
    if not news_enabled() or not corp_name.strip():
        return []
    if analyze_k is None:
        analyze_k = _ANALYZE_BODY_K

    # 3일 윈도우를 확실히 담으려면 넉넉히 받아 날짜로 거른다(고빈도 기업 대비 display 크게).
    params = {"query": corp_name, "sort": "date", "display": 100}
    headers = {
        **_HEADERS,
        "X-Naver-Client-Id": _client_id(),
        "X-Naver-Client-Secret": _client_secret(),
    }
    try:
        resp = httpx.get(_NAVER_NEWS_API, params=params, headers=headers, timeout=_API_TIMEOUT)
        resp.raise_for_status()
        items = resp.json().get("items", [])
    except (httpx.HTTPError, ValueError) as e:
        print(f"⚠️ [news] 네이버 API 호출 실패(빈 결과로 degrade): {e!r}")
        return []

    # 인링크만(아웃링크는 본문 파싱 불안정, §6) → 최근 N일 윈도우(최신순 유지)
    inlinks = [it for it in items if _INLINK_RE.match((it.get("link") or "").strip())]
    cutoff = _cutoff_date(_LOOKBACK_DAYS)
    windowed = [
        it for it in inlinks if _within_lookback(_normalize_pub_date(it.get("pubDate", "")), cutoff)
    ]
    # 윈도우가 너무 비면(조용한 주) 최신 _MIN_ITEMS 건까지는 보장 — 탭이 비어 보이지 않게.
    if len(windowed) < _MIN_ITEMS:
        windowed = inlinks[: max(_MIN_ITEMS, len(windowed))]
    candidates = windowed[: max(1, int(n))]
    if not candidates:
        return []

    # 메타 카드 먼저(본문 "") — 상위 analyze_k 건만 본문을 병렬로 채운다(느린 HTTP 고정).
    results: list[NewsItem] = [_meta_item(it) for it in candidates]
    body_targets = list(enumerate(candidates))[: max(0, int(analyze_k))]
    if body_targets:
        # 전부 한 배치로 동시 fetch — worker 수를 인위적으로 8로 잘라 2배치(직렬)로
        # 늘어지지 않게 한다. 개수가 적어(NEWS_PER_COMPANY 상한) 동시 접속 부담도 적다.
        with ThreadPoolExecutor(max_workers=len(body_targets)) as pool:
            futures = {
                pool.submit(_extract_body, (it.get("link") or "").strip()): i
                for i, it in body_targets
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    body = fut.result()
                except Exception as e:
                    print(f"⚠️ [news] 본문 fetch 실패(무시): {e!r}")
                    body = ""
                if body:
                    results[idx] = _meta_item(candidates[idx], body)

    return results
