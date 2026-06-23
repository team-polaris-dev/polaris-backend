# tests/test_news_analysis.py
import json

# 뉴스 분석 탭 — 순수 로직 단위 테스트(네트워크·LLM 없이).
#  - companies_from_query: 질문 → 대상 기업(등장순·긴 이름 우선·cap)
#  - news_client 헬퍼: 태그 제거 / 인링크 필터 / oid→언론사 / 날짜 정규화
#  - core/news 병합 헬퍼: LLM 결과를 신뢰 필드로 안전 병합 + 폴백
from core.news import (
    _coerce_cards,
    _coerce_event_type,
    _coerce_sentiment,
    _fallback_company,
    _merge_llm_company,
    companies_from_query,
)
from tool import news_client as nc

_MAP = {
    "삼성전자": "00126380",
    "SK하이닉스": "00164779",
    "SK": "00100001",          # 부분문자열 후보 — 'SK하이닉스'에 흡수돼야 한다
    "한미반도체": "00100002",
    "LG에너지솔루션": "00100003",
}


def test_companies_single():
    assert companies_from_query("삼성전자 공시 알려줘", _MAP) == [("삼성전자", "00126380")]


def test_companies_longest_name_wins():
    # 'SK' 가 'SK하이닉스'의 부분문자열이므로 SK하이닉스만 잡힌다.
    out = companies_from_query("SK하이닉스 실적", _MAP)
    assert out == [("SK하이닉스", "00164779")]


def test_companies_case_insensitive():
    # 사용자가 'sk하이닉스'(소문자)로 쳐도 매핑의 'SK하이닉스'와 매칭돼야 한다.
    out = companies_from_query("sk하이닉스와 삼성전자 관계는?", _MAP)
    assert out == [("SK하이닉스", "00164779"), ("삼성전자", "00126380")]


def test_companies_appearance_order():
    # 질문에 먼저 등장한 기업이 앞에 온다.
    out = companies_from_query("한미반도체와 삼성전자 비교", _MAP)
    assert out == [("한미반도체", "00100002"), ("삼성전자", "00126380")]


def test_companies_cap():
    q = "삼성전자 SK하이닉스 한미반도체 LG에너지솔루션 동향"
    out = companies_from_query(q, _MAP, cap=3)
    assert len(out) == 3
    assert ("LG에너지솔루션", "00100003") not in out


def test_companies_none_when_no_match():
    assert companies_from_query("반도체 업계 전반 구조", _MAP) == []
    assert companies_from_query("", _MAP) == []


def test_strip_tags():
    assert nc._strip_tags("<b>삼성</b>전자 &amp; SK") == "삼성전자 & SK"


def test_inlink_filter():
    assert nc._INLINK_RE.match("https://n.news.naver.com/mnews/article/015/1")
    assert not nc._INLINK_RE.match("https://www.hankyung.com/article/1")


def test_press_from_oid():
    assert nc._press_from_oid("https://n.news.naver.com/mnews/article/015/0001") == "한국경제"
    assert nc._press_from_oid("https://n.news.naver.com/mnews/article/999/0001") == ""


def test_normalize_pub_date():
    assert nc._normalize_pub_date("Mon, 21 Jun 2026 09:00:00 +0900") == "2026-06-21"
    assert nc._normalize_pub_date("") == ""
    assert nc._normalize_pub_date("garbage") == ""


def test_news_disabled_returns_empty(monkeypatch):
    monkeypatch.delenv("NEWS_ENABLED", raising=False)
    assert nc.fetch_company_news("삼성전자") == []


def test_coerce_sentiment():
    assert _coerce_sentiment("positive") == "positive"
    assert _coerce_sentiment("POSITIVE") == "positive"
    assert _coerce_sentiment("bullish") == "neutral"
    assert _coerce_sentiment(None) == "neutral"


_NEWS = [
    {"title": "삼성 HBM 수출", "url": "https://n.news.naver.com/a/1", "description": "d1",
     "body": "본문1", "press": "한국경제", "pub_date": "2026-06-21"},
    {"title": "삼성 실적", "url": "https://n.news.naver.com/a/2", "description": "d2",
     "body": "", "press": "", "pub_date": "2026-06-20"},
]


def test_coerce_event_type():
    assert _coerce_event_type("실적") == "실적"
    assert _coerce_event_type("HBM·신제품") == "HBM·신제품"
    assert _coerce_event_type("M&A") == "기타"      # enum 밖 → 기타
    assert _coerce_event_type(None) == "기타"


def test_coerce_cards_validates_and_caps():
    raw = [
        {"headline": "시총 역전", "desc": "SK가 삼성 추월", "event_type": "실적", "sentiment": "positive"},
        {"headline": "", "desc": "제목 없음"},                # headline 없음 → 버려짐
        {"headline": "이상 유형", "event_type": "외계인", "sentiment": "bullish"},  # 강제 보정
        {"headline": "넷째", "desc": "상한 초과"},
        {"headline": "다섯째", "desc": "상한 초과"},
    ]
    cards = _coerce_cards(raw)
    assert len(cards) == 3                           # 상한 3 + 빈 headline 제외
    assert cards[0]["headline"] == "시총 역전"
    assert cards[1]["event_type"] == "기타"           # enum 밖 보정
    assert cards[1]["sentiment"] == "neutral"        # 유효하지 않은 논조 보정
    assert _coerce_cards(None) == []
    assert _coerce_cards("nope") == []


def test_fallback_company_keeps_articles():
    out = _fallback_company("삼성전자", "00126380", _NEWS)
    assert out["corp_name"] == "삼성전자"
    assert out["summary"] == ""
    assert len(out["articles"]) == 2
    assert out["articles"][0]["url"] == "https://n.news.naver.com/a/1"


def test_merge_llm_index_based_trusted_fields():
    # LLM 은 index 로만 짚고, title/url/press/pub_date 는 항상 입력값(신뢰 필드)에서 온다.
    llm_obj = {
        "summary": "최근 HBM 수출 관련 동향",
        "relevance": "사업보고서와 연결",
        "sentiment": "positive",
        "articles": [
            {"index": 1, "sentiment": "positive", "evidence": ["사업보고서 §II"]},
        ],
    }
    out = _merge_llm_company("삼성전자", "00126380", _NEWS, llm_obj)
    assert out["summary"] == "최근 HBM 수출 관련 동향"
    assert out["sentiment"] == "positive"
    # 입력 뉴스 전부가 카드로 남는다(표시용). LLM 이 짚은 1번만 positive, 안 짚은 2번은 neutral.
    assert len(out["articles"]) == 2
    card = out["articles"][0]
    assert card["url"] == "https://n.news.naver.com/a/1"
    assert card["press"] == "한국경제"           # 입력값
    assert card["pub_date"] == "2026-06-21"      # 입력값
    assert card["sentiment"] == "positive"
    assert card["evidence"] == ["사업보고서 §II"]
    assert out["articles"][1]["sentiment"] == "neutral"   # 안 짚은 기사 기본값
    assert out["articles"][1]["evidence"] == []


def test_merge_llm_includes_cards():
    llm_obj = {
        "summary": "s", "relevance": "", "sentiment": "positive",
        "cards": [{"headline": "HBM 증설", "desc": "평택 라인", "event_type": "증설", "sentiment": "positive"}],
        "articles": [],
    }
    out = _merge_llm_company("삼성전자", "00126380", _NEWS, llm_obj)
    assert len(out["cards"]) == 1
    assert out["cards"][0]["headline"] == "HBM 증설"
    assert out["cards"][0]["event_type"] == "증설"


def test_fallback_company_has_empty_cards():
    out = _fallback_company("삼성전자", "00126380", _NEWS)
    assert out["cards"] == []


def test_merge_llm_empty_articles_keeps_all_news():
    # LLM 이 articles 를 비워도 입력 뉴스 전부가 카드로 보존된다(0건 처리 안 함).
    llm_obj = {"summary": "s", "relevance": "", "sentiment": "neutral", "articles": []}
    out = _merge_llm_company("삼성전자", "00126380", _NEWS, llm_obj)
    assert len(out["articles"]) == 2


# ---- build_news_analysis 전체 배선 (fetch/검색/LLM 전부 가짜 주입) ----
from langchain_core.messages import AIMessage

import core.news as core_news
from core.news import build_news_analysis


def test_build_news_analysis_wiring(monkeypatch):
    """뉴스 수집→공시 검색→LLM 병합 배선을 네트워크·LLM 없이 검증."""
    fake_chunks = [
        {"name": "삼성전자 · 사업보고서 (2023.12)", "value": {"embedding_text": "HBM 관련 설비투자"},
         "extra": {"section_path": "§II"}},
    ]

    def fake_fetch(name, n):
        return _NEWS

    def fake_search(text, top_k=3, corp_codes=None):
        assert corp_codes == ["00126380"]   # 회사 pre-filter 가 전달됐는지
        return fake_chunks

    canned = {
        "companies": [
            {"corp_name": "삼성전자", "summary": "최근 HBM 동향", "relevance": "사업보고서와 연결",
             "sentiment": "positive",
             "articles": [
                 {"index": 1, "sentiment": "positive",
                  "evidence": ["삼성전자 · 사업보고서 (2023.12) §II"]},
             ]},
        ]
    }
    monkeypatch.setattr(
        core_news.json_llm, "invoke",
        lambda msgs: AIMessage(content=json.dumps(canned, ensure_ascii=False)),
    )

    out = build_news_analysis([("삼성전자", "00126380")], "삼성전자 HBM",
                              fetch_news=fake_fetch, vector_search=fake_search)
    assert len(out) == 1
    co = out[0]
    assert co["corp_name"] == "삼성전자"
    assert co["summary"] == "최근 HBM 동향"
    assert co["sentiment"] == "positive"
    # 신뢰 필드는 입력 NewsItem 으로 복원됐는지
    assert co["articles"][0]["press"] == "한국경제"
    assert co["articles"][0]["evidence"] == ["삼성전자 · 사업보고서 (2023.12) §II"]


def test_build_news_analysis_llm_failure_degrades(monkeypatch):
    """LLM 이 깨진 JSON 을 내도 뉴스 카드는 보존(0건 처리 안 함)."""
    def fake_fetch(name, n):
        return _NEWS

    monkeypatch.setattr(core_news.json_llm, "invoke",
                        lambda msgs: AIMessage(content="not json {{{"))
    out = build_news_analysis([("삼성전자", "00126380")], "삼성전자",
                              fetch_news=fake_fetch, vector_search=lambda *a, **k: [])
    assert len(out) == 1
    assert out[0]["summary"] == ""          # 분석은 비었지만
    assert len(out[0]["articles"]) == 2     # 카드는 살아있다


def test_build_news_analysis_empty_companies():
    assert build_news_analysis([], "x", fetch_news=lambda *a: _NEWS) == []


# ---- 최근 N일 윈도우(3일치) 순수 헬퍼 ----
def test_cutoff_date_inclusive_window():
    from datetime import date, timedelta

    # 3일치 = 오늘 포함 이틀 전까지(오늘, 어제, 그제).
    expected = (date.today() - timedelta(days=2)).isoformat()
    assert nc._cutoff_date(3) == expected
    assert nc._cutoff_date(1) == date.today().isoformat()
    assert nc._cutoff_date(0) == ""        # 0 이면 날짜 필터 비활성


def test_within_lookback():
    assert nc._within_lookback("2026-06-22", "2026-06-20") is True   # 윈도우 안
    assert nc._within_lookback("2026-06-20", "2026-06-20") is True   # 경계 포함
    assert nc._within_lookback("2026-06-19", "2026-06-20") is False  # 윈도우 밖(오래됨)
    assert nc._within_lookback("", "2026-06-20") is True             # 날짜 없으면 통과(드롭 안 함)
    assert nc._within_lookback("2026-06-19", "") is True             # cutoff 없으면 통과(필터 끔)
