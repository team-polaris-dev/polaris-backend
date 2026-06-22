from graphrag.matcher import _exact_alias_codes, _strip_link_stopwords


def test_exact_alias_codes_prefer_sk_hynix_over_sk_group():
    codes = _exact_alias_codes("SK하이닉스와 삼성전자 둘 다와 연결된 협력사")

    assert codes[:2] == ["00164779", "00126380"]
    assert "00181712" not in codes


def test_exact_alias_codes_handles_korean_sk_hynix_name():
    codes = _exact_alias_codes("에스케이하이닉스에 제품을 공급하는 회사")

    assert codes == ["00164779"]


# ── _strip_link_stopwords: 불용어(지표어·일반명사) 독립 토큰만 제거(회사명 토큰 보존) ──

def test_strip_link_stopwords_removes_standalone_metric_token():
    # "매출" 은 회사명이 아니라 랭킹 차원 → 풀텍스트 검색어에서 빠진다(Product 오염 차단).
    assert _strip_link_stopwords("삼성전자 매출 1위 계열사") == "삼성전자 1위 계열사"


def test_strip_link_stopwords_removes_metric_with_josa():
    assert _strip_link_stopwords("삼성전자 매출은 얼마") == "삼성전자 얼마"
    assert _strip_link_stopwords("당기순이익이 가장 큰 곳") == "가장 큰 곳"


def test_strip_link_stopwords_preserves_company_name_token():
    # '자산'은 지표어지만 '삼성자산운용'은 한 토큰이라 제거되지 않는다(부분치환 금지).
    assert _strip_link_stopwords("삼성자산운용 자산 규모") == "삼성자산운용 규모"


def test_strip_link_stopwords_metric_only_becomes_empty():
    # 지표어만 있고 회사명 토큰이 없으면 빈 검색어 → 매처는 빈 시드로 정상 degrade.
    assert _strip_link_stopwords("매출").strip() == ""


def test_strip_link_stopwords_removes_generic_org_noun():
    # "기업"은 보통명사 → 풀텍스트에서 빠진다(FULLTEXT cjk 가 '기업은행'에 매칭하는 가짜 앵커 차단).
    assert _strip_link_stopwords("수혜를 볼만한 기업").split() == ["수혜를", "볼만한"]
    assert _strip_link_stopwords("가장 수혜 보는 기업은").split() == ["가장", "수혜", "보는"]


def test_strip_link_stopwords_preserves_real_company_with_generic_substring():
    # '기업'이 들어간 실재 회사명(기업은행)은 한 토큰이라 보존된다(부분치환 금지).
    assert _strip_link_stopwords("기업은행 주가") == "기업은행 주가"
    assert _strip_link_stopwords("기업은행은 어디") == "기업은행은 어디"
