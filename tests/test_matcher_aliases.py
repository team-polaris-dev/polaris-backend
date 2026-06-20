from graphrag.matcher import _exact_alias_codes, _strip_metric_terms


def test_exact_alias_codes_prefer_sk_hynix_over_sk_group():
    codes = _exact_alias_codes("SK하이닉스와 삼성전자 둘 다와 연결된 협력사")

    assert codes[:2] == ["00164779", "00126380"]
    assert "00181712" not in codes


def test_exact_alias_codes_handles_korean_sk_hynix_name():
    codes = _exact_alias_codes("에스케이하이닉스에 제품을 공급하는 회사")

    assert codes == ["00164779"]


# ── _strip_metric_terms: 지표어 독립 토큰만 제거(회사명 토큰 보존) ──────────────

def test_strip_metric_terms_removes_standalone_metric_token():
    # "매출" 은 회사명이 아니라 랭킹 차원 → 풀텍스트 검색어에서 빠진다(Product 오염 차단).
    assert _strip_metric_terms("삼성전자 매출 1위 계열사") == "삼성전자 1위 계열사"


def test_strip_metric_terms_removes_metric_with_josa():
    assert _strip_metric_terms("삼성전자 매출은 얼마") == "삼성전자 얼마"
    assert _strip_metric_terms("당기순이익이 가장 큰 곳") == "가장 큰 곳"


def test_strip_metric_terms_preserves_company_name_token():
    # '자산'은 지표어지만 '삼성자산운용'은 한 토큰이라 제거되지 않는다(부분치환 금지).
    assert _strip_metric_terms("삼성자산운용 자산 규모") == "삼성자산운용 규모"


def test_strip_metric_terms_metric_only_becomes_empty():
    # 지표어만 있고 회사명 토큰이 없으면 빈 검색어 → 매처는 빈 시드로 정상 degrade.
    assert _strip_metric_terms("매출").strip() == ""
