from graphrag.matcher import _exact_alias_codes


def test_exact_alias_codes_prefer_sk_hynix_over_sk_group():
    codes = _exact_alias_codes("SK하이닉스와 삼성전자 둘 다와 연결된 협력사")

    assert codes[:2] == ["00164779", "00126380"]
    assert "00181712" not in codes


def test_exact_alias_codes_handles_korean_sk_hynix_name():
    codes = _exact_alias_codes("에스케이하이닉스에 제품을 공급하는 회사")

    assert codes == ["00164779"]
