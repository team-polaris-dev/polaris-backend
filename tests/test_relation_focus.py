"""search._relation_focus — 질문 키워드 → 앞세울 관계 유형 매핑 회귀 가드.

겸직(INTERLOCKING_DIRECTORATE)이 FOCUS_KEYWORD_GROUPS 에서 누락돼 망 엣지로 못
그려지던 회귀를 막는다. 순수 함수라 DB·LLM 없이 검증한다.
"""
from graphrag.search import _relation_focus


def test_interlocking_directorate_focus_from_gyeomjik() -> None:
    assert "INTERLOCKING_DIRECTORATE" in _relation_focus("SK 임원 겸직 현황 알려줘")
    assert "INTERLOCKING_DIRECTORATE" in _relation_focus("삼성생명 겸임 임원 보여줘")


def test_shareholder_focus_unaffected() -> None:
    focus = _relation_focus("삼성전자 주요주주는?")
    assert "IS_MAJOR_SHAREHOLDER_OF" in focus
    assert "INTERLOCKING_DIRECTORATE" not in focus


def test_no_keyword_yields_empty_focus() -> None:
    assert _relation_focus("삼성전자에 대해 알려줘") == set()
