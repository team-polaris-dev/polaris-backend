"""SUPPLIES_TO 끝점 1차 분류 — 결정론(사실 + 닫힌집합)만. LLM·DB 호출 없음.

SUPPLIES_TO 는 '회사 → 회사' 여야 한다. 추출 모델(qwen 등)이 끝점으로 국가·지역·
제품·일반어를 잘못 뽑는다. 이 모듈은 **결정론으로 확정 가능한 것만** 판정한다:

  company   : corp_code 보유 / 법인 접미사(㈜·Inc·Ltd…) / corp_master 매칭 → 회사(사실)
  geo       : 국가·지역 닫힌집합 일치 → 비회사 확정(국가는 유한·열거가능)
  unresolved: 그 외 전부 → 결정론으로 단정 불가. 회사일 수도(외국·자회사), 제품·
              일반어일 수도. **판정을 LLM(apimaker)에게 위임** (detect_conflicts →
              entity-judge). 휴리스틱 추정(일반어·제품 denylist)은 폐기했다 — 무한
              집합이라 목록으로 닫을 수 없고(두더지잡기), 원문 맥락을 봐야 정확하다.

설계 원칙: needs_er ≠ 쓰레기. corp_code 없는 진짜 외국/자회사(TSMC·지멘스 등)를
끄지 않도록, 결정론 단계는 '확정 가능한 것만' 처리하고 모호하면 unresolved 로 넘긴다.
"""
from __future__ import annotations

import re

# 국가·지역·대륙 — 공급 주체/객체가 될 수 없음. 유한·열거가능한 닫힌집합이므로
# denylist 가 곧 완전(휴리스틱 아님). 전체이름 일치('한국타이어' 부분일치 오탐 방지).
GEO = {
    "일본", "미국", "중국", "대만", "유럽", "아시아", "아시아태평양", "아태", "아태지역",
    "동남아", "동남아시아", "동북아", "북미", "남미", "중남미", "인도", "베트남", "독일",
    "영국", "프랑스", "한국", "국내", "해외", "글로벌", "전세계", "싱가포르", "말레이시아",
    "태국", "인도네시아", "필리핀", "홍콩", "러시아", "브라질", "멕시코", "캐나다", "호주",
    "중동", "아프리카", "이스라엘", "네덜란드", "스위스", "이탈리아", "스페인", "유럽연합",
    "eu", "대만성", "중화권", "구주", "미주", "유럽지역", "아시아지역", "북미지역",
    "중국대륙", "기타지역", "기타국가",
}

# 법인 접미사 — 있으면 회사(사실 근거). 짧은 라틴 약어는 단어경계.
CORP_SUFFIX = re.compile(
    r"(주식회사|유한회사|유한책임회사|합자회사|합명회사|사단법인|재단법인|㈜|\(주\)|\(유\)|"
    r"\bco\.?\s*,?\s*ltd\b|\bltd\b|\binc\b|\bcorp\b|\bcorporation\b|\bcompany\b|\bllc\b|"
    r"\bl\.?\s*p\.?\b|\bllp\b|\bgmbh\b|\bag\b|\bs\.?\s*a\.?\b|\bb\.?\s*v\.?\b|\bpte\b|"
    r"\bn\.?\s*v\.?\b|\bs\.?\s*r\.?\s*l\b|\bpvt\b|\bplc\b|\bsas\b|\bsarl\b|\boy\b|\bbhd\b|"
    r"\bsdn\b)",
    re.IGNORECASE,
)
_HANGUL_CORP = re.compile(r"(\(주\)|㈜|\(유\)|\(재\)|\(사\))")


def normalize(name: str) -> str:
    """닫힌집합 일치 비교용 정규화: 소문자·각주((*1))·공백 제거. 접미사는 보존."""
    s = (name or "").strip().lower()
    s = re.sub(r"\(\s*\*?\s*\d+\s*\)", "", s)   # (*1) (1) 각주 제거
    s = re.sub(r"\s+", "", s)
    return s


def classify(name: str, *, has_corp_code: bool, in_corp_master: bool) -> tuple[str, str]:
    """끝점 1개 1차 분류 (결정론).

    반환 (label, reason):
      label ∈ {"company", "geo", "unresolved"}
      company   = 유지(사실 근거)
      geo       = 비회사 확정(닫힌집합)
      unresolved = LLM 판정 대상 (회사/제품/일반어/인물 불명)
    """
    raw = (name or "").strip()
    if not raw:
        return "company", ""                       # 빈값 → 보수적 유지
    if has_corp_code:
        return "company", "corp_code 보유"
    if _HANGUL_CORP.search(raw) or CORP_SUFFIX.search(raw):
        return "company", "법인 접미사 보유"
    if in_corp_master:
        return "company", "corp_master 매칭"
    if normalize(raw) in GEO:
        return "geo", f"국가/지역('{raw}')은 공급 주체가 될 수 없음"
    return "unresolved", ""                         # 휴리스틱 추정 금지 → LLM 위임


# LLM verdict(entity-judge) → 표시 라벨 + 쓰레기 여부
VERDICT_LABEL = {
    "company": "회사",
    "country_region": "국가/지역",
    "product": "제품",
    "generic": "일반어",
    "person": "인물",
    "uncertain": "불확실",
}
# 결정론 geo + LLM 쓰레기 verdict
JUNK_VERDICTS = {"country_region", "product", "generic", "person"}
