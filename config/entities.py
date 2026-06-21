"""회사 엔티티 정규화·별칭·일반어 블록리스트 단일 진실원천(SSOT).

이전엔 회사명 정규화가 db.py / matcher.py / serialize.py 3벌로 갈라져 접미사 패턴이
다 다르고 별칭 방향까지 반대(serialize 만 에스케이→SK)였다. 여기서 한 벌로 합친다.

정규형(canonical) = 한글 그룹명(에스케이·엘지 …). DART 공시·그래프 노드 표기 기준.
적재(pipeline_scripts)·검색(graphrag)·직렬화(core) 모두 이 모듈을 import 한다.

단어집(그룹 별칭·일반어 블록리스트·라벨맵)은 config/entities.json 에 정의하고 이 모듈은
읽어 제공한다. 정규화 정규식·함수만 코드로 둔다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_VOCAB_PATH = Path(__file__).with_name("entities.json")
with _VOCAB_PATH.open("r", encoding="utf-8") as _f:
    _VOCAB = json.load(_f)

# 법인 접미사 (3벌 합집합 — db/matcher/serialize 가 각자 갖던 것 통합)
_SUFFIX_RE = re.compile(
    r"주식회사|유한회사|유한책임회사|\(주\)|\(유\)|\(재\)|주\)|㈜|"
    r"Co\.,?\s*Ltd\.?|Inc\.?|Corp\.?|Ltd\.?|LLC|GmbH",
    re.IGNORECASE,
)
_PAREN_RE = re.compile(r"\([^)]*\)")   # (SEA) 같은 괄호 약어/주석
_WS_RE = re.compile(r"\s+")

# 재벌 그룹 로마자 → 한글 정규형 (접두 한정). 'SK하이닉스'와 '에스케이하이닉스'를 같은 키로.
CORP_ALIASES: dict[str, str] = _VOCAB["corp_aliases"]
_ALIASES_BY_LEN = sorted(CORP_ALIASES.items(), key=lambda kv: -len(kv[0]))  # 긴 접두부터(kcc 가 kt 보다 먼저)

# 추상·일반어 placeholder — 실제 회사가 아니라서 노드로 만들면 RELATED_PARTY 허브가 되고
# FULLTEXT 시드 1순위로 잡혀 검색을 오염시킨다. 적재 시 노드화 차단 + 검색 시 시드 제외.
GENERIC_ORG_BLOCKLIST: frozenset[str] = frozenset(_VOCAB["generic_org_blocklist"])

# 질문 속 '회사/기업'류 일반명사 — 회사명이 아니라 "어떤 회사?"의 보통명사인데, FULLTEXT cjk 가
# "기업"을 '기업은행' 같은 실재 노드에 매칭해 가짜 앵커를 만든다. 엔티티 링킹 풀텍스트 질의에서
# 이 *독립 토큰*만 떼어 오염을 막는다(부분문자열 아님 — '기업은행' 같은 회사명 토큰은 보존).
GENERIC_ORG_TERMS: tuple[str, ...] = tuple(_VOCAB["generic_org_terms"])

# Neo4j 라벨 → 내부 taxonomy 라벨 (ppr.py / matcher.py 공용)
LABEL_MAP: dict[str, str] = _VOCAB["label_map"]


def normalize_corp_name(name: str) -> str:
    """ER 키용 정규화: 괄호주석·법인접미사·공백 제거 + 그룹 로마자 별칭 통일 후 소문자.

    예) 'SK하이닉스(주)' → '에스케이하이닉스', '에스케이하이닉스(주)' → '에스케이하이닉스'
    """
    if not name:
        return ""
    s = _PAREN_RE.sub("", name.strip())
    s = _SUFFIX_RE.sub("", s)
    s = _WS_RE.sub("", s).lower()
    for romanized, korean in _ALIASES_BY_LEN:
        if s.startswith(romanized):
            return korean + s[len(romanized):]
    return s


def is_generic_org(name: str) -> bool:
    """정규화한 이름이 일반어 블록리스트에 해당하면 True(노드화·시드 금지)."""
    return normalize_corp_name(name) in GENERIC_ORG_BLOCKLIST
