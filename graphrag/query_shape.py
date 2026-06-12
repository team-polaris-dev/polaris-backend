"""Rule-based query router — 6+1 타입 분류 (한국어).

타입: factoid · multi_hop · comparative · aggregational · temporal · contrastive
      · global (커뮤니티/클러스터 전체 조망 — community card 응답)
"""
from __future__ import annotations

import re


# 키워드/패턴 정의
P_GLOBAL = re.compile(
    r"(클러스터|커뮤니티|생태계|관계망|네트워크\s*구조|전체\s*구조|구조\s*요약|계열\s*묶음)"
)
P_COMPARATIVE = re.compile(
    r"(vs\b|비교|더\s*(?:큰|많은|높은|낮은|작은|적은)|어느\s*쪽이|어느\s*회사가|중\s*(?:최|가장))"
)
P_STRONG_TEMPORAL = re.compile(r"(변화|증감|추이|성장률|연도별|매년)")
P_AGGREG = re.compile(
    r"(합계|총\s*(?:몇|개수|매출|자산|부채|자본|수)|총합|평균|개수|몇\s*개|몇\s*명|모두)"
)
P_TEMPORAL = re.compile(
    r"(\d{4}\s*년\s*(?:대비|이후|이전|부터|까지)|증감|변화|추이|성장률|연도별|매년)"
)
P_CONTRASTIVE = re.compile(
    r"(공급\s*(?:하는가|받는가|방향)|에\s*공급|로부터\s*공급|→|←)"
)
P_MULTIHOP = re.compile(
    r"(최대주주(?:가|의|\(법인\))|자회사(?:의|들)|모회사(?:의|와)|"
    r"종속회사\s*\d+개\s*중|종속회사.*?가장|출자한.*?가장|출자한.*?\d+개\s*중|"
    r"임원(?:이|을|중)\s.+?(?:다른|또\s*다른|소속|있는)|"
    r"공급(?:사|받는|업체)(?:의|이|가)|관계기업(?:의|에)|같은\s*임원)"
)


def classify(query: str) -> str:
    q = query.strip()
    if not q:
        return "factoid"
    if P_GLOBAL.search(q):
        return "global"
    if P_CONTRASTIVE.search(q):
        return "contrastive"
    # 시계열 강 키워드는 비교/팩토이드보다 우선
    if P_STRONG_TEMPORAL.search(q):
        return "temporal"
    if P_COMPARATIVE.search(q):
        return "comparative"
    if P_AGGREG.search(q):
        return "aggregational"
    if P_MULTIHOP.search(q):
        return "multi_hop"
    if P_TEMPORAL.search(q):
        return "temporal"
    return "factoid"
