"""GraphRAG Global Search — 커뮤니티 요약 query-time map-reduce.

매크로/업계/주제형 질문(intent="global")과 DRIFT 결합(ctx 경로에서 앵커 군집)의
검색측. Cypher 생성이 아니라, 인덱스 시점에 미리 만들어둔 Community 노드(군집별 LLM
요약)를 읽어 관련 군집을 고르고(select), 군집마다 LLM 으로 질문 맞춤 부분답+점수를
만든 뒤(map), 점수로 거르고 정렬한다. 종합(reduce)은 gen 노드가 한다.

MS GraphRAG global search 의 map-reduce 를 질의 시점에 수행한다:
  select → map(_map_community) → filter(GLOBAL_MAP_MIN_SCORE) → sort → cap.
graphrag.node.graph_search_node 가 호출한다:
  - intent="global": global_search(query)            (질의 멤버명 매칭으로 군집 선택)
  - 그 외(ctx/local, DRIFT): global_search(query, anchor_corp_codes=[...])
    (로컬 시드 corp_code 가 멤버인 군집만 — 정밀 매칭)

견고성:
  - GLOBAL_MAP_REDUCE=0 → 기존 정적-요약 읽기 경로로 폴백(map LLM 미호출).
  - map LLM 실패/타임아웃/파싱오류 → 군집 정적 summary + 중립 점수로 폴백(예외 안 던짐).
  - Neo4j 불가 → 빈 리스트 degrade(파이프라인 보호).
"""
from __future__ import annotations

import json
import logging

from config.graphrag import (
    GLOBAL_MAP_MAX_COMMUNITIES,
    GLOBAL_MAP_MIN_SCORE,
    GLOBAL_MAP_REDUCE,
)
from config.llm import json_llm
from tool.graph_client import neo4j_driver

log = logging.getLogger(__name__)

# map LLM 폴백 시(정적요약 사용) 부여하는 중립 점수. 너무 높지도 낮지도 않게 둬서
# 근거는 살리되 진짜 부분답보다 뒤로 정렬되게 한다(GLOBAL_MAP_MIN_SCORE 보다는 위).
_FALLBACK_SCORE = 50


def _load_communities() -> list[dict]:
    """Neo4j 의 모든 Community 노드를 dict 리스트로 로드. 실패 시 []."""
    try:
        with neo4j_driver.session() as s:
            rows = s.run(
                "MATCH (c:Community) "
                "RETURN c.cluster_id AS cluster_id, c.summary AS summary, "
                "       c.size AS size, c.members AS members, "
                "       c.member_names AS member_names, "
                "       c.anchor_names AS anchor_names, c.edge_dist AS edge_dist "
                "ORDER BY c.size DESC"
            ).data()
        return rows or []
    except Exception as e:
        log.warning("global_search: Community 로드 실패(Neo4j 불가?): %s", e)
        return []


def _matches(query: str, community: dict) -> bool:
    """재구성 질의에 이 군집의 멤버명/대표명이 하나라도 등장하면 True."""
    members = (community.get("member_names") or []) + (community.get("anchor_names") or [])
    for nm in members:
        if not nm:
            continue
        # '삼성전자(주)' 같은 정식명·접미사 흔들림 흡수를 위해 양방향 포함 검사.
        if nm in query or query.find(nm.replace("(주)", "").replace("주식회사", "").strip()) != -1:
            return True
    return False


def _select_communities(
    communities: list[dict],
    query: str,
    anchor_corp_codes: list[str] | None,
) -> list[dict]:
    """map 대상 군집 선택. 결정성(입력 순서=size desc 보존) 유지.

    - anchor_corp_codes 주어지면(DRIFT 로컬 경로): 군집 members(corp_code) 와 앵커의
      교집합이 있는 군집만. 정밀 매칭이라 이름 문자열 매칭보다 헤어볼/오탐이 적다.
      교집합이 하나도 없으면 [](군집 없음 = 노이즈 0).
    - 없으면(순수 global): 질의에 멤버명이 등장하는 군집 → 하나도 없으면 전체(광범위 질문).
    어느 경로든 size 상위 GLOBAL_MAP_MAX_COMMUNITIES 개로 cap(map LLM 호출 bound).
    """
    if anchor_corp_codes:
        anchors = set(anchor_corp_codes)
        selected = [
            c for c in communities
            if anchors & set(c.get("members") or [])
        ]
    else:
        selected = [c for c in communities if _matches(query or "", c)]
        if not selected:
            selected = list(communities)
    return selected[:GLOBAL_MAP_MAX_COMMUNITIES]


def _edge_dist_obj(community: dict) -> dict:
    """edge_dist(JSON 문자열 또는 dict) → dict. 파싱 실패 시 {}."""
    edge_dist = community.get("edge_dist")
    if isinstance(edge_dist, str):
        try:
            return json.loads(edge_dist)
        except Exception:
            return {}
    return edge_dist or {}


def _map_community(question: str, community: dict) -> tuple[str, int]:
    """군집 근거 위에서 질문 맞춤 부분답 + 관련성 점수(0~100) 생성 (map 1건).

    json_llm(json_mode) 로 {"answer": str, "score": int} 를 받는다. 실패/타임아웃/
    파싱오류 시 정적 summary + 중립 점수로 폴백한다(예외 위로 안 던짐 — 질의 보호).
    """
    summary = community.get("summary") or ""
    members = ", ".join(community.get("member_names") or [])
    edge_dist = _edge_dist_obj(community)
    dist_txt = ", ".join(f"{rel} {cnt}건" for rel, cnt in edge_dist.items())
    prompt = (
        "당신은 한국 기업 지배구조·계열 관계 분석가입니다.\n"
        "아래는 그래프 커뮤니티 검출로 묶인 한 기업 군집의 요약·구성원·관계 분포입니다.\n"
        "이 군집 정보만 근거로, 사용자 질문에 답하는 데 이 군집이 기여하는 부분답을\n"
        "2~4문장의 자연스러운 한국어로 작성하고, 이 군집이 질문과 얼마나 관련 있는지를\n"
        "0~100 정수 점수로 매기세요. 군집에 없는 사실은 절대 지어내지 말고, 관련이 거의\n"
        "없으면 낮은 점수와 짧은 부분답을 주세요.\n\n"
        f"[질문]\n{question}\n\n"
        f"[군집 요약]\n{summary}\n\n"
        f"[구성원]\n{members}\n\n"
        f"[관계 분포]\n{dist_txt}\n\n"
        '출력 형식(JSON): {"answer": "<부분답>", "score": <0~100 정수>}'
    )
    try:
        raw = json_llm.invoke(prompt).content
        data = json.loads(raw)
        answer = str(data.get("answer") or "").strip()
        score = int(data.get("score"))
        if not answer:
            raise ValueError("빈 부분답")
        score = max(0, min(100, score))
        return answer, score
    except Exception as e:
        log.warning("global_search: map 실패 → 정적요약 폴백 (cluster=%s): %s",
                    community.get("cluster_id"), e)
        return summary, _FALLBACK_SCORE


def _to_unified(community: dict, value: str | None = None, score: int | None = None) -> dict:
    """Community dict → UnifiedResult(type='community').

    value 미지정 시 정적 summary 사용(정적-요약 폴백 경로). score 주어지면 extra 에 부착.
    """
    anchors = community.get("anchor_names") or community.get("member_names") or []
    cid = community.get("cluster_id")
    name = anchors[0] if anchors else f"군집 {cid}"
    extra = {
        "size": community.get("size"),
        "edge_dist": _edge_dist_obj(community),
        "member_names": community.get("member_names") or [],
        # 멤버 corp_code 모음. serialize 가 document_index 역조회로 우측 '원본 문서'
        # 카드를 짤 때 쓴다(글로벌 루트의 카드 채움) + build_global_graph 가 군집 내 엣지
        # 쿼리 입력으로 쓴다(우측 관계도 패널 채움).
        "members": community.get("members") or [],
    }
    if score is not None:
        extra["score"] = score
    return {
        "type": "community",
        "code": str(cid),
        "name": str(name),
        "value": value if value is not None else (community.get("summary") or ""),
        "extra": extra,
        "source": f"community:{cid}",
    }


def _static_select(communities: list[dict], query: str, anchor_corp_codes: list[str] | None) -> list[dict]:
    """정적-요약 폴백(GLOBAL_MAP_REDUCE=0): select 만 하고 정적 summary 그대로 반환."""
    selected = _select_communities(communities, query, anchor_corp_codes)
    return [_to_unified(c) for c in selected]


def global_search(query: str, anchor_corp_codes: list[str] | None = None) -> list[dict]:
    """질의 → 관련 Community 부분답 UnifiedResult 리스트 (map-reduce 의 map+filter).

    anchor_corp_codes: 주어지면 그 corp_code 가 멤버인 군집만 선택(DRIFT 로컬 결합).
                       없으면 질의 멤버명 매칭/전체(순수 global).
    Neo4j 불가 시 []. GLOBAL_MAP_REDUCE=0 이면 정적-요약 경로로 폴백.
    """
    communities = _load_communities()
    if not communities:
        return []

    if not GLOBAL_MAP_REDUCE:
        return _static_select(communities, query, anchor_corp_codes)

    selected = _select_communities(communities, query, anchor_corp_codes)
    if not selected:
        return []

    mapped: list[dict] = []
    for c in selected:
        answer, score = _map_community(query or "", c)
        if score < GLOBAL_MAP_MIN_SCORE:
            continue
        mapped.append(_to_unified(c, value=answer, score=score))

    # 점수 desc, 동점 cluster_id asc (결정성). cluster_id 는 _to_unified 에서 code(str)로
    # 보관되므로 정렬 안정성을 위해 정수 변환해 tie-break.
    def _cid(u: dict) -> int:
        try:
            return int(u.get("code"))
        except (TypeError, ValueError):
            return 0

    mapped.sort(key=lambda u: (-int(u["extra"].get("score", 0)), _cid(u)))
    return mapped[:GLOBAL_MAP_MAX_COMMUNITIES]


# build_communities 가 군집 검출에 쓴 같은 5종 관계. 같은 관계로 미니 관계망을 그려야
# "이 군집이 무엇으로 묶였는지"가 답변(군집 요약) ↔ 패널(관계도) 사이에 일치한다.
_COMMUNITY_GRAPH_REL_TYPES = [
    "IS_SUBSIDIARY_OF",
    "IS_MAJOR_SHAREHOLDER_OF",
    "INVESTS_IN",
    "SUPPLIES_TO",
    "INTERLOCKING_DIRECTORATE",
]

# 군집당 엣지 cap. 5군집 × 40 = 최대 200 엣지(force-graph 무리 없음).
# 군집은 간접 관계로도 묶이므로 anchor 사이 직접 엣지만 보면 거의 비는 경우가 있다 —
# 군집 멤버 전체에서 끌고 ORDER BY 로 구조화 관계 우선 정렬해 노이즈를 줄인다.
GLOBAL_GRAPH_EDGES_PER_COMMUNITY = 40


def _community_edges(member_codes: list[str]) -> list[dict]:
    """군집 멤버 corp_code 사이 valid_to=NULL 엣지를 Neo4j 에서 cap 만큼 로드.

    구조화 관계(자회사 1·주주 2·투자 3)를 추출 관계(공급/임원겸직 4)보다 우선 정렬해
    같은 군집이라도 시각화에 의미 있는 엣지가 먼저 들어오게 한다. a.corp_code < b.corp_code
    로 같은 엣지 1회. r.rcept_no 는 추출 엣지에만 있고 구조화 엣지는 NULL. 실패 시 [].
    """
    if not member_codes:
        return []
    try:
        with neo4j_driver.session() as s:
            rows = s.run(
                "MATCH (a:Organization)-[r]-(b:Organization) "
                "WHERE a.corp_code IN $codes AND b.corp_code IN $codes "
                "  AND a.corp_code < b.corp_code "
                "  AND type(r) IN $rel_types "
                "  AND r.valid_to IS NULL AND r.qc_disabled_at IS NULL "
                "RETURN a.name AS src, type(r) AS rel, b.name AS tgt, "
                "       coalesce(r.rcept_no, '') AS rcept "
                "ORDER BY CASE type(r) "
                "  WHEN 'IS_SUBSIDIARY_OF' THEN 1 "
                "  WHEN 'IS_MAJOR_SHAREHOLDER_OF' THEN 2 "
                "  WHEN 'INVESTS_IN' THEN 3 "
                "  ELSE 4 END "
                "LIMIT $cap",
                codes=member_codes,
                rel_types=_COMMUNITY_GRAPH_REL_TYPES,
                cap=GLOBAL_GRAPH_EDGES_PER_COMMUNITY,
            ).data()
        return rows or []
    except Exception as e:
        log.warning("global_search: 군집 엣지 로드 실패: %s", e)
        return []


def build_global_graph(community_results: list[dict]) -> dict:
    """글로벌 루트용 미니 관계망. 군집별 멤버 corp_code 사이 엣지로 graph_paths/sources/chunks.

    글로벌은 graph_facts 를 채우지 않아(답변 본문에 관계망 텍스트 안 끼게) 우측 관계도
    패널이 비었다. 군집 검출 입력이었던 회사 ↔ 회사 엣지가 Neo4j 에 이미 있으므로
    군집당 GLOBAL_GRAPH_EDGES_PER_COMMUNITY 엣지만 끌어 미니 관계망을 그린다.
    serialize.build_graph 가 graph_paths 를 받아 노드/엣지를 만들고 panel='graph' 가
    자동 선택된다. graph_chunks 는 글로벌엔 청크 출처가 없어 빈 문자열로 정렬만 맞춘다.
    """
    paths: list[list[str]] = []
    sources: list[str] = []
    chunks: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    for c in community_results:
        member_codes = (c.get("extra") or {}).get("members") or []
        for edge in _community_edges(member_codes):
            src = str(edge.get("src") or "")
            tgt = str(edge.get("tgt") or "")
            rel = str(edge.get("rel") or "")
            if not src or not tgt or not rel:
                continue
            key = (src, tgt, rel)
            if key in seen:
                continue
            seen.add(key)
            paths.append([src, rel, tgt])
            sources.append(str(edge.get("rcept") or ""))
            chunks.append("")
    return {
        "graph_paths": paths,
        "graph_path_sources": sources,
        "graph_path_chunks": chunks,
    }
