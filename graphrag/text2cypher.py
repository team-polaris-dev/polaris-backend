"""Neo4j 공식 Text2CypherRetriever 기반 관계 검색 (Layer A).

손수 짠 plan-kind 분류기 대신, Neo4j 공식
``neo4j_graphrag.retrievers.Text2CypherRetriever`` 가 그래프 스키마 + few-shot
예시로 read-only Cypher 한 개를 생성·실행한다. 라이브러리가 EXPLAIN 으로
read-only 여부를 Neo4j 자체 query_type 으로 판정해 쓰기 쿼리를 실행 전에 거부하므로,
정규식 기반 read-only 가드보다 견고하다. 우리는 그 위에 도메인 화이트리스트
(_uses_only_whitelisted) 를 생성된 Cypher(metadata['cypher'])에 한 번 더 적용한다.

LLM 접점은 config.llm(plain) 을 LLMBase 로 감싼 어댑터(_ConfigLLM)다. config.llm 은
로컬 gemini CLI 구독 세션이라 API 키가 없으므로, 라이브러리 내장 프로바이더(OpenAI 등)
대신 이 어댑터를 주입한다.

재무 지표 랭킹(매출 1위 등)은 그래프가 아니라 MariaDB 에 있으므로 이 경로가 아니라
structured_executor 의 결정적 SQL 랭킹(_fetch_metric_values)이 담당한다. 이 모듈은
관계/구조/존재 탐색만 책임진다.
"""
from __future__ import annotations

import re
from typing import Any

from config import graphrag as cfg
from config.relations import DOMAIN_RELS, RELATIONS

# ── 도메인 가드 (config.relations SSOT 파생) ──
# 라이브러리(Text2CypherRetriever)는 라벨/관계 타입을 제약하지 않으므로, 생성 Cypher 에
# Organization + 9 DOMAIN_RELS 화이트리스트를 한 번 더 적용한다(표준 위에 얹는 도메인 레이어).
_ALLOWED_LABELS: frozenset[str] = frozenset({"Organization"})
_ALLOWED_REL_TYPES: frozenset[str] = frozenset(DOMAIN_RELS)
_LABEL_RE = re.compile(r":\s*([A-Za-z_][A-Za-z0-9_]*)")           # :Label / :REL_TYPE 모두 매칭
_REL_RE = re.compile(r"\[[^\]]*?:\s*([A-Za-z_][A-Za-z0-9_]*)")    # [r:REL_TYPE ...]


def _rel_types_in(cypher: str) -> set[str]:
    return {m.group(1) for m in _REL_RE.finditer(cypher)}


def _labels_in(cypher: str) -> set[str]:
    """:X 토큰 중 관계 타입이 아닌 것(=노드 라벨)만 추린다."""
    rel_types = _rel_types_in(cypher)
    return {m.group(1) for m in _LABEL_RE.finditer(cypher) if m.group(1) not in rel_types}


def _uses_only_whitelisted(cypher: str) -> bool:
    """모든 노드 라벨이 Organization 이고 모든 관계 타입이 DOMAIN_RELS 안인가."""
    if not cypher:
        return False
    if any(rel not in _ALLOWED_REL_TYPES for rel in _rel_types_in(cypher)):
        return False
    return all(label in _ALLOWED_LABELS for label in _labels_in(cypher))


def _schema_description() -> str:
    rel_lines = "\n".join(f"- {r.type}: {r.ko_label}" for r in RELATIONS)
    return (
        "노드 라벨은 단 하나: Organization(회사).\n"
        "Organization 속성: corp_code(고유 코드), er_name(정규화 회사명), "
        "name(표시 회사명), stock_code(상장 종목코드, 비상장이면 없음).\n"
        "허용 관계 타입(이 외에는 절대 사용 금지):\n"
        f"{rel_lines}"
    )


# 라이브러리 custom_prompt 템플릿. Text2CypherRetriever 가
# .format(schema=, examples=, query_text=, **prompt_params) 로 채운다.
# anchors 는 prompt_params 로 주입한 corp_code 리터럴 리스트 문자열이다.
# 라이브러리는 생성 Cypher 를 파라미터 없이 실행하므로 $anchors 바인딩을 못 쓴다 →
# 앵커 corp_code 를 WHERE ... IN [...] 리터럴로 인라인하도록 지시한다(read-only 리터럴).
_CUSTOM_PROMPT = """너는 한국어 회사 관계 질문을 그래프 위 read-only Cypher 한 개로 바꾸는 변환기다.
그래프 스키마는 다음과 같다.

{schema}

엄격한 규칙:
- 노드 라벨은 Organization 만, 관계 타입은 위 허용 목록만 사용한다(이 외 라벨/관계 절대 금지).
- 쓰기 절(CREATE/MERGE/DELETE/SET/REMOVE/DETACH/FOREACH/LOAD CSV)은 절대 쓰지 않는다(read-only).
- 앵커 회사는 다음 corp_code 리터럴 리스트로만 참조한다(회사명을 쿼리에 직접 쓰지 말 것):
  {anchors}
  예: MATCH (anchor:Organization) WHERE anchor.corp_code IN {anchors}
- 재무 지표(매출/영업이익/순이익/자산)로 줄세우는 랭킹·최댓값·1위 질문은 그래프가 아니라
  다른 시스템이 처리한다 → 그런 질문이면 빈 결과가 나오는 무해한 MATCH 를 만들지 말고,
  관계/구조/존재 탐색만 Cypher 로 만든다.
- 결과는 엣지 한 행마다 다음 컬럼을 반드시 반환한다:
  from_id, from_name, to_id, to_name, rel_type, source, chunk_id, from_role, to_role
  (id 는 coalesce(n.corp_code, n.er_name, n.name), source 는 관계의 rcept_no,
   chunk_id 는 관계의 chunk_id, role 은 anchor/bridge/sibling/neighbor 중 하나. 없으면 '' )

few-shot 예시:
{examples}

코드펜스나 설명 없이 Cypher 한 개만 출력하라.
질문: {query_text}
"""

# few-shot 예시(라이브러리는 "\n".join(examples) 로 프롬프트에 붙인다).
_EXAMPLES: list[str] = [
    (
        "USER INPUT: '삼성전자의 공급사' "
        "QUERY: MATCH (cand:Organization)-[r:SUPPLIES_TO]->(anchor:Organization) "
        "WHERE anchor.corp_code IN ['00126380'] "
        "RETURN coalesce(cand.corp_code, cand.er_name, cand.name) AS from_id, "
        "cand.name AS from_name, "
        "coalesce(anchor.corp_code, anchor.er_name, anchor.name) AS to_id, "
        "anchor.name AS to_name, type(r) AS rel_type, "
        "coalesce(r.rcept_no,'') AS source, coalesce(r.chunk_id,'') AS chunk_id, "
        "'supplier' AS from_role, 'anchor' AS to_role"
    ),
    (
        "USER INPUT: 'SK하이닉스의 대주주' "
        "QUERY: MATCH (cand:Organization)-[r:IS_MAJOR_SHAREHOLDER_OF]->(anchor:Organization) "
        "WHERE anchor.corp_code IN ['00164779'] "
        "RETURN coalesce(cand.corp_code, cand.er_name, cand.name) AS from_id, "
        "cand.name AS from_name, "
        "coalesce(anchor.corp_code, anchor.er_name, anchor.name) AS to_id, "
        "anchor.name AS to_name, type(r) AS rel_type, "
        "coalesce(r.rcept_no,'') AS source, coalesce(r.chunk_id,'') AS chunk_id, "
        "'neighbor' AS from_role, 'anchor' AS to_role"
    ),
]


def _anchors_literal(anchor_codes: list[str]) -> str:
    """corp_code 리스트를 Cypher 리터럴 리스트 문자열로. 인젝션 방지: 코드만 통과."""
    safe = [c for c in dict.fromkeys(anchor_codes) if c and c.replace("-", "").isalnum()]
    return "[" + ", ".join(f"'{c}'" for c in safe) + "]"


def _make_adapter():
    """config.llm(plain) 을 neo4j_graphrag.LLMBase 로 감싼 어댑터 인스턴스.

    지연 import: neo4j_graphrag / config.llm 의 임포트 부작용(gemini 초기화)을
    모듈 로드가 아니라 호출 시점으로 미룬다. 테스트는 _make_adapter 를 monkeypatch.
    """
    from neo4j_graphrag.llm import LLMBase, LLMResponse  # noqa: PLC0415

    class _ConfigLLM(LLMBase):
        def __init__(self) -> None:
            super().__init__(model_name="apimaker-config-llm")

        def _content(self, prompt: str, system_instruction: str | None) -> str:
            from config.llm import llm as _plain  # noqa: PLC0415

            if system_instruction:
                from langchain_core.messages import (  # noqa: PLC0415
                    HumanMessage,
                    SystemMessage,
                )

                res = _plain.invoke([
                    SystemMessage(content=system_instruction),
                    HumanMessage(content=prompt),
                ])
            else:
                res = _plain.invoke(prompt)
            content = getattr(res, "content", None)
            return content if isinstance(content, str) else str(res)

        def invoke(self, input, message_history=None, system_instruction=None, **kwargs):  # type: ignore[override]
            text = input if isinstance(input, str) else str(input)
            return LLMResponse(content=self._content(text, system_instruction))

        async def ainvoke(self, input, message_history=None, system_instruction=None, **kwargs):  # type: ignore[override]
            return self.invoke(input, message_history, system_instruction, **kwargs)

    return _ConfigLLM()


def _build_retriever():
    """Text2CypherRetriever 구성. 드라이버·LLM·스키마·예시·custom_prompt 주입."""
    from neo4j_graphrag.retrievers import Text2CypherRetriever  # noqa: PLC0415

    from tool.graph_client import neo4j_driver  # noqa: PLC0415

    return Text2CypherRetriever(
        driver=neo4j_driver,
        llm=_make_adapter(),
        neo4j_schema=_schema_description(),
        examples=_EXAMPLES,
        custom_prompt=_CUSTOM_PROMPT,
    )


def run_relationship_query(
    query: str,
    anchor_codes: list[str],
) -> tuple[list[dict[str, Any]], str] | None:
    """질문 → (행 dict 리스트, 생성된 Cypher) 또는 None(fail-closed).

    flag off / 앵커 없음 / 생성·실행 실패 / 화이트리스트 위반 → 전부 None.
    라이브러리가 read-only EXPLAIN 가드를 내장하므로 쓰기 쿼리는 실행 전 거부된다.
    그 위에 도메인 라벨/관계 화이트리스트를 생성 Cypher 에 한 번 더 적용한다.
    """
    if not cfg.TEXT2CYPHER_ENABLED:
        return None
    codes = [c for c in dict.fromkeys(anchor_codes or []) if c]
    if not codes:
        return None
    try:
        retriever = _build_retriever()
        result = retriever.get_search_results(
            query_text=query,
            prompt_params={"anchors": _anchors_literal(codes)},
        )
    except Exception as exc:
        print(f"⚠️ [text2cypher] retriever failed: {exc!r}")
        return None

    cypher = str((result.metadata or {}).get("cypher") or "")
    if not cypher or not _uses_only_whitelisted(cypher):
        return None
    rows = [r.data() for r in result.records]
    return rows, cypher
