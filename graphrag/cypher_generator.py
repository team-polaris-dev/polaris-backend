"""Schema-guarded text-to-Cypher generator (구조 검색 전용).

손수 짠 plan-kind 분류기(planner/llm_planner) 대신 LLM 이 그래프 스키마에서
read-only Cypher MATCH 한 개를 생성한다. 단, 재무 지표(매출·영업이익 등)는
그래프가 아니라 MariaDB(structured_executor._fetch_metric_values)에 있으므로,
이 생성기는 관계/구조/존재만 다루고 랭킹·지표 질문은 supported=false 로 흘려
기존 경로(structured_executor)가 폴백 처리하게 한다.

전부 fail-closed·결정적. 정적 가드(read-only / whitelist / determinism)는 순수
함수로 단위 테스트 가능하고, LLM·Neo4j 접점(_invoke_llm, _explain_ok)은
monkeypatch 가능한 모듈 수준 seam 이다(llm_planner._invoke_llm 패턴 미러).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from config import graphrag as cfg
from config.relations import DOMAIN_RELS, RELATIONS

# ── 화이트리스트 SSOT (config.relations 에서 파생 — 9종을 따로 하드코딩하지 않는다) ──
_ALLOWED_LABELS: frozenset[str] = frozenset({"Organization"})
_ALLOWED_REL_TYPES: frozenset[str] = frozenset(DOMAIN_RELS)
# 쓰기측 안전을 위해 속성 접근(특히 SET 대상이 될 수 있는 키)을 제한한다. 읽기 자체는
# 관대하지만, 가드 핵심은 label+rel type 이며 속성은 보너스 안전망이다.
_ALLOWED_PROPERTIES: frozenset[str] = frozenset({"corp_code", "er_name", "name", "stock_code"})

# 쓰기 절(read-only 위반). 토큰/단어경계 인식, 대소문자 무시.
_WRITE_CLAUSES: tuple[str, ...] = (
    "CREATE", "MERGE", "DELETE", "SET", "REMOVE", "DETACH", "FOREACH", "LOAD",
)
# 비결정성 함수 — 결정성 보장 불가 → 거부.
_NONDETERMINISTIC = re.compile(r"\b(rand|randomUUID)\s*\(", re.IGNORECASE)

# 토큰 추출 정규식.
_LABEL_RE = re.compile(r":\s*([A-Za-z_][A-Za-z0-9_]*)")           # :Label / :REL_TYPE 모두 매칭
_REL_RE = re.compile(r"\[[^\]]*?:\s*([A-Za-z_][A-Za-z0-9_]*)")    # [r:REL_TYPE ...]
_HAS_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)
_HAS_LIMIT = re.compile(r"\bLIMIT\b", re.IGNORECASE)
# CALL { ... } 서브쿼리 안의 쓰기는 위 _WRITE_CLAUSES 검사로 함께 걸린다.


@dataclass(frozen=True)
class GeneratedCypher:
    """검증·결정성 주입을 마친 read-only Cypher 와 파라미터."""

    cypher: str
    params: dict
    reason: str


# ─────────────────────────────────────────────────────────────
# 가드레일 (순수 함수 — Neo4j/LLM 없이 단위 테스트 가능)
# ─────────────────────────────────────────────────────────────

def _word_present(text: str, word: str) -> bool:
    """단어 경계 기준으로 키워드가 있는가(대소문자 무시). 식별자 내부 부분일치 제외."""
    return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None


def _is_read_only(cypher: str) -> bool:
    """쓰기 절(CREATE/MERGE/DELETE/SET/REMOVE/DETACH/FOREACH/LOAD CSV)이 없으면 True."""
    if not cypher:
        return False
    for clause in _WRITE_CLAUSES:
        if _word_present(cypher, clause):
            return False
    return True


def _rel_types_in(cypher: str) -> set[str]:
    types: set[str] = set()
    for m in _REL_RE.finditer(cypher):
        types.add(m.group(1))
    return types


def _labels_in(cypher: str) -> set[str]:
    """:X 토큰 중 관계 타입이 아닌 것(=노드 라벨)을 추린다.

    [r:REL] 안의 토큰은 관계 타입이므로 제외하고 남은 :Label 만 라벨로 본다.
    """
    rel_types = _rel_types_in(cypher)
    labels: set[str] = set()
    for m in _LABEL_RE.finditer(cypher):
        tok = m.group(1)
        if tok in rel_types:
            continue
        labels.add(tok)
    return labels


def _uses_only_whitelisted(cypher: str) -> bool:
    """모든 노드 라벨이 Organization 이고 모든 관계 타입이 DOMAIN_RELS 안인가."""
    if not cypher:
        return False
    for rel in _rel_types_in(cypher):
        if rel not in _ALLOWED_REL_TYPES:
            return False
    for label in _labels_in(cypher):
        if label not in _ALLOWED_LABELS:
            return False
    return True


def _ensure_deterministic(cypher: str, limit: int) -> str | None:
    """ORDER BY 없으면 이름 컬럼 asc 로, LIMIT 없으면 LIMIT <limit> 를 append.

    비결정성 함수(rand/randomUUID)가 있으면 None(거부). 이미 있는 ORDER BY/LIMIT 는 보존.
    """
    if not cypher:
        return None
    if _NONDETERMINISTIC.search(cypher):
        return None
    out = cypher.rstrip().rstrip(";").rstrip()
    if not _HAS_ORDER_BY.search(out):
        out = f"{out}\nORDER BY from_name ASC, to_name ASC"
    if not _HAS_LIMIT.search(out):
        out = f"{out}\nLIMIT {int(limit)}"
    return out


# ─────────────────────────────────────────────────────────────
# 스키마 설명 (config.relations SSOT 에서 파생)
# ─────────────────────────────────────────────────────────────

def _schema_description() -> str:
    rel_lines = "\n".join(f"- {r.type}: {r.ko_label}" for r in RELATIONS)
    return (
        "노드 라벨은 단 하나: Organization(회사).\n"
        "Organization 속성: corp_code(고유 코드), er_name(정규화 회사명), "
        "name(표시 회사명), stock_code(상장 종목코드, 비상장이면 없음).\n"
        "허용 관계 타입(이 외에는 절대 사용 금지):\n"
        f"{rel_lines}"
    )


_SYSTEM_PROMPT = """너는 한국어 회사 관계 질문을 그래프 위 read-only Cypher 한 개로 바꾸는 변환기다.
그래프 스키마는 다음과 같다.

{schema}

엄격한 규칙:
- 노드 라벨은 Organization 만, 관계 타입은 위 허용 목록만 사용한다.
- 쓰기 절(CREATE/MERGE/DELETE/SET/REMOVE/DETACH/FOREACH/LOAD CSV)은 절대 쓰지 않는다(read-only).
- 회사명을 쿼리에 직접 넣지 말고, 앵커는 반드시 파라미터 $anchors (corp_code 문자열 리스트)로 참조한다.
  예: MATCH (anchor:Organization) WHERE anchor.corp_code IN $anchors
- 재무 지표(매출/영업이익/순이익/자산)로 줄세우는 랭킹·최댓값·1위 질문은 그래프가 아니라
  다른 시스템이 처리한다 → 반드시 "supported": false 로 답한다(Cypher 만들지 말 것).
- 그래프가 할 수 있는 일은 관계/구조/존재 탐색 뿐이다(공급사·주주·자회사·특수관계 등).
- 결과는 엣지 한 행마다 다음 컬럼을 반드시 반환한다:
  from_id, from_name, to_id, to_name, rel_type, source, chunk_id, from_role, to_role
  (role 은 anchor/bridge/sibling/neighbor 중 하나. id 는 coalesce(n.corp_code, n.er_name, n.name),
   source 는 관계의 rcept_no, chunk_id 는 관계의 chunk_id 를 쓴다. 없으면 빈 문자열.)

출력은 JSON 객체 하나만 한다."""

_USER_TEMPLATE = """질문을 아래 JSON 스키마로만 변환하라.

{{
  "supported": true 또는 false,
  "cypher": "read-only Cypher MATCH 한 개 (supported=false 면 빈 문자열)",
  "params": {{}} (필요한 추가 파라미터, 보통 비어 있음. $anchors 는 시스템이 주입하므로 넣지 말 것),
  "reason": "짧은 한국어 근거"
}}

질문: {query}"""

_REPAIR_TEMPLATE = """방금 만든 Cypher 가 EXPLAIN 단계에서 컴파일에 실패했다.
오류: {error}

같은 규칙을 지키면서 컴파일 가능한 read-only Cypher 로 고쳐 같은 JSON 스키마로 다시 답하라.
질문: {query}"""


# ─────────────────────────────────────────────────────────────
# LLM / Neo4j 접점 (monkeypatch 가능한 seam)
# ─────────────────────────────────────────────────────────────

def _as_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return json.loads(content)
    raise ValueError("cypher generator response is not JSON")


def _invoke_llm(query: str, repair_error: str | None = None) -> dict[str, Any]:
    """질문(필요 시 EXPLAIN 오류 포함)을 LLM 에 보내 JSON dict 를 받는다.

    llm_planner._invoke_llm 패턴 미러. 테스트는 이 함수를 monkeypatch 한다.
    """
    from config.llm import json_llm  # noqa: PLC0415
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    system = _SYSTEM_PROMPT.format(schema=_schema_description())
    if repair_error:
        user = _REPAIR_TEMPLATE.format(error=repair_error, query=query)
    else:
        user = _USER_TEMPLATE.format(query=query)
    raw = json_llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return _as_dict(raw)


def _explain_ok(cypher: str, params: dict) -> bool:
    """EXPLAIN <cypher> 가 컴파일되면 True, 어떤 예외든 False(fail-closed).

    structured_executor 와 동일한 드라이버/세션 패턴 사용. 테스트는 monkeypatch.
    """
    try:
        from tool.graph_client import neo4j_driver  # noqa: PLC0415

        with neo4j_driver.session() as session:
            session.run(f"EXPLAIN {cypher}", **(params or {}))
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# 공개 API
# ─────────────────────────────────────────────────────────────

def _validate_and_finalize(data: dict[str, Any]) -> GeneratedCypher | None:
    """LLM dict → 정적 가드 + 결정성 주입. EXPLAIN 은 호출자가 따로 돌린다.

    정적 가드 실패는 즉시 None(수리 금지). 통과하면 결정성 주입한 GeneratedCypher 반환.
    """
    if not isinstance(data, dict) or data.get("supported") is False:
        return None
    cypher = str(data.get("cypher") or "").strip()
    if not cypher:
        return None
    if not _is_read_only(cypher):
        return None
    if not _uses_only_whitelisted(cypher):
        return None
    finalized = _ensure_deterministic(cypher, cfg.TEXT2CYPHER_RESULT_LIMIT)
    if not finalized:
        return None
    params = data.get("params")
    if not isinstance(params, dict):
        params = {}
    reason = str(data.get("reason") or "")
    return GeneratedCypher(cypher=finalized, params=params, reason=reason)


def generate(query: str, anchors: list[dict]) -> GeneratedCypher | None:
    """질문 → 검증된 read-only Cypher(GeneratedCypher) 또는 None.

    flag off / 앵커 없음 / 지표 랭킹(supported=false) / 가드 실패 / LLM 다운 →
    전부 None(fail-closed). EXPLAIN 실패만 TEXT2CYPHER_REPAIR_MAX 회 수리.
    """
    if not cfg.TEXT2CYPHER_ENABLED:
        return None
    if not anchors:
        return None
    try:
        data = _invoke_llm(query)
    except Exception:
        return None

    candidate = _validate_and_finalize(data)
    if candidate is None:
        return None  # 정적 가드 실패(또는 supported=false)는 수리하지 않는다.
    if _explain_ok(candidate.cypher, candidate.params):
        return candidate

    # EXPLAIN 실패 → 수리 루프.
    repair_left = max(0, int(cfg.TEXT2CYPHER_REPAIR_MAX))
    error = "EXPLAIN failed for generated query"
    for _ in range(repair_left):
        try:
            data = _invoke_llm(query, repair_error=error)
        except Exception:
            return None
        candidate = _validate_and_finalize(data)
        if candidate is None:
            return None  # 수리본이 정적 가드를 위반 → 폐기.
        if _explain_ok(candidate.cypher, candidate.params):
            return candidate
    return None
