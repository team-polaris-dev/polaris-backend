"""GraphAgentRunner — Notion 흐름도의 `관계망 검색(Text-to-Cypher)` 블록 본체.

04_graphrag.md §2 흐름:
  [1] 의도분류 (LLM 12의도 + route + stage)
  [2] route 게이트 — graph 비대상(rdb/vec/provenance)은 reject 신호
  [3] 엔티티/슬롯 채움 (corp_code, account_id)
  [4] 템플릿 매칭 (intent + stage) — 1단계 단답 / 2 멀티홉 / 3 교차 / 4 온톨로지
        hit → 가드 후 실행
        miss → text_to_cypher → 가드 → 실행 (1회 retry) → 실패 시 anchor_chunks 안전망
  [5] anchor_chunks 보완 (멀티홉/교차/온톨로지 또는 결과 빈약)
  [6] 집계 (count 키워드)
  [7] score (의도×fact_type 가중치)
  [8] self_check (요청 커버리지)

재무 단건/추이(fin_value/fin_trend)는 route='rdb' 로 reject — graph 가 처리 안 함.
"""
from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from ...retrievers.graph import GraphRetriever
from .cypher_guard import GuardError, validate_cypher
from .intent_classifier import INTENT_ROUTE, IntentClassifier
from .intent_router import match_template
from .score import apply_score_weights
from .self_check import self_check
from .text_to_cypher import CypherGenError, NoLLMAvailable, TextToCypher


# 자주 쓰는 IFRS 계정 한국어 → account_id (backend/app/account_dict 핵심 5개 미러)
_ACCOUNT_KO: dict[str, str] = {
    "매출액": "ifrs-full_Revenue", "매출": "ifrs-full_Revenue",
    "자산총액": "ifrs-full_Assets", "자산": "ifrs-full_Assets",
    "부채총액": "ifrs-full_Liabilities", "부채": "ifrs-full_Liabilities",
    "자본총액": "ifrs-full_Equity", "자본": "ifrs-full_Equity",
    "당기순이익": "ifrs-full_ProfitLoss", "순이익": "ifrs-full_ProfitLoss",
}
_AGG_RE = re.compile(r"(몇\s*[개명곳]|총\s*(?:몇|개수|수)|개수|총합|합계|모두)")


class GraphAgentInput(TypedDict, total=False):
    query: str
    intent: str | None
    entities: list[str]
    slots: dict[str, Any]
    requested_relations: list[str] | None
    max_hops: int


class GraphAgentOutput(TypedDict, total=False):
    graph_facts: list[dict[str, Any]]
    graph_chunk_ids: list[str]
    graph_paths: list[list[str]]
    intent: str
    route: str
    stage: int
    rejected: bool          # route != graph 면 True (graph 가 처리 안 함)
    template_used: str | None
    cypher_executed: list[str]
    self_check: dict[str, Any]
    latency_ms: float
    errors: list[str]


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _resolve_account(query: str, slots: dict[str, Any]) -> str | None:
    if slots.get("account_id"):
        return slots["account_id"]
    for ko, code in _ACCOUNT_KO.items():
        if ko in query:
            return code
    return None


class GraphAgentRunner:
    def __init__(
        self,
        retriever: GraphRetriever | None = None,
        llm: TextToCypher | None = None,
        intent_clf: IntentClassifier | None = None,
    ):
        self.r = retriever or GraphRetriever()
        self.clf = intent_clf or IntentClassifier()
        try:
            self.llm: TextToCypher | None = llm or TextToCypher()
        except NoLLMAvailable:
            self.llm = None

    def _exec_cypher(
        self, cypher: str, params: dict[str, Any], fact_type: str | None
    ) -> list[dict[str, Any]]:
        with self.r.driver.session() as s:
            rows = s.run(cypher, **params).data()
        facts: list[dict[str, Any]] = []
        for row in rows:
            row = dict(row)
            row["type"] = fact_type or "text_to_cypher"
            facts.append(row)
        return facts

    def _llm_path(
        self, query: str, entities: list[str], slots: dict[str, Any], errors: list[str]
    ) -> tuple[list[dict[str, Any]], list[str], str | None]:
        if not self.llm:
            return [], [], None
        prior_error: str | None = None
        for attempt in range(2):
            try:
                cypher, params, _ = self.llm.generate(query, entities, slots, prior_error=prior_error)
                with self.r.driver.session() as s:
                    normalized = validate_cypher(cypher, session=s, params=params)
                facts = self._exec_cypher(normalized, params, fact_type=None)
                return facts, [normalized], "text_to_cypher"
            except (GuardError, CypherGenError) as e:
                prior_error = str(e)
                errors.append(f"llm_attempt_{attempt}: {e}")
            except Exception as e:
                prior_error = str(e)
                errors.append(f"llm_exec_{attempt}: {e}")
        return [], [], None

    def run(self, inp: GraphAgentInput) -> GraphAgentOutput:
        t0 = time.perf_counter()
        query = (inp.get("query") or "").strip()
        errors: list[str] = []

        # [1] 의도분류 — inp.intent 가 12의도면 사용, 아니면 분류기
        given = inp.get("intent")
        if given in INTENT_ROUTE:
            route, default_stage = INTENT_ROUTE[given]
            res = {"intent": given, "route": route, "stage": default_stage,
                   "slots": {}, "relations": []}
        else:
            res = self.clf.classify(query)

        intent, route, stage = res["intent"], res["route"], res["stage"]
        slots = {**(res.get("slots") or {}), **(inp.get("slots") or {})}
        relations = inp.get("requested_relations") or res.get("relations") or []

        # [2] route 게이트 — graph 비대상은 reject
        if route != "graph":
            return GraphAgentOutput(
                graph_facts=[], graph_chunk_ids=[], graph_paths=[],
                intent=intent, route=route, stage=stage, rejected=True,
                template_used=None, cypher_executed=[],
                self_check={"has_results": False, "missing": [f"route={route} (graph 비대상)"]},
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                errors=errors,
            )

        # [3] 엔티티/슬롯 채움
        entities = list(inp.get("entities") or [])
        if not entities:
            probe = query
            if slots.get("org_name"):
                probe = f"{slots['org_name']} {query}"
            entities = self.r.detect_entities(probe)
        acct = _resolve_account(query, slots)
        if acct:
            slots["account_id"] = acct

        # [4] 템플릿 매칭 (intent + stage)
        tpl_key, cypher, params, fact_type = match_template(intent, query, entities, slots, stage=stage)

        facts: list[dict[str, Any]] = []
        chunk_ids: list[str] = []
        paths: list[list[str]] = []
        cypher_executed: list[str] = []
        template_used: str | None = None

        if tpl_key and cypher is not None and params is not None:
            try:
                with self.r.driver.session() as s:
                    normalized = validate_cypher(cypher, session=s, params=params)
                facts = self._exec_cypher(normalized, params, fact_type=fact_type)
                cypher_executed.append(normalized)
                template_used = tpl_key
            except Exception as e:
                errors.append(f"template_exec[{tpl_key}]: {e}")
                tpl_key = None

        # [4-B] 템플릿 miss → LLM Cypher
        if not facts and not tpl_key:
            llm_facts, llm_cyphers, llm_tag = self._llm_path(query, entities, slots, errors)
            facts = llm_facts
            cypher_executed.extend(llm_cyphers)
            template_used = llm_tag

        # [5] anchor_chunks 보완 (멀티홉/교차/온톨로지 또는 결과 빈약)
        if entities and (stage >= 2 or not facts):
            try:
                for h in self.r.anchor_chunks(entities)[: self.r.top_n]:
                    if h.chunk_id:
                        chunk_ids.append(h.chunk_id)
                    if h.path:
                        paths.append(h.path)
            except Exception as e:
                errors.append(f"anchor_chunks: {e}")

        # [6] 집계 (count 키워드)
        if entities and _AGG_RE.search(query):
            try:
                for h in self.r.aggregate_counts(entities):
                    if h.fact_card:
                        facts.append(dict(h.fact_card))
            except Exception as e:
                errors.append(f"aggregate_counts: {e}")

        # [7] score
        facts = apply_score_weights(facts, intent)

        # [8] self_check
        sc = self_check(
            {"entities": entities, "requested_relations": relations, "slots": slots},
            facts, chunk_ids,
        )

        return GraphAgentOutput(
            graph_facts=facts,
            graph_chunk_ids=_dedup_keep_order(chunk_ids),
            graph_paths=paths,
            intent=intent, route=route, stage=stage, rejected=False,
            template_used=template_used,
            cypher_executed=cypher_executed,
            self_check=sc,
            latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            errors=errors,
        )

    def close(self) -> None:
        self.r.close()
