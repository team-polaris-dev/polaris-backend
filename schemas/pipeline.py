"""관리자 파이프라인 API 의 Request/Response Pydantic 모델."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
StepState = Literal["pending", "running", "succeeded", "failed", "skipped", "cancelled"]
StepId = Literal["fetch", "chunk", "mariadb", "qdrant", "neo4j_struct", "extract", "qc", "canon", "cleanup"]
ExtractProvider = Literal["ollama", "apimaker"]


class StepConfig(BaseModel):
    id: StepId
    enabled: bool = True
    # extract 단계 전용 예: {"provider":"apimaker","model":null,"limit":200,"positive_only":true}
    # provider 기본 "ollama" (로컬·결정론), limit 생략 시 pending 전체.
    # 구 "chunk_window" 파라미터는 폐기 — extract_step.py 가 --limit 으로 단일화.
    params: dict[str, Any] = Field(default_factory=dict)


class JobCreateRequest(BaseModel):
    corp_codes: list[str] = Field(min_length=1)
    steps: list[StepConfig] = Field(min_length=1)
    from_date: str | None = None  # YYYY-MM-DD, fetch 증분
    to_date: str | None = None
    label: str | None = None


class StepStatus(BaseModel):
    step_id: StepId
    corp_code: str
    state: StepState
    progress: float = 0.0
    counters: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    error: str | None = None


class JobResponse(BaseModel):
    job_id: str
    state: JobState
    corp_codes: list[str]
    label: str | None = None
    steps: list[StepStatus]
    created_at: datetime
    updated_at: datetime


class CorpInfo(BaseModel):
    corp_code: str
    corp_name: str
    raw_dir: str | None = None
    has_raw: bool = False
    doc_count: int = 0
    chunk_count: int = 0
    last_fetch: datetime | None = None


class CancelResponse(BaseModel):
    job_id: str
    cancelled: bool
    was_running: bool


class QcResolveRequest(BaseModel):
    """QC 모순 해결 요청 — 사람이 [적용]을 누른 단건 삭제만 수행."""

    kind: Literal["self_loop", "bidirectional_supplies", "non_company_supplies"]
    # self_loop 용
    org: str | None = None
    rel: str | None = None
    chunk_id: str | None = None
    # bidirectional_supplies / non_company_supplies 용 — 비활성화할 방향의 (from, to).
    # 값은 corp_code 또는 er_name. non_company 는 노이즈 엣지 1개를 그대로 끈다.
    from_id: str | None = None
    to_id: str | None = None


class QcJudgeRequest(BaseModel):
    """양방향 SUPPLIES_TO 방향 판정 요청 — in-process LLM(apimaker)이 본문 근거로
    올바른 방향을 제안한다. 적용(삭제)은 사람이 별도 confirm 후 /qc/resolve 로."""

    a: str            # 회사 A 표시명
    b: str            # 회사 B 표시명
    a_id: str         # corp_code 또는 er_name
    b_id: str
    fwd_chunk: str | None = None   # A→B 방향의 근거 chunk_id
    rev_chunk: str | None = None   # B→A 방향의 근거 chunk_id


class QcEntityJudgeRequest(BaseModel):
    """미해소 끝점 1개 타입 판정 요청 — apimaker(Gemini)가 회사/비회사 분류."""

    entity_key: str               # 정규화 이름 (캐시 키)
    name: str                     # 표시명(원문 엔티티명)
    chunk_id: str | None = None   # 판단 근거 본문 chunk_id


class DBStatusResponse(BaseModel):
    mariadb: dict[str, int]
    qdrant: dict[str, dict[str, int]]
    neo4j: dict[str, dict[str, int]]
    measured_at: datetime
