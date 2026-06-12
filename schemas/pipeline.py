"""관리자 파이프라인 API 의 Request/Response Pydantic 모델."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
StepState = Literal["pending", "running", "succeeded", "failed", "skipped", "cancelled"]
StepId = Literal["fetch", "chunk", "mariadb", "qdrant", "neo4j_struct", "extract"]
ExtractProvider = Literal["ollama", "claude"]


class StepConfig(BaseModel):
    id: StepId
    enabled: bool = True
    # extract 단계 전용. 예: {"provider":"claude","model":"claude-haiku-4-5","chunk_window":[0,200],"positive_only":true}
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


class DBStatusResponse(BaseModel):
    mariadb: dict[str, int]
    qdrant: dict[str, dict[str, int]]
    neo4j: dict[str, dict[str, int]]
    measured_at: datetime
