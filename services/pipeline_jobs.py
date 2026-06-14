"""잡 영속화 (MariaDB) + 부팅 시 부트스트랩 + stale sweep.

테이블: pipeline_jobs, pipeline_step_runs (운영 메타. DART 데이터 영역과 분리.)
기존 polaris-backend `tool.rdb_client.mariadb_conn()` 컨텍스트매니저 재사용.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import pymysql.cursors

# 기존 polaris-backend tool 모듈 재사용. SELECT-only 게이트(execute_sql_query)는 우회.
from tool.rdb_client import mariadb_conn

from schemas.pipeline import (
    JobCreateRequest,
    JobResponse,
    JobState,
    StepConfig,
    StepStatus,
)


DDL = """
CREATE TABLE IF NOT EXISTS pipeline_jobs (
  job_id        CHAR(36)        PRIMARY KEY,
  state         VARCHAR(16)     NOT NULL,
  corp_codes    JSON            NOT NULL,
  config        JSON            NOT NULL,
  label         VARCHAR(200),
  pid           INT,
  created_at    DATETIME(3)     NOT NULL,
  updated_at    DATETIME(3)     NOT NULL,
  INDEX idx_state_created (state, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS pipeline_step_runs (
  run_id        BIGINT          AUTO_INCREMENT PRIMARY KEY,
  job_id        CHAR(36)        NOT NULL,
  corp_code     CHAR(8)         NOT NULL,
  step_id       VARCHAR(32)     NOT NULL,
  state         VARCHAR(16)     NOT NULL,
  progress      DOUBLE          NOT NULL DEFAULT 0,
  counters      JSON,
  log_path      VARCHAR(500),
  error         TEXT,
  started_at    DATETIME(3),
  ended_at      DATETIME(3),
  INDEX idx_job_step (job_id, corp_code, step_id),
  CONSTRAINT fk_step_job FOREIGN KEY (job_id) REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


def init_pipeline_tables() -> None:
    """부팅 시 1회 호출. CREATE TABLE IF NOT EXISTS 라 재실행 안전."""
    with mariadb_conn() as conn, conn.cursor() as cur:
        for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
            cur.execute(stmt)
        conn.commit()


def sweep_stale_jobs() -> int:
    """워커가 죽기 전에 'running' 으로 남긴 잡 정리. 부팅 시 1회 호출."""
    try:
        import psutil  # type: ignore

        def alive(pid: int | None) -> bool:
            return bool(pid) and psutil.pid_exists(pid)
    except ImportError:
        # psutil 없으면 보수적으로 모두 죽었다고 간주(재부팅했으니 워커도 사라짐)
        def alive(pid: int | None) -> bool:  # type: ignore[no-redef]
            return False

    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT job_id, pid FROM pipeline_jobs WHERE state IN ('running','queued')")
        candidates = cur.fetchall()
        dead = [row["job_id"] for row in candidates if not alive(row.get("pid"))]
        if dead:
            placeholders = ",".join(["%s"] * len(dead))
            cur.execute(
                f"UPDATE pipeline_jobs SET state='failed', updated_at=NOW(3) "
                f"WHERE job_id IN ({placeholders})",
                dead,
            )
            cur.execute(
                f"UPDATE pipeline_step_runs SET state='failed', "
                f"error=COALESCE(error,'worker_died_before_restart'), ended_at=NOW(3) "
                f"WHERE job_id IN ({placeholders}) AND state IN ('running','pending')",
                dead,
            )
            conn.commit()
        return len(dead)


def create_job(req: JobCreateRequest) -> str:
    """UUID 발급 + pipeline_jobs row 생성 + 단계별 step_runs pending row 미리 생성."""
    job_id = str(uuid.uuid4())
    now = datetime.now()
    with mariadb_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pipeline_jobs "
            "(job_id, state, corp_codes, config, label, created_at, updated_at) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                job_id,
                "queued",
                json.dumps(req.corp_codes, ensure_ascii=False),
                req.model_dump_json(),
                req.label,
                now,
                now,
            ),
        )
        for corp_code in req.corp_codes:
            for step in req.steps:
                cur.execute(
                    "INSERT INTO pipeline_step_runs "
                    "(job_id, corp_code, step_id, state, progress, counters) "
                    "VALUES (%s,%s,%s,%s,0,JSON_OBJECT())",
                    (job_id, corp_code, step.id, "pending"),
                )
        conn.commit()
    return job_id


def update_job_state(job_id: str, state: JobState, *, pid: int | None = None) -> None:
    with mariadb_conn() as conn, conn.cursor() as cur:
        if pid is not None:
            cur.execute(
                "UPDATE pipeline_jobs SET state=%s, pid=%s, updated_at=NOW(3) WHERE job_id=%s",
                (state, pid, job_id),
            )
        else:
            cur.execute(
                "UPDATE pipeline_jobs SET state=%s, updated_at=NOW(3) WHERE job_id=%s",
                (state, job_id),
            )
        conn.commit()


def update_step(
    job_id: str,
    corp_code: str,
    step_id: str,
    *,
    state: str | None = None,
    progress: float | None = None,
    counters: dict[str, Any] | None = None,
    started_at: datetime | None = None,
    ended_at: datetime | None = None,
    log_path: str | None = None,
    error: str | None = None,
) -> None:
    fields: list[str] = []
    values: list[Any] = []
    if state is not None:
        fields.append("state=%s"); values.append(state)
    if progress is not None:
        fields.append("progress=%s"); values.append(progress)
    if counters is not None:
        fields.append("counters=%s"); values.append(json.dumps(counters, ensure_ascii=False))
    if started_at is not None:
        fields.append("started_at=%s"); values.append(started_at)
    if ended_at is not None:
        fields.append("ended_at=%s"); values.append(ended_at)
    if log_path is not None:
        fields.append("log_path=%s"); values.append(log_path)
    if error is not None:
        fields.append("error=%s"); values.append(error)
    if not fields:
        return
    values.extend([job_id, corp_code, step_id])
    with mariadb_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE pipeline_step_runs SET {', '.join(fields)} "
            f"WHERE job_id=%s AND corp_code=%s AND step_id=%s",
            values,
        )
        conn.commit()


def get_job_config(job_id: str) -> dict[str, Any] | None:
    """잡의 원본 JobCreateRequest(config JSON 컬럼)를 파싱해 반환.

    runner 가 단계별 enabled/params + from_date/to_date 를 복원하는 데 사용.
    """
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT config FROM pipeline_jobs WHERE job_id=%s", (job_id,))
        row = cur.fetchone()
    if not row:
        return None
    return _load_json(row["config"])


def get_job(job_id: str) -> JobResponse | None:
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT * FROM pipeline_jobs WHERE job_id=%s", (job_id,))
        job_row = cur.fetchone()
        if not job_row:
            return None
        cur.execute(
            "SELECT * FROM pipeline_step_runs WHERE job_id=%s ORDER BY run_id",
            (job_id,),
        )
        step_rows = cur.fetchall()
    return _to_response(job_row, step_rows)


def list_jobs(limit: int = 20, offset: int = 0) -> list[JobResponse]:
    limit = max(1, min(100, limit))
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            "SELECT * FROM pipeline_jobs ORDER BY created_at DESC LIMIT %s OFFSET %s",
            (limit, offset),
        )
        job_rows = cur.fetchall()
        if not job_rows:
            return []
        job_ids = [j["job_id"] for j in job_rows]
        placeholders = ",".join(["%s"] * len(job_ids))
        cur.execute(
            f"SELECT * FROM pipeline_step_runs WHERE job_id IN ({placeholders}) ORDER BY run_id",
            job_ids,
        )
        all_steps = cur.fetchall()
    by_job: dict[str, list[dict[str, Any]]] = {}
    for s in all_steps:
        by_job.setdefault(s["job_id"], []).append(s)
    return [_to_response(j, by_job.get(j["job_id"], [])) for j in job_rows]


def _to_response(job_row: dict[str, Any], step_rows: list[dict[str, Any]]) -> JobResponse:
    return JobResponse(
        job_id=job_row["job_id"],
        state=job_row["state"],
        corp_codes=_load_json(job_row["corp_codes"]),
        label=job_row.get("label"),
        steps=[
            StepStatus(
                step_id=s["step_id"],
                corp_code=s["corp_code"],
                state=s["state"],
                progress=float(s.get("progress") or 0),
                counters=_load_json(s.get("counters")) or {},
                started_at=s.get("started_at"),
                ended_at=s.get("ended_at"),
                error=s.get("error"),
            )
            for s in step_rows
        ],
        created_at=job_row["created_at"],
        updated_at=job_row["updated_at"],
    )


def _load_json(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    return json.loads(v)
