"""subprocess 오케스트레이션 — BackgroundTasks 진입.

- 동시 잡 = 1 (asyncio.Lock).
- 각 단계는 pipeline_scripts/ 의 .py 파일을 subprocess 로 실행.
- stdout 라인을 SSE 로 push + 파일 로그 백업 + POLARIS_PIPELINE 마커 파싱.
- 자식 프로세스 그룹 분리(Windows CREATE_NEW_PROCESS_GROUP) + 2단 취소(SIGTERM → SIGKILL).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from config.admin import LOGS_DIR, PIPELINE_WORKDIR
from services.pipeline_jobs import (
    get_job,
    get_job_config,
    update_job_state,
    update_step,
)
from services.pipeline_steps import STEP_REGISTRY, JobCtx
from services.sse import drop_job_buffer, push_sse

logger = logging.getLogger(__name__)

_WORKER_LOCK = asyncio.Lock()
# 동기 Popen 사용 — Windows + uvicorn reload 는 SelectorEventLoop 라
# asyncio.create_subprocess_exec 이 NotImplementedError 를 던진다.
# stdout 읽기는 asyncio.to_thread 로 감싸서 메인 루프(push_sse/SSE 큐)를 유지한다.
_ACTIVE_PROC: dict[str, subprocess.Popen] = {}
_MARKER_PREFIX = "POLARIS_PIPELINE "


class StepFailed(Exception):
    def __init__(self, step_id: str, rc: int) -> None:
        super().__init__(f"step {step_id} failed (exit={rc})")
        self.step_id = step_id
        self.rc = rc


async def run_job(job_id: str, corp_name_map: dict[str, str]) -> None:
    """라우터의 BackgroundTasks 가 호출하는 진입점.

    corp_name_map: 회사코드→회사명 (raw 폴더명) 매핑. 라우터에서 미리 만들어 주입.
    """
    async with _WORKER_LOCK:
        update_job_state(job_id, "running", pid=os.getpid())
        job = get_job(job_id)
        if job is None:
            logger.error("run_job: job %s not found", job_id)
            return
        # 원본 JobCreateRequest(enabled/params/from_date/to_date)를 config JSON 컬럼에서 복원
        cfg = get_job_config(job_id) or {}
        from_date = cfg.get("from_date")
        to_date = cfg.get("to_date")
        steps: list[dict[str, Any]] = cfg.get("steps") or []
        try:
            for corp_code in job.corp_codes:
                ctx = JobCtx(
                    from_date=from_date,
                    to_date=to_date,
                    corp_name=corp_name_map.get(corp_code, corp_code),
                )
                for step in steps:
                    step_id = step["id"]
                    params = step.get("params") or {}
                    if step.get("enabled", True) is False:
                        update_step(job_id, corp_code, step_id, state="skipped")
                        push_sse(job_id, {"type": "step_end", "corp_code": corp_code,
                                          "step": step_id, "state": "skipped"})
                        continue
                    await _run_step(job_id, corp_code, step_id, params, ctx)
            update_job_state(job_id, "succeeded")
            push_sse(job_id, {"type": "job_end", "state": "succeeded"})
        except StepFailed:
            update_job_state(job_id, "failed")
            push_sse(job_id, {"type": "job_end", "state": "failed"})
        except asyncio.CancelledError:
            update_job_state(job_id, "cancelled")
            push_sse(job_id, {"type": "job_end", "state": "cancelled"})
            raise
        except Exception as e:  # noqa: BLE001 — 예측 못한 예외도 잡 실패로 마킹
            logger.exception("job %s crashed", job_id)
            update_job_state(job_id, "failed")
            push_sse(job_id, {"type": "job_end", "state": "failed", "error": str(e)})


async def _run_step(
    job_id: str,
    corp_code: str,
    step_id: str,
    params: dict[str, Any],
    ctx: JobCtx,
) -> None:
    spec = STEP_REGISTRY[step_id]
    args = spec.build_args(corp_code, params, ctx)
    extra_env = spec.extra_env(params)
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "POLARIS_CORPS": corp_code,
        "POLARIS_CORP_NAMES": ctx.corp_name,
        **extra_env,
    }
    log_path = LOGS_DIR / job_id / corp_code / f"{step_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    update_step(
        job_id, corp_code, step_id,
        state="running",
        started_at=datetime.now(),
        log_path=str(log_path),
    )
    push_sse(job_id, {"type": "step_start", "corp_code": corp_code, "step": step_id})

    script_path = PIPELINE_WORKDIR / spec.script
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # type: ignore[attr-defined]
    proc = subprocess.Popen(
        [sys.executable, "-X", "utf8", str(script_path), *args],
        cwd=str(PIPELINE_WORKDIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    _ACTIVE_PROC[job_id] = proc

    counters: dict[str, Any] = {}
    last_progress = 0.0
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            assert proc.stdout is not None
            while True:
                # blocking readline 은 스레드로 — 라인 처리(push_sse 등)는 메인 루프에서
                raw = await asyncio.to_thread(proc.stdout.readline)
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                logf.write(line + "\n")
                push_sse(job_id, {
                    "type": "log", "corp_code": corp_code, "step": step_id, "line": line,
                })
                progress, marker_counters = _parse_marker(line)
                if marker_counters is not None:
                    counters.update(marker_counters)
                if progress is not None and progress != last_progress:
                    last_progress = progress
                    update_step(job_id, corp_code, step_id, progress=progress, counters=counters)
        rc = await asyncio.to_thread(proc.wait)
    finally:
        _ACTIVE_PROC.pop(job_id, None)

    if rc != 0:
        update_step(
            job_id, corp_code, step_id,
            state="failed",
            ended_at=datetime.now(),
            counters=counters,
            error=f"exit_code={rc}",
        )
        push_sse(job_id, {
            "type": "step_end", "corp_code": corp_code, "step": step_id, "state": "failed",
        })
        raise StepFailed(step_id, rc)

    update_step(
        job_id, corp_code, step_id,
        state="succeeded",
        progress=1.0,
        ended_at=datetime.now(),
        counters=counters,
    )
    push_sse(job_id, {
        "type": "step_end", "corp_code": corp_code, "step": step_id, "state": "succeeded",
    })


def _parse_marker(line: str) -> tuple[float | None, dict[str, Any] | None]:
    """`POLARIS_PIPELINE {...}` 한 줄 → (progress, counters dict).

    형식 어긋나면 둘 다 None 반환(에러 던지지 않음).
    """
    if not line.startswith(_MARKER_PREFIX):
        return None, None
    try:
        marker = json.loads(line[len(_MARKER_PREFIX):])
    except json.JSONDecodeError:
        return None, None
    progress: float | None = None
    done = marker.get("done")
    total = marker.get("total")
    if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
        progress = max(0.0, min(1.0, float(done) / float(total)))
    if marker.get("phase") == "end":
        progress = 1.0
    counters = {k: v for k, v in marker.items()
                if k not in {"v", "step", "corp_code", "phase", "ts", "done", "total"}}
    return progress, counters or None


async def cancel_job(job_id: str) -> tuple[bool, bool]:
    """(cancelled, was_running). 큐 대기 중이면 즉시 state='cancelled', 실행 중이면 신호 전송."""
    proc = _ACTIVE_PROC.get(job_id)
    if proc is None:
        # 큐 대기 중이거나 이미 종료. DB 상태 보고 결정
        from services.pipeline_jobs import get_job as _get_job

        cur = _get_job(job_id)
        if cur is None:
            return False, False
        if cur.state in {"queued"}:
            update_job_state(job_id, "cancelled")
            push_sse(job_id, {"type": "job_end", "state": "cancelled"})
            return True, False
        return False, False

    if os.name == "nt":
        proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
    else:
        proc.terminate()
    try:
        await asyncio.wait_for(asyncio.to_thread(proc.wait), timeout=5)
    except asyncio.TimeoutError:
        proc.kill()
    return True, True


def cleanup_after_job(job_id: str) -> None:
    """잡 완전 종료 후 SSE 버퍼 해제."""
    drop_job_buffer(job_id)
