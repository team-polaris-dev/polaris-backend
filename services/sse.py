"""SSE 헬퍼 — 잡 단위 asyncio.Queue + 메모리 ring buffer.

브라우저 EventSource 는 자동 재연결 시도. 재연결 시 ring buffer 의 과거 라인을 먼저 흘려서
중간 끊김을 메움. 15초 idle 마다 ': keep-alive' SSE comment.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Any, AsyncIterator

_JOB_QUEUES: dict[str, list[asyncio.Queue]] = defaultdict(list)
_RING_BUFFER: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=1000))

_KEEPALIVE_INTERVAL = 15.0


def push_sse(job_id: str, event: dict[str, Any]) -> None:
    """잡 워커가 호출 — 모든 구독자(queue)에 이벤트 fan-out + ring buffer 보관."""
    if "ts" not in event:
        event["ts"] = time.time()
    payload = json.dumps(event, ensure_ascii=False)
    _RING_BUFFER[job_id].append(payload)
    for q in list(_JOB_QUEUES[job_id]):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            # 느린 구독자 무시 — ring buffer 에는 남으니 재연결시 따라잡힘
            pass


async def stream_job(job_id: str) -> AsyncIterator[str]:
    """라우터에서 StreamingResponse(... media_type='text/event-stream') 의 generator."""
    q: asyncio.Queue[str] = asyncio.Queue(maxsize=2000)
    _JOB_QUEUES[job_id].append(q)
    try:
        # 과거 라인 replay (재연결 보강)
        for past in list(_RING_BUFFER[job_id]):
            yield f"data: {past}\n\n"

        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=_KEEPALIVE_INTERVAL)
                yield f"data: {payload}\n\n"
            except asyncio.TimeoutError:
                yield ": keep-alive\n\n"  # SSE comment
    finally:
        try:
            _JOB_QUEUES[job_id].remove(q)
        except ValueError:
            pass


def drop_job_buffer(job_id: str) -> None:
    """잡 완전 종료 후 버퍼 해제 (메모리 회수). 라우터/runner 의 후처리에서 호출."""
    _RING_BUFFER.pop(job_id, None)
