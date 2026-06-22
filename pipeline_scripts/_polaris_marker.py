"""POLARIS_PIPELINE 진행률 마커 헬퍼.

관리자 파이프라인 콘솔(services/pipeline_runner.py)이 이 형식의 stdout 라인을
파싱해 단계별 진행률을 갱신한다. 형식: `POLARIS_PIPELINE <json-1line>\\n`

스크립트에서 사용:
    from _polaris_marker import marker
    marker(step="fetch", phase="start")
    marker(step="fetch", done=12, total=250, rcept_no="20240514001326")
    marker(step="fetch", phase="end", added=3, skipped=247)

마커는 선택 사항이다 — 없으면 진행률이 0→1 로 점프할 뿐 동작에는 지장 없다.
"""
from __future__ import annotations

import json
import os
import sys
import time

_CORP = os.getenv("POLARIS_CORPS", "").split(",")[0] or "unknown"


def marker(step: str, **fields) -> None:
    payload = {"v": 1, "step": step, "corp_code": _CORP, "ts": time.time(), **fields}
    sys.stdout.write("POLARIS_PIPELINE " + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()
