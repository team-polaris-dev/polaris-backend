"""관리자 콘솔 전용 설정 — env 로딩 + 토큰 비교.

기존 polaris-backend `config/llm.py` 옆에 배치. 별도 파일로 분리한 이유:
- ADMIN_TOKEN 등 운영 시크릿은 LLM 설정과 라이프사이클·민감도가 다름
- 모듈 임포트만으로 KeyError 가 나도록 해서 토큰 미설정 부팅을 막음
"""
from __future__ import annotations

import os
from pathlib import Path

# 부팅 시 한 번 평가 — 미설정이면 KeyError 로 부팅 실패(의도된 안전 디폴트)
ADMIN_TOKEN: str = os.environ["ADMIN_TOKEN"]

# pipeline_scripts/ 절대경로. 컨테이너 안은 /app/pipeline_scripts 가 디폴트.
PIPELINE_WORKDIR: Path = Path(
    os.getenv("PIPELINE_WORKDIR") or (Path(__file__).resolve().parents[1] / "pipeline_scripts")
).resolve()

# 잡별 stdout 백업 로그 디렉터리
LOGS_DIR: Path = Path(os.getenv("LOGS_DIR") or (Path(__file__).resolve().parents[1] / "logs")).resolve()
LOGS_DIR.mkdir(parents=True, exist_ok=True)
