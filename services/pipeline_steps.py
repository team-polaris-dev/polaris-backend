"""단계 레지스트리 — step_id → 실제 스크립트 경로·인자·env 매핑.

이 한 곳만 수정하면 새 단계 추가/스크립트 위치 변경에 대응. routers/services 코드에 스크립트
경로가 박혀있지 않게 분리.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StepSpec:
    # pipeline_scripts/ 기준 상대 경로 (config.admin.PIPELINE_WORKDIR + 이 경로)
    script: str
    # (corp_code, step.params, job_cfg) -> list[str] CLI 인자
    build_args: Callable[[str, dict[str, Any], "JobCtx"], list[str]]
    # (step.params) -> 추가 환경변수
    extra_env: Callable[[dict[str, Any]], dict[str, str]] = lambda _p: {}


@dataclass(frozen=True)
class JobCtx:
    """build_args 에 넘기는 잡 단위 컨텍스트."""
    from_date: str | None
    to_date: str | None
    corp_name: str  # POLARIS_CORP_NAMES 세팅용


def _fetch_args(corp_code: str, params: dict[str, Any], ctx: JobCtx) -> list[str]:
    args: list[str] = []
    if ctx.from_date:
        args += ["--from-date", ctx.from_date]
    if ctx.to_date:
        args += ["--to-date", ctx.to_date]
    return args


def _extract_args(corp_code: str, params: dict[str, Any], ctx: JobCtx) -> list[str]:
    # extract_step.py(prep→추출→load 일괄) CLI. 기본 = pending 전부, limit 은 선택 상한.
    args = [corp_code]
    limit = params.get("limit")
    if isinstance(limit, (int, float)) and limit > 0:
        args += ["--limit", str(int(limit))]
    if params.get("positive_only"):
        args.append("--positive")
    args += ["--provider", str(params.get("provider", "ollama"))]
    if params.get("model"):
        args += ["--model", str(params["model"])]
    return args


def _extract_env(params: dict[str, Any]) -> dict[str, str]:
    # provider: ollama(기본·로컬) | apimaker(in-process Gemini CLI — API 키 불필요).
    # 구 'claude'(ANTHROPIC_API_KEY) 경로는 2026-06-13 폐기 — API 미사용 방침.
    return {
        "POLARIS_EXTRACT_PROVIDER": str(params.get("provider", "ollama")),
        "POLARIS_EXTRACT_MODEL": str(params.get("model", "")),
    }


STEP_REGISTRY: dict[str, StepSpec] = {
    "fetch": StepSpec(script="fetch_dart.py", build_args=_fetch_args),
    "chunk": StepSpec(script="chunk/chunk.py", build_args=lambda *_a: []),
    "mariadb": StepSpec(script="load/load_mariadb.py", build_args=lambda *_a: []),
    "qdrant": StepSpec(script="load/embed_qdrant.py", build_args=lambda *_a: []),
    "neo4j_struct": StepSpec(script="graph/load_structured.py", build_args=lambda *_a: []),
    "extract": StepSpec(script="graph/extract_step.py", build_args=_extract_args, extra_env=_extract_env),
    "qc": StepSpec(script="graph/qc_step.py", build_args=lambda corp_code, _p, _c: [corp_code]),
    # 글로벌 엔티티 통합 — needs_er→corp_code 병합 + Product/Tech 재캐논. 전역·인자 없음.
    "canon": StepSpec(script="graph/canonicalize.py", build_args=lambda *_a: []),
    # 원본 정리 — 인자 없음(POLARIS_CORP_NAMES env 로 대상 회사 인식)
    "cleanup": StepSpec(script="cleanup_raw.py", build_args=lambda *_a: []),
}
