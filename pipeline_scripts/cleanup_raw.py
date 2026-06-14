"""적재 완료 후 원본 작업물(raw·청크 JSONL) 정리 — 디스크 슬림.

raw/{회사}/ 와 chunk/output/{회사}_*.jsonl 은 적재 단계(2 청킹·3 MariaDB·5 Neo4j
정형)의 입력일 뿐, 적재가 끝나면 데이터는 전부 DB(dart_raw_index·document_index·
chunk_index·Neo4j)에 들어가 있다. 따라서 영구 보관 불필요 — 재청킹이 필요하면
DART 에서 다시 fetch(무료·멱등). 이 스텝은 파이프라인 끝에서 해당 회사의 작업물만
삭제한다. extract(6)·QC 는 DB/그래프만 읽으므로 영향 없음.

env POLARIS_CORP_NAMES(회사명) 로 대상 회사 폴더를 안다.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
CHUNK_OUT = HERE / "chunk" / "output"


def marker(**kw) -> None:
    """SSE 진행률 마커 — 다른 스텝과 동일 포맷 (services/pipeline_runner._parse_marker)."""
    print("POLARIS_PIPELINE " + json.dumps(kw, ensure_ascii=False), flush=True)


def _dir_size_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024


def main() -> int:
    names = [n.strip() for n in os.getenv("POLARIS_CORP_NAMES", "").split(",") if n.strip()]
    if not names:
        print("POLARIS_CORP_NAMES 없음 — 정리 대상 회사 미지정, 건너뜀")
        marker(step="cleanup", phase="end", freed_mb=0)
        return 0

    freed = 0.0
    total = len(names)
    for i, name in enumerate(names, 1):
        comp_raw = RAW / name
        if comp_raw.exists():
            freed += _dir_size_mb(comp_raw)
            shutil.rmtree(comp_raw, ignore_errors=True)
            print(f"  raw/{name} 삭제")
        # chunk/output/{회사}_*.jsonl
        n_jsonl = 0
        for f in CHUNK_OUT.glob(f"{name}_*.jsonl"):
            freed += f.stat().st_size / 1024 / 1024
            f.unlink(missing_ok=True)
            n_jsonl += 1
        if n_jsonl:
            print(f"  chunk/output/{name}_*.jsonl {n_jsonl}개 삭제")
        marker(step="cleanup", done=i, total=total, freed_mb=round(freed, 1))

    marker(step="cleanup", phase="end", freed_mb=round(freed, 1))
    print(f"원본 정리 완료: {freed:.0f}MB 회수 (데이터는 DB에 보존)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
