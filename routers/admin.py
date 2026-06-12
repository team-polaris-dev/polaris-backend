"""관리자 라우터 — /api/admin/*

엔드포인트:
- GET  /health
- GET  /db/status
- GET  /corps
- POST /pipeline/jobs
- GET  /pipeline/jobs
- GET  /pipeline/jobs/{job_id}
- GET  /pipeline/jobs/{job_id}/stream  (SSE)
- POST /pipeline/jobs/{job_id}/cancel
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import pymysql.cursors
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse

from config.admin import ADMIN_TOKEN, PIPELINE_WORKDIR
from schemas.pipeline import (
    CancelResponse,
    CorpInfo,
    DBStatusResponse,
    JobCreateRequest,
    JobResponse,
)
from services import pipeline_jobs, pipeline_runner
from services.sse import stream_job
from tool.rdb_client import mariadb_conn

router = APIRouter(prefix="/api/admin", tags=["admin"])


async def verify_admin_token(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    token: str | None = Query(default=None),
) -> None:
    received = x_admin_token or token
    if not received or not secrets.compare_digest(received, ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="invalid admin token")


@router.get("/health")
async def health(_=Depends(verify_admin_token)) -> dict[str, bool]:
    return {"ok": True}


@router.get("/db/status", response_model=DBStatusResponse)
async def db_status(_=Depends(verify_admin_token)) -> DBStatusResponse:
    return DBStatusResponse(
        mariadb=_count_mariadb(),
        qdrant=_count_qdrant(),
        neo4j=_count_neo4j(),
        measured_at=datetime.now(),
    )


@router.get("/corps", response_model=list[CorpInfo])
async def corps(_=Depends(verify_admin_token)) -> list[CorpInfo]:
    raw_root = PIPELINE_WORKDIR / "raw"
    raw_dirs = {p.name for p in raw_root.iterdir() if p.is_dir()} if raw_root.exists() else set()

    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            "SELECT corp_code, corp_name, COUNT(*) AS doc_count "
            "FROM document_index GROUP BY corp_code, corp_name"
        )
        doc_rows = cur.fetchall()
        cur.execute(
            "SELECT corp_code, COUNT(*) AS chunk_count FROM chunk_index GROUP BY corp_code"
        )
        chunk_rows = {r["corp_code"]: int(r["chunk_count"]) for r in cur.fetchall()}

    out: dict[str, CorpInfo] = {}
    for r in doc_rows:
        code = (r["corp_code"] or "").zfill(8)
        name = r["corp_name"] or ""
        out[code] = CorpInfo(
            corp_code=code,
            corp_name=name,
            raw_dir=name if name in raw_dirs else None,
            has_raw=name in raw_dirs,
            doc_count=int(r["doc_count"]),
            chunk_count=chunk_rows.get(code, 0),
        )
    # raw 폴더만 있고 DB 에 적재 안 된 회사도 노출
    for name in raw_dirs:
        if not any(c.corp_name == name for c in out.values()):
            out[f"__raw__{name}"] = CorpInfo(
                corp_code="00000000", corp_name=name, raw_dir=name, has_raw=True,
            )
    return sorted(out.values(), key=lambda c: c.corp_name)


@router.post("/pipeline/jobs", response_model=JobResponse, status_code=201)
async def create_job(
    req: JobCreateRequest,
    bg: BackgroundTasks,
    _=Depends(verify_admin_token),
) -> JobResponse:
    # 회사명 매핑 — runner 가 POLARIS_CORP_NAMES 로 주입
    corp_name_map = _resolve_corp_names(req.corp_codes)

    job_id = pipeline_jobs.create_job(req)
    bg.add_task(pipeline_runner.run_job, job_id, corp_name_map)
    job = pipeline_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=500, detail="failed to create job")
    return job


@router.get("/pipeline/jobs", response_model=list[JobResponse])
async def list_jobs_route(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _=Depends(verify_admin_token),
) -> list[JobResponse]:
    return pipeline_jobs.list_jobs(limit=limit, offset=offset)


@router.get("/pipeline/jobs/{job_id}", response_model=JobResponse)
async def get_job_route(job_id: str, _=Depends(verify_admin_token)) -> JobResponse:
    job = pipeline_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return job


@router.get("/pipeline/jobs/{job_id}/stream")
async def stream_job_route(job_id: str, _=Depends(verify_admin_token)) -> StreamingResponse:
    job = pipeline_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    return StreamingResponse(
        stream_job(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/pipeline/jobs/{job_id}/cancel", response_model=CancelResponse)
async def cancel_job_route(job_id: str, _=Depends(verify_admin_token)) -> CancelResponse:
    job = pipeline_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
    cancelled, was_running = await pipeline_runner.cancel_job(job_id)
    return CancelResponse(job_id=job_id, cancelled=cancelled, was_running=was_running)


# ----- 내부 헬퍼 -----

def _resolve_corp_names(corp_codes: list[str]) -> dict[str, str]:
    if not corp_codes:
        return {}
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        placeholders = ",".join(["%s"] * len(corp_codes))
        cur.execute(
            f"SELECT DISTINCT corp_code, corp_name FROM document_index "
            f"WHERE corp_code IN ({placeholders})",
            corp_codes,
        )
        rows = cur.fetchall()
    mapping = {r["corp_code"]: (r["corp_name"] or r["corp_code"]) for r in rows}
    for code in corp_codes:
        mapping.setdefault(code, code)
    return mapping


def _count_mariadb() -> dict[str, int]:
    tables = [
        "dart_raw_index", "document_index", "chunk_index",
        "fin_metric", "extraction_provenance",
        "pipeline_jobs", "pipeline_step_runs",
    ]
    out: dict[str, int] = {}
    # mariadb_conn() 은 DictCursor 를 반환하므로 COUNT 에 별칭을 붙여 dict 키로 읽는다.
    with mariadb_conn() as conn, conn.cursor() as cur:
        for t in tables:
            try:
                cur.execute(f"SELECT COUNT(*) AS c FROM {t}")
                row = cur.fetchone()
                out[t] = int(row["c"]) if row else 0
            except Exception:
                out[t] = -1  # 테이블 미존재 등
    return out


def _count_qdrant() -> dict[str, dict[str, int]]:
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return {"_error": {"reason": "qdrant_client not installed"}}
    # 클라이언트/서버 마이너 버전이 어긋나도 카운트는 정상 동작하므로 호환성 체크를 끈다.
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        check_compatibility=False,
    )
    out: dict[str, dict[str, int]] = {}
    for coll in ["polaris-chunks", "polaris-org-er"]:
        try:
            count = client.count(collection_name=coll, exact=True).count
            points = getattr(client.get_collection(coll), "points_count", None)
            out[coll] = {
                "points_count": int(points if points is not None else count),
                "vectors_count": int(count),
            }
        except Exception:
            out[coll] = {"points_count": -1, "vectors_count": -1}
    return out


def _count_neo4j() -> dict[str, dict[str, int]]:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        return {"nodes": {"_error": -1}, "rels": {}}
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )
    nodes: dict[str, int] = {}
    rels: dict[str, int] = {}
    try:
        with driver.session() as sess:
            for label in ["Organization", "Person", "Chunk", "FinMetric", "Product", "Technology"]:
                r = sess.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()
                nodes[label] = int(r["c"]) if r else 0
            for rel in ["EXECUTIVE_OF", "IS_MAJOR_SHAREHOLDER_OF", "PRODUCES",
                        "USES_TECH", "SUPPLIES_TO", "RELATED_PARTY", "HAS_METRIC",
                        "INVESTS_IN", "IS_SUBSIDIARY_OF", "INTERLOCKING_DIRECTORATE"]:
                r = sess.run(f"MATCH ()-[r:`{rel}`]->() RETURN count(r) AS c").single()
                rels[rel] = int(r["c"]) if r else 0
    finally:
        driver.close()
    return {"nodes": nodes, "rels": rels}
