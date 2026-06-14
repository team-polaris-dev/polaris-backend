"""관리자 라우터 — /api/admin/*

엔드포인트:
  운영 점검
  - GET  /health
  - GET  /connections          의존 서비스(MariaDB·Neo4j·Qdrant·Ollama·apimaker) 연결 점검
  - GET  /db/status            모든 테이블 동적 카운트 (SHOW TABLES 기반)
  회사 조회
  - GET  /corps                보유 회사(raw·DB 기준)
  - GET  /corps/search?q=…     corp_master 전체 검색 (보유/미보유 구분)
  적재 파이프라인
  - GET  /extract/pending      extract 실행 전 미처리 청크 예측치
  - POST /pipeline/jobs        잡 생성 (BackgroundTasks)
  - GET  /pipeline/jobs        잡 목록
  - GET  /pipeline/jobs/{id}
  - GET  /pipeline/jobs/{id}/stream  (SSE)
  - POST /pipeline/jobs/{id}/cancel
  QC (그래프 모순/노이즈 해소)
  - GET  /qc/report            검토 큐 + 회사별 QC 산출물
  - POST /qc/rescan            detect_conflicts 재실행
  - POST /qc/judge             양방향 1건 LLM 방향 판정
  - POST /qc/judge-all         미판정 양방향 백그라운드 일괄 판정
  - POST /qc/judge-all/stop
  - GET  /qc/batch/status
  - POST /qc/apply-all         확정 판정 일괄 적용 (소프트 비활성)
  - POST /qc/resolve           단건 적용 (3종)
  - POST /qc/restore           적용 되돌리기
  - POST /qc/acknowledge       정상 양방향으로 인정
  - GET  /qc/chunk?ids=…       근거 청크 원문 조회
  - GET  /qc/disabled          비활성화된 엣지 목록
  - POST /qc/entity-judge      비회사 끝점 1건 LLM 타입 판정
  - POST /qc/entity-judge-all  미판정 끝점 백그라운드 일괄 판정
  - POST /qc/entity-judge-all/stop
  - GET  /qc/entity-batch/status
  챗봇 통계
  - GET  /analytics/overview
  - GET  /analytics/volume
  - GET  /analytics/intents
  - GET  /analytics/tools
  - GET  /analytics/users
  - GET  /analytics/sessions
"""
from __future__ import annotations

import json
import os
import re
import secrets
import sys
import threading
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
    QcEntityJudgeRequest,
    QcJudgeRequest,
    QcResolveRequest,
)
from services import chat_analytics, pipeline_jobs, pipeline_runner
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


@router.get("/connections")
def connections(_=Depends(verify_admin_token)) -> dict[str, Any]:
    """의존 서비스 연결 점검 (업계 표준 health check 패턴).

    주소는 .env 가 단일 출처 — 여기서는 읽기전용으로 표시만 한다 (UI 에서 주소를
    바꾸는 런타임 설정 변경은 설정 이중화·보안 문제로 의도적으로 제공하지 않음).
    sync def — 블로킹 점검이 이벤트 루프를 막지 않게 스레드풀에서 실행.
    """
    return {
        "services": [
            _check_mariadb(),
            _check_neo4j(),
            _check_qdrant(),
            _check_ollama(),
            _check_apimaker(),
        ],
        "measured_at": datetime.now().isoformat(),
    }


@router.get("/db/status", response_model=DBStatusResponse)
async def db_status(_=Depends(verify_admin_token)) -> DBStatusResponse:
    return DBStatusResponse(
        mariadb=_count_mariadb(),
        qdrant=_count_qdrant(),
        neo4j=_count_neo4j(),
        measured_at=datetime.now(),
    )


@router.get("/corps/search")
def corps_search(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=50),
    _=Depends(verify_admin_token),
) -> list[dict[str, Any]]:
    """전체 상장사(corp_master, DART corpCode 기반)에서 검색 — 보유/미보유 구분 포함.

    has_data=True 면 이미 적재된 회사(document_index 존재), False 면 신규로
    파이프라인을 돌릴 수 있는 회사.
    """
    like = f"%{q.strip()}%"
    with mariadb_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT m.corp_code, m.corp_name, m.stock_code, "
            "       COALESCE(d.doc_count, 0) AS doc_count "
            "FROM corp_master m "
            "LEFT JOIN (SELECT corp_code, COUNT(*) AS doc_count "
            "           FROM document_index GROUP BY corp_code) d "
            "  ON d.corp_code = m.corp_code "
            "WHERE m.corp_name LIKE %s OR m.corp_code LIKE %s OR m.stock_code LIKE %s "
            "ORDER BY (m.corp_name = %s) DESC, CHAR_LENGTH(m.corp_name), m.corp_name "
            "LIMIT %s",
            (like, like, like, q.strip(), limit),
        )
        return [
            {
                "corp_code": r["corp_code"],
                "corp_name": r["corp_name"],
                "stock_code": r["stock_code"],
                "doc_count": int(r["doc_count"]),
                "has_data": int(r["doc_count"]) > 0,
            }
            for r in cur.fetchall()
        ]


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


@router.get("/extract/pending")
def extract_pending(
    corp_codes: str = Query(..., description="콤마구분 corp_code 목록"),
    positive: bool = Query(default=False),
    _=Depends(verify_admin_token),
) -> list[dict[str, Any]]:
    """extract 스텝 사전 정보 — 회사별 대상(eligible)/처리됨(원장)/남음(pending) 청크 수.

    UI 가 "전부 추출"의 규모를 실행 전에 보여주기 위한 조회. WHERE 절과 원장은
    pipeline_scripts/graph 의 정의를 그대로 가져온다 (필터 이중 관리 금지).
    """
    codes = [c.strip() for c in corp_codes.split(",") if c.strip()]
    if not codes:
        return []
    where, done = _extract_filter_and_ledger(positive)
    out: list[dict[str, Any]] = []
    with mariadb_conn() as conn, conn.cursor() as cur:
        for code in codes:
            cur.execute(
                f"SELECT chunk_id FROM chunk_index WHERE corp_code=%s AND {where.replace('%', '%%')}",
                (code,),
            )
            ids = [r["chunk_id"] for r in cur.fetchall()]
            pending = sum(1 for i in ids if i not in done)
            cur.execute(
                "SELECT corp_name FROM document_index WHERE corp_code=%s LIMIT 1", (code,)
            )
            row = cur.fetchone()
            out.append({
                "corp_code": code,
                "corp_name": (row or {}).get("corp_name") or code,
                "eligible": len(ids),
                "done": len(ids) - pending,
                "pending": pending,
            })
    return out


def _extract_filter_and_ledger(positive: bool) -> tuple[str, set[str]]:
    """pipeline_scripts/graph 의 WHERE 절 정의·원장(ledger)을 임포트해 재사용."""
    graph_dir = str(PIPELINE_WORKDIR / "graph")
    if graph_dir not in sys.path:
        sys.path.insert(0, graph_dir)
    import auto_runner  # noqa: PLC0415 — 지연 import (graph 스크립트 트리)
    from extract_prompt import SKIP_WHERE  # noqa: PLC0415

    where = auto_runner.POSITIVE_WHERE if positive else SKIP_WHERE
    return where, auto_runner.ledger_ids()


@router.get("/qc/report")
def qc_report(
    suspects_limit: int = Query(default=200, ge=1, le=1000),
    _=Depends(verify_admin_token),
) -> dict[str, Any]:
    """추출 QC 산출물 뷰어 — qc/extract 스텝이 남긴 파일들을 읽어 반환 (읽기 전용).

    - graph/conflicts_queue.json          : detect_conflicts 검토 큐 (그래프 전역)
    - graph/_auto/qc_<corp>_summary.json  : 회사별 산출물 QC 통계
    - graph/_auto/qc_<corp>_suspects.jsonl: 회사별 의심 항목 (사람 검토용)
    """
    graph_dir = PIPELINE_WORKDIR / "graph"
    auto = graph_dir / "_auto"

    def _mtime(p: Path) -> str:
        return datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")

    conflicts: dict[str, Any] | None = None
    qpath = graph_dir / "conflicts_queue.json"
    if qpath.exists():
        try:
            items = json.loads(qpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            items = []
        conflicts = {
            "generated_at": _mtime(qpath),
            "items": items if isinstance(items, list) else [],
        }

    corps: list[dict[str, Any]] = []
    if auto.exists():
        for spath in sorted(auto.glob("qc_*_summary.json")):
            m = re.match(r"^qc_(\d{8})_summary\.json$", spath.name)
            if not m:
                continue
            try:
                summary = json.loads(spath.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            suspects: list[dict[str, Any]] = []
            sus_path = auto / f"qc_{m.group(1)}_suspects.jsonl"
            if sus_path.exists():
                for line in sus_path.read_text(encoding="utf-8").splitlines():
                    if len(suspects) >= suspects_limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        suspects.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            corps.append({
                "corp_code": m.group(1),
                "generated_at": _mtime(spath),
                "summary": summary,
                "suspects": suspects,
            })

    # 양방향 항목엔 방향 판정, 비회사 pending 항목엔 LLM 엔티티 판정(verdict) 병합
    if conflicts and conflicts["items"]:
        judgments = _load_qc_judgments()
        entity_judgments = _load_entity_judgments()
        for item in conflicts["items"]:
            if item.get("kind") == "bidirectional_supplies":
                j = judgments.get(_pair_key(item.get("a_id"), item.get("b_id")))
                if j:
                    item["judgment"] = j
            elif item.get("kind") == "non_company_supplies" and item.get("decision") == "pending":
                ej = entity_judgments.get(item.get("entity_key"))
                if ej:
                    item["verdict"] = ej.get("verdict")
                    item["verdict_reason"] = ej.get("reason")
                    item["judged_at"] = ej.get("judged_at")

    return {"conflicts": conflicts, "corps": corps,
            "measured_at": datetime.now().isoformat(timespec="seconds")}


@router.post("/qc/rescan")
def qc_rescan(_=Depends(verify_admin_token)) -> dict[str, Any]:
    """detect_conflicts 재실행 (읽기 전용 — conflicts_queue.json 갱신)."""
    graph_dir = str(PIPELINE_WORKDIR / "graph")
    if graph_dir not in sys.path:
        sys.path.insert(0, graph_dir)
    import detect_conflicts  # noqa: PLC0415

    detect_conflicts.main()
    qpath = PIPELINE_WORKDIR / "graph" / "conflicts_queue.json"
    n = -1
    if qpath.exists():
        items = json.loads(qpath.read_text(encoding="utf-8"))
        if isinstance(items, list):
            n = len(items)
    return {"ok": True, "conflicts": n}


def _judge_pair(a: str, b: str, a_id: str, b_id: str,
                fwd_chunk: str | None, rev_chunk: str | None) -> dict[str, Any]:
    """양방향 SUPPLIES_TO 방향 판정 본체 — 단건/일괄 엔드포인트가 공유."""
    texts: dict[str, str] = {}
    chunk_ids = [c for c in (fwd_chunk, rev_chunk) if c]
    if chunk_ids:
        with mariadb_conn() as conn, conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(chunk_ids))
            cur.execute(
                f"SELECT chunk_id, embedding_text FROM chunk_index "
                f"WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            )
            for r in cur.fetchall():
                texts[str(r["chunk_id"])] = str(r["embedding_text"] or "")[:3000]

    fwd_text = texts.get(fwd_chunk or "", "(본문 미확보)")
    rev_text = texts.get(rev_chunk or "", "(본문 미확보)")

    from config.llm import json_llm  # noqa: PLC0415 — 서버의 단일 LLM 경로 재사용
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    system = (
        "너는 한국 공시(DART) 본문에서 기업 간 공급(SUPPLIES_TO) 방향을 판정하는 "
        "엄격한 검수자다. 본문 근거가 명확할 때만 방향을 정하고, 애매하면 "
        "uncertain 으로 답한다. 추측·일반상식 금지 — 제시된 본문만 근거로 쓴다."
    )
    user = (
        f"회사 A = \"{a}\", 회사 B = \"{b}\".\n"
        f"그래프에는 A→B 와 B→A 두 방향이 동시에 존재한다(모순). 올바른 방향을 판정하라.\n\n"
        f"[근거1 — A→B(A가 B에 공급) 방향의 출처 본문]\n{fwd_text}\n\n"
        f"[근거2 — B→A(B가 A에 공급) 방향의 출처 본문]\n{rev_text}\n\n"
        '다음 JSON 하나만 출력: {"direction": "a_to_b" 또는 "b_to_a" 또는 "uncertain", '
        '"reason": "본문 표현을 인용한 한 문장 근거"}'
    )
    raw = json_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    parsed = json.loads(str(raw.content))  # 실패 시 호출자가 처리
    direction = parsed.get("direction")
    if direction not in ("a_to_b", "b_to_a", "uncertain"):
        raise ValueError(f"LLM 방향 값 이상: {direction!r}")

    judgment = {
        "direction": direction,
        "reason": str(parsed.get("reason") or ""),
        "judged_at": datetime.now().isoformat(timespec="seconds"),
        "a": a, "b": b,
    }
    judgments = _load_qc_judgments()
    judgments[_pair_key(a_id, b_id)] = judgment
    _save_qc_judgments(judgments)
    return judgment


@router.post("/qc/judge")
def qc_judge(req: QcJudgeRequest, _=Depends(verify_admin_token)) -> dict[str, Any]:
    """단건 방향 판정. 적용(삭제)은 사람이 confirm 후 /qc/resolve."""
    try:
        return _judge_pair(req.a, req.b, req.a_id, req.b_id, req.fwd_chunk, req.rev_chunk)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"LLM 판정 실패: {e}") from e


# ── QC 일괄 판정 — 서버 백그라운드 스레드 (브라우저/페이지와 무관하게 진행) ──
_QC_BATCH: dict[str, Any] = {
    "running": False, "done": 0, "total": 0, "errors": 0,
    "stop_requested": False, "started_at": None, "finished_at": None,
}
_QC_BATCH_LOCK = threading.Lock()


def _qc_batch_worker(items: list[dict[str, Any]]) -> None:
    for item in items:
        if _QC_BATCH["stop_requested"]:
            break
        try:
            _judge_pair(
                str(item.get("a") or ""), str(item.get("b") or ""),
                str(item.get("a_id") or ""), str(item.get("b_id") or ""),
                item.get("fwd_chunk"), item.get("rev_chunk"),
            )
        except Exception:  # noqa: BLE001 — 한 건 실패해도 계속 (미판정으로 남음)
            _QC_BATCH["errors"] += 1
        _QC_BATCH["done"] += 1
    _QC_BATCH["running"] = False
    _QC_BATCH["finished_at"] = datetime.now().isoformat(timespec="seconds")


@router.post("/qc/judge-all")
def qc_judge_all(_=Depends(verify_admin_token)) -> dict[str, Any]:
    """미판정 양방향 전부를 백그라운드에서 순차 판정. 진행은 /qc/batch/status."""
    with _QC_BATCH_LOCK:
        if _QC_BATCH["running"]:
            return {"ok": False, "reason": "already_running", **_qc_batch_status()}

        qpath = PIPELINE_WORKDIR / "graph" / "conflicts_queue.json"
        items: list[dict[str, Any]] = []
        if qpath.exists():
            try:
                items = json.loads(qpath.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                items = []
        judgments = _load_qc_judgments()
        targets = [
            it for it in items
            if it.get("kind") == "bidirectional_supplies"
            and _pair_key(it.get("a_id"), it.get("b_id")) not in judgments
        ]
        _QC_BATCH.update(running=True, done=0, total=len(targets), errors=0,
                         stop_requested=False, finished_at=None,
                         started_at=datetime.now().isoformat(timespec="seconds"))
        if targets:
            threading.Thread(target=_qc_batch_worker, args=(targets,),
                             daemon=True, name="qc-judge-all").start()
        else:
            _QC_BATCH["running"] = False
    return {"ok": True, **_qc_batch_status()}


@router.post("/qc/judge-all/stop")
def qc_judge_all_stop(_=Depends(verify_admin_token)) -> dict[str, Any]:
    _QC_BATCH["stop_requested"] = True
    return {"ok": True, **_qc_batch_status()}


@router.get("/qc/batch/status")
def qc_batch_status(_=Depends(verify_admin_token)) -> dict[str, Any]:
    return _qc_batch_status()


def _qc_batch_status() -> dict[str, Any]:
    return {k: _QC_BATCH[k] for k in
            ("running", "done", "total", "errors", "stop_requested",
             "started_at", "finished_at")}


@router.post("/qc/apply-all")
def qc_apply_all(_=Depends(verify_admin_token)) -> dict[str, Any]:
    """판정 확정(uncertain 제외)된 양방향 전부 일괄 적용 — 잘못된 방향 삭제 후 재검사.

    동기 처리(수 초) — 클라이언트가 떠나도 스레드풀에서 끝까지 실행된다.
    """
    qpath = PIPELINE_WORKDIR / "graph" / "conflicts_queue.json"
    items: list[dict[str, Any]] = []
    if qpath.exists():
        try:
            items = json.loads(qpath.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            items = []
    judgments = _load_qc_judgments()

    applied = 0
    disabled_total = 0
    for it in items:
        if it.get("kind") != "bidirectional_supplies":
            continue
        j = judgments.get(_pair_key(it.get("a_id"), it.get("b_id")))
        if not j or j.get("direction") not in ("a_to_b", "b_to_a"):
            continue
        if j["direction"] == "a_to_b":   # A→B 유지 → B→A 비활성화
            del_from, del_to = str(it.get("b_id")), str(it.get("a_id"))
        else:                            # B→A 유지 → A→B 비활성화
            del_from, del_to = str(it.get("a_id")), str(it.get("b_id"))
        n = _disable_supplies_edge(del_from, del_to)
        disabled_total += n
        applied += 1
        _append_qc_audit({"kind": "bidirectional_supplies", "action": "disable",
                          "affected": n, "via": "apply_all",
                          "params": {"from_id": del_from, "to_id": del_to},
                          "judgment": j})

    # 일괄 적용 후 재검사 1회로 큐 갱신
    graph_dir = str(PIPELINE_WORKDIR / "graph")
    if graph_dir not in sys.path:
        sys.path.insert(0, graph_dir)
    import detect_conflicts  # noqa: PLC0415

    detect_conflicts.main()
    return {"ok": True, "applied": applied, "disabled": disabled_total}


def _neo4j_driver():
    from neo4j import GraphDatabase  # noqa: PLC0415

    return GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
    )


# ── 소프트 삭제 정책 ──────────────────────────────────────────────────
# QC 적용은 엣지를 DELETE 하지 않고 r.qc_disabled_at 타임스탬프를 찍어 "비활성화"
# 한다. 읽기 경로(supplies_pair·supply_chain_2hop·detect_conflicts·anchor_chunks)는
# qc_disabled_at IS NULL 만 본다. 따라서 모든 적용은 /qc/restore 로 100% 되돌릴 수
# 있고, chunk_id·rcept_no 등 근거 속성도 보존된다. (DELETE 였다면 복구 불가)
def _disable_supplies_edge(from_id: str, to_id: str, reason: str = "qc") -> int:
    driver = _neo4j_driver()
    try:
        with driver.session() as s:
            res = s.run(
                "MATCH (x:Organization)-[r:SUPPLIES_TO]->(y:Organization) "
                "WHERE coalesce(x.corp_code, x.er_name) = $f "
                "  AND coalesce(y.corp_code, y.er_name) = $t "
                "  AND r.qc_disabled_at IS NULL "
                "SET r.qc_disabled_at = datetime(), r.qc_disabled_reason = $reason "
                "RETURN count(r) AS n",
                f=from_id, t=to_id, reason=reason,
            )
            return int(res.single()["n"])
    finally:
        driver.close()


def _restore_supplies_edge(from_id: str, to_id: str) -> int:
    driver = _neo4j_driver()
    try:
        with driver.session() as s:
            res = s.run(
                "MATCH (x:Organization)-[r:SUPPLIES_TO]->(y:Organization) "
                "WHERE coalesce(x.corp_code, x.er_name) = $f "
                "  AND coalesce(y.corp_code, y.er_name) = $t "
                "  AND r.qc_disabled_at IS NOT NULL "
                "REMOVE r.qc_disabled_at, r.qc_disabled_reason "
                "RETURN count(r) AS n",
                f=from_id, t=to_id,
            )
            return int(res.single()["n"])
    finally:
        driver.close()


def _append_qc_audit(record: dict[str, Any]) -> None:
    audit = PIPELINE_WORKDIR / "graph" / "qc_resolved.jsonl"
    with audit.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(
            {"ts": datetime.now().isoformat(timespec="seconds"), **record},
            ensure_ascii=False) + "\n")


@router.post("/qc/resolve")
def qc_resolve(req: QcResolveRequest, _=Depends(verify_admin_token)) -> dict[str, Any]:
    """사람이 [적용]을 누른 모순 1건 해소 — 엣지 비활성화(되돌리기 가능) + 감사 로그.

    bidirectional_supplies: (from_id → to_id) 방향의 SUPPLIES_TO 비활성화.
    non_company_supplies: (from_id → to_id) 비회사 노이즈 엣지 1개 비활성화.
    self_loop: org 자기참조 관계 비활성화 (rel·chunk_id 일치 조건).
    """
    if req.kind in ("bidirectional_supplies", "non_company_supplies"):
        if not req.from_id or not req.to_id:
            raise HTTPException(status_code=422, detail="from_id/to_id 필요")
        affected = _disable_supplies_edge(req.from_id, req.to_id, reason=req.kind)
    else:  # self_loop
        if not req.org or not req.rel:
            raise HTTPException(status_code=422, detail="org/rel 필요")
        driver = _neo4j_driver()
        try:
            with driver.session() as s:
                res = s.run(
                    "MATCH (a:Organization)-[r]->(a) "
                    "WHERE a.name = $org AND type(r) = $rel "
                    "  AND ($cid IS NULL OR r.chunk_id = $cid) "
                    "  AND r.qc_disabled_at IS NULL "
                    "SET r.qc_disabled_at = datetime(), r.qc_disabled_reason = 'qc' "
                    "RETURN count(r) AS n",
                    org=req.org, rel=req.rel, cid=req.chunk_id,
                )
                affected = int(res.single()["n"])
        finally:
            driver.close()

    _append_qc_audit({"kind": req.kind, "action": "disable", "affected": affected,
                      "params": req.model_dump(exclude_none=True)})
    return {"ok": True, "disabled": affected}


@router.post("/qc/restore")
def qc_restore(req: QcResolveRequest, _=Depends(verify_admin_token)) -> dict[str, Any]:
    """비활성화한 엣지를 되살린다 (QC 적용 되돌리기). SUPPLIES_TO 방향 엣지 전용."""
    if req.kind not in ("bidirectional_supplies", "non_company_supplies") \
            or not req.from_id or not req.to_id:
        raise HTTPException(status_code=422, detail="from_id/to_id 필요 (SUPPLIES_TO)")
    affected = _restore_supplies_edge(req.from_id, req.to_id)
    _append_qc_audit({"kind": req.kind, "action": "restore", "affected": affected,
                      "params": req.model_dump(exclude_none=True)})
    return {"ok": True, "restored": affected}


_QC_ACK_PATH = PIPELINE_WORKDIR / "graph" / "qc_acknowledged.json"


@router.post("/qc/acknowledge")
def qc_acknowledge(req: QcJudgeRequest, _=Depends(verify_admin_token)) -> dict[str, Any]:
    """양방향 SUPPLIES_TO 를 '정상(실제 양방향 거래)'으로 인정 — 모순 목록에서 제외.

    detect_conflicts 가 이 목록(qc_acknowledged.json)의 쌍은 양방향이어도 충돌로
    보고하지 않는다. 엣지는 양쪽 다 활성 유지(삭제·비활성 아님)."""
    ack = _load_json_dict(_QC_ACK_PATH)
    key = _pair_key(req.a_id, req.b_id)
    rkey = _pair_key(req.b_id, req.a_id)  # 방향 무관하게 인식되도록 양쪽 키 저장
    entry = {"a": req.a, "b": req.b, "acknowledged_at": datetime.now().isoformat(timespec="seconds")}
    ack[key] = entry
    ack[rkey] = entry
    _QC_ACK_PATH.write_text(json.dumps(ack, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_qc_audit({"kind": "bidirectional_supplies", "action": "acknowledge",
                      "params": {"a_id": req.a_id, "b_id": req.b_id}})
    return {"ok": True, "acknowledged": key}


def _load_json_dict(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


@router.get("/qc/chunk")
def qc_chunk(
    ids: str = Query(..., description="콤마구분 chunk_id 목록"),
    _=Depends(verify_admin_token),
) -> list[dict[str, Any]]:
    """청크 원문 조회 — 방향 판정 전 사람이 근거 본문을 직접 확인하기 위한 것.

    chunk_index(본문) + document_index(회사·제목) 조인. 입력 순서를 보존해 반환.
    """
    chunk_ids = [c.strip() for c in ids.split(",") if c.strip()]
    if not chunk_ids:
        return []
    placeholders = ",".join(["%s"] * len(chunk_ids))
    with mariadb_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT c.chunk_id, c.corp_code, c.section_path, c.embedding_text, "
            f"       d.corp_name, d.title "
            f"FROM chunk_index c "
            f"LEFT JOIN document_index d ON d.rcept_no = c.rcept_no "
            f"WHERE c.chunk_id IN ({placeholders})",
            chunk_ids,
        )
        by_id = {str(r["chunk_id"]): r for r in cur.fetchall()}
    out: list[dict[str, Any]] = []
    for cid in chunk_ids:
        r = by_id.get(cid)
        if r is None:
            out.append({"chunk_id": cid, "found": False})
            continue
        out.append({
            "chunk_id": cid, "found": True,
            "corp_name": r.get("corp_name") or r.get("corp_code") or "",
            "title": r.get("title") or "",
            "section_path": r.get("section_path") or "",
            "text": str(r.get("embedding_text") or ""),
        })
    return out


@router.get("/qc/disabled")
def qc_disabled(_=Depends(verify_admin_token)) -> list[dict[str, Any]]:
    """QC 로 비활성화된 SUPPLIES_TO 엣지 목록 (되돌리기 이력 뷰)."""
    driver = _neo4j_driver()
    try:
        with driver.session() as s:
            rows = s.run(
                "MATCH (x:Organization)-[r:SUPPLIES_TO]->(y:Organization) "
                "WHERE r.qc_disabled_at IS NOT NULL "
                "RETURN x.name AS from_name, y.name AS to_name, "
                "       coalesce(x.corp_code, x.er_name) AS from_id, "
                "       coalesce(y.corp_code, y.er_name) AS to_id, "
                "       toString(r.qc_disabled_at) AS disabled_at, "
                "       r.qc_disabled_reason AS reason, r.chunk_id AS chunk_id "
                "ORDER BY r.qc_disabled_at DESC"
            ).data()
    finally:
        driver.close()
    return rows


_QC_JUDGMENTS_PATH = PIPELINE_WORKDIR / "graph" / "qc_judgments.json"


def _pair_key(a_id: Any, b_id: Any) -> str:
    return f"{a_id}|{b_id}"


def _load_qc_judgments() -> dict[str, Any]:
    if _QC_JUDGMENTS_PATH.exists():
        try:
            data = json.loads(_QC_JUDGMENTS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {}


def _save_qc_judgments(data: dict[str, Any]) -> None:
    _QC_JUDGMENTS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── 비회사 엔티티 LLM 판정 (휴리스틱 폐기분 대체) ──────────────────────
# detect_conflicts 가 decision='pending' 으로 넘긴 미해소 끝점을, in-process
# apimaker(Gemini)가 원문 근거로 회사/국가지역/제품/일반어/인물 분류한다. 결과는
# entity_key(정규화 이름)별로 캐시 — 같은 엔티티는 1회만 호출. /qc/report 가 병합.
_QC_ENTITY_JUDGMENTS_PATH = PIPELINE_WORKDIR / "graph" / "qc_entity_judgments.json"
_VALID_VERDICTS = {"company", "country_region", "product", "generic", "person", "uncertain"}


def _load_entity_judgments() -> dict[str, Any]:
    return _load_json_dict(_QC_ENTITY_JUDGMENTS_PATH)


def _save_entity_judgments(data: dict[str, Any]) -> None:
    _QC_ENTITY_JUDGMENTS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _judge_entity(entity_key: str, name: str, chunk_id: str | None) -> dict[str, Any]:
    """미해소 끝점 1개 타입 판정 — 외국기업·자회사도 company. 결과 캐시 후 반환."""
    text = ""
    if chunk_id:
        with mariadb_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT embedding_text FROM chunk_index WHERE chunk_id=%s", (chunk_id,)
            )
            row = cur.fetchone()
            if row:
                text = str(row["embedding_text"] or "")[:3000]

    from config.llm import json_llm  # noqa: PLC0415 — 서버 단일 LLM 경로 재사용
    from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415

    system = (
        "너는 한국 공시(DART)에서 추출된 엔티티가 SUPPLIES_TO(기업간 공급) 관계의 "
        "끝점으로 적합한 '회사(법인)'인지, 아니면 국가/지역·제품/품목·일반업계어·인물 "
        "중 무엇인지 분류하는 엄격한 검수자다. 외국 기업·해외 자회사도 회사(company)다. "
        "본문 맥락만 근거로 판단하고, 단정하기 어려우면 uncertain 으로 답한다."
    )
    user = (
        f'분류할 엔티티: "{name}"\n\n'
        f"[이 엔티티가 등장한 본문]\n{text or '(본문 미확보)'}\n\n"
        '다음 JSON 하나만 출력: {"verdict": '
        '"company"|"country_region"|"product"|"generic"|"person"|"uncertain", '
        '"reason": "본문 표현을 인용한 한 문장 근거"}'
    )
    raw = json_llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    parsed = json.loads(str(raw.content))  # 실패 시 호출자가 처리
    verdict = parsed.get("verdict")
    if verdict not in _VALID_VERDICTS:
        raise ValueError(f"LLM verdict 이상: {verdict!r}")

    judgment = {
        "verdict": verdict,
        "reason": str(parsed.get("reason") or ""),
        "judged_at": datetime.now().isoformat(timespec="seconds"),
        "name": name,
    }
    data = _load_entity_judgments()
    data[entity_key] = judgment
    _save_entity_judgments(data)
    return judgment


@router.post("/qc/entity-judge")
def qc_entity_judge(req: QcEntityJudgeRequest, _=Depends(verify_admin_token)) -> dict[str, Any]:
    """단건 엔티티 타입 판정 (회사/비회사). 적용(비활성화)은 사람이 confirm 후 /qc/resolve."""
    try:
        return _judge_entity(req.entity_key, req.name, req.chunk_id)
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=502, detail=f"LLM 판정 실패: {e}") from e


_QC_ENTITY_BATCH: dict[str, Any] = {
    "running": False, "done": 0, "total": 0, "errors": 0,
    "stop_requested": False, "started_at": None, "finished_at": None,
}
_QC_ENTITY_BATCH_LOCK = threading.Lock()


def _qc_entity_batch_status() -> dict[str, Any]:
    return {k: _QC_ENTITY_BATCH[k] for k in
            ("running", "done", "total", "errors", "stop_requested",
             "started_at", "finished_at")}


def _qc_entity_batch_worker(items: list[dict[str, Any]]) -> None:
    for it in items:
        if _QC_ENTITY_BATCH["stop_requested"]:
            break
        try:
            _judge_entity(str(it.get("entity_key") or ""),
                          str(it.get("entity_name") or ""), it.get("sample_chunk"))
        except Exception:  # noqa: BLE001 — 한 건 실패해도 계속 (미판정 유지)
            _QC_ENTITY_BATCH["errors"] += 1
        _QC_ENTITY_BATCH["done"] += 1
    _QC_ENTITY_BATCH["running"] = False
    _QC_ENTITY_BATCH["finished_at"] = datetime.now().isoformat(timespec="seconds")


@router.post("/qc/entity-judge-all")
def qc_entity_judge_all(_=Depends(verify_admin_token)) -> dict[str, Any]:
    """판정 대기(decision='pending') 비회사 후보 엔티티 전부를 백그라운드 LLM 판정.

    이미 판정된(entity_key 캐시) 건은 건너뛴다. 진행은 /qc/entity-batch/status.
    """
    with _QC_ENTITY_BATCH_LOCK:
        if _QC_ENTITY_BATCH["running"]:
            return {"ok": False, "reason": "already_running", **_qc_entity_batch_status()}

        qpath = PIPELINE_WORKDIR / "graph" / "conflicts_queue.json"
        items: list[dict[str, Any]] = []
        if qpath.exists():
            try:
                items = json.loads(qpath.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                items = []
        judged = _load_entity_judgments()
        targets = [
            it for it in items
            if it.get("kind") == "non_company_supplies"
            and it.get("decision") == "pending"
            and it.get("entity_key") not in judged
        ]
        _QC_ENTITY_BATCH.update(running=True, done=0, total=len(targets), errors=0,
                                stop_requested=False, finished_at=None,
                                started_at=datetime.now().isoformat(timespec="seconds"))
        if targets:
            threading.Thread(target=_qc_entity_batch_worker, args=(targets,),
                             daemon=True, name="qc-entity-judge-all").start()
        else:
            _QC_ENTITY_BATCH["running"] = False
    return {"ok": True, **_qc_entity_batch_status()}


@router.post("/qc/entity-judge-all/stop")
def qc_entity_judge_all_stop(_=Depends(verify_admin_token)) -> dict[str, Any]:
    _QC_ENTITY_BATCH["stop_requested"] = True
    return {"ok": True, **_qc_entity_batch_status()}


@router.get("/qc/entity-batch/status")
def qc_entity_batch_status(_=Depends(verify_admin_token)) -> dict[str, Any]:
    return _qc_entity_batch_status()


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


# ===== 챗봇 통계 (analytics) =====

@router.get("/analytics/overview")
async def analytics_overview(_=Depends(verify_admin_token)) -> dict:
    return chat_analytics.overview()


@router.get("/analytics/volume")
async def analytics_volume(
    days: int = Query(default=30, ge=1, le=180),
    _=Depends(verify_admin_token),
) -> list[dict]:
    return chat_analytics.volume_series(days=days)


@router.get("/analytics/intents")
async def analytics_intents(
    limit: int = Query(default=12, ge=1, le=50),
    _=Depends(verify_admin_token),
) -> list[dict]:
    return chat_analytics.intent_distribution(limit=limit)


@router.get("/analytics/tools")
async def analytics_tools(_=Depends(verify_admin_token)) -> list[dict]:
    return chat_analytics.tool_usage()


@router.get("/analytics/users")
async def analytics_users(
    limit: int = Query(default=10, ge=1, le=100),
    _=Depends(verify_admin_token),
) -> list[dict]:
    return chat_analytics.top_users(limit=limit)


@router.get("/analytics/sessions")
async def analytics_sessions(
    limit: int = Query(default=15, ge=1, le=100),
    _=Depends(verify_admin_token),
) -> list[dict]:
    return chat_analytics.recent_sessions(limit=limit)


# ----- 연결 점검 헬퍼 -----

def _timed_check(name: str, address: str, fn) -> dict[str, Any]:
    """fn() 실행 → {name, address, ok, latency_ms, detail}. 예외는 ok=False 로 흡수."""
    import time as _time

    t0 = _time.perf_counter()
    try:
        detail = fn() or "ok"
        ok = True
    except Exception as e:  # noqa: BLE001 — 점검 실패는 상태로 보고
        detail = f"{type(e).__name__}: {e}"
        ok = False
    return {
        "name": name,
        "address": address,
        "ok": ok,
        "latency_ms": round((_time.perf_counter() - t0) * 1000.0, 1),
        "detail": str(detail)[:300],
    }


def _check_mariadb() -> dict[str, Any]:
    host = os.getenv("MARIADB_HOST", "localhost")
    port = os.getenv("MARIADB_PORT") or "3307"
    db = os.getenv("MARIADB_DATABASE", "polaris")

    def fn():
        with mariadb_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            cur.fetchone()
        return f"database={db}"

    return _timed_check("MariaDB", f"{host}:{port}/{db}", fn)


def _check_neo4j() -> dict[str, Any]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")

    def fn():
        from neo4j import GraphDatabase

        d = GraphDatabase.driver(
            uri, auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
        )
        try:
            d.verify_connectivity()
        finally:
            d.close()
        return "verify_connectivity ok"

    return _timed_check("Neo4j", uri, fn)


def _check_qdrant() -> dict[str, Any]:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = os.getenv("QDRANT_PORT", "6333")
    address = f"http://{host}:{port}"

    def fn():
        import httpx

        r = httpx.get(f"{address}/", timeout=3.0)
        r.raise_for_status()
        version = (r.json() or {}).get("version", "?")
        return f"server v{version}"

    return _timed_check("Qdrant", address, fn)


def _check_ollama() -> dict[str, Any]:
    base = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")

    def fn():
        import httpx

        ver = httpx.get(f"{base}/api/version", timeout=3.0)
        ver.raise_for_status()
        tags = httpx.get(f"{base}/api/tags", timeout=3.0)
        tags.raise_for_status()
        models = [m.get("name", "") for m in (tags.json() or {}).get("models", [])]
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:latest")
        has_embed = any(m.startswith(embed_model.split(":")[0]) for m in models)
        return (
            f"v{(ver.json() or {}).get('version', '?')} · 모델 {len(models)}개"
            f" · 임베딩({embed_model}) {'있음' if has_embed else '없음!'}"
        )

    return _timed_check("Ollama", base, fn)


def _check_apimaker() -> dict[str, Any]:
    """챗봇/추출용 in-process LLM(Gemini CLI) 가용성 — 쿼터 아끼려 실호출은 안 한다."""
    def fn():
        from config.llm import APIMAKER_PROVIDER, _resolve_gemini_launcher

        if APIMAKER_PROVIDER != "gemini":
            return f"provider={APIMAKER_PROVIDER} (런처 점검은 gemini 전용)"
        launcher = _resolve_gemini_launcher()
        if not launcher:
            raise RuntimeError("gemini CLI 런처 해석 실패 — @google/gemini-cli 설치/PATH 확인")
        return f"gemini CLI ok ({Path(launcher[1]).name})"

    return _timed_check("LLM (apimaker/Gemini)", "in-process", fn)


# ----- 내부 헬퍼 -----

def _resolve_corp_names(corp_codes: list[str]) -> dict[str, str]:
    if not corp_codes:
        return {}
    placeholders = ",".join(["%s"] * len(corp_codes))
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            f"SELECT DISTINCT corp_code, corp_name FROM document_index "
            f"WHERE corp_code IN ({placeholders})",
            corp_codes,
        )
        mapping = {r["corp_code"]: (r["corp_name"] or r["corp_code"]) for r in cur.fetchall()}
        # 미보유(신규) 회사는 corp_master 에서 이름 해석 — raw/{회사명} 폴더·청킹에 필요
        missing = [c for c in corp_codes if c not in mapping]
        if missing:
            ph2 = ",".join(["%s"] * len(missing))
            cur.execute(
                f"SELECT corp_code, corp_name FROM corp_master WHERE corp_code IN ({ph2})",
                missing,
            )
            for r in cur.fetchall():
                mapping[r["corp_code"]] = r["corp_name"]
    for code in corp_codes:
        mapping.setdefault(code, code)
    return mapping


def _count_mariadb() -> dict[str, int]:
    """DB 의 모든 테이블을 동적으로 집계 — 하드코딩 목록은 새 테이블(chat_* 등)이
    생길 때마다 누락됐던 전적이 있어 SHOW TABLES 기반으로 전환 (2026-06-13)."""
    out: dict[str, int] = {}
    # mariadb_conn() 은 DictCursor 를 반환하므로 COUNT 에 별칭을 붙여 dict 키로 읽는다.
    with mariadb_conn() as conn, conn.cursor() as cur:
        cur.execute("SHOW TABLES")
        tables = sorted(str(next(iter(r.values()))) for r in cur.fetchall())
        for t in tables:
            try:
                cur.execute(f"SELECT COUNT(*) AS c FROM `{t}`")
                row = cur.fetchone()
                out[t] = int(row["c"]) if row else 0
            except Exception:
                out[t] = -1  # 권한/락 등 집계 실패
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
