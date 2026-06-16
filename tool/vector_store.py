"""Vector 하이브리드 검색 — dump 스키마(MariaDB chunk_index/document_index +
Qdrant `polaris-chunks`) 기준 self-contained 구현.

Dense(Qdrant) + BM25(rank_bm25) + RRF(k=60).
모듈 캐시는 프로세스 단위로 1회만 적재한다 — 캐시 무효화는 프로세스 재시작.

The node owns user-facing graceful error formatting. This tool either returns
normalized rows or raises the underlying exception.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from tool.rdb_client import mariadb_conn

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

RRF_K = 60
DENSE_FETCH_MIN = 30
DENSE_FETCH_MULTIPLIER = 5
MAX_CHUNKS_PER_DOC = 3  # 결과 다양화: 한 공시(rcept_no)에서 가져올 최대 청크 수

_TOKEN_RE = re.compile(r"[가-힣a-zA-Z0-9]+")
_YEAR_RE = re.compile(r"(20\d{2})\s*년?")
_COMPARISON_RE = re.compile(r"비교|대비|\bvs\b|VS|둘\s*중|차이")
# 제목의 회계기간 '(YYYY.MM)' — 연도 필터를 접수연도가 아닌 회계연도 기준으로 잡기 위함
_TITLE_YEAR_RE = re.compile(r"\((20\d{2})\.\d{1,2}")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    corp_code: str
    corp_name: str
    rcept_no: str
    year: int | None
    doc_type: str
    title: str
    section_path: str
    chunk_type: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "corp_code": self.corp_code,
            "corp_name": self.corp_name,
            "rcept_no": self.rcept_no,
            "year": self.year,
            "doc_type": self.doc_type,
            "title": self.title,
            "section_path": self.section_path,
            "chunk_type": self.chunk_type,
            "score": self.score,
        }


@dataclass
class _ChunkRow:
    chunk_id: str
    corp_code: str
    corp_name: str
    rcept_no: str
    year: int | None
    doc_type: str
    title: str
    section_path: str
    chunk_type: str
    text: str


@dataclass
class _ChunkIndex:
    rows: list[_ChunkRow]
    by_chunk_id: dict[str, _ChunkRow]
    tokenized: list[list[str]]


# ---------------------------------------------------------------- 프로세스 캐시
_CHUNK_INDEX_CACHE: _ChunkIndex | None = None
_CORP_NAME_TO_CODE_CACHE: dict[str, str] | None = None
_BM25_CACHE: Any | None = None
_QDRANT_CLIENT: Any | None = None
_LOCK = threading.Lock()


def chunk_id_to_uuid(chunk_id: str) -> str:
    """chunk_id(16자리 hex) → Qdrant point UUID (md5 기반, 결정론)."""
    digest = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:]}"


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text or "")


def _resolve_year(title: str, rcept_no: str) -> int | None:
    """연도 필터용 회계연도 결정 — 제목의 '(YYYY.MM)' 회계기간을 우선 사용하고
    없으면 접수연도(rcept_no[:4])로 폴백한다.

    사업보고서는 다음 해에 접수되므로(예: 2024 사업보고서 → 2025-03 접수)
    접수연도로 필터하면 '2024' 질의에서 정답이 누락된다.
    """
    m = _TITLE_YEAR_RE.search(title)
    if m:
        return int(m.group(1))
    return int(rcept_no[:4]) if rcept_no[:4].isdigit() else None


# ---------------------------------------------------------------- chunk index 적재
def _load_chunk_index() -> _ChunkIndex:
    global _CHUNK_INDEX_CACHE
    if _CHUNK_INDEX_CACHE is not None:
        return _CHUNK_INDEX_CACHE
    with _LOCK:
        if _CHUNK_INDEX_CACHE is not None:
            return _CHUNK_INDEX_CACHE

        sql = """
            SELECT
                ci.chunk_id, ci.corp_code, ci.rcept_no, ci.chunk_type,
                ci.section_path, ci.embedding_text,
                di.corp_name, di.doc_type, di.title
            FROM chunk_index ci
            LEFT JOIN document_index di ON di.rcept_no = ci.rcept_no
            WHERE ci.ingest_status = 'ready' AND ci.embedding_text IS NOT NULL
        """
        with mariadb_conn() as conn, conn.cursor() as cur:
            cur.execute(sql)
            db_rows = cur.fetchall()

        rows: list[_ChunkRow] = []
        by_chunk_id: dict[str, _ChunkRow] = {}
        tokenized: list[list[str]] = []
        for r in db_rows:
            chunk_id = str(r["chunk_id"])
            rcept_no = str(r.get("rcept_no") or "")
            title = str(r.get("title") or "")
            year = _resolve_year(title, rcept_no)
            text = str(r.get("embedding_text") or "")
            row = _ChunkRow(
                chunk_id=chunk_id,
                corp_code=str(r.get("corp_code") or "").zfill(8),
                corp_name=str(r.get("corp_name") or ""),
                rcept_no=rcept_no,
                year=year,
                doc_type=str(r.get("doc_type") or ""),
                title=title,
                section_path=str(r.get("section_path") or ""),
                chunk_type=str(r.get("chunk_type") or ""),
                text=text,
            )
            rows.append(row)
            by_chunk_id[chunk_id] = row
            tokenized.append(_tokenize(text))

        _CHUNK_INDEX_CACHE = _ChunkIndex(
            rows=rows,
            by_chunk_id=by_chunk_id,
            tokenized=tokenized,
        )
        return _CHUNK_INDEX_CACHE


def _get_corp_name_to_code() -> dict[str, str]:
    global _CORP_NAME_TO_CODE_CACHE
    if _CORP_NAME_TO_CODE_CACHE is not None:
        return _CORP_NAME_TO_CODE_CACHE
    mapping: dict[str, str] = {}
    for row in _load_chunk_index().rows:
        if row.corp_name and row.corp_code:
            mapping.setdefault(row.corp_name, row.corp_code)
    _CORP_NAME_TO_CODE_CACHE = mapping
    return mapping


# ---------------------------------------------------------------- 질의 → 필터 추출
def _dedup(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _select_owner_company(query: str, mentioned: list[str]) -> str | None:
    """"A의/A에서/A 사업보고서" 패턴의 "문서 소유" 회사를 찾는다."""
    for name in mentioned:
        idx = query.find(name)
        if idx == -1:
            continue
        rest = query[idx + len(name):]
        if rest.startswith(("의", "에서")):
            return name
        if f"{name} 사업보고서" in query or f"{name} 보고서" in query:
            return name
    return None


def extract_filter_signals(query: str) -> tuple[list[str], int | None]:
    """질문에서 corp_code 필터 후보와 연도를 추출한다.

    여러 회사가 언급된 경우, 비교형 질문("비교/대비/vs/둘 중")은 언급된 회사를
    모두 필터로 쓰고, 그 외에는 "문서 소유" 회사("A의/A에서/A 사업보고서") 1개만
    필터로 쓴다(나머지는 단순 비교 대상으로 보고 필터에서 제외).
    """
    name_to_code = _get_corp_name_to_code()
    candidates = sorted(
        (name for name in name_to_code if name and name in query), key=len, reverse=True
    )
    # 긴 이름 우선("SK하이닉스" > "SK") — 이미 채택된 이름의 부분문자열인 후보는 제외
    mentioned: list[str] = []
    for name in candidates:
        if any(name in kept for kept in mentioned):
            continue
        mentioned.append(name)

    year_match = _YEAR_RE.search(query)
    year = int(year_match.group(1)) if year_match else None

    if not mentioned:
        return [], year

    if len(mentioned) == 1 or _COMPARISON_RE.search(query):
        return _dedup([name_to_code[name] for name in mentioned]), year

    owner = _select_owner_company(query, mentioned)
    if owner:
        return [name_to_code[owner]], year
    return _dedup([name_to_code[name] for name in mentioned]), year


# ---------------------------------------------------------------- 임베딩 (Ollama)
def _embed(text: str) -> list[float]:
    import httpx

    base = os.getenv("OLLAMA_BASE", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:latest")

    try:
        resp = httpx.post(f"{base}/api/embed", json={"model": model, "input": text}, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        if data.get("embeddings"):
            return data["embeddings"][0]
        if data.get("embedding"):
            return data["embedding"]
    except httpx.HTTPError:
        pass

    # 구버전 Ollama 폴백
    resp = httpx.post(f"{base}/api/embeddings", json={"model": model, "prompt": text}, timeout=60.0)
    resp.raise_for_status()
    return resp.json()["embedding"]


# ---------------------------------------------------------------- Dense (Qdrant)
def _qdrant_client():
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is None:
        from qdrant_client import QdrantClient

        _QDRANT_CLIENT = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
    return _QDRANT_CLIENT


def _dense_search(vector: list[float], limit: int) -> list[tuple[str, float]]:
    """Qdrant 최근접 검색 → (chunk_id, score).

    적재된 `polaris-chunks` 컬렉션은 포인트 ID 가 정수이고 chunk_id 는 payload 에
    들어있다(md5-UUID 포인트 ID 가 아님). payload['chunk_id'] 로 직접 복원한다.
    """
    client = _qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", "polaris-chunks")

    result = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        with_payload=["chunk_id"],
        with_vectors=False,
    )
    out: list[tuple[str, float]] = []
    for point in result.points:
        chunk_id = (point.payload or {}).get("chunk_id")
        if chunk_id is None:
            continue
        out.append((str(chunk_id), float(point.score)))
    return out


# ---------------------------------------------------------------- Sparse (BM25)
def _get_bm25():
    global _BM25_CACHE
    if _BM25_CACHE is None:
        with _LOCK:
            if _BM25_CACHE is None:
                from rank_bm25 import BM25Okapi

                _BM25_CACHE = BM25Okapi(_load_chunk_index().tokenized)
    return _BM25_CACHE


def _bm25_search(query: str, limit: int) -> list[tuple[str, float]]:
    """BM25 top-N. 전체(~17만) 인덱스에 대한 파이썬 람다 정렬을 numpy argsort 로 교체.

    기존 `sorted(range(n), key=lambda i: scores[i], reverse=True)` 는 17만 회
    파이썬 람다 호출로 느렸다. `np.argsort(-scores, kind='stable')` 는 C 레벨
    정렬이라 훨씬 빠르고, 점수 내림차순·동점 시 인덱스 오름차순으로 **기존 결과
    순서를 그대로 보존**한다(동점 경계 포함).
    """
    index = _load_chunk_index()
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = np.asarray(_get_bm25().get_scores(tokens), dtype=float)
    if scores.shape[0] == 0:
        return []
    order = np.argsort(-scores, kind="stable")  # 점수 내림차순, 동점은 인덱스 오름차순
    out: list[tuple[str, float]] = []
    for i in order[:limit]:
        s = float(scores[i])
        if s <= 0:
            break
        out.append((index.rows[int(i)].chunk_id, s))
    return out


# ---------------------------------------------------------------- 메타 필터 / RRF
def _passes_meta_filter(
    row: _ChunkRow,
    corp_codes: list[str] | None,
    year: int | None,
    chunk_type_filter: str | None,
) -> bool:
    if corp_codes and row.corp_code not in corp_codes:
        return False
    if year and row.year != year:
        return False
    if chunk_type_filter and row.chunk_type != chunk_type_filter:
        return False
    return True


def _rrf_fuse(
    dense: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion (k=60) — 두 랭킹의 순위를 점수로 합산."""
    scores: dict[str, float] = {}
    for rank, (chunk_id, _score) in enumerate(dense, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for rank, (chunk_id, _score) in enumerate(bm25, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _to_retrieved(row: _ChunkRow, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.chunk_id,
        text=row.text,
        corp_code=row.corp_code,
        corp_name=row.corp_name,
        rcept_no=row.rcept_no,
        year=row.year,
        doc_type=row.doc_type,
        title=row.title,
        section_path=row.section_path,
        chunk_type=row.chunk_type,
        score=score,
    )


# ---------------------------------------------------------------- 결과 다양화
def _select_diverse(
    candidates: list[tuple[_ChunkRow, float]],
    top_k: int,
    per_doc_cap: int | None,
) -> list[tuple[_ChunkRow, float]]:
    """관련도순 후보에서 top_k 선별 — 한 공시(rcept_no)당 청크 수를 cap 으로 제한.

    같은 공시가 여러 청크로 쪼개져 상위를 독차지하는 쏠림을 완화해 더 많은 문서를
    커버한다. cap 으로 자리가 남으면 넘친 청크로 다시 채워 top_k 개수를 보장한다
    (관련도 순서 유지). per_doc_cap 이 None/0 이하면 다양화 없이 상위 top_k.
    """
    if per_doc_cap is None or per_doc_cap <= 0:
        return candidates[:top_k]
    picked: list[tuple[_ChunkRow, float]] = []
    overflow: list[tuple[_ChunkRow, float]] = []
    counts: dict[str, int] = {}
    for row, score in candidates:
        key = row.rcept_no or row.chunk_id  # 문서 식별(없으면 청크 단위로 폴백)
        if counts.get(key, 0) < per_doc_cap:
            picked.append((row, score))
            counts[key] = counts.get(key, 0) + 1
            if len(picked) >= top_k:
                return picked
        else:
            overflow.append((row, score))
    for rs in overflow:  # 다양화로 모자란 만큼 넘친 청크로 backfill
        if len(picked) >= top_k:
            break
        picked.append(rs)
    return picked


# ---------------------------------------------------------------- 진입점
def hybrid_search(
    query: str,
    top_k: int = 10,
    *,
    corp_codes: list[str] | None = None,
    year: int | None = None,
    use_bm25: bool = True,
    chunk_type_filter: str | None = None,
    per_doc_cap: int | None = MAX_CHUNKS_PER_DOC,
) -> list[RetrievedChunk]:
    """Dense(Qdrant) + BM25 + RRF(k=60) + 문서 단위 다양화(per_doc_cap)."""
    if not query.strip() or top_k <= 0:
        return []

    index = _load_chunk_index()
    fetch_limit = max(top_k * DENSE_FETCH_MULTIPLIER, DENSE_FETCH_MIN)

    dense = _dense_search(_embed(query), fetch_limit)
    bm25_results = _bm25_search(query, fetch_limit) if use_bm25 else []
    fused = _rrf_fuse(dense, bm25_results)

    # 메타필터를 통과한 후보를 관련도순으로 모은 뒤, 문서 단위로 다양화해 top_k 선별
    candidates: list[tuple[_ChunkRow, float]] = []
    for chunk_id, score in fused:
        row = index.by_chunk_id.get(chunk_id)
        if row is None:
            continue
        if not _passes_meta_filter(row, corp_codes, year, chunk_type_filter):
            continue
        candidates.append((row, score))

    selected = _select_diverse(candidates, top_k, per_doc_cap)
    return [_to_retrieved(row, score) for row, score in selected]


def search_vector_db(query: str, top_k: int = 10) -> list[dict[str, Any]]:

    print(f"🛠️ [vectorDB]  검색 시뮬레이션 중: {query}")
    if not query.strip():
        return []
    corp_codes, year = extract_filter_signals(query)
    rows = hybrid_search(query, top_k=top_k, corp_codes=corp_codes or None, year=year)
    return [row.to_dict() for row in rows]


def warmup() -> None:
    """chunk 인덱스·BM25 를 미리 적재해 첫 검색 요청의 콜드스타트(~2분)를 제거한다.

    첫 호출은 MariaDB 174k 행 적재 + BM25 빌드로 수십 초~2분이 걸리고 이후엔
    프로세스 캐시로 즉시 응답한다. FastAPI lifespan/startup 등 서버 기동 시점에
    이 함수를 호출하면 첫 사용자 요청이 지연되지 않는다.
    """
    _load_chunk_index()
    _get_bm25()
