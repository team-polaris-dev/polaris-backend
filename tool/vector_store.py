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

from dotenv import load_dotenv

from tool.rdb_client import mariadb_conn

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

RRF_K = 60
DENSE_FETCH_MIN = 30
DENSE_FETCH_MULTIPLIER = 5

_TOKEN_RE = re.compile(r"[가-힣a-zA-Z0-9]+")
_YEAR_RE = re.compile(r"(20\d{2})\s*년?")
_COMPARISON_RE = re.compile(r"비교|대비|\bvs\b|VS|둘\s*중|차이")


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
    uuid_to_chunk_id: dict[str, str]
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
        uuid_to_chunk_id: dict[str, str] = {}
        tokenized: list[list[str]] = []
        for r in db_rows:
            chunk_id = str(r["chunk_id"])
            rcept_no = str(r.get("rcept_no") or "")
            year = int(rcept_no[:4]) if rcept_no[:4].isdigit() else None
            text = str(r.get("embedding_text") or "")
            row = _ChunkRow(
                chunk_id=chunk_id,
                corp_code=str(r.get("corp_code") or "").zfill(8),
                corp_name=str(r.get("corp_name") or ""),
                rcept_no=rcept_no,
                year=year,
                doc_type=str(r.get("doc_type") or ""),
                title=str(r.get("title") or ""),
                section_path=str(r.get("section_path") or ""),
                chunk_type=str(r.get("chunk_type") or ""),
                text=text,
            )
            rows.append(row)
            by_chunk_id[chunk_id] = row
            uuid_to_chunk_id[chunk_id_to_uuid(chunk_id)] = chunk_id
            tokenized.append(_tokenize(text))

        _CHUNK_INDEX_CACHE = _ChunkIndex(
            rows=rows,
            by_chunk_id=by_chunk_id,
            uuid_to_chunk_id=uuid_to_chunk_id,
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
    index = _load_chunk_index()
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = _get_bm25().get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    out: list[tuple[str, float]] = []
    for i in ranked[:limit]:
        if scores[i] <= 0:
            break
        out.append((index.rows[i].chunk_id, float(scores[i])))
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


# ---------------------------------------------------------------- 진입점
def hybrid_search(
    query: str,
    top_k: int = 10,
    *,
    corp_codes: list[str] | None = None,
    year: int | None = None,
    use_bm25: bool = True,
    chunk_type_filter: str | None = None,
) -> list[RetrievedChunk]:
    """Dense(Qdrant) + BM25 + RRF(k=60)."""
    if not query.strip() or top_k <= 0:
        return []

    index = _load_chunk_index()
    fetch_limit = max(top_k * DENSE_FETCH_MULTIPLIER, DENSE_FETCH_MIN)

    dense = _dense_search(_embed(query), fetch_limit)
    bm25_results = _bm25_search(query, fetch_limit) if use_bm25 else []
    fused = _rrf_fuse(dense, bm25_results)

    candidates: list[tuple[_ChunkRow, float]] = []
    for chunk_id, score in fused:
        row = index.by_chunk_id.get(chunk_id)
        if row is None:
            continue
        if not _passes_meta_filter(row, corp_codes, year, chunk_type_filter):
            continue
        candidates.append((row, score))
        if len(candidates) >= top_k:
            break

    return [_to_retrieved(row, score) for row, score in candidates]


def search_vector_db(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    corp_codes, year = extract_filter_signals(query)
    rows = hybrid_search(query, top_k=top_k, corp_codes=corp_codes or None, year=year)
    return [row.to_dict() for row in rows]
