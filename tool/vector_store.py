"""Vector 하이브리드 검색 — dump 스키마(MariaDB chunk_index/document_index +
Qdrant `polaris-chunks`) 기준 self-contained 구현.

Dense(Qdrant) + BM25(rank_bm25) + RRF(k=60).
모듈 캐시는 프로세스 단위로 1회만 적재한다 — 캐시 무효화는 프로세스 재시작.

The node owns user-facing graceful error formatting. This tool either returns
normalized rows or raises the underlying exception.
"""
from __future__ import annotations

import functools
import hashlib
import os
import pickle
import re
import threading
import time
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

# ---- 관련도 하한 (off-topic 게이트) ----
# RRF 점수는 '순위'의 함수라 무관한 질문에도 항상 0보다 커서(상위 N 이 늘 존재)
# 관련도 컷의 잣대가 못 된다. 그래서 의미적 관련도는 dense(bge-m3) 코사인 유사도로
# 잰다. 이 코퍼스(반도체 공시)에서 실측한 top-1 코사인 분포(2026-06):
#   관련 질문(공급망/기술) 0.58 · 관련(재무) 0.72  ↔  무관(잡담) 0.43 · 무관(타도메인) 0.41
# → 0.50 이면 관련/무관 사이에 양방향 마진(±0.07~0.08)이 생긴다. dense 최고 코사인이
#   이 하한 미만이면 '코퍼스에 의미적으로 관련된 청크가 없다'고 보고 빈 결과를 낸다
#   (0건이 정답일 수 있다 — rdb 규칙 9 와 같은 철학). 코퍼스가 바뀌면 재보정한다.
DENSE_SCORE_FLOOR = float(os.getenv("VECTOR_DENSE_FLOOR", "0.50"))

_TOKEN_RE = re.compile(r"[가-힣a-zA-Z0-9]+")
_YEAR_RE = re.compile(r"(20\d{2})\s*년?")
_COMPARISON_RE = re.compile(r"비교|대비|\bvs\b|VS|둘\s*중|차이")
# 제목의 회계기간 '(YYYY.MM)' — 연도 필터를 접수연도가 아닌 회계연도 기준으로 잡기 위함
_TITLE_YEAR_RE = re.compile(r"\((20\d{2})\.\d{1,2}")

# ---- 섹션 인지 재가중 (내용성 질문에서 보일러플레이트 디프리오, 사업의 내용 부스트) ----
# 재무 수치성 질문이면 '재무제표 주석'이 정답이므로 페널티를 끄고 전부 중립(×1.0)으로 둔다.
_FINANCIAL_QUERY_RE = re.compile(
    r"매출|영업이익|순이익|손익|자산|부채|자본|현금흐름|재무|실적|배당|EPS|주당"
)
# section_path 가 재무제표·감사·지분거래 보일러플레이트인지 / 사업의 내용(본문)인지 판별.
# 실제 코퍼스 section_path 10개 전수 introspection 으로 확정(2026-06): 기존 패턴이
# 'IX. 계열회사 등에 관한 사항'(출자표)과 'XI. 그 밖에 투자자 보호를 위하여 필요한 사항'
# (사회공헌 등 기타표) 두 섹션을 못 잡아 내용성 질문에서 노이즈로 샜다 → 추가.
_BOILERPLATE_SECTION_RE = re.compile(
    r"감사보고서|감사의견|주석|대주주|특수관계|이해관계자|계열회사|투자자\s*보호"
)
_BUSINESS_SECTION_RE = re.compile(r"사업의\s*내용")
# 내용성 질문에서 적용할 섹션 배수 (RRF 점수에 곱함). 시작값이며 검증하며 조정.
_SECTION_WEIGHT_BOILERPLATE = 0.3
_SECTION_WEIGHT_BUSINESS = 1.3

# ---- 본문 근접중복 제거 (같은 문단이 분기마다 반복 게재되는 문제) ----
# per_doc_cap 은 같은 공시(rcept_no) 안에서만 막으므로, 공시를 넘는 동일 문단은
# 본문 토큰 Jaccard 로 판정해 제거한다. 임계 이상이면 근접중복으로 본다.
_DEDUP_JACCARD_THRESHOLD = float(os.getenv("VECTOR_DEDUP_JACCARD", "0.8"))

# ---- 관련도 하한 (꼬리 필러 제거) ----
# 최종 재가중 점수가 1위의 이 비율 미만이면 버린다. dense 코사인이 아니라 '상대 비율'을
# 쓰는 이유: 신호 청크가 bm25-only(코사인 없음)이고 노이즈 표는 코사인이 높아 절대 코사인
# 하한이 정반대로 작동하기 때문(실측). 점수 절벽이 있으면 자르고, 다 비슷하면 그대로 둔다.
_RELEVANCE_FLOOR_RATIO = float(os.getenv("VECTOR_RELEVANCE_FLOOR_RATIO", "0.5"))

# ---- 절대 점수 하한 (무관 쿼리 전체 게이트) ----
# 재가중 후 후보 전체의 최고 점수가 이 값 미만이면 0건 반환(RDB/Graph 에 위임).
# 상대 비율(_RELEVANCE_FLOOR_RATIO)은 "1위 대비 꼬리"를 자르지만, 전체가 일제히 낮은
# 쿼리(이사회 구성 max 0.008, 감사인 독립성 혼동 등)는 비율이 유지돼 노이즈가 통과한다.
# 실측(30개 대표 질문, 2026-06):
#   코퍼스에 답 있는 쿼리 최고 점수: 0.020 ~ 0.042
#   코퍼스 커버리지 낮은 쿼리(이사회): 0.008 / 대표이사 임원: 0.011
# → 0.010 하한이면 이사회 쿼리만 0건 처리, 나머지 28/30 는 영향 없음.
_ABSOLUTE_SCORE_FLOOR = float(os.getenv("VECTOR_ABSOLUTE_FLOOR", "0.010"))


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


# 형태소 토크나이저 — 조사/어미를 떼어 표면형 차이('기술에' vs '기술은')를 흡수한다.
# 내용어 태그만 보존: 명사(NNG/NNP/NR)·외국어(SL)·숫자(SN)·한자(SH)·기타기호어(SW)·어근(XR).
_CONTENT_TAGS = frozenset({"NNG", "NNP", "NR", "SL", "SN", "SH", "SW", "XR"})
# 연속 병합 대상(반도체 약어 보존: HBM3E·DDR5 가 hbm/3/e 로 쪼개지지 않게)
_MERGE_TAGS = frozenset({"SL", "SN", "SH", "SW"})
_KIWI: Any | None = None
_KIWI_FAILED = False
# Kiwi 전용 락 — _LOCK 을 쓰면 안 된다. _load_chunk_index 가 _LOCK 을 쥔 채
# _tokenize_many → _get_kiwi 를 호출하는데, 같은 _LOCK 을 다시 잡으면(재진입 불가)
# 콜드빌드가 토큰화 시작 직전에 자기 자신을 기다리며 영원히 멈춘다(데드락).
_KIWI_LOCK = threading.Lock()


def _get_kiwi():
    """Kiwi 형태소 분석기 싱글톤(프로세스 1회 생성). import/로드 실패 시 None →
    호출부가 정규식 폴백으로 graceful degrade(이 코드베이스의 실패-삼킴 철학)."""
    global _KIWI, _KIWI_FAILED
    if _KIWI is not None or _KIWI_FAILED:
        return _KIWI
    with _KIWI_LOCK:  # _LOCK 아님 — _load_chunk_index 락 안에서 호출되므로 데드락 회피
        if _KIWI is not None or _KIWI_FAILED:
            return _KIWI
        try:
            from kiwipiepy import Kiwi

            _KIWI = Kiwi()
        except Exception:
            _KIWI_FAILED = True
            _KIWI = None
        return _KIWI


def _merge_morphemes(tokens) -> list[str]:
    """Kiwi 토큰 리스트(한 문서) → 내용어만 소문자화하고 연속 영문/숫자/기호어를
    하나로 병합(약어 보존: HBM3E·DDR5). _tokenize(단건)와 배치 경로가 공유한다."""
    out: list[str] = []
    prev_end = -1
    prev_mergeable = False
    for tok in tokens:
        if tok.tag not in _CONTENT_TAGS:
            prev_mergeable = False
            continue
        form = tok.form.lower()
        mergeable = tok.tag in _MERGE_TAGS
        # 직전 토큰과 원문상 인접(공백 없음)하고 둘 다 병합대상이면 약어로 이어붙인다
        if mergeable and prev_mergeable and tok.start == prev_end and out:
            out[-1] += form
        else:
            out.append(form)
        prev_end = tok.start + len(tok.form)
        prev_mergeable = mergeable
    return out


def _tokenize(text: str) -> list[str]:
    """형태소 기반 토큰화. 내용어 태그만 남기고 소문자화하며, 원문에서 연속된
    영문/숫자/기호어는 하나로 병합(약어 보존)한다. Kiwi 미가용 시 정규식 폴백."""
    if not text:
        return []
    kiwi = _get_kiwi()
    if kiwi is None:
        return _TOKEN_RE.findall(text)
    return _merge_morphemes(kiwi.tokenize(text))


_TOKENIZE_PROGRESS_EVERY = 10000  # 대량 빌드 시 진행률 로그 간격(행)


def _tokenize_many(texts: list[str]) -> list[list[str]]:
    """배치 토큰화 — 17만 행을 행별 호출 대신 Kiwi 배치로 한 번에 처리(콜드빌드 가속).
    Kiwi 미가용 시 정규식 폴백. 입력 순서를 그대로 보존한다.

    대량(≥1만 행) 빌드는 수십 초~수 분 걸리므로 진행률(행수/속도/ETA)을 즉시 flush 로
    찍어 '멈춤'과 '진행 중'을 구분한다(콜드빌드 가시성)."""
    if not texts:
        return []
    kiwi = _get_kiwi()
    if kiwi is None:
        return [_TOKEN_RE.findall(t or "") for t in texts]

    total = len(texts)
    verbose = total >= _TOKENIZE_PROGRESS_EVERY
    if verbose:
        print(f"🔧 [vector] 형태소 토큰화 시작: {total:,}행 (kiwi)", flush=True)
    out: list[list[str]] = []
    t0 = time.time()
    for i, toks in enumerate(kiwi.tokenize(texts), 1):
        out.append(_merge_morphemes(toks))
        if verbose and (i % _TOKENIZE_PROGRESS_EVERY == 0 or i == total):
            el = time.time() - t0
            rate = i / el if el else 0.0
            eta = (total - i) / rate if rate else 0.0
            print(
                f"🔧 [vector] 토큰화 {i:,}/{total:,} ({i * 100 // total}%) "
                f"· {rate:.0f}행/s · 경과 {el / 60:.1f}분 · ETA {eta / 60:.1f}분",
                flush=True,
            )
    return out


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


# ---------------------------------------------------------------- 디스크 캐시
# 17만 행 토큰화(kiwi ~수십초)를 프로세스 재시작마다 반복하지 않도록 토큰화된
# chunk 인덱스를 디스크에 영속화한다. 코퍼스 '지문'(ready 행 수 + chunk_id CRC XOR)이
# 같으면 디스크에서 즉시 로드, 적재가 바뀌면(지문 불일치) 무효화 후 재빌드한다.
_CACHE_VERSION = 1  # 직렬화 포맷/토크나이저 규약이 바뀌면 올린다(구 캐시 자동 무효화)


def _cache_path() -> Path:
    """캐시 파일 경로. BM25_CACHE_PATH 로 덮어쓸 수 있다.

    기본 위치는 repo 밖 사용자 캐시 디렉토리다 — 토큰화 인덱스가 ~600MB라
    repo 안(.cache/)에 두면 git 추적·실수 커밋 위험이 있어서다(.gitignore 를
    건드리지 않고 repo 오염을 원천 차단). Windows=%LOCALAPPDATA%\\polaris,
    그 외=~/.cache/polaris.
    """
    override = os.getenv("BM25_CACHE_PATH")
    if override:
        return Path(override)
    base = os.getenv("LOCALAPPDATA") or str(Path.home() / ".cache")
    d = Path(base) / "polaris"
    d.mkdir(parents=True, exist_ok=True)
    return d / "bm25_chunk_index.pkl"


def _read_cached_index(fingerprint: str, path: Path) -> _ChunkIndex | None:
    """지문이 일치하는 캐시만 복원. 없음/불일치/버전불일치/손상 → None(재빌드 유도)."""
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
    except Exception:
        return None
    if not isinstance(blob, dict):
        return None
    if blob.get("version") != _CACHE_VERSION or blob.get("fingerprint") != fingerprint:
        return None
    index = blob.get("index")
    return index if isinstance(index, _ChunkIndex) else None


def _write_cached_index(index: _ChunkIndex, fingerprint: str, path: Path) -> None:
    """원자적 저장(temp 쓰기 후 rename). 실패는 삼킨다(캐시는 최적화일 뿐 필수 아님)."""
    blob = {"version": _CACHE_VERSION, "fingerprint": fingerprint, "index": index}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except Exception:
        pass


# ---------------------------------------------------------------- chunk index 적재
def _load_chunk_index() -> _ChunkIndex:
    global _CHUNK_INDEX_CACHE
    if _CHUNK_INDEX_CACHE is not None:
        return _CHUNK_INDEX_CACHE
    with _LOCK:
        if _CHUNK_INDEX_CACHE is not None:
            return _CHUNK_INDEX_CACHE

        fp_sql = (
            "SELECT COUNT(*) c, COALESCE(BIT_XOR(CRC32(chunk_id)), 0) x "
            "FROM chunk_index WHERE ingest_status='ready' AND embedding_text IS NOT NULL"
        )
        sql = """
            SELECT
                ci.chunk_id, ci.corp_code, ci.rcept_no, ci.chunk_type,
                ci.section_path, ci.embedding_text,
                di.corp_name, di.doc_type, di.title
            FROM chunk_index ci
            LEFT JOIN document_index di ON di.rcept_no = ci.rcept_no
            WHERE ci.ingest_status = 'ready' AND ci.embedding_text IS NOT NULL
        """
        # 1) 코퍼스 지문(싼 집계 쿼리)으로 디스크 캐시 적중 여부 판단 — 적중 시 재토큰화 생략
        cache_path = _cache_path()
        with mariadb_conn() as conn, conn.cursor() as cur:
            cur.execute(fp_sql)
            fp_row = cur.fetchone()
            fingerprint = f"v{_CACHE_VERSION}:{fp_row['c']}:{fp_row['x']}"
            cached = _read_cached_index(fingerprint, cache_path)
            if cached is not None:
                # 디스크 캐시 적중 — 콜드빌드(수십 초~2분) 생략. 이후 검색 0건은
                # '인덱스 미적재(콜드)'가 아니라 진짜 결과 없음으로 해석해야 한다.
                print(f"ℹ️ [vector] chunk 인덱스 디스크 캐시 적중({fp_row['c']}행) — 콜드빌드 생략")
                _CHUNK_INDEX_CACHE = cached
                return _CHUNK_INDEX_CACHE
            # 2) 캐시 미스 → 콜드빌드(행 적재 + 배치 토큰화). 이 로그가 찍힌 직후
            #    들어온 검색이 0건이면 '콜드스타트 중'이 원인일 수 있다(완료 로그 전까지).
            print(f"🧊 [vector] chunk 인덱스 콜드빌드 시작({fp_row['c']}행) — MariaDB 적재+토큰화")
            cur.execute(sql)
            db_rows = cur.fetchall()

        rows: list[_ChunkRow] = []
        by_chunk_id: dict[str, _ChunkRow] = {}
        texts: list[str] = []
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
            texts.append(text)

        tokenized = _tokenize_many(texts)  # 행별 호출 대신 배치(콜드빌드 가속)

        _CHUNK_INDEX_CACHE = _ChunkIndex(
            rows=rows,
            by_chunk_id=by_chunk_id,
            tokenized=tokenized,
        )
        print(f"✅ [vector] chunk 인덱스 콜드빌드 완료({len(rows)}행) — 이후 검색은 캐시로 즉시 응답")
        # 3) 다음 프로세스 기동을 위해 디스크에 영속화(원자적, 실패는 삼킴)
        _write_cached_index(_CHUNK_INDEX_CACHE, fingerprint, cache_path)
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


def _extract_year_filter(query: str) -> int | None:
    """연도 하드필터 값. distinct 연도가 정확히 1개일 때만 그 연도를 쓰고,
    0개면 None, 2개 이상이면 None 으로 둔다.

    예전엔 `_YEAR_RE.search`로 '첫' 연도만 잡아 '2023년 대비 2024년' 같은 비교형이
    2023 으로 굳어 2024 청크가 통째로 드롭됐다(과필터). 다중 연도면 한쪽을 잃지
    않도록 하드필터를 끄고(전체 허용) 관련도·회사필터에 맡긴다. 단일 연도(회사 비교
    포함)는 연도 스코핑이 정당하므로 유지한다.
    """
    distinct = sorted({int(y) for y in _YEAR_RE.findall(query)})
    return distinct[0] if len(distinct) == 1 else None


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

    year = _extract_year_filter(query)

    if not mentioned:
        return [], year

    if len(mentioned) == 1 or _COMPARISON_RE.search(query):
        return _dedup([name_to_code[name] for name in mentioned]), year

    owner = _select_owner_company(query, mentioned)
    if owner:
        return [name_to_code[owner]], year
    return _dedup([name_to_code[name] for name in mentioned]), year


# ---------------------------------------------------------------- 임베딩 (Ollama)
# 임베딩 캐시 크기. 같은 쿼리 재임베딩을 막아 (1) Ollama 왕복 비용과 (2) 원격 GPU
# 부동소수점 미세 흔들림(같은 입력에도 1e-6 단위 차이→HNSW 이웃 변동)을 함께 제거한다.
# 캐시 무효화는 프로세스 재시작. (VECTOR_EMBED_CACHE 로 override)
_EMBED_CACHE_SIZE = int(os.getenv("VECTOR_EMBED_CACHE", "512"))


def _embed_fetch(text: str) -> list[float]:
    """Ollama(원격)로 실제 임베딩 1회 왕복. 캐시는 _embed 가 담당(이 함수는 순수 fetch)."""
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


@functools.lru_cache(maxsize=_EMBED_CACHE_SIZE)
def _embed_cached_key(text: str) -> tuple[float, ...]:
    """불변 tuple 로 캐시한다(호출자 변형으로부터 캐시 보호). 내부 fetch 는 _embed_fetch."""
    return tuple(_embed_fetch(text))


def _embed(text: str) -> list[float]:
    """텍스트 → 임베딩 벡터(list). 같은 text 는 캐시에서 재사용해 결정성을 보장한다.

    캐시는 불변 tuple 로 들고, 매 호출 새 list 로 복사해 돌려준다 — 호출자가 결과
    list 를 변형해도 캐시가 오염되지 않는다.
    """
    return list(_embed_cached_key(text))


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


def _merge_dense_candidates(
    primary: list[tuple[str, float]], supplemental: list[tuple[str, float]]
) -> list[tuple[str, float]]:
    """두 dense 결과(혼합 top-N + 본문 보강)를 chunk_id 기준 합집합·점수 내림차순으로 병합.

    같은 chunk_id 가 양쪽에 있으면 더 높은 코사인을 남긴다. 표(table_nl)가 코사인
    상위를 독식해 본문(text_micro)이 dense top-N 에서 통째로 빠지는 문제를, 본문을
    별도 쿼터로 뽑아 여기서 합쳐 후보풀 진입을 보장하기 위한 순수함수.
    """
    best: dict[str, float] = {}
    for chunk_id, score in [*primary, *supplemental]:
        if chunk_id not in best or score > best[chunk_id]:
            best[chunk_id] = score
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def _dense_search(
    vector: list[float], limit: int, *, corp_codes: list[str] | None = None,
    chunk_type: str | None = None,
) -> list[tuple[str, float]]:
    """Qdrant 최근접 검색 → (chunk_id, score).

    적재된 `polaris-chunks` 컬렉션은 포인트 ID 가 정수이고 chunk_id 는 payload 에
    들어있다(md5-UUID 포인트 ID 가 아님). payload['chunk_id'] 로 직접 복원한다.

    `corp_codes` 가 주어지면 Qdrant `query_filter` 로 **검색 단계에서** 회사를
    한정한다(pre-filtering). 전체 코퍼스에서 top-N 을 뽑고 사후에 거르면 타겟
    회사 청크가 전역 순위 밖으로 밀려 0건이 되는 문제를 원천 차단한다. payload
    `corp_code` 는 KEYWORD 인덱스가 적재돼 있어(embed_qdrant) 인덱스 기반 필터다.
    """
    client = _qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", "polaris-chunks")

    query_filter = None
    if corp_codes or chunk_type:
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        must = []
        if corp_codes:
            must.append(FieldCondition(key="corp_code", match=MatchAny(any=list(corp_codes))))
        if chunk_type:
            must.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
        query_filter = Filter(must=must)

    # exact=True: HNSW 근사 대신 전수 검색 → 같은 벡터는 항상 같은 이웃(결정성).
    # 근사 탐색은 fetch_limit 경계의 청크가 실행마다 들락거려 top-N 이 흔들렸다.
    # 코퍼스(~17만 청크)에서 전수 검색 추가 비용은 ~0.5s/쿼리로, LLM 대비 미미하다.
    from qdrant_client.models import SearchParams

    result = client.query_points(
        collection_name=collection,
        query=vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=["chunk_id"],
        with_vectors=False,
        search_params=SearchParams(exact=True),
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


def _bm25_search(
    query: str, limit: int, *, row_filter: Any | None = None
) -> list[tuple[str, float]]:
    """BM25 top-N. 전체(~17만) 인덱스에 대한 파이썬 람다 정렬을 numpy argsort 로 교체.

    기존 `sorted(range(n), key=lambda i: scores[i], reverse=True)` 는 17만 회
    파이썬 람다 호출로 느렸다. `np.argsort(-scores, kind='stable')` 는 C 레벨
    정렬이라 훨씬 빠르고, 점수 내림차순·동점 시 인덱스 오름차순으로 **기존 결과
    순서를 그대로 보존**한다(동점 경계 포함).

    BM25 인덱스는 전역 1개라 재구축할 수 없으므로, 회사/연도 한정이 필요할 때는
    `row_filter`(행 술어)로 **top-N 선별 단계에서** 회사 내부 행만 모은다. dense 의
    Qdrant pre-filter 와 대칭을 이뤄 RRF 가 '회사 내부 양쪽 랭킹'을 융합하게 한다.
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
    for i in order:
        idx = int(i)
        s = float(scores[idx])
        if s <= 0:
            break
        if row_filter is not None and not row_filter(index.rows[idx]):
            continue
        out.append((index.rows[idx].chunk_id, s))
        if len(out) >= limit:
            break
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


def _is_offtopic(
    dense: list[tuple[str, float]],
    bm25: list[tuple[str, float]],
    floor: float = DENSE_SCORE_FLOOR,
) -> bool:
    """off-topic 게이트: 코퍼스에 의미·어휘적으로 무관한 질문인지 판정.

    dense(bge-m3) 최고 코사인이 floor 미만이고 **BM25 정확 매치도 없을 때만** True.
    예전엔 BM25 를 보기 전에 dense 단독으로 컷해서, dense 가 약하지만 sparse 가 강한
    질문(약어 'HBM3E'·코드 '11011'·정확 명칭)을 잘못 0건 처리했다(①의 BM25 개선과
    충돌). 두 신호가 모두 약할 때만 무관으로 본다.

    dense 가 비면(Qdrant/Ollama 장애 가능) 게이트를 끈다 — 외부 장애를 '무관'으로
    오인하지 않고 BM25 단독으로 graceful degrade.
    """
    if not dense:
        return False
    top_dense = max(score for _, score in dense)
    if top_dense >= floor:
        return False
    return not bm25


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


def _is_financial_query(query: str) -> bool:
    """재무 수치성 질문인지 — 결정론적 키워드 판별.

    재무 질문은 '재무제표 주석' 섹션이 정답일 수 있어 섹션 페널티를 꺼야 한다
    (_section_weight 가 이 결과로 content_mode 를 끈다).
    """
    return bool(_FINANCIAL_QUERY_RE.search(query or ""))


def _dedup_body_tokens(text: str) -> frozenset[str]:
    """근접중복 판정용 본문 토큰 집합. 머리말 메타('[회사 · 문서(날짜) · 섹션]')는
    날짜가 달라 동일 본문도 다르게 보이게 하므로 떼어내고 본문만 토큰화한다."""
    body = text.split("]", 1)[-1] if text.startswith("[") else text
    return frozenset(t.lower() for t in _TOKEN_RE.findall(body))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """두 토큰 집합의 Jaccard 유사도(교집합/합집합). 한쪽이 비면 0."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedup_near_duplicates(
    candidates: list[tuple["_ChunkRow", float]],
    threshold: float = _DEDUP_JACCARD_THRESHOLD,
) -> list[tuple["_ChunkRow", float]]:
    """관련도순(내림차순) 후보에서 근접중복을 제거한다.

    앞에서부터 보며 이미 채택된 대표 중 하나와 본문 Jaccard 가 threshold 이상이면
    버린다(가장 관련도 높은 대표 1건만 남김). 같은 문단이 여러 분기 보고서에 반복
    게재돼 결과를 독식하던 문제를 제거 — per_doc_cap(공시 내부)으로는 못 막는 케이스.
    """
    kept: list[tuple[_ChunkRow, float]] = []
    kept_tokens: list[frozenset[str]] = []
    for row, score in candidates:
        toks = _dedup_body_tokens(row.text)
        if any(_jaccard(toks, kt) >= threshold for kt in kept_tokens):
            continue
        kept.append((row, score))
        kept_tokens.append(toks)
    return kept


def _apply_relevance_floor(
    candidates: list[tuple["_ChunkRow", float]],
    ratio: float = _RELEVANCE_FLOOR_RATIO,
) -> list[tuple["_ChunkRow", float]]:
    """관련도순(내림차순) 후보에서 1위 점수의 ratio 미만을 버린다.

    점수에 절벽이 있으면(신호 다음 필러가 뚝 떨어짐) 필러를 잘라 top_k 를 억지로
    채우지 않고, 점수가 매끄럽게 다 높으면 그대로 둔다(거짓 드롭 방지). 1위가 0 이하인
    퇴화 케이스는 비율 계산을 건너뛴다.
    """
    if not candidates:
        return candidates
    top = candidates[0][1]
    if top <= 0:
        return candidates
    cutoff = top * ratio
    return [(row, score) for row, score in candidates if score >= cutoff]


def _apply_absolute_score_floor(
    candidates: list[tuple["_ChunkRow", float]],
    floor: float = _ABSOLUTE_SCORE_FLOOR,
) -> list[tuple["_ChunkRow", float]]:
    """재가중 후 최고 점수가 floor 미만이면 전체 0건을 반환한다.

    상대 비율(_apply_relevance_floor)은 꼬리를 자르지만, 이사회 구성처럼 코퍼스 커버리지가
    낮아 전체 점수가 일제히 낮은 쿼리는 비율이 유지돼 노이즈가 그대로 통과한다.
    이 게이트는 "최고 점수조차 의미 없는 수준" 을 잡아 RDB/Graph 에 위임한다.
    floor=0.0 이면 항상 통과(기능 끄기).
    """
    if not candidates or floor <= 0:
        return candidates
    top = candidates[0][1]
    if top < floor:
        return []
    return candidates


def _section_weight(row: _ChunkRow, content_mode: bool) -> float:
    """section_path 분류 기반 RRF 점수 배수.

    content_mode(내용성 질문)에서만 보일러플레이트(감사보고서·재무제표 주석·지분거래)를
    디프리오(×0.3)하고 사업의 내용을 부스트(×1.3)한다. 재무 질문(content_mode=False)은
    재무제표 주석이 정답일 수 있어 전부 중립(×1.0)으로 둔다.

    공급망·기술 같은 본문 질문에서 감사·재무주석 보일러플레이트가 동률 RRF 점수로
    상위를 차지해 '사업의 내용' 본문을 밀어내던 문제(관련성 약화)를 결정론적으로 보정.
    """
    if not content_mode:
        return 1.0
    # 내용성 질문(공급망·기술 등)이 원하는 건 서술 본문(text_micro)이다. 표(table_nl)는
    # 섹션과 무관하게 디프리오한다 — 코퍼스의 71%가 표라 retrieval 이 표를 과대표집하고,
    # 특히 '사업의 내용' 섹션에 섞인 표(368개)가 본문 부스트(×1.3)를 가로채던 문제를 차단.
    # (재무 질문은 위 content_mode=False 분기에서 이미 중립이라 표가 정답일 때 영향 없음)
    if (row.chunk_type or "") == "table_nl":
        return _SECTION_WEIGHT_BOILERPLATE
    path = row.section_path or ""
    if _BUSINESS_SECTION_RE.search(path):
        return _SECTION_WEIGHT_BUSINESS
    if _BOILERPLATE_SECTION_RE.search(path):
        return _SECTION_WEIGHT_BOILERPLATE
    return 1.0


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

    # 회사/연도 한정을 검색 단계로 끌어올린다(pre-filtering): dense 는 Qdrant
    # query_filter(corp_code), bm25 는 행 술어로 회사 내부에서 top-N 을 뽑는다.
    # year/chunk_type 은 제목 회계기간 기반이라 Qdrant 로 밀지 않고 사후 필터로 둔다.
    row_filter = None
    if corp_codes or year or chunk_type_filter:
        def row_filter(r: _ChunkRow) -> bool:
            return _passes_meta_filter(r, corp_codes, year, chunk_type_filter)

    qvec = _embed(query)
    dense = _dense_search(qvec, fetch_limit, corp_codes=corp_codes)
    # 본문(text_micro) 보강: bge-m3 는 회사명·관계어로 빽빽한 표(table_nl)를 본문보다
    # 일괄 높게 점수내어(실측: 표 0.64~0.65 vs 본문 ≤0.63) 본문이 dense top-N 에서 통째로
    # 빠진다. 본문을 별도 쿼터로 추가 retrieval 해 후보풀 진입을 보장한다(additive — 표
    # recall 은 그대로). 이후 RRF·섹션 재가중이 최종 순위를 정한다.
    micro = _dense_search(qvec, max(fetch_limit // 2, DENSE_FETCH_MIN // 2),
                          corp_codes=corp_codes, chunk_type="text_micro")
    dense = _merge_dense_candidates(dense, micro)
    bm25_results = _bm25_search(query, fetch_limit, row_filter=row_filter) if use_bm25 else []

    # off-topic 게이트는 dense·BM25 를 모두 본 뒤 적용한다(BM25 강한 매치를 죽이지
    # 않기 위해 — dense 단독 컷이 약어/코드 질문을 0건 처리하던 결함 수정).
    if _is_offtopic(dense, bm25_results):
        print(f"ℹ️ [vector] off-topic 게이트(dense<{DENSE_SCORE_FLOOR:.2f} & BM25 무매치) "
              f"→ 빈 결과(무관 질문, 정상 0건): {query[:40]!r}")
        return []

    fused = _rrf_fuse(dense, bm25_results)

    # 섹션 인지 재가중: 내용성 질문이면 보일러플레이트(감사·재무주석·지분거래)를 누르고
    # 사업의 내용을 띄운다. 재무 수치성 질문은 중립(×1.0)이라 영향이 없다.
    content_mode = not _is_financial_query(query)

    # 메타필터를 통과한 후보를 (재가중된) 관련도순으로 모은 뒤, 문서 단위로 다양화해 top_k 선별
    candidates: list[tuple[_ChunkRow, float]] = []
    for chunk_id, score in fused:
        row = index.by_chunk_id.get(chunk_id)
        if row is None:
            continue
        if not _passes_meta_filter(row, corp_codes, year, chunk_type_filter):
            continue
        candidates.append((row, score * _section_weight(row, content_mode)))

    candidates.sort(key=lambda rs: rs[1], reverse=True)  # 재가중으로 순위가 바뀌므로 재정렬
    # 본문 근접중복 제거: 같은 문단이 여러 분기 보고서에 반복 게재돼 top_k 를 독식하던
    # 문제 제거(per_doc_cap 은 같은 공시 안에서만 막음). 관련도 높은 대표 1건만 남긴다.
    candidates = _dedup_near_duplicates(candidates)
    # 관련도 하한: 1위 점수의 절반도 안 되는 꼬리 필러(주석·감사·표)를 잘라 top_k 를
    # 억지로 채우지 않는다. 점수 절벽이 있을 때만 작동, 다 관련되면 그대로 둔다.
    candidates = _apply_relevance_floor(candidates)
    # 절대 점수 하한: 재가중 후 최고 점수조차 임계 미만이면 전체 0건 반환.
    # 전체가 일제히 낮은 쿼리(이사회 구성 등 코퍼스 커버리지 낮음)는 상대 비율로 못 잡으므로
    # 이 게이트가 잡아 RDB/Graph 에 위임한다.
    candidates = _apply_absolute_score_floor(candidates)
    selected = _select_diverse(candidates, top_k, per_doc_cap)
    return [_to_retrieved(row, score) for row, score in selected]


def search_vector_db(
    query: str, top_k: int = 10, *, corp_codes: list[str] | None = None
) -> list[dict[str, Any]]:
    """하이브리드 검색 진입점. `corp_codes` 가 주어지면 회사 필터로 그걸 쓴다.

    호출부(노드)가 관계도에서 뽑은 앵커 corp_code 를 넘기면 벡터 검색을 그 회사들로
    한정한다 — 질문 텍스트로만 회사를 추측하던(extract_filter_signals) 오염을 막는다.
    앵커가 없으면(None/빈) 기존대로 질문에서 회사·연도를 추출해 폴백한다. 임베딩/BM25
    질의는 어느 경우든 question 그대로 — 회사 한정만 그래프 기준으로 바뀐다.
    """
    print(f"🛠️ [vectorDB]  검색 시뮬레이션 중: {query}")
    if not query.strip():
        return []
    extracted_codes, year = extract_filter_signals(query)
    filter_codes = corp_codes if corp_codes else extracted_codes
    rows = hybrid_search(query, top_k=top_k, corp_codes=filter_codes or None, year=year)
    return [row.to_dict() for row in rows]


def warmup() -> None:
    """chunk 인덱스·BM25·Qdrant 커넥션·임베딩 경로를 미리 데워 첫 검색의
    콜드스타트(~2분)를 제거한다.

    첫 호출은 MariaDB 174k 행 적재 + BM25 빌드로 수십 초~2분이 걸리고 이후엔
    프로세스 캐시로 즉시 응답한다. Qdrant 커넥션과 원격 Ollama 임베딩 경로도
    더미 임베딩 1회로 함께 데워, 첫 사용자 요청의 외부 핸드셰이크 지연까지 없앤다.

    외부 의존(Qdrant/Ollama)이 일시적으로 죽어 있어도 chunk/BM25 캐시는 적재되며,
    개별 단계 실패는 삼켜 서버 기동을 막지 않는다(다음 실제 검색에서 재시도).
    FastAPI lifespan/startup 에서 (블로킹 방지 위해 백그라운드로) 호출하면 첫
    사용자 요청이 지연되지 않는다.
    """
    print("🔥 [vector] warmup 시작 — chunk/BM25 인덱스 + Qdrant/임베딩 경로 예열")
    _load_chunk_index()  # 콜드빌드/캐시히트 여부는 _load_chunk_index 가 직접 로깅
    _get_bm25()
    print("✅ [vector] warmup: chunk/BM25 인덱스 준비 완료")
    # Qdrant 커넥션 + 임베딩 경로 데우기 — 외부 의존이라 실패해도 기동을 막지 않는다.
    # 단, 어느 단계가 죽었는지는 로깅한다 — 이후 검색 0건이 '관련 없음'이 아니라
    # 이 외부 의존 장애(예외) 때문임을 구분하기 위함(콜드스타트 0건과도 구분).
    try:
        _qdrant_client()
    except Exception as e:
        print(f"⚠️ [vector] warmup: Qdrant 커넥션 예열 실패(검색 시 재시도): {e!r}")
    try:
        _embed("warmup")
    except Exception as e:
        print(f"⚠️ [vector] warmup: 임베딩(Ollama) 경로 예열 실패(검색 시 재시도): {e!r}")
    print("🏁 [vector] warmup 완료")
