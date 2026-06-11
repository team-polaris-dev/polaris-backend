"""Vector search tool wrapper.

The node owns user-facing graceful error formatting. This tool should either
return normalized rows or raise the underlying exception.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv


def _load_backend_env() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _ensure_pola_on_path() -> None:
    """Allow local development before `pip install -e ../pola` is run."""
    root = Path(__file__).resolve().parents[2]
    src = root / "pola" / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def search_vector_db(query: str, top_k: int = 10) -> list[dict[str, Any]]:

    print(f"🛠️ [Mock Vector DB] 하이브리드 검색 시뮬레이션 중: {query}")
    return [
        {
            "chunk_id": "vec_doc_001",
            "corp_code": "12345678",
            "corp_name": "테스트기업(Vector)",
            "text": f"'{query}'와 관련된 핵심 사업 개요 요약 텍스트입니다.",
            "score": 0.88,
            "year": "2024",
            "doc_type": "사업보고서",
            "section_path": "사업의 내용 > 주요 제품"
        }
    ]

    # if not query.strip():
    #     return []
    # _load_backend_env()
    # _ensure_pola_on_path()
    # from polaris.retrieve import hybrid_search
    # rows = hybrid_search(query, top_k=top_k)
    # out: list[dict[str, Any]] = []
    # for row in rows:
    #     if hasattr(row, "to_dict"):
    #         out.append(row.to_dict())
    #     else:
    #         out.append(dict(row))
    # return out