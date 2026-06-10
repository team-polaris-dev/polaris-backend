"""Vector search tool wrapper.

The node owns user-facing graceful error formatting. This tool should either
return normalized rows or raise the underlying exception.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys
from typing import Any


def _load_backend_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if "#" in value and not value.startswith(('"', "'")):
            value = value.split("#", 1)[0].strip()
        value = value.strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ensure_pola_on_path() -> None:
    """Allow local development before `pip install -e ../pola` is run."""
    root = Path(__file__).resolve().parents[2]
    src = root / "pola" / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def search_vector_db(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    if not query.strip():
        return []
    _load_backend_env()
    _ensure_pola_on_path()
    from polaris.retrieve import hybrid_search

    rows = hybrid_search(query, top_k=top_k)
    out: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "to_dict"):
            out.append(row.to_dict())
        else:
            out.append(dict(row))
    return out
