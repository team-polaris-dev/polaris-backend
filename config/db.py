# config/db.py — MariaDB 읽기 전용 커넥션 (backend/app/db.py 패턴 재사용)
from __future__ import annotations

import os
from contextlib import contextmanager

import pymysql


def mariadb() -> "pymysql.connections.Connection":
    """MariaDB 커넥션 생성. .env 의 MARIADB_* 사용. 결과는 dict 로 받는다."""
    return pymysql.connect(
        host=os.getenv("MARIADB_HOST", "localhost"),
        port=int(os.getenv("MARIADB_PORT") or 3307),
        user=os.getenv("MARIADB_USER", "polaris"),
        password=os.getenv("MARIADB_PASSWORD", "polaris_dev_only"),
        database=os.getenv("MARIADB_DATABASE", "polaris"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


@contextmanager
def mariadb_conn():
    """커넥션 컨텍스트매니저 — 예외가 나도 close 보장(누수 방지)."""
    conn = mariadb()
    try:
        yield conn
    finally:
        conn.close()
