"""챗봇 대화 영속화 — 사용자/세션/메시지 적재 + 부팅 부트스트랩.

/api/chat 엔드포인트가 한 턴(사용자 발화 + 어시스턴트 응답)을 끝낼 때마다 log_turn()
한 번을 호출하면 통계용 3테이블(chat_users/chat_sessions/chat_messages)이 채워진다.

LangGraph 결과(AgentState)에서 바로 얻을 수 있는 메타(intent/search_plan/is_sufficient/
retry_count)와 엔드포인트에서 측정한 latency_ms 를 어시스턴트 메시지에 같이 적재한다.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from tool.rdb_client import mariadb_conn

DDL = """
CREATE TABLE IF NOT EXISTS chat_users (
  user_id       VARCHAR(64)  PRIMARY KEY,
  display_name  VARCHAR(120),
  first_seen_at DATETIME(3)  NOT NULL,
  last_seen_at  DATETIME(3)  NOT NULL,
  meta          JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_sessions (
  session_id    VARCHAR(64)  PRIMARY KEY,
  user_id       VARCHAR(64)  NOT NULL,
  started_at    DATETIME(3)  NOT NULL,
  last_at       DATETIME(3)  NOT NULL,
  message_count INT          NOT NULL DEFAULT 0,
  meta          JSON,
  INDEX idx_user (user_id),
  INDEX idx_started (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_messages (
  message_id    BIGINT       AUTO_INCREMENT PRIMARY KEY,
  session_id    VARCHAR(64)  NOT NULL,
  user_id       VARCHAR(64)  NOT NULL,
  role          VARCHAR(16)  NOT NULL,
  content       MEDIUMTEXT,
  intent        VARCHAR(64),
  search_plan   JSON,
  is_sufficient TINYINT,
  retry_count   INT,
  latency_ms    INT,
  created_at    DATETIME(3)  NOT NULL,
  INDEX idx_session (session_id),
  INDEX idx_user_time (user_id, created_at),
  INDEX idx_intent (intent),
  INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


def init_chat_tables() -> None:
    """부팅 시 1회. CREATE TABLE IF NOT EXISTS 라 재실행 안전."""
    with mariadb_conn() as conn, conn.cursor() as cur:
        for stmt in [s.strip() for s in DDL.split(";") if s.strip()]:
            cur.execute(stmt)
        conn.commit()


def log_turn(
    *,
    user_id: str,
    session_id: str,
    user_message: str,
    assistant_message: str,
    display_name: str | None = None,
    intent: str | None = None,
    search_plan: list[str] | None = None,
    is_sufficient: bool | None = None,
    retry_count: int | None = None,
    latency_ms: int | None = None,
    at: datetime | None = None,
) -> None:
    """한 대화 턴(user 발화 + assistant 응답)을 통계 테이블에 적재.

    멱등성은 message_id auto-increment 라 보장 안 함(매 호출이 2행 추가).
    같은 턴을 중복 호출하지 않도록 엔드포인트에서 1회만 부른다.
    """
    now = at or datetime.now()
    sp = json.dumps(search_plan, ensure_ascii=False) if search_plan is not None else None
    suff = None if is_sufficient is None else (1 if is_sufficient else 0)

    with mariadb_conn() as conn, conn.cursor() as cur:
        # 사용자 upsert
        cur.execute(
            "INSERT INTO chat_users (user_id, display_name, first_seen_at, last_seen_at) "
            "VALUES (%s,%s,%s,%s) "
            "ON DUPLICATE KEY UPDATE last_seen_at=VALUES(last_seen_at), "
            "display_name=COALESCE(VALUES(display_name), display_name)",
            (user_id, display_name, now, now),
        )
        # 세션 upsert (+2 메시지)
        cur.execute(
            "INSERT INTO chat_sessions (session_id, user_id, started_at, last_at, message_count) "
            "VALUES (%s,%s,%s,%s,2) "
            "ON DUPLICATE KEY UPDATE last_at=VALUES(last_at), "
            "message_count=message_count+2",
            (session_id, user_id, now, now),
        )
        # 사용자 메시지
        cur.execute(
            "INSERT INTO chat_messages (session_id, user_id, role, content, created_at) "
            "VALUES (%s,%s,'user',%s,%s)",
            (session_id, user_id, user_message, now),
        )
        # 어시스턴트 메시지 + 메타
        cur.execute(
            "INSERT INTO chat_messages "
            "(session_id, user_id, role, content, intent, search_plan, is_sufficient, retry_count, latency_ms, created_at) "
            "VALUES (%s,%s,'assistant',%s,%s,%s,%s,%s,%s,%s)",
            (session_id, user_id, assistant_message, intent, sp, suff, retry_count, latency_ms, now),
        )
        conn.commit()
