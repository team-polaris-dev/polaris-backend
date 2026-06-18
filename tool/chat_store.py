# tool/chat_store.py — 로그인/회원가입 + 채팅 세션·메시지 기록 (chat_users/chat_sessions/chat_messages)
"""사용자이름 기반의 아주 단순한 로그인과 대화 영속화를 담당한다.

설계 메모
- 로그인 키 = 사용자이름. 별도 비밀번호 없이 user_id 에 사용자이름을 그대로 쓴다.
  처음 보는 사용자이름이면 chat_users 에 새로 INSERT 되어 자동 '회원가입' 된다.
- rdb_client 는 SELECT 전용(읽기) 도구라 여기서는 쓰기까지 필요하므로
  같은 커넥션 헬퍼(mariadb_conn)만 빌려 쓰고 INSERT/UPDATE 를 직접 실행한다.
- 모든 시간 컬럼은 datetime(3) 이라 파이썬 datetime 을 그대로 바인딩한다.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from tool.rdb_client import mariadb_conn


def _now() -> datetime:
    return datetime.now()


# ──────────────────────────────────────────────────────────────
# 로그인 / 회원가입
# ──────────────────────────────────────────────────────────────
def login_or_signup(username: str) -> dict[str, Any]:
    """사용자이름으로 로그인. 없으면 새로 만들어(회원가입) 준다.

    반환: {user_id, display_name, is_new}
    user_id 는 사용자이름을 그대로 사용한다(공백 제거).
    """
    uid = (username or "").strip()
    if not uid:
        raise ValueError("사용자이름이 비어 있습니다.")

    now = _now()
    with mariadb_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id, display_name FROM chat_users WHERE user_id=%s", (uid,))
            row = cur.fetchone()
            if row:
                # 기존 사용자 → 마지막 접속 시각만 갱신
                cur.execute("UPDATE chat_users SET last_seen_at=%s WHERE user_id=%s", (now, uid))
                conn.commit()
                return {
                    "user_id": row["user_id"],
                    "display_name": row["display_name"] or row["user_id"],
                    "is_new": False,
                }

            # 새 사용자 → 회원가입
            cur.execute(
                "INSERT INTO chat_users (user_id, display_name, first_seen_at, last_seen_at) "
                "VALUES (%s, %s, %s, %s)",
                (uid, uid, now, now),
            )
            conn.commit()
            return {"user_id": uid, "display_name": uid, "is_new": True}


# ──────────────────────────────────────────────────────────────
# 세션 / 메시지 기록
# ──────────────────────────────────────────────────────────────
def ensure_session(session_id: str, user_id: str) -> None:
    """세션이 없으면 생성한다(있으면 그대로 둠)."""
    now = _now()
    with mariadb_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (session_id, user_id, started_at, last_at, message_count) "
                "VALUES (%s, %s, %s, %s, 0) "
                "ON DUPLICATE KEY UPDATE last_at=VALUES(last_at)",
                (session_id, user_id, now, now),
            )
            conn.commit()


def save_message(
    session_id: str,
    user_id: str,
    role: str,
    content: str,
    *,
    intent: Optional[str] = None,
    latency_ms: Optional[int] = None,
    is_sufficient: Optional[bool] = None,
    retry_count: Optional[int] = None,
    panel: Optional[dict] = None,
) -> int:
    """메시지 한 건을 저장하고 세션 카운트/마지막 시각을 갱신한다. message_id 반환.

    panel(우측 패널 페이로드: {graph, documents, panel})이 있으면
    스키마 변경 없이 기존 search_plan(longtext) 컬럼에 JSON 으로 보관한다.
    세션을 다시 열 때 관계도/원본문서 버튼을 그대로 복원하기 위함이다.

    latency_ms/is_sufficient/retry_count 는 에이전트 처리 메타로,
    어시스턴트 메시지에 한해 채워진다(사용자 메시지에는 보통 None).
    """
    now = _now()
    search_plan = json.dumps(panel, ensure_ascii=False) if panel else None
    # is_sufficient 는 tinyint 컬럼이라 bool → 0/1 로 변환(None 은 그대로 둠)
    is_suff_val = None if is_sufficient is None else int(bool(is_sufficient))
    with mariadb_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_messages "
                "(session_id, user_id, role, content, intent, search_plan, "
                " is_sufficient, retry_count, latency_ms, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    session_id, user_id, role, content, intent, search_plan,
                    is_suff_val, retry_count, latency_ms, now,
                ),
            )
            message_id = cur.lastrowid
            cur.execute(
                "UPDATE chat_sessions SET last_at=%s, message_count=message_count+1 "
                "WHERE session_id=%s",
                (now, session_id),
            )
            conn.commit()
            return message_id


def list_sessions(user_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """사이드바용 — 사용자의 대화 세션 목록(최신순). 제목은 첫 사용자 메시지로 만든다."""
    with mariadb_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT s.session_id, s.started_at, s.last_at, s.message_count, "
                "  (SELECT m.content FROM chat_messages m "
                "    WHERE m.session_id=s.session_id AND m.role='user' "
                "    ORDER BY m.message_id ASC LIMIT 1) AS first_user_msg "
                "FROM chat_sessions s "
                "WHERE s.user_id=%s "
                "ORDER BY s.last_at DESC LIMIT %s",
                (user_id, int(limit)),
            )
            rows = cur.fetchall()

    sessions: list[dict[str, Any]] = []
    for r in rows:
        first = (r.get("first_user_msg") or "").strip()
        title = (first[:30] + "…") if len(first) > 30 else (first or "새 대화")
        sessions.append(
            {
                "session_id": r["session_id"],
                "title": title,
                "preview": first[:60],
                "message_count": r.get("message_count") or 0,
                "last_at": str(r.get("last_at") or ""),
            }
        )
    return sessions


def list_messages(session_id: str, limit: int = 500) -> list[dict[str, Any]]:
    """특정 세션의 메시지 목록(시간순). search_plan 에 보관된 패널 데이터도 복원한다."""
    with mariadb_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT message_id, role, content, intent, search_plan, created_at "
                "FROM chat_messages WHERE session_id=%s "
                "ORDER BY message_id ASC LIMIT %s",
                (session_id, int(limit)),
            )
            rows = cur.fetchall()

    messages: list[dict[str, Any]] = []
    for r in rows:
        # search_plan(JSON) → 우측 패널(graph/documents/panel) 복원. 없거나 깨졌으면 기본값.
        panel: dict[str, Any] = {}
        raw = r.get("search_plan")
        if raw:
            try:
                panel = json.loads(raw)
            except (ValueError, TypeError):
                panel = {}
        messages.append(
            {
                "message_id": r["message_id"],
                "role": r["role"],
                "content": r["content"] or "",
                "intent": r.get("intent") or "",
                "created_at": str(r.get("created_at") or ""),
                "panel": panel.get("panel", "none"),
                "graph": panel.get("graph", {"nodes": [], "edges": []}),
                "documents": panel.get("documents", []),
                "financials": panel.get("financials", []),
            }
        )
    return messages
