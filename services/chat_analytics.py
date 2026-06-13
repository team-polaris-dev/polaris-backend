"""챗봇 통계 집계 — 관리자 대시보드용 읽기 쿼리.

전부 chat_users/chat_sessions/chat_messages 를 GROUP BY/집계. 읽기 전용.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pymysql.cursors

from tool.rdb_client import mariadb_conn


def overview() -> dict[str, Any]:
    """KPI 한 묶음."""
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute("SELECT COUNT(*) AS c FROM chat_users")
        total_users = int(cur.fetchone()["c"])
        cur.execute("SELECT COUNT(*) AS c FROM chat_sessions")
        total_sessions = int(cur.fetchone()["c"])
        cur.execute(
            "SELECT "
            "COUNT(*) AS total, "
            "SUM(role='user') AS user_msgs, "
            "SUM(role='assistant') AS asst_msgs "
            "FROM chat_messages"
        )
        m = cur.fetchone()
        total_messages = int(m["total"] or 0)
        user_msgs = int(m["user_msgs"] or 0)
        asst_msgs = int(m["asst_msgs"] or 0)

        # 최근 7일 활성 사용자
        since7 = datetime.now() - timedelta(days=7)
        cur.execute(
            "SELECT COUNT(DISTINCT user_id) AS c FROM chat_messages WHERE created_at >= %s",
            (since7,),
        )
        active_7d = int(cur.fetchone()["c"])

        # RAG 충분율 + 평균 지연 (assistant 턴 기준)
        cur.execute(
            "SELECT "
            "AVG(CASE WHEN is_sufficient IS NULL THEN NULL ELSE is_sufficient END) AS suff_rate, "
            "AVG(latency_ms) AS avg_latency, "
            "AVG(retry_count) AS avg_retry "
            "FROM chat_messages WHERE role='assistant'"
        )
        a = cur.fetchone()

        # 단발성 세션(메시지 2개 이하 = 1턴) 비율
        cur.execute("SELECT COUNT(*) AS c FROM chat_sessions WHERE message_count <= 2")
        single = int(cur.fetchone()["c"])

    return {
        "total_users": total_users,
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "user_messages": user_msgs,
        "assistant_messages": asst_msgs,
        "active_users_7d": active_7d,
        "avg_messages_per_session": round(total_messages / total_sessions, 2) if total_sessions else 0,
        "avg_sessions_per_user": round(total_sessions / total_users, 2) if total_users else 0,
        "sufficient_rate": round(float(a["suff_rate"]), 4) if a["suff_rate"] is not None else None,
        "avg_latency_ms": int(a["avg_latency"]) if a["avg_latency"] is not None else None,
        "avg_retry_count": round(float(a["avg_retry"]), 2) if a["avg_retry"] is not None else None,
        "single_turn_session_rate": round(single / total_sessions, 4) if total_sessions else 0,
    }


def volume_series(days: int = 30) -> list[dict[str, Any]]:
    """일별 메시지/세션/활성유저 시계열 (빈 날은 0으로 채움)."""
    days = max(1, min(180, days))
    since = (datetime.now() - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            "SELECT DATE(created_at) AS d, "
            "COUNT(*) AS messages, "
            "COUNT(DISTINCT user_id) AS active_users, "
            "COUNT(DISTINCT session_id) AS sessions "
            "FROM chat_messages WHERE created_at >= %s "
            "GROUP BY DATE(created_at) ORDER BY d",
            (since,),
        )
        rows = {str(r["d"]): r for r in cur.fetchall()}

    out: list[dict[str, Any]] = []
    for i in range(days):
        day = (since + timedelta(days=i)).date()
        key = str(day)
        r = rows.get(key)
        out.append({
            "date": key,
            "messages": int(r["messages"]) if r else 0,
            "active_users": int(r["active_users"]) if r else 0,
            "sessions": int(r["sessions"]) if r else 0,
        })
    return out


def intent_distribution(limit: int = 12) -> list[dict[str, Any]]:
    """assistant 턴의 intent 분포 (NULL/빈 값은 'unknown')."""
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            "SELECT COALESCE(NULLIF(intent,''),'unknown') AS intent, COUNT(*) AS c "
            "FROM chat_messages WHERE role='assistant' "
            "GROUP BY COALESCE(NULLIF(intent,''),'unknown') ORDER BY c DESC LIMIT %s",
            (limit,),
        )
        return [{"intent": r["intent"], "count": int(r["c"])} for r in cur.fetchall()]


def tool_usage() -> list[dict[str, Any]]:
    """search_plan 에 등장한 도구(rdb/vec/graph)별 사용 횟수."""
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        # search_plan 은 JSON 배열. JSON_CONTAINS 로 각 도구 카운트.
        out = []
        for tool in ["rdb", "vec", "graph"]:
            cur.execute(
                "SELECT COUNT(*) AS c FROM chat_messages "
                "WHERE role='assistant' AND JSON_CONTAINS(search_plan, %s)",
                (f'"{tool}"',),
            )
            out.append({"tool": tool, "count": int(cur.fetchone()["c"])})
        return out


def top_users(limit: int = 10) -> list[dict[str, Any]]:
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            "SELECT u.user_id, u.display_name, u.last_seen_at, "
            "COUNT(DISTINCT s.session_id) AS sessions, "
            "COALESCE(SUM(s.message_count),0) AS messages "
            "FROM chat_users u LEFT JOIN chat_sessions s ON s.user_id=u.user_id "
            "GROUP BY u.user_id, u.display_name, u.last_seen_at "
            "ORDER BY messages DESC LIMIT %s",
            (limit,),
        )
        return [
            {
                "user_id": r["user_id"],
                "display_name": r["display_name"],
                "sessions": int(r["sessions"]),
                "messages": int(r["messages"]),
                "last_seen_at": r["last_seen_at"].isoformat() if r["last_seen_at"] else None,
            }
            for r in cur.fetchall()
        ]


def recent_sessions(limit: int = 15) -> list[dict[str, Any]]:
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            "SELECT s.session_id, s.user_id, s.started_at, s.last_at, s.message_count, "
            "(SELECT intent FROM chat_messages m WHERE m.session_id=s.session_id "
            " AND m.role='assistant' ORDER BY m.message_id DESC LIMIT 1) AS last_intent "
            "FROM chat_sessions s ORDER BY s.last_at DESC LIMIT %s",
            (limit,),
        )
        return [
            {
                "session_id": r["session_id"],
                "user_id": r["user_id"],
                "started_at": r["started_at"].isoformat() if r["started_at"] else None,
                "last_at": r["last_at"].isoformat() if r["last_at"] else None,
                "message_count": int(r["message_count"]),
                "last_intent": r["last_intent"],
            }
            for r in cur.fetchall()
        ]
