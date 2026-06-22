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

        # 활성 사용자 — DAU(1일)/WAU(7일)/MAU(30일). 같은 커서로 윈도우만 바꿔 집계.
        now = datetime.now()

        def _active_since(days: int) -> int:
            cur.execute(
                "SELECT COUNT(DISTINCT user_id) AS c FROM chat_messages WHERE created_at >= %s",
                (now - timedelta(days=days),),
            )
            return int(cur.fetchone()["c"])

        dau = _active_since(1)
        active_7d = _active_since(7)
        mau = _active_since(30)

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
        "dau": dau,
        "wau": active_7d,
        "mau": mau,
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
    """search_plan 에 등장한 도구(rdb/vec/graph)별 사용 횟수.

    search_plan 컬럼은 패널 JSON({graph, documents, panel, tools})이고, 도구 목록은
    그 안의 $.tools 배열에 있다(main.py 가 save_message 에 함께 보관). JSON_EXTRACT 로
    $.tools 를 꺼내 JSON_CONTAINS 한다. tools 키가 없는 과거 행은 NULL → 미집계.
    """
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        out = []
        for tool in ["rdb", "vec", "graph"]:
            cur.execute(
                "SELECT COUNT(*) AS c FROM chat_messages "
                "WHERE role='assistant' "
                "AND JSON_CONTAINS(JSON_EXTRACT(search_plan, '$.tools'), %s)",
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


# 응답 지연 히스토그램 버킷 경계(ms). 평균만 보면 꼬리(tail)가 가려져 분위수로 본다.
_LATENCY_BUCKETS: list[tuple[str, int, int | None]] = [
    ("<0.5s", 0, 500),
    ("0.5–1s", 500, 1000),
    ("1–2s", 1000, 2000),
    ("2–4s", 2000, 4000),
    ("4–8s", 4000, 8000),
    ("8s+", 8000, None),
]


def _percentile(sorted_vals: list[int], p: float) -> int | None:
    """정렬된 값에서 nearest-rank 분위수. 빈 리스트면 None."""
    if not sorted_vals:
        return None
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(idx, len(sorted_vals) - 1))]


def latency_stats() -> dict[str, Any]:
    """assistant 응답 지연 분포 — p50/p90/p95/p99 + 히스토그램 버킷.

    기존 chat_messages.latency_ms 만 사용(수집 추가 없음).
    """
    with mariadb_conn() as conn, conn.cursor(pymysql.cursors.DictCursor) as cur:
        cur.execute(
            "SELECT latency_ms FROM chat_messages "
            "WHERE role='assistant' AND latency_ms IS NOT NULL ORDER BY latency_ms"
        )
        vals = [int(r["latency_ms"]) for r in cur.fetchall()]

    buckets = [
        {
            "label": label,
            "count": sum(1 for v in vals if v >= lo and (hi is None or v < hi)),
        }
        for label, lo, hi in _LATENCY_BUCKETS
    ]
    return {
        "count": len(vals),
        "p50": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "p99": _percentile(vals, 99),
        "buckets": buckets,
    }
