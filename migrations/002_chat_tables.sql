-- 002_chat_tables.sql — 챗봇 사용자/세션/메시지 + 통계용
-- 실행: services.chat_logging.init_chat_tables() 가 lifespan 에서 호출.
-- 본 파일은 수동 적용용 사본(레퍼런스).
--
-- 주의: 별도 인증 시스템이 도입되면 chat_users 를 그 users 테이블과 통합(또는 FK)할 수 있다.
-- 지금은 /api/chat 의 user_id 문자열을 그대로 PK 로 쓴다(스키마 독립).

CREATE TABLE IF NOT EXISTS chat_users (
  user_id       VARCHAR(64)  PRIMARY KEY,
  display_name  VARCHAR(120),
  first_seen_at DATETIME(3)  NOT NULL,
  last_seen_at  DATETIME(3)  NOT NULL,
  meta          JSON
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS chat_sessions (
  session_id    VARCHAR(64)  PRIMARY KEY,   -- LangGraph thread_id
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
  role          VARCHAR(16)  NOT NULL,        -- 'user' | 'assistant'
  content       MEDIUMTEXT,
  intent        VARCHAR(64),                  -- assistant 턴: 분류된 의도
  search_plan   JSON,                         -- assistant 턴: ['rdb','vec','graph']
  is_sufficient TINYINT,                      -- assistant 턴: RAG 충분 여부
  retry_count   INT,                          -- assistant 턴: 반성 재시도 횟수
  latency_ms    INT,                          -- assistant 턴: 응답 지연
  created_at    DATETIME(3)  NOT NULL,
  INDEX idx_session (session_id),
  INDEX idx_user_time (user_id, created_at),
  INDEX idx_intent (intent),
  INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
