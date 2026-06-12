-- 001_pipeline_tables.sql — 관리자 콘솔 운영 메타 테이블 2개
-- 실행: services.pipeline_jobs.init_pipeline_tables() 가 lifespan 에서 호출.
-- 본 파일은 수동 적용용 사본(레퍼런스).

CREATE TABLE IF NOT EXISTS pipeline_jobs (
  job_id        CHAR(36)        PRIMARY KEY,
  state         VARCHAR(16)     NOT NULL,
  corp_codes    JSON            NOT NULL,
  config        JSON            NOT NULL,
  label         VARCHAR(200),
  pid           INT,
  created_at    DATETIME(3)     NOT NULL,
  updated_at    DATETIME(3)     NOT NULL,
  INDEX idx_state_created (state, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS pipeline_step_runs (
  run_id        BIGINT          AUTO_INCREMENT PRIMARY KEY,
  job_id        CHAR(36)        NOT NULL,
  corp_code     CHAR(8)         NOT NULL,
  step_id       VARCHAR(32)     NOT NULL,
  state         VARCHAR(16)     NOT NULL,
  progress      DOUBLE          NOT NULL DEFAULT 0,
  counters      JSON,
  log_path      VARCHAR(500),
  error         TEXT,
  started_at    DATETIME(3),
  ended_at      DATETIME(3),
  INDEX idx_job_step (job_id, corp_code, step_id),
  CONSTRAINT fk_step_job FOREIGN KEY (job_id) REFERENCES pipeline_jobs(job_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
