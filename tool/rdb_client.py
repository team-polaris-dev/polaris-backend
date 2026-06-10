# tool/rdb_client.py — RDB 스키마 설명 + SELECT-only 안전 실행기
from __future__ import annotations

import re

from config.db import mariadb_conn

MAX_ROWS = 50

# SELECT 외 변경/DDL 키워드 + 파일 쓰기(INTO OUTFILE/DUMPFILE) 차단 (이중 방어).
# REPLACE/MERGE 는 SELECT 내 함수명과 충돌하므로 제외 — DML 문장은 어차피
# "SELECT/WITH 로 시작" 검사에서 걸러진다.
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"CALL|LOCK|RENAME|HANDLER)\b"
    r"|\bINTO\s+(OUTFILE|DUMPFILE)\b",
    re.IGNORECASE,
)


def _strip(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def is_safe_select(sql: str) -> bool:
    """단일 읽기 쿼리(SELECT/WITH)이고 변경·파일쓰기 키워드가 없으면 True."""
    s = _strip(sql)
    if not s:
        return False
    if ";" in s:  # 다중 문장 차단 (끝 세미콜론은 _strip 이 이미 제거)
        return False
    if not re.match(r"(?is)^\s*(SELECT|WITH)\b", s):  # CTE(WITH) 허용
        return False
    if _FORBIDDEN.search(s):
        return False
    return True


def enforce_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    """끝에 단순 LIMIT N 이면 상한으로 클램프, LIMIT 이 없으면 부여.

    LIMIT 이 OFFSET/콤마 형태로 이미 있으면 SQL 을 훼손하지 않고 그대로 둔다
    (행 수 상한은 execute_sql_query 가 fetch 후 슬라이스로 최종 보장).
    """
    s = _strip(sql)
    m = re.search(r"(?is)\blimit\s+(\d+)\s*$", s)
    if m:
        if int(m.group(1)) > max_rows:
            s = re.sub(r"(?is)\blimit\s+\d+\s*$", f"LIMIT {max_rows}", s)
        return s
    if re.search(r"(?is)\blimit\b", s):  # OFFSET/콤마 등 복합 LIMIT → 건드리지 않음
        return s
    return f"{s} LIMIT {max_rows}"


def get_schema_prompt() -> str:
    """LLM 그라운딩용 큐레이션 스키마 설명.

    SSOT(docs/DBdocs/디비설계.md)는 8개 테이블·run_id 컬럼을 포함한
    설계를 문서화하지만, 현재 적재된 덤프(dump/maria.sql)의 실제 테이블은
    이와 다르다 — chunk_summary/document_unified/news_raw/active_run_manifest
    4개 테이블 자체가 없고, 남은 3개 테이블에도 run_id/key_facts 컬럼이 없다
    (DESCRIBE로 직접 확인함). SSOT를 그대로 옮기면 LLM이 존재하지 않는
    테이블·컬럼을 조회해 매번 실패하므로, 실제 적재된 컬럼만 기술한다.
    덤프가 SSOT 스키마로 갱신되면 이 함수도 함께 갱신한다.
    """
    return """\
MariaDB (MySQL 호환). 한국 반도체 기업 공시·뉴스 데이터. 아래 테이블만 사용한다.
(주의: run_id, active_run_manifest, key_facts 등은 이 DB에 존재하지 않는다 —
 절대 조회하지 말 것)

[chunk_index] 청크 메타 + 본문 텍스트. PK(chunk_id)
  - corp_code(8자리 회사코드), rcept_no(14자리 공시접수번호)
  - chunk_type('table_nl'|'text_micro'|'text_macro'), section_path(섹션 경로)
  - embedding_text(청크 원문 텍스트), token_count, ingest_status
[document_index] 공시문서 메타+요약. PK(rcept_no)
  - corp_code, corp_name, doc_type, date(공시일), title, summary_short
[dart_raw_index] DART 원본 JSON. PK(corp_code, endpoint, hash8)
  - rcept_no, body_json(LONGTEXT), status, collected_at

회사 corp_code: 삼성전자=00126380, SK하이닉스=00164779, 한미반도체=00161383.

## 질의 작성 규칙 (정확도 핵심)
1) 답에 "회사"가 등장하면 corp_code(숫자) 말고 corp_name(회사명)을 SELECT 한다.
   비교/집계도 GROUP BY 는 corp_code 로 하되 SELECT 절에 corp_name 을 반드시 포함한다.
   예) 공시 많은 회사: SELECT corp_name, COUNT(*) c FROM document_index
       WHERE corp_code IN (...) GROUP BY corp_code, corp_name ORDER BY c DESC LIMIT 1
2) 제목·주제어로 문서를 찾을 땐 doc_type='...' 정확매칭보다 title LIKE '%키워드%' 를
   우선한다. 키워드가 doc_type/title 어디에 있을지 모르면 (title LIKE '%키워드%'
   OR doc_type LIKE '%키워드%') 로 넓게 잡는다.
   특히 '~관련/~관해서' 처럼 넓은 의미를 물으면 반드시 title LIKE 핵심어로 넓게 잡는다.
   doc_type 정확매칭은 특정 공시 종류명을 콕 집어 물을 때만 쓴다.
   예) '기업설명회(IR) 관련 공시' → title LIKE '%기업설명회%' (doc_type 정확매칭 아님)
       '기업설명회(IR)개최 공시' → doc_type='기업설명회(IR)개최(안내공시)' (종류명 지정)
3) 정기보고서류(사업보고서·반기보고서·분기보고서·감사보고서)는 doc_type 이 아니라
   title 에 들어있다. 예: title LIKE '%분기보고서%'. doc_type='분기보고서' 는 0건이다.
4) doc_type 값에는 가운뎃점 'ㆍ'(U+318D) 가 그대로 쓰인다(일반 점 '·'·중점 아님).
   직접 매칭이 필요하면 아래 실제 값을 글자 그대로 복사해 쓰고, 애매하면 title LIKE 로 우회한다.
   ※ 사용자는 중점 '·' 나 공백을 넣어 쓰기 쉬운데, DB 실제값은 가운뎃점 'ㆍ' + 공백 없음
     이라 그대로 정확매칭하면 0건이 된다. 이럴 땐 둘 중 하나로 처리한다:
       ① 아래 예시 목록에 있는 값이면 글자 그대로(ㆍ 포함, 공백 없이) 복사해 정확매칭
       ② 특수문자·공백을 제거한 연속 핵심어로 LIKE
     예) 사용자가 '현금·현물배당 결정' 이라 써도
         → doc_type='현금ㆍ현물배당결정'  또는  title LIKE '%현물배당결정%'
   실제 doc_type 예시:
     '임원ㆍ주요주주특정증권등소유상황보고서', '기업설명회(IR)개최(안내공시)',
     '현금ㆍ현물배당결정', '연결재무제표기준영업(잠정)실적(공정공시)',
     '주식등의대량보유상황보고서(일반)', '주요사항보고서(자기주식처분결정)'
5) 청크 "원문/본문"을 직접 보여달라는 게 아니라 "공시가 있는지/몇 건인지/목록"을 묻는
   질문은 chunk_index(본문)가 아니라 document_index(공시 메타)를 조회한다.
6) 답의 형태(개수 vs 목록)는 질문의 표현으로 정한다:
   - '몇 건/몇 번/몇 개/몇 가지/개수' 같은 수량 표현이 있으면 행 목록이 아니라 COUNT(*) 로 답한다.
   - '보여줘/목록/어떤 게 있는지/제목' 처럼 나열을 요청하면 행을 SELECT 한다.
     사용자가 "제목/종류가 어떤 게 있는지"처럼 고유 항목 예시를 묻는 경우에는
     중복 행을 반복하지 말고 DISTINCT title 또는 DISTINCT doc_type 을 우선한다.
   - 둘 다 있으면(예: '제목 몇 개만 보여줘') 나열을 우선한다.
7) 날짜 표현은 경계 포함 여부를 엄격히 지킨다.
   - 'YYYY-MM-DD 이후/부터'는 해당 날짜를 포함하므로 date >= 'YYYY-MM-DD' 로 쓴다.
   - 'YYYY-MM-DD 초과/뒤'처럼 명시적으로 제외하는 표현일 때만 date > 'YYYY-MM-DD' 를 쓴다.
"""


def execute_sql_query(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """SELECT 만 안전 실행. 반환: {"ok", "rows", "error", "sql"}."""
    if not is_safe_select(sql):
        return {
            "ok": False,
            "rows": [],
            "error": "안전하지 않은 SQL — SELECT 단일 문장만 허용됩니다.",
            "sql": sql,
        }
    safe_sql = enforce_limit(sql, max_rows)
    try:
        with mariadb_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(safe_sql)
                rows = cur.fetchall()
        # SQL LIMIT 형태와 무관하게 행 수 상한을 최종 보장 (OFFSET/콤마 LIMIT 대비)
        return {"ok": True, "rows": list(rows)[:max_rows], "error": None, "sql": safe_sql}
    except Exception as e:
        return {"ok": False, "rows": [], "error": str(e), "sql": safe_sql}
