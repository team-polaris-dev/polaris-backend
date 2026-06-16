# tool/rdb_client.py — MariaDB 커넥션 + RDB 스키마 설명 + SELECT-only 안전 실행기
#                       + DB introspection / 결정론 resolve 헬퍼
from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager

import pymysql
import sqlglot
from sqlglot import exp

MAX_ROWS = 50

# sqlglot 은 미지원 구문(CALL 등)을 Command 로 폴백하며 warning 을 남긴다 — 임의의
# LLM SQL 을 검증하는 용도라 이 경고는 정상 동작(거부 대상)이므로 로그 소음만 줄인다.
logging.getLogger("sqlglot").setLevel(logging.ERROR)


def _connect() -> "pymysql.connections.Connection":
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
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()

# 서버 파일 읽기 함수 차단 — 변경/DDL/CALL/다중문/INTO 파일쓰기는 AST 구조
# (읽기 쿼리 루트 여부)에서 자연히 걸러지므로 키워드 블록리스트가 필요 없다.
_FILE_FUNCS = {"load_file", "load_data"}


def _strip(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def is_safe_select(sql: str) -> bool:
    """단일 읽기 쿼리(SELECT/CTE/UNION)인지 sqlglot AST 로 검증한다.

    정규식 블록리스트 대신 구문 트리를 본다 — 'updated_at' 같은 컬럼명이
    'UPDATE' 로 오탐되지 않고, 다중문·DML·DDL·CALL·INTO OUTFILE·LOAD DATA 는
    파싱 실패 또는 루트 노드 타입(읽기 쿼리 아님)에서 걸러진다.
    파싱 불가/예외는 fail-closed(거부)로 처리한다.
    """
    s = _strip(sql)
    if not s:
        return False
    try:
        statements = [st for st in sqlglot.parse(s, read="mysql") if st is not None]
    except Exception:
        return False  # 파싱 실패 = 거부 (fail-closed)
    if len(statements) != 1:  # 다중 문장 차단
        return False
    root = statements[0]
    if not isinstance(root, (exp.Query, exp.Subquery)):  # 읽기 쿼리만 (DML/DDL/CALL 등 제외)
        return False
    if list(root.find_all(exp.Into)):  # SELECT ... INTO OUTFILE/DUMPFILE/@var 차단
        return False
    if any(fn.name.lower() in _FILE_FUNCS for fn in root.find_all(exp.Anonymous)):
        return False
    return True


def enforce_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    """LIMIT 이 없으면 부여하고, 상한 초과면 클램프한다(AST 기반).

    콤마/OFFSET 형태(`LIMIT 10, 20`)도 sqlglot 이 count/offset 으로 분해하므로
    정규식과 달리 안전하게 클램프된다. 상한 이내면 원본을 그대로 둬 불필요한
    재생성을 피한다. (행 수 상한은 execute_sql_query 가 fetch 후 슬라이스로도 최종 보장)
    """
    s = _strip(sql)
    try:
        root = sqlglot.parse_one(s, read="mysql")
    except Exception:
        return s
    limit = root.args.get("limit")
    if limit is None:
        return root.limit(max_rows).sql(dialect="mysql")
    try:
        current = int(limit.expression.sql())
    except (ValueError, AttributeError):
        return s  # 파라미터·표현식 LIMIT 은 건드리지 않음
    if current > max_rows:
        return root.limit(max_rows).sql(dialect="mysql")
    return s


# 회사/doc_type 하드코딩을 DB introspection 으로 대체하기 위한 폴백 상수
# (DB 미가용 시에만 사용 — 테스트/eval 무중단 보장).
_CORP_CODES_FALLBACK = "삼성전자=00126380, SK하이닉스=00164779, 한미반도체=00161383"
_DOC_TYPE_EXAMPLES_FALLBACK = (
    "'임원ㆍ주요주주특정증권등소유상황보고서', '기업설명회(IR)개최(안내공시)',\n"
    "     '현금ㆍ현물배당결정', '연결재무제표기준영업(잠정)실적(공정공시)',\n"
    "     '주식등의대량보유상황보고서(일반)', '주요사항보고서(자기주식처분결정)'"
)


def _format_corp_codes() -> str:
    """회사 corp_code 안내 문자열 — DB introspection 우선, 미가용 시 폴백.

    코퍼스가 3사에서 수십 개로 늘어 하드코딩이 한계다. 실제 적재된 회사를
    전부 노출한다(이름순 정렬).
    """
    mapping = get_corp_name_to_code()
    if not mapping:
        return _CORP_CODES_FALLBACK
    return ", ".join(f"{name}={code}" for name, code in sorted(mapping.items()))


def _format_doc_type_examples(limit: int = 12) -> str:
    """doc_type 예시 문자열 — DB introspection 우선, 미가용 시 폴백.

    손으로 적던 예시 대신 실제 적재값을 노출한다. 'ㆍ'(U+318D) 포함 값을 우선
    보여 특수문자 패턴을 학습시키고, 나머지로 채운다. 이는 '예시'이며 목록에
    없는 값은 규칙 4의 LIKE 우회로 처리되므로 부분 노출이어도 안전하다.
    """
    values = get_doc_type_values()
    if not values:
        return _DOC_TYPE_EXAMPLES_FALLBACK
    special = sorted(v for v in values if "ㆍ" in v)
    others = sorted(v for v in values if "ㆍ" not in v)
    picked = (special + others)[:limit]
    return ", ".join(f"'{v}'" for v in picked)


def get_schema_prompt() -> str:
    """LLM 그라운딩용 큐레이션 스키마 설명.

    SSOT(docs/DBdocs/디비설계.md)는 8개 테이블·run_id 컬럼을 포함한
    설계를 문서화하지만, 현재 적재된 덤프(dump/maria.sql)의 실제 테이블은
    이와 다르다 — chunk_summary/document_unified/news_raw/active_run_manifest
    4개 테이블 자체가 없고, 검색 대상 테이블에도 run_id/key_facts 컬럼이 없다
    (DESCRIBE로 직접 확인함). SSOT를 그대로 옮기면 LLM이 존재하지 않는
    테이블·컬럼을 조회해 매번 실패하므로, 실제 적재된 컬럼만 기술한다.

    노출 테이블은 chunk_index/document_index/dart_raw_index 에 더해
    fin_metric(재무지표)까지 4개다. fin_metric 은 재무 수치 질의의 핵심이라 추가했다
    (이전엔 누락돼 재무 질문이 chunk 본문만 뒤졌다 — RDB_KNOWN_ISSUES A1-b).
    extraction_provenance(그래프 추출 출처/confidence)는 KG 빌드 내부 계보라
    자연어 질의 표면이 없고 graph 에이전트 영역과 겹쳐 Text-to-SQL 엔 노출하지 않는다.

    회사 corp_code 와 doc_type 예시는 하드코딩하지 않고 DB introspection
    (_format_corp_codes / _format_doc_type_examples)으로 채운다. DB 미가용 시에만
    폴백 상수를 쓴다. 덤프 스키마(테이블/컬럼)가 바뀌면 이 함수의 구조 설명만 갱신.
    """
    corp_codes = _format_corp_codes()
    doc_type_examples = _format_doc_type_examples()
    return f"""\
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
[fin_metric] 재무지표(매출·영업이익·순이익·자산 등 숫자값). PK(metric_id)
  - corp_code, rcept_no, bsns_year(회계연도 SMALLINT, 예: 2024)
  - reprt_code(보고서 구분: '11011'=사업/연간, '11012'=반기, '11013'=1분기, '11014'=3분기)
  - account_id(IFRS/DART 택소노미 코드 — 한국어 계정명이 아니라 아래 매핑 사용)
  - value(DECIMAL 금액), unit('KRW'), fs_div('CFS'=연결, 'OFS'=별도)

회사 corp_code: {corp_codes}.

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
   실제 doc_type 예시(DB에서 추출, 전체 목록 아님):
     {doc_type_examples}
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
8) 재무 수치(매출·영업이익·순이익·자산·부채·자본 등 "금액·숫자")를 물으면
   document_index/chunk_index 가 아니라 [fin_metric] 을 조회한다.
   account_id 는 한국어가 아니라 IFRS/DART 코드이므로 아래 매핑으로 변환해 매칭한다:
     매출/매출액/수익        → account_id='ifrs-full_Revenue'
     영업이익               → account_id='dart_OperatingIncomeLoss'
     당기순이익/순이익       → account_id='ifrs-full_ProfitLoss'
     법인세차감전순이익      → account_id='ifrs-full_ProfitLossBeforeTax'
     매출총이익             → account_id='ifrs-full_GrossProfit'
     자산총계/총자산        → account_id='ifrs-full_Assets'
     부채총계/총부채        → account_id='ifrs-full_Liabilities'
     자본총계/자기자본       → account_id='ifrs-full_Equity'
   매핑에 없는 계정을 물으면 account_id 를 지어내지 말고 corp_code+bsns_year 로만
   조회하거나 사용자에게 어떤 계정인지 되묻는다(틀린 코드는 0건이 된다).
   ※ 같은 계정도 fs_div(연결 CFS/별도 OFS)·reprt_code(연간/분기)·bsns_year 별로
     여러 행이 있다. 사용자가 따로 말하지 않으면 연결·연간 기준
     (fs_div='CFS' AND reprt_code='11011')으로 한정하고 bsns_year 도 명시한다.
   예) '삼성전자 2024년 매출' → SELECT d.corp_name, f.value FROM fin_metric f
       LEFT JOIN document_index d ON d.rcept_no=f.rcept_no
       WHERE f.corp_code='00126380' AND f.account_id='ifrs-full_Revenue'
         AND f.bsns_year=2024 AND f.fs_div='CFS' AND f.reprt_code='11011'
   (fin_metric 엔 corp_name 이 없으니 회사명이 필요하면 document_index 와 LEFT JOIN 한다)
"""


def execute_sql_query(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """실제 MariaDB에 SELECT 구문을 실행합니다."""
    print(f"🛠️ [MariaDB]  검색 시뮬레이션 중: {sql}")
    if not is_safe_select(sql):
        return {
            "ok": False,
            "rows": [],
            "error": "안전하지 않은 SQL — SELECT 단일 문장만 허용됩니다.",
            "sql": sql,
        }

    # doc_type 리터럴을 실제 DB 값으로 정규화(특수문자 '·'→'ㆍ' 등). 매칭 없으면 no-op.
    sql = normalize_doc_type_literals(sql)
    safe_sql = enforce_limit(sql, max_rows)
    
    try:
        with mariadb_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(safe_sql)
                rows = cur.fetchall()
                
        # SQL LIMIT 형태와 무관하게 행 수 상한을 최종 보장
        return {"ok": True, "rows": list(rows)[:max_rows], "error": None, "sql": safe_sql}

    except Exception as e:
        return {"ok": False, "rows": [], "error": str(e), "sql": safe_sql}


# ======================================================================
# DB introspection — get_schema_prompt 하드코딩 제거용 (회사 코드/doc_type 실제값)
# ----------------------------------------------------------------------
# 회사명→코드, doc_type 실제값을 DB 에서 직접 끌어와 프롬프트에 채운다(하드코딩 없이).
# 질문 구조화(planner/QueryEnvelope)는 상류 노드 담당이며 여기서 하지 않는다.
# 프로세스 캐시(read-only) — 무효화는 프로세스 재시작.
# ======================================================================


# ---------------------------------------------------------------- introspection
_DOC_TYPE_CACHE: list[str] | None = None
_CORP_MAP_CACHE: dict[str, str] | None = None
_INTROSPECT_LOCK = threading.Lock()


def get_doc_type_values() -> list[str]:
    """document_index 의 실제 doc_type DISTINCT 목록(프로세스 캐시, read-only).

    손으로 적은 예시 대신 실제 적재값을 쓰기 위함. DB 미가용 시 [] 로 degrade.
    """
    global _DOC_TYPE_CACHE
    if _DOC_TYPE_CACHE is not None:
        return _DOC_TYPE_CACHE
    with _INTROSPECT_LOCK:
        if _DOC_TYPE_CACHE is not None:
            return _DOC_TYPE_CACHE
        try:
            with mariadb_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT doc_type FROM document_index WHERE doc_type IS NOT NULL"
                )
                rows = cur.fetchall()
            _DOC_TYPE_CACHE = [str(r["doc_type"]) for r in rows if r.get("doc_type")]
        except Exception:
            _DOC_TYPE_CACHE = []
        return _DOC_TYPE_CACHE


def get_corp_name_to_code() -> dict[str, str]:
    """회사명 → corp_code(8자리) 매핑(프로세스 캐시, read-only).

    document_index 기준. DB 미가용 시 {} 로 degrade.
    """
    global _CORP_MAP_CACHE
    if _CORP_MAP_CACHE is not None:
        return _CORP_MAP_CACHE
    with _INTROSPECT_LOCK:
        if _CORP_MAP_CACHE is not None:
            return _CORP_MAP_CACHE
        mapping: dict[str, str] = {}
        try:
            with mariadb_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT corp_name, corp_code FROM document_index "
                    "WHERE corp_name IS NOT NULL AND corp_code IS NOT NULL"
                )
                for r in cur.fetchall():
                    name = str(r.get("corp_name") or "").strip()
                    code = str(r.get("corp_code") or "").strip().zfill(8)
                    if name and code:
                        mapping.setdefault(name, code)
        except Exception:
            mapping = {}
        _CORP_MAP_CACHE = mapping
        return _CORP_MAP_CACHE


# ---------------------------------------------------------------- doc_type 정규화
_DOCTYPE_NORM_RE = re.compile(r"[^가-힣a-zA-Z0-9]")


def _normalize_doctype(s: str) -> str:
    """doc_type 비교용 정규화 — 특수문자/공백 제거. 'ㆍ'(U+318D)·중점 '·'·공백 차이를 흡수."""
    return _DOCTYPE_NORM_RE.sub("", s or "")


def normalize_doc_type_literals(sql: str) -> str:
    """생성된 SQL 의 `doc_type = '리터럴'` 을 실제 DB 값으로 정규화(결정론).

    LLM 이 `현금·현물배당`(일반점)·공백 등 표기로 doc_type 정확매칭을 쓰면 0건이 된다.
    실제 DB 값(`현금ㆍ현물배당결정`, U+318D)과 **정규화(특수문자/공백 제거) 동일**일
    때만 리터럴을 실제값으로 치환한다. 부분일치는 쓰지 않아(의미 변형 위험) 멀쩡한
    쿼리를 깨지 않는다. `doc_type LIKE` 는 의도적 fuzzy 라 건드리지 않는다.

    프롬프트를 손대지 않고 #23(특수문자 매칭)을 결정론적으로 해결하는 실행계층 보정.
    DB 미가용/파싱 실패/매칭 없음 → 원본 SQL 그대로 반환(no-op).
    """
    values = get_doc_type_values()
    if not values:
        return sql
    norm_to_val: dict[str, str] = {}
    for v in values:
        norm_to_val.setdefault(_normalize_doctype(v), v)

    try:
        root = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return sql

    changed = False
    for eq in root.find_all(exp.EQ):
        # 양변 중 하나가 doc_type 컬럼, 다른 하나가 문자열 리터럴인 경우만
        for col_side, lit_side in ((eq.left, eq.right), (eq.right, eq.left)):
            if (
                isinstance(col_side, exp.Column)
                and col_side.name.lower() == "doc_type"
                and isinstance(lit_side, exp.Literal)
                and lit_side.is_string
            ):
                real = norm_to_val.get(_normalize_doctype(lit_side.this))
                if real and real != lit_side.this:
                    lit_side.set("this", real)
                    changed = True
                break

    return root.sql(dialect="mysql") if changed else sql
