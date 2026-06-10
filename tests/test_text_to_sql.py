from tool.text_to_sql import _extract_sql


def test_extract_plain_sql():
    assert _extract_sql("SELECT 1 FROM chunk_index") == "SELECT 1 FROM chunk_index"


def test_extract_strips_code_fence():
    raw = "```sql\nSELECT 1 FROM chunk_index\n```"
    assert _extract_sql(raw) == "SELECT 1 FROM chunk_index"


def test_extract_strips_prose_before_select():
    raw = "다음 쿼리를 쓰면 됩니다:\nSELECT corp_code FROM document_index"
    assert _extract_sql(raw) == "SELECT corp_code FROM document_index"


def test_extract_cuts_at_first_semicolon():
    raw = "SELECT 1; 추가 설명입니다"
    assert _extract_sql(raw) == "SELECT 1"


def test_extract_cuts_at_blank_line():
    raw = "SELECT title FROM news_raw\n\n이 쿼리는 뉴스 제목을 가져옵니다."
    assert _extract_sql(raw) == "SELECT title FROM news_raw"


def test_extract_allows_with_cte():
    raw = "WITH t AS (SELECT 1) SELECT * FROM t"
    assert _extract_sql(raw) == "WITH t AS (SELECT 1) SELECT * FROM t"
