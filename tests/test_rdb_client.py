from tool.rdb_client import enforce_limit, is_safe_select


def test_allows_plain_select():
    assert is_safe_select("SELECT corp_code FROM chunk_index") is True


def test_allows_select_with_trailing_semicolon():
    assert is_safe_select("SELECT 1;") is True


def test_rejects_insert():
    assert is_safe_select("INSERT INTO chunk_index VALUES (1)") is False


def test_rejects_drop():
    assert is_safe_select("SELECT 1; DROP TABLE chunk_index") is False


def test_rejects_multi_statement():
    assert is_safe_select("SELECT 1; SELECT 2") is False


def test_rejects_update_keyword():
    assert is_safe_select("UPDATE chunk_index SET x=1") is False


def test_enforce_limit_adds_when_missing():
    assert enforce_limit("SELECT 1") == "SELECT 1 LIMIT 50"


def test_enforce_limit_clamps_when_too_large():
    assert enforce_limit("SELECT 1 LIMIT 9999") == "SELECT 1 LIMIT 50"


def test_enforce_limit_keeps_small_limit():
    assert enforce_limit("SELECT 1 LIMIT 10") == "SELECT 1 LIMIT 10"


def test_allows_with_cte():
    assert is_safe_select("WITH t AS (SELECT 1) SELECT * FROM t") is True


def test_allows_replace_function():
    assert is_safe_select("SELECT REPLACE(title,'a','b') FROM document_index") is True


def test_rejects_select_into_outfile():
    assert is_safe_select("SELECT * FROM chunk_index INTO OUTFILE '/tmp/x'") is False


def test_enforce_limit_does_not_corrupt_offset():
    sql = "SELECT 1 LIMIT 10 OFFSET 5"
    assert enforce_limit(sql) == sql


def test_enforce_limit_does_not_corrupt_comma_form():
    sql = "SELECT 1 LIMIT 5, 10"
    assert enforce_limit(sql) == sql
