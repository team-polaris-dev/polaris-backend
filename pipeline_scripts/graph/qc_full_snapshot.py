"""03_neo4j.md §7 전체 QC 스냅샷 — 현재 그래프 노이즈 전수 측정(읽기전용).

결과를 graph/qc_full_snapshot.json(UTF-8)에 기록 + 콘솔 요약.
실행: cd db && uv run python graph/qc_full_snapshot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from db import neo4j_driver, mariadb_conn  # noqa: E402

GENERIC = ["신제품", "생산품", "상품", "제품", "공정용 화학재료", "Set제품",
           "이차전지용 전자재료", "소재", "부품", "장비", "계", "소계", "합계"]
COMPANY_HINTS = ["CO., LTD", "CO.,LTD", "CORPORATION", "주식회사", "(주)", "Inc.", "Ltd.",
                 "GmbH", "LLC", "삼성디스플레이", "HONGKONG"]


def main() -> None:
    drv = neo4j_driver()
    out: dict = {}
    with drv.session() as s:
        def val(q, **p):
            r = s.run(q, **p).single()
            return r[0] if r else None

        def rows(q, **p):
            return s.run(q, **p).data()

        # ① self-loop (전 관계)
        out["self_loops"] = rows(
            "MATCH (a)-[r]->(a) RETURN type(r) AS rel, count(r) AS n ORDER BY n DESC")

        # ② SUPPLIES_TO 양방향
        out["supplies_bidirectional_total"] = val(
            "MATCH (a:Organization)-[:SUPPLIES_TO]->(b:Organization) "
            "MATCH (b)-[:SUPPLIES_TO]->(a) WHERE elementId(a)<elementId(b) RETURN count(*)")
        out["supplies_bidir_both_domestic"] = val(
            "MATCH (a:Organization)-[:SUPPLIES_TO]->(b:Organization) "
            "MATCH (b)-[:SUPPLIES_TO]->(a) "
            "WHERE elementId(a)<elementId(b) AND a.corp_code IS NOT NULL AND b.corp_code IS NOT NULL "
            "RETURN count(*)")

        # ③ 고아 Product/Tech (관계 0)
        out["orphan_product"] = val(
            "MATCH (n:Product) WHERE NOT (n)--() RETURN count(n)")
        out["orphan_tech"] = val(
            "MATCH (n:Technology) WHERE NOT (n)--() RETURN count(n)")

        # ④ 회사명이 Product/Tech로 새어들어옴
        out["company_in_producttech"] = rows(
            "MATCH (n) WHERE (n:Product OR n:Technology) AND "
            "any(h IN $hints WHERE toUpper(n.name) CONTAINS toUpper(h)) "
            "OPTIONAL MATCH (n)-[r]-() RETURN labels(n) AS labels, n.name AS name, count(r) AS rels "
            "ORDER BY rels DESC LIMIT 50", hints=COMPANY_HINTS)

        # ⑤ 일반어/1글자/줄바꿈 타겟
        out["generic_nodes"] = rows(
            "MATCH (n) WHERE (n:Product OR n:Technology) AND n.name IN $g "
            "RETURN labels(n) AS labels, n.name AS name", g=GENERIC)
        out["short_nodes"] = rows(
            "MATCH (n) WHERE (n:Product OR n:Technology) AND size(trim(n.name))<=1 "
            "RETURN labels(n) AS labels, n.name AS name LIMIT 30")
        out["newline_nodes"] = val(
            "MATCH (n) WHERE (n:Product OR n:Technology) AND (n.name CONTAINS '\\n' OR n.name CONTAINS '\\r') RETURN count(n)")

        # ⑥ 깨진문자(mojibake) 노드
        out["mojibake_nodes"] = val(
            "MATCH (n) WHERE (n:Product OR n:Technology OR n:Organization) AND n.name CONTAINS '?' "
            "RETURN count(n)")

        # ⑦ 근거 chunk_id 끊김 (추출 엣지가 가리키는 Chunk 노드 부재)
        out["broken_provenance_chunks"] = val(
            "MATCH ()-[r]->() WHERE r.extracted_by='claude' AND r.chunk_id IS NOT NULL "
            "AND NOT EXISTS { MATCH (c:Chunk {chunk_id:r.chunk_id}) } RETURN count(r)")

        # 노드/엣지 규모
        out["edge_counts"] = rows(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS n ORDER BY n DESC")
        out["node_counts"] = rows(
            "MATCH (n) UNWIND labels(n) AS l RETURN l AS label, count(*) AS n ORDER BY n DESC")
    drv.close()

    # MariaDB extraction_provenance 원장 행수
    try:
        conn = mariadb_conn()
        cur = conn.cursor()
        cur.execute("SELECT extracted_by, COUNT(*) FROM extraction_provenance GROUP BY extracted_by")
        out["provenance_ledger"] = {(by or "NULL"): n for by, n in cur.fetchall()}
        cur.close(); conn.close()
    except Exception as e:
        out["provenance_ledger"] = f"ERR {e}"

    (HERE / "qc_full_snapshot.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("=== §7 QC 스냅샷 ===")
    print("self_loops:", out["self_loops"] or "0 (통과)")
    print("SUPPLIES_TO 양방향 total:", out["supplies_bidirectional_total"],
          "| 둘다 적재사(QC6 정식대상):", out["supplies_bidir_both_domestic"])
    print("고아 Product:", out["orphan_product"], "| 고아 Tech:", out["orphan_tech"])
    print("회사명 Product/Tech 유입:", len(out["company_in_producttech"]))
    for r in out["company_in_producttech"][:20]:
        print("   ", r["labels"], r["name"], "rels=", r["rels"])
    print("일반어 노드:", len(out["generic_nodes"]), [r["name"] for r in out["generic_nodes"]])
    print("1글자 노드:", len(out["short_nodes"]), [r["name"] for r in out["short_nodes"]])
    print("줄바꿈 노드:", out["newline_nodes"], "| mojibake:", out["mojibake_nodes"])
    print("근거 chunk 끊김:", out["broken_provenance_chunks"])
    print("provenance 원장:", out["provenance_ledger"])
    print("\n전체 → graph/qc_full_snapshot.json")


if __name__ == "__main__":
    main()
