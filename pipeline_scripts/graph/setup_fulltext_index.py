"""entity_fulltext FULLTEXT INDEX 1회성 DDL + APOC 가용성 점검.

대상: Organization.name/er_name, Person.name, Product.name, Technology.name
analyzer: cjk (한국어 부분일치/오타 보강. 플러그인 불필요)

실행: python -m pipeline_scripts.graph.setup_fulltext_index
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tool.graph_client import neo4j_driver


INDEX_DDL = """
CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
FOR (n:Organization|Person|Product|Technology)
ON EACH [n.name, n.er_name]
OPTIONS {indexConfig: {
  `fulltext.analyzer`: 'cjk',
  `fulltext.eventually_consistent`: false
}}
"""


def main() -> int:
    with neo4j_driver.session() as s:
        s.run(INDEX_DDL).consume()
        idx = s.run(
            "SHOW INDEXES YIELD name, type, state "
            "WHERE name = 'entity_fulltext' "
            "RETURN name, type, state"
        ).single()
        if idx is None:
            print("[error] entity_fulltext index missing after CREATE", file=sys.stderr)
            return 1
        print(f"[ok] index={idx['name']} type={idx['type']} state={idx['state']}")

        try:
            apoc = s.run("RETURN apoc.version() AS v").single()
            print(f"[ok] APOC available: {apoc['v']}")
        except Exception as e:
            print(f"[warn] APOC not available — fallback subgraph will use variable-length Cypher: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
