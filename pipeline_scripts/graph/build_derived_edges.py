"""온톨로지 4단계 파생엣지 배치 생성 (forward-chaining materialization).

03_neo4j.md §2-3 의 파생관계를 정형 엣지로부터 사전 추론해 MERGE.
GraphRAG 런타임은 이 엣지를 읽기만 한다(생성 금지 — CLAUDE.md 5번).

생성 대상:
  - CONTROLS_INDIRECTLY  : IS_MAJOR_SHAREHOLDER_OF*2.. 전 구간 qota_rt>=50 → 간접지배
      ⚠ 실측(2026-06): 현재 데이터로 0건 (지분율 50%↑이 337개 중 5개뿐, 종속 연쇄 0).
        지분 데이터 다단계 확장 전에는 생성 안 됨 — 향후 데이터 늘면 자동 활성.
  - INTERLOCKING_DIRECTORATE : 같은 이름 Person 이 두 회사 EXECUTIVE_OF → 인적연결
                               (동명이인 위험 → confidence='low'). 실측 66건 생성.

멱등: --rebuild 시 derived_by='rule' 엣지 전부 삭제 후 재생성.

실행 (db/graph 디렉토리에서):
  python build_derived_edges.py --dry-run   # 생성될 개수만
  python build_derived_edges.py --rebuild   # 삭제 후 재생성
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -m 실행 시 db 모듈 해석

from db import neo4j_driver


DELETE_DERIVED = """
MATCH ()-[d]->()
WHERE d.derived_by = 'rule'
DELETE d
RETURN count(d) AS deleted
"""

# 전이지배 — 전 구간 qota_rt>=50. qota_rt 가 문자열일 수 있어 toFloat. null 은 제외.
TRANSITIVE_CONTROL = """
MATCH path = (a:Organization)-[:IS_MAJOR_SHAREHOLDER_OF*2..4]->(c:Organization)
WHERE a.corp_code IS NOT NULL AND c.corp_code IS NOT NULL
  AND a.corp_code <> c.corp_code
  AND all(r IN relationships(path)
          WHERE r.qota_rt IS NOT NULL AND toFloat(r.qota_rt) >= 50)
WITH a, c, min(length(path)) AS hops
MERGE (a)-[d:CONTROLS_INDIRECTLY]->(c)
  SET d.via_hops = hops,
      d.derived_by = 'rule',
      d.rule = 'transitive_control'
RETURN count(d) AS created
"""

# 공통임원 — 같은 이름 Person 이 서로 다른 두 회사 임원. elementId 비교로 (a,b) 한 방향만.
COMMON_DIRECTOR = """
MATCH (p1:Person)-[:EXECUTIVE_OF]->(a:Organization),
      (p2:Person)-[:EXECUTIVE_OF]->(b:Organization)
WHERE p1.name = p2.name
  AND a.corp_code IS NOT NULL AND b.corp_code IS NOT NULL
  AND a.corp_code <> b.corp_code
  AND elementId(a) < elementId(b)
  AND size(p1.name) >= 2
MERGE (a)-[d:INTERLOCKING_DIRECTORATE {via: p1.name}]->(b)
  SET d.derived_by = 'rule',
      d.rule = 'common_director',
      d.confidence = 'low'
RETURN count(d) AS created
"""

# dry-run 용 카운트 (MERGE 없이 매칭 개수만)
COUNT_TRANSITIVE = """
MATCH path = (a:Organization)-[:IS_MAJOR_SHAREHOLDER_OF*2..4]->(c:Organization)
WHERE a.corp_code IS NOT NULL AND c.corp_code IS NOT NULL
  AND a.corp_code <> c.corp_code
  AND all(r IN relationships(path)
          WHERE r.qota_rt IS NOT NULL AND toFloat(r.qota_rt) >= 50)
RETURN count(DISTINCT [a.corp_code, c.corp_code]) AS n
"""

COUNT_COMMON = """
MATCH (p1:Person)-[:EXECUTIVE_OF]->(a:Organization),
      (p2:Person)-[:EXECUTIVE_OF]->(b:Organization)
WHERE p1.name = p2.name
  AND a.corp_code IS NOT NULL AND b.corp_code IS NOT NULL
  AND a.corp_code <> b.corp_code
  AND elementId(a) < elementId(b)
  AND size(p1.name) >= 2
RETURN count(*) AS n
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="생성 개수만 출력, 쓰기 없음")
    ap.add_argument("--rebuild", action="store_true", help="기존 rule 엣지 삭제 후 재생성")
    args = ap.parse_args()

    driver = neo4j_driver()
    try:
        with driver.session() as s:
            if args.dry_run:
                nt = s.run(COUNT_TRANSITIVE).single()["n"]
                nc = s.run(COUNT_COMMON).single()["n"]
                print(f"[dry-run] CONTROLS_INDIRECTLY 후보: {nt}")
                print(f"[dry-run] INTERLOCKING_DIRECTORATE 후보: {nc} (동명이인 포함, confidence=low)")
                return 0

            if args.rebuild:
                deleted = s.run(DELETE_DERIVED).single()["deleted"]
                print(f"[rebuild] 기존 rule 엣지 삭제: {deleted}")

            ct = s.run(TRANSITIVE_CONTROL).single()["created"]
            print(f"[build] CONTROLS_INDIRECTLY: {ct}")
            cc = s.run(COMMON_DIRECTOR).single()["created"]
            print(f"[build] INTERLOCKING_DIRECTORATE: {cc} (confidence=low)")

            # 검증: self-loop 0 확인
            sl = s.run(
                "MATCH (n)-[d]->(n) WHERE d.derived_by='rule' RETURN count(d) AS c"
            ).single()["c"]
            if sl:
                print(f"WARN: 파생 self-loop {sl}건 — 규칙 점검 필요")
            print("[done] 파생엣지 materialize 완료. GraphRAG 런타임은 읽기만 수행.")
            return 0
    finally:
        driver.close()


if __name__ == "__main__":
    raise SystemExit(main())
