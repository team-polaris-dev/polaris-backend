"""글로벌 엔티티 통합(캐논화) — 추출 후 그래프 전역 정리. 멱등.

청크 단위 캐논화(verify_and_canonicalize)는 추출 적재 시 청크마다 돈다. 이건 그
"이후"의 그래프 전역 통합:

  1) consolidate_er — needs_er 떠다니는 노드를 실제 corp_code 노드로 병합.
       추출 시점에 resolve_org 가 corp_code 를 몰라 분리됐던 회사 노드를, 이제
       그래프에 그 회사 corp_code 노드가 있으면 합친다(엣지 이동). 신규 회사를
       적재·추출하면 그 회사를 가리키던 타사 보고서의 needs_er 가 비로소 해소된다.
  2) recanon — Product/Technology 변형 노드를 강화 단어집으로 재병합 + 일반어 삭제.

전역 작업이라 회사 인자 없음. 멱등(이미 통합된 건 변화 없음)이라 매 추출 후 자동
실행 안전. POLARIS_PIPELINE 마커로 진행 보고.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def marker(**kw) -> None:
    print("POLARIS_PIPELINE " + json.dumps(kw, ensure_ascii=False), flush=True)


def main() -> int:
    # 1) needs_er → corp_code 병합
    er_merged = -1
    try:
        import consolidate_er

        consolidate_er.main()
        from db import neo4j_driver  # noqa: PLC0415

        d = neo4j_driver()
        with d.session() as s:
            er_merged = s.run(
                "MATCH (o:Organization) WHERE coalesce(o.needs_er,false)=true "
                "RETURN count(*) AS n"
            ).single()["n"]
        d.close()
        print(f"캐논화① needs_er 병합 후 잔여 needs_er: {er_merged}개")
    except Exception as e:  # noqa: BLE001 — 캐논화 실패가 적재를 무효화하지 않음
        print(f"[warn] consolidate_er 실패: {type(e).__name__}: {e}")
    marker(step="canon", done=1, total=2, needs_er_remaining=er_merged)

    # 2) Product/Technology 재캐논
    try:
        import recanon

        recanon.main()
        print("캐논화② Product/Technology 재병합 완료")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] recanon 실패: {type(e).__name__}: {e}")

    marker(step="canon", phase="end", needs_er_remaining=er_merged)
    print("글로벌 캐논화 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
