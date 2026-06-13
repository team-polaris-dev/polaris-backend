"""관리자 콘솔 QC 스텝 — 추출 품질 점검 (읽기 전용, 적재와 분리해 언제든 실행).

  ① 산출물 QC  — _auto 에 추출 산출물(batch/review/result)이 남아 있으면 검사해
                  qc_<corp>_summary.json / qc_<corp>_suspects.jsonl 갱신.
                  (정상 적재 후엔 batch/result 가 정리되므로 보통 직전 실행 잔여분이
                   대상 — 없으면 생략하고 알린다)
  ② 그래프 모순 — detect_conflicts: 양방향 SUPPLIES_TO · self-loop · 원장-그래프
                  방향충돌 → graph/conflicts_queue.json 검토 큐 갱신.

extract 스텝의 인라인 QC 와 같은 코드를 재사용한다 — 이 스텝은 "추출 없이
재검진만" 하고 싶을 때(예: 수동 정리 후 확인, 주기 점검) 단독으로 돌린다.
Neo4j/MariaDB 에 아무것도 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def marker(**kw) -> None:
    print("POLARIS_PIPELINE " + json.dumps(kw, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corp_code")
    args = ap.parse_args()
    corp = args.corp_code

    # ① 산출물 QC (잔여 산출물이 있을 때만)
    from qc_all_ollama_results import AUTO, discover_codes, qc_corp

    suspects_n = 0
    if corp in discover_codes():
        qc_summary, suspects = qc_corp(corp)
        (AUTO / f"qc_{corp}_summary.json").write_text(
            json.dumps(qc_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with (AUTO / f"qc_{corp}_suspects.jsonl").open("w", encoding="utf-8") as fh:
            for item in suspects:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        suspects_n = len(suspects)
        print(f"QC① 산출물: 청크 {qc_summary['chunks']} · clean {qc_summary['clean_edges']}"
              f" · 거부 {qc_summary['rejected']} · suspects {suspects_n}"
              f" → _auto/qc_{corp}_suspects.jsonl")
    else:
        print("QC① 산출물: _auto 에 추출 잔여 산출물 없음 — 생략 (그래프 검사만 수행)")
    marker(step="qc", done=1, total=2, suspects=suspects_n)

    # ② 그래프 모순 검출 (전역, 읽기 전용)
    import detect_conflicts

    detect_conflicts.main()
    conflicts_n = -1
    qpath = Path(detect_conflicts.OUT_PATH)
    if qpath.exists():
        queue = json.loads(qpath.read_text(encoding="utf-8"))
        if isinstance(queue, list):
            conflicts_n = len(queue)

    marker(step="qc", phase="end", suspects=suspects_n, conflicts=conflicts_n)
    print(f"QC 스텝 완료: suspects {suspects_n} · conflicts {conflicts_n}"
          f" (검토 큐: graph/conflicts_queue.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
