"""관리자 콘솔 extract 스텝 — prep → LLM 추출 → load 일괄 실행.

기본 동작은 "미처리(pending) 청크 전부" 추출이다 — 사용자가 pending 규모를
모른 채 개수를 미리 정하게 하지 않는다. 상한이 필요하면 --limit N 으로 지정.

  1) auto_runner.prep()           — 미처리 청크를 _auto/batch_*.json 으로 준비
  2) ollama_extract.process_batch — LLM 추출 (provider: ollama 기본 | apimaker QC용)
  3) QC① 산출물 검사              — qc_all_ollama_results.qc_corp (batch/result/review
                                     가 살아있는 load "전" 시점에만 가능 — 읽기 전용)
  4) auto_runner.load()           — 앵커검증·캐논화 → Neo4j/MariaDB 적재 + 원장 mark
  5) QC② 그래프 모순 검출         — detect_conflicts (양방향 SUPPLIES_TO·self-loop·
                                     원장 방향충돌, 읽기 전용 → conflicts_queue.json)

QC 는 보조 검사라 실패해도 적재를 막지 않는다(경고만). suspects/conflicts 건수는
마커 counters 로 콘솔에 노출된다.

진행률은 POLARIS_PIPELINE 마커(services/pipeline_runner._parse_marker 형식:
{"done": n, "total": m} / {"phase": "end"})로 stdout 에 찍어 SSE 진행바와 연동.
배치(기본 40청크) 단위로 보고하므로 전부 추출 시에도 진행률이 흐른다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import auto_runner  # noqa: E402
import ollama_extract as oe  # noqa: E402

DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
ALL_WAVES = 10**6  # prep 의 rows[:wave*bsize] 슬라이스가 전부를 덮도록


def marker(**kw) -> None:
    print("POLARIS_PIPELINE " + json.dumps(kw, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corp_code")
    ap.add_argument("--limit", type=int, default=None,
                    help="이번 실행에서 처리할 청크 수 상한 (생략 시 pending 전부)")
    ap.add_argument("--bsize", type=int, default=40,
                    help="내부 배치 크기 — 진행률 보고 단위")
    ap.add_argument("--positive", action="store_true", help="anchor 청크만 추출")
    ap.add_argument("--provider", choices=["ollama", "apimaker"], default="ollama")
    ap.add_argument("--model", default=None,
                    help="ollama 모델명(기본 qwen3.5:9b) 또는 apimaker(Gemini) 모델명")
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    corp = args.corp_code
    bsize = max(1, args.bsize)
    if args.limit is not None and args.limit > 0:
        # limit 은 "초과 금지" 상한 — prep 은 wave*bsize 배수 단위라,
        # limit 미만 배수로 내림한다 (limit < bsize 면 배치 크기를 limit 으로 축소).
        if args.limit < bsize:
            bsize = args.limit
            wave = 1
        else:
            wave = args.limit // bsize
    else:
        wave = ALL_WAVES  # 전부

    # [1] prep — 원장에 없는 청크만 배치파일로
    nb = auto_runner.prep(corp, wave, bsize, positive=args.positive)
    if nb == 0:
        print("미처리 청크 없음 — 추출/적재 생략")
        marker(step="extract", phase="end", batches=0, chunks=0, edges=0)
        return 0

    # [2] LLM 추출 — 배치 단위 진행률 보고
    batches = sorted(auto_runner.AUTO.glob(f"batch_{corp}_*.json"))
    if args.provider == "apimaker":
        ollama_model = DEFAULT_OLLAMA_MODEL  # 미사용
        apimaker_model = args.model or None  # None → Gemini CLI 기본 모델
    else:
        ollama_model = args.model or DEFAULT_OLLAMA_MODEL
        apimaker_model = None

    total_chunks = total_edges = total_rejected = 0
    for i, batch in enumerate(batches, 1):
        n_chunk, n_edge, n_reject = oe.process_batch(
            batch, base=args.base, model=ollama_model, timeout=args.timeout,
            provider=args.provider, apimaker_model=apimaker_model,
        )
        total_chunks += n_chunk
        total_edges += n_edge
        total_rejected += n_reject
        marker(step="extract", done=i, total=len(batches),
               chunks=total_chunks, edges=total_edges, rejected=total_rejected)

    # [3] QC① — 산출물 검사 (batch/result/review 가 살아있는 load 전 시점)
    suspects_n = -1
    try:
        from qc_all_ollama_results import qc_corp

        qc_summary, suspects = qc_corp(corp)
        (auto_runner.AUTO / f"qc_{corp}_summary.json").write_text(
            json.dumps(qc_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        with (auto_runner.AUTO / f"qc_{corp}_suspects.jsonl").open("w", encoding="utf-8") as fh:
            for item in suspects:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        suspects_n = len(suspects)
        print(f"QC① 산출물: suspects={suspects_n} → _auto/qc_{corp}_suspects.jsonl")
    except Exception as e:  # noqa: BLE001 — QC 실패는 적재를 막지 않는다
        print(f"[warn] QC①(산출물) 실패 — 적재는 계속: {type(e).__name__}: {e}")

    # [4] load — Neo4j/MariaDB 적재 + 원장 mark (extracted_by = provider 명)
    auto_runner.load(corp, extracted_by=args.provider)

    # [5] QC② — 그래프 모순 검출 (적재 후, 읽기 전용)
    conflicts_n = -1
    try:
        import detect_conflicts

        detect_conflicts.main()
        qpath = Path(detect_conflicts.OUT_PATH)
        if qpath.exists():
            queue = json.loads(qpath.read_text(encoding="utf-8"))
            if isinstance(queue, list):
                conflicts_n = len(queue)
        print(f"QC② 그래프 모순: conflicts={conflicts_n} → graph/conflicts_queue.json")
    except Exception as e:  # noqa: BLE001 — QC 실패해도 적재는 유효
        print(f"[warn] QC②(그래프 모순) 실패 — 적재는 유효: {type(e).__name__}: {e}")

    marker(step="extract", phase="end",
           batches=nb, chunks=total_chunks, edges=total_edges, rejected=total_rejected,
           suspects=suspects_n, conflicts=conflicts_n)
    print(f"extract 스텝 완료: provider={args.provider} 배치 {nb} · 청크 {total_chunks}"
          f" · 엣지 {total_edges} · 거부 {total_rejected}"
          f" · QC suspects {suspects_n} · conflicts {conflicts_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
