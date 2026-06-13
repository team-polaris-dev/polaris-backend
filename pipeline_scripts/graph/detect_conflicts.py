"""그래프 모순 검출 배치 (읽기 전용) — 검토 큐 출력.

검출 3종:
  1) 양방향 SUPPLIES_TO — A→B 와 B→A 가 동시에 존재 (방향 노이즈, QC ⑥ 위반)
  2) self-loop — 시작=도착 같은 Organization (QC ⑦ 위반)
  3) 원장-그래프 방향 충돌 — extraction_provenance 에는 A→B 인데 그래프에는
     B→A 만 존재 (q095 실측 사례: 06-05 그래프 정리분이 불변 원장에 잔존.
     원장 직답 금지 — 정리 반영분은 그래프가 SSOT 라는 규약의 모니터링).

출력: 콘솔 요약 + conflicts_queue.json (검토 큐).
이 결과는 score 4축의 consistency 입력으로도 사용 가능
(충돌 엣지 = consistency 0.5, langgraph_app/graph/score.py 참조).

사용:
  cd db && uv run python graph/detect_conflicts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from db import mariadb_conn, neo4j_driver, normalize_corp_name  # noqa: E402

OUT_PATH = HERE / "conflicts_queue.json"
ACK_PATH = HERE / "qc_acknowledged.json"  # '정상 양방향'으로 인정된 쌍 (충돌 제외)


def _acknowledged_pairs() -> set[tuple[str, str]]:
    if not ACK_PATH.exists():
        return set()
    try:
        data = json.loads(ACK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    out: set[tuple[str, str]] = set()
    for key in data:  # key = "a_id|b_id"
        if "|" in key:
            a, b = key.split("|", 1)
            out.add((a, b))
    return out


def graph_supply_pairs(s) -> set[tuple[str, str]]:
    """그래프의 활성 SUPPLIES_TO (from_id, to_id). id = corp_code 우선, 없으면 er_name.

    QC 로 비활성화(qc_disabled_at)된 엣지는 제외 — 이미 해소된 모순 재출현 방지.
    """
    pairs: set[tuple[str, str]] = set()
    for r in s.run(
        "MATCH (a:Organization)-[r:SUPPLIES_TO]->(b:Organization) "
        "WHERE r.qc_disabled_at IS NULL "
        "RETURN coalesce(a.corp_code, a.er_name) AS f, "
        "       coalesce(b.corp_code, b.er_name) AS t"
    ):
        if r["f"] and r["t"]:
            pairs.add((r["f"], r["t"]))
    return pairs


def _detect_non_company(queue: list[dict]) -> dict:
    """4) 비회사 SUPPLIES_TO 검출 — 추출 노이즈(국가/지역·제품·일반어 오분류).

    SUPPLIES_TO 는 '회사 → 회사' 여야 한다. 끝점을 결정론 1차 분류(noise_filter):
      - company    : corp_code/법인접미사/corp_master → 사실상 회사. 제외.
      - geo        : 국가·지역 닫힌집합 → 비회사 확정. decision='geo' (즉시 조치 가능).
      - unresolved : 그 외 → 결정론 단정 불가. decision='pending' → LLM(apimaker)이
                     원문 근거로 회사/제품/일반어/인물 판정(entity-judge). 휴리스틱
                     추정은 폐기 — needs_er 진짜 외국/자회사(TSMC·지멘스)를 끄지 않도록.

    엔티티(정규화 이름) 단위로 그룹화한다 — 같은 쓰레기가 여러 엣지에 걸쳐 있어도
    판정은 1회(LLM 캐시), 적용은 그룹의 모든 엣지를 한 번에. 반환은 집계 dict.
    """
    from noise_filter import classify, normalize as nf_norm  # noqa: PLC0415

    # corp_master 정규화 집합 (DART 상장사 = 회사 보호). 없어도 검출은 진행.
    cm: set[str] = set()
    try:
        conn = mariadb_conn()
        try:
            cur = conn.cursor()
            cur.execute("SELECT corp_name FROM corp_master")
            for row in cur.fetchall():
                nm = row["corp_name"] if isinstance(row, dict) else row[0]
                key = normalize_corp_name(nm)
                if key:
                    cm.add(key)
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 — corp_master 미존재 시 가드만 약화
        print(f"  (corp_master 로드 실패 — 회사보호 가드 약화: {e})")

    d = neo4j_driver()
    with d.session() as s:
        rows = s.run(
            "MATCH (a:Organization)-[r:SUPPLIES_TO]->(b:Organization) "
            "WHERE r.qc_disabled_at IS NULL "
            "RETURN a.name AS a_name, b.name AS b_name, "
            "       a.corp_code AS a_code, b.corp_code AS b_code, "
            "       coalesce(a.corp_code, a.er_name) AS a_id, "
            "       coalesce(b.corp_code, b.er_name) AS b_id, "
            "       r.chunk_id AS chunk_id"
        ).data()
    d.close()

    groups: dict[str, dict] = {}  # entity_key -> 그룹 항목
    for r in rows:
        a_id, b_id = r["a_id"], r["b_id"]
        if not a_id or not b_id:
            continue
        # 두 끝점 중 비회사(geo/unresolved)인 쪽을 from 우선으로 1개 기록
        for side, nm, code in (("from", r["a_name"], r["a_code"]),
                               ("to", r["b_name"], r["b_code"])):
            label, reason = classify(
                nm or "", has_corp_code=bool(code),
                in_corp_master=normalize_corp_name(nm or "") in cm,
            )
            if label == "company":
                continue
            key = nf_norm(nm or "") or (nm or "")
            g = groups.get(key)
            if g is None:
                g = {
                    "kind": "non_company_supplies",
                    "entity_name": nm, "entity_key": key,
                    "decision": "geo" if label == "geo" else "pending",
                    "junk_kind": "geo" if label == "geo" else None,
                    "reason": reason,
                    "sample_chunk": r["chunk_id"],
                    "edges": [],
                }
                groups[key] = g
            if not g.get("sample_chunk") and r["chunk_id"]:
                g["sample_chunk"] = r["chunk_id"]
            g["edges"].append({
                "from_id": a_id, "to_id": b_id,
                "from_name": r["a_name"], "to_name": r["b_name"],
                "junk_side": side, "chunk_id": r["chunk_id"],
            })
            break  # 한 엣지당 비회사 끝점 1개만

    n_geo = n_pending = n_edges = 0
    for g in groups.values():
        g["n_edges"] = len(g["edges"])
        n_edges += g["n_edges"]
        if g["decision"] == "geo":
            n_geo += 1
        else:
            n_pending += 1
        queue.append(g)
    return {"geo": n_geo, "pending": n_pending, "edges": n_edges}


def graph_supply_pairs_disabled(s) -> set[tuple[str, str]]:
    """QC 로 비활성화된 SUPPLIES_TO (from_id, to_id) — 방향충돌 면제 대상."""
    pairs: set[tuple[str, str]] = set()
    for r in s.run(
        "MATCH (a:Organization)-[r:SUPPLIES_TO]->(b:Organization) "
        "WHERE r.qc_disabled_at IS NOT NULL "
        "RETURN coalesce(a.corp_code, a.er_name) AS f, "
        "       coalesce(b.corp_code, b.er_name) AS t"
    ):
        if r["f"] and r["t"]:
            pairs.add((r["f"], r["t"]))
    return pairs


def main() -> int:
    queue: list[dict] = []
    acked = _acknowledged_pairs()
    d = neo4j_driver()
    with d.session() as s:
        # 1) 양방향 SUPPLIES_TO (QC 비활성화 엣지 + '정상 양방향' 인정 쌍 제외)
        rows = s.run(
            "MATCH (a:Organization)-[r1:SUPPLIES_TO]->(b:Organization) "
            "MATCH (b)-[r2:SUPPLIES_TO]->(a) "
            "WHERE elementId(a) < elementId(b) "
            "  AND r1.qc_disabled_at IS NULL AND r2.qc_disabled_at IS NULL "
            "RETURN a.name AS a, b.name AS b, "
            "       coalesce(a.corp_code, a.er_name) AS a_id, "
            "       coalesce(b.corp_code, b.er_name) AS b_id, "
            "       r1.chunk_id AS fwd_chunk, r2.chunk_id AS rev_chunk, "
            "       r1.source AS fwd_src, r2.source AS rev_src"
        ).data()
        n_bi = 0
        for r in rows:
            if (r["a_id"], r["b_id"]) in acked or (r["b_id"], r["a_id"]) in acked:
                continue  # 사람이 '정상 양방향'으로 인정한 쌍
            queue.append({"kind": "bidirectional_supplies", **r,
                          "조치": "본문(chunk) 확인 후 한 방향 삭제 — event 보강(source=event:*)이 있으면 그쪽이 우선"})
            n_bi += 1
        print(f"[1] 양방향 SUPPLIES_TO: {n_bi}건 (인정 제외 {len(rows) - n_bi})")

        # 2) self-loop (모든 관계타입, QC 비활성화 제외)
        rows = s.run(
            "MATCH (a:Organization)-[r]->(a) "
            "WHERE r.qc_disabled_at IS NULL "
            "RETURN a.name AS org, type(r) AS rel, r.chunk_id AS chunk_id"
        ).data()
        for r in rows:
            queue.append({"kind": "self_loop", **r, "조치": "삭제 (QC ⑦)"})
        print(f"[2] self-loop: {len(rows)}건")

        # 3) 원장-그래프 방향 충돌
        gpairs = graph_supply_pairs(s)            # 활성 엣지
        disabled_pairs = graph_supply_pairs_disabled(s)  # QC 로 끈 엣지
    d.close()

    conn = mariadb_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT subject_id, predicate, object_id, chunk_id, rcept_no "
            "FROM extraction_provenance WHERE predicate='SUPPLIES_TO'"
        )
        prov = cur.fetchall()
    finally:
        conn.close()

    n_dir = 0
    for row in prov:
        # DictCursor 가 아닐 수 있어 양쪽 지원
        if isinstance(row, dict):
            subj, obj = row["subject_id"], row["object_id"]
            chunk, rcept = row["chunk_id"], row["rcept_no"]
        else:
            subj, _, obj, chunk, rcept = row
        fwd = (subj, obj)
        rev = (obj, subj)
        # QC 로 의도적으로 비활성화한 방향(disabled_pairs)은 충돌이 아니다 —
        # 사람이 검토 후 끈 것이라 원장에 남아도 정상. 진짜 충돌은 그래프에
        # 흔적조차 없는데 역방향만 활성인 경우.
        if fwd not in gpairs and fwd not in disabled_pairs and rev in gpairs:
            n_dir += 1
            queue.append({
                "kind": "ledger_graph_direction_conflict",
                "ledger": f"{subj} -> {obj}", "graph": f"{obj} -> {subj}",
                "chunk_id": chunk, "rcept_no": rcept,
                "조치": "그래프가 SSOT(정리 반영분) — 원장 직답 금지 확인. 본문 재확인 시 chunk_id 사용",
            })
    print(f"[3] 원장-그래프 방향 충돌: {n_dir}건 (원장 SUPPLIES_TO {len(prov)}건 중)")

    # 4) 비회사 SUPPLIES_TO (geo 확정 + LLM 판정대기 — 추출 노이즈)
    nc = _detect_non_company(queue)
    print(f"[4] 비회사 SUPPLIES_TO: geo확정 {nc['geo']}종 / "
          f"LLM판정대기 {nc['pending']}종 (엣지 {nc['edges']})")

    OUT_PATH.write_text(json.dumps(queue, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n검토 큐 {len(queue)}건 → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
