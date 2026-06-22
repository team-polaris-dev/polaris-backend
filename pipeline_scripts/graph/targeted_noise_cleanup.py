"""Targeted graph noise cleanup with a dry-run first workflow.

Default:
  uv run python graph/targeted_noise_cleanup.py

Apply:
  uv run python graph/targeted_noise_cleanup.py --apply

This script only handles reviewed, narrow cleanup rules:
- Drop obvious non-entities from Product/Technology.
- Remove Hybrid Bonder edges whose source chunk does not explicitly mention a
  Hybrid Bonder, then merge only the Hybrid Bonder Product/Technology duplicate.
- Merge reviewed same-kind Product/Technology aliases without crossing
  process/equipment/material boundaries.
- Merge only the WUIXI/WUXI FST subsidiary spelling variants, not the Korean
  FST parent company.

It updates MariaDB extraction_provenance together with Neo4j so evidence lookup
stays consistent after node merges.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from db import mariadb_conn, neo4j_driver  # noqa: E402
from extract_helpers import entity_id, prov_id  # noqa: E402


BLOCK_ENTITY_NAMES = {
    "SDC",
    "DS 부문",
    "DX 부문",
    "ISO14001",
    "ISO 14001",
    "ISO45001",
    "ISO 45001",
    # 2026-06-12 QC 잔여 (NOISE_TODO §3·§4-2026-06-06) — 회사명·일반어가 Product로 유입
    "삼성디스플레이",
    "WUXI FST.CO.,LTD",
    "FINE SEMITECH USA CORPORATION",
    "SOLMICS HONGKONG CO., LTD.",
    "XIAN FST.CO.LTD",
    "상품",
    "공정용 화학재료",
    "신제품",
    "이차전지용 전자재료",
    # 2026-06-16 재누적분 — corp_master 등록 회사명이 접미사 없이 Product 로 유입
    # (기존 COMPANY_HINTS 가 법인접미사만 봐서 맨이름을 놓침). corp_master 대조로 확정.
    "삼성전자",
    "SK하이닉스",
    "LG이노텍",
    "삼성전기",
    "삼성SDI",
    "삼성물산",
    "삼성바이오로직스",
    "삼성에스디에스",
    "삼성에피스홀딩스",
    "제일기획",
    "코리아써키트",
    "SFA반도체",
    "시그네틱스",
    "잉크테크",
    "디엔에프",
    "레인보우로보틱스",
    "광전자",
    "ISC",
    "APS",
    # 2026-06-16 고아(0엣지) 회계계정·일반어가 Product/Tech 로 유입
    "재고자산", "저장품", "유형자산", "투자부동산", "부재료", "반도체구성",
    "공장", "컴퓨터소프트웨어", "소프트웨어", "산업재산권", "생산시스템",
    "기계제작", "연구개발시설", "Chat GPT", "2000년대 중반", "Life-Cycle",
    # 2026-06-16 고아 한국 회사명 Product 유입 ((주) 접두 — _ORG_SUFFIX 미포착)
    "(주)솔레오", "(주)이엠테크", "(주)이오엘", "(주)잉크테크", "(주)한울반도체",
}

HYBRID_BONDER_EVIDENCE_RE = re.compile(
    r"hybrid\s*bonder|하이브리드\s*본더",
    re.IGNORECASE,
)

# source_label, source_name, target_label, target_name, reason
ENTITY_MERGES = [
    ("Technology", "Hybrid Bonder", "Product", "Hybrid Bonder", "cross_label_duplicate"),
    (
        "Technology",
        "Hollow Silica 저굴절·저유전·단열 기술",
        "Product",
        "Hollow Silica",
        "hollow_silica_material",
    ),
    (
        "Technology",
        "CVD(Chemical Vapor Deposition) 박막 증착 기술",
        "Technology",
        "CVD",
        "cvd_process_alias",
    ),
    ("Technology", "CVD 공정", "Technology", "CVD", "cvd_process_alias"),
    (
        "Technology",
        "CVD SiC Coating기술",
        "Technology",
        "CVD SiC Coating 기술",
        "cvd_sic_coating_spacing",
    ),
    (
        "Technology",
        "ALD(Atomic Layer Deposition) 원자층 증착 기술",
        "Technology",
        "ALD",
        "ald_process_alias",
    ),
    (
        "Product",
        "반도체 CVD 증착 장비(GEMINI/QUARTO/LEVATA)",
        "Product",
        "CVD 증착 장비",
        "cvd_equipment_alias",
    ),
    (
        "Product",
        "반도체 ALD(Atomic Layer Deposition) 증착 장비(HYETA/PRESTO/CLARO/VELOCE)",
        "Product",
        "ALD 증착 장비",
        "ald_equipment_alias",
    ),
    # ── 2026-06-12 마스크/펠리클 표기분산 캐논 (NOISE_TODO 2026-06-06 잔여) ──
    # 원칙: 같은 제품의 띄어쓰기·한영·괄호 표기변형만 병합. 서브타입(DUV/CNT/Blank/프레임/
    # 검사장비/공정)은 의미가 달라 보존 — 과병합 금지(03_neo4j.md §7-3 "세대구분은 병합 안 함").
    ("Product", "Photo Mask", "Product", "Photomask", "photomask_spacing"),
    ("Product", "Phase-shift mask", "Product", "Phase Shift Mask", "psm_hyphen_case"),
    ("Technology", "Photomask", "Product", "Photomask", "photomask_cross_label"),
    ("Product", "펠리클", "Product", "Pellicle", "pellicle_ko_en"),
    ("Product", "반도체펠리클", "Product", "반도체 펠리클", "semi_pellicle_spacing"),
    ("Product", "반도체 펠리클(Pellicle)", "Product", "반도체 펠리클", "semi_pellicle_paren"),
    ("Product", "Semiconductor Pellicle", "Product", "반도체 펠리클", "semi_pellicle_en"),
    ("Product", "EUV용 펠리클", "Product", "EUV Pellicle", "euv_pellicle_ko"),
    ("Product", "EUV용펠리클", "Product", "EUV Pellicle", "euv_pellicle_ko2"),
    ("Product", "FPD 펠리클", "Product", "FPD Pellicle", "fpd_pellicle_ko"),
    ("Product", "FPD펠리클", "Product", "FPD Pellicle", "fpd_pellicle_ko2"),
    ("Product", "FPD용 펠리클", "Product", "FPD Pellicle", "fpd_pellicle_ko3"),
    ("Product", "FPD 펠리클(TFT-LCD/OLED용)", "Product", "FPD Pellicle", "fpd_pellicle_paren"),
    ("Product", "펠리클용 프레임", "Product", "펠리클 프레임", "pellicle_frame_spacing"),
    ("Product", "반도체 펠리클용 프레임", "Product", "펠리클 프레임", "pellicle_frame_semi"),
    # ── 2026-06-16 HBM 표기분산 캐논 ──
    # 원칙: 제네릭 "HBM"을 뜻하는 한영·괄호·칩/제품 접미·생산문구 변형만 Product "HBM"으로 병합.
    # 세대(HBM3/HBM3E/HBM4/HBM5)·별개 제품(TC 본더·6-SIDE 검사·소재기술·적층방식)은 의미가
    # 달라 보존 — 03_neo4j.md §7-3 "세대구분은 병합 안 함".
    ("Technology", "HBM", "Product", "HBM", "hbm_cross_label"),
    ("Technology", "HBM(광대역폭메모리)", "Product", "HBM", "hbm_ko_paren"),
    ("Technology", "HBM: High Bandwidth Memory", "Product", "HBM", "hbm_en_expansion"),
    ("Product", "HBM: High Bandwidth Memory", "Product", "HBM", "hbm_en_expansion2"),
    ("Product", "광대역폭메모리반도체 (HBM: High Bandwidth Memory)", "Product", "HBM", "hbm_ko_en_paren"),
    ("Product", "광대역폭메모리반도체", "Product", "HBM", "hbm_ko_broadband"),
    ("Product", "고대역폭메모리", "Product", "HBM", "hbm_ko_high_bw"),
    ("Product", "HBM제품", "Product", "HBM", "hbm_product_suffix"),
    ("Product", "HBM칩", "Product", "HBM", "hbm_chip"),
    ("Product", "HBM 칩", "Product", "HBM", "hbm_chip_spacing"),
    ("Product", "HBM 칩(Die)", "Product", "HBM", "hbm_chip_die"),
    ("Technology", "HBM칩생산", "Product", "HBM", "hbm_chip_production"),
    ("Technology", "AI 반도체 구현을 위한 HBM 칩 생산", "Product", "HBM", "hbm_chip_production_phrase"),
    ("Technology", "AI 반도체 구현을 위한 HBM칩 생산", "Product", "HBM", "hbm_chip_production_phrase2"),
    ("Product", "HBM 제조용", "Product", "HBM", "hbm_manufacturing_fragment"),
    ("Technology", "Core HBM", "Product", "HBM", "core_hbm_cross_label"),
    ("Product", "Core HBM", "Product", "HBM", "core_hbm"),
    ("Product", "HBM3E DRAM", "Product", "HBM3E", "hbm3e_dram_generation"),
]

WUXI_FST_TARGET = {"name": "WUXI FST", "er_name": "wuxifst"}
WUXI_FST_SOURCE_NAMES = {
    "WUIXI FST",
    "WUIXI FST CO.L,TD.(*1)",
    "WUXI FST.CO.LTD",
}


def key_field(label: str) -> str:
    if label == "Product":
        return "product_id"
    if label == "Technology":
        return "tech_id"
    raise ValueError(label)


def entity_key(label: str, name: str) -> str:
    return entity_id(name.lower())


def fetch_chunk_texts(chunk_ids: set[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    conn = mariadb_conn()
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(chunk_ids))
    cur.execute(
        f"SELECT chunk_id, embedding_text FROM chunk_index WHERE chunk_id IN ({placeholders})",
        tuple(chunk_ids),
    )
    out = {cid: text or "" for cid, text in cur.fetchall()}
    cur.close()
    conn.close()
    return out


def fetch_entity_nodes(session, names: set[str]) -> list[dict[str, Any]]:
    rows = session.run(
        """
        MATCH (n)
        WHERE (n:Product OR n:Technology) AND n.name IN $names
        OPTIONAL MATCH (n)-[r]-()
        RETURN elementId(n) AS element_id, labels(n) AS labels, n.name AS name,
               n.product_id AS product_id, n.tech_id AS tech_id, count(r) AS rel_count
        ORDER BY n.name, labels(n)
        """,
        names=sorted(names),
    ).data()
    return rows


def fetch_entity_merge_sources(session) -> list[dict[str, Any]]:
    wanted = sorted({m[1] for m in ENTITY_MERGES} | {m[3] for m in ENTITY_MERGES})
    return fetch_entity_nodes(session, set(wanted))


def get_entity_ids_for_names(session, names: set[str]) -> set[str]:
    ids = set()
    for row in fetch_entity_nodes(session, names):
        if row.get("product_id"):
            ids.add(row["product_id"])
        if row.get("tech_id"):
            ids.add(row["tech_id"])
    return ids


def count_provenance_for_ids(conn, ids: set[str]) -> int:
    if not ids:
        return 0
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(ids))
    cur.execute(
        f"""
        SELECT COUNT(*)
        FROM extraction_provenance
        WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})
        """,
        tuple(ids) + tuple(ids),
    )
    count = cur.fetchone()[0]
    cur.close()
    return int(count)


def delete_provenance_for_ids(conn, ids: set[str]) -> int:
    if not ids:
        return 0
    cur = conn.cursor()
    placeholders = ",".join(["%s"] * len(ids))
    cur.execute(
        f"""
        DELETE FROM extraction_provenance
        WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})
        """,
        tuple(ids) + tuple(ids),
    )
    count = cur.rowcount
    cur.close()
    return int(count)


def rewrite_provenance_ids(conn, id_map: dict[str, str]) -> dict[str, int]:
    """Rewrite subject_id/object_id and recompute prov_id.

    If the rewritten prov_id already exists, the old duplicate provenance row is
    removed. This preserves idempotency after Neo4j node merges.
    """
    id_map = {old: new for old, new in id_map.items() if old and new and old != new}
    if not id_map:
        return {"rows": 0, "updated": 0, "dedup_deleted": 0}

    cur = conn.cursor()
    olds = sorted(id_map)
    placeholders = ",".join(["%s"] * len(olds))
    cur.execute(
        f"""
        SELECT prov_id, subject_id, predicate, object_id, chunk_id, rcept_no, extracted_by, confidence
        FROM extraction_provenance
        WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})
        """,
        tuple(olds) + tuple(olds),
    )
    rows = cur.fetchall()

    updated = 0
    dedup_deleted = 0
    for old_pid, subject_id, predicate, object_id, chunk_id, *_ in rows:
        new_subject = id_map.get(subject_id, subject_id)
        new_object = id_map.get(object_id, object_id)
        if new_subject == subject_id and new_object == object_id:
            continue
        new_pid = prov_id(new_subject, predicate, new_object, chunk_id)
        cur.execute("SELECT 1 FROM extraction_provenance WHERE prov_id=%s", (new_pid,))
        if cur.fetchone():
            cur.execute("DELETE FROM extraction_provenance WHERE prov_id=%s", (old_pid,))
            dedup_deleted += 1
        else:
            cur.execute(
                """
                UPDATE extraction_provenance
                SET prov_id=%s, subject_id=%s, object_id=%s
                WHERE prov_id=%s
                """,
                (new_pid, new_subject, new_object, old_pid),
            )
            updated += 1
    cur.close()
    return {"rows": len(rows), "updated": updated, "dedup_deleted": dedup_deleted}


def delete_relationship_by_element_id(session, rel_id: str) -> int:
    result = session.run(
        """
        MATCH ()-[r]->()
        WHERE elementId(r)=$rel_id
        WITH r, count(r) AS c
        DELETE r
        RETURN c
        """,
        rel_id=rel_id,
    ).single()
    return int(result["c"] if result else 0)


def fetch_invalid_hybrid_edges(session) -> list[dict[str, Any]]:
    rows = session.run(
        """
        MATCH (o:Organization)-[r:PRODUCES|USES_TECH]->(n)
        WHERE n.name='Hybrid Bonder' AND r.chunk_id IS NOT NULL
        RETURN elementId(r) AS rel_id, type(r) AS predicate,
               coalesce(r.extracted_by, '') AS extracted_by,
               r.chunk_id AS chunk_id,
               coalesce(o.name, o.er_name, o.corp_code) AS subject_name,
               o.corp_code AS subject_corp_code,
               coalesce(n.product_id, n.tech_id) AS object_id
        ORDER BY extracted_by, subject_name, predicate
        """
    ).data()
    texts = fetch_chunk_texts({r["chunk_id"] for r in rows if r.get("chunk_id")})
    invalid = []
    for row in rows:
        text = texts.get(row["chunk_id"], "")
        has_evidence = bool(HYBRID_BONDER_EVIDENCE_RE.search(text))
        row["has_exact_evidence"] = has_evidence
        if not has_evidence:
            row["reason"] = "hybrid_bonder_without_exact_evidence"
            invalid.append(row)
    return invalid


def delete_invalid_hybrid_edges(session, conn, invalid_edges: list[dict[str, Any]]) -> dict[str, int]:
    rel_deleted = 0
    prov_deleted = 0
    cur = conn.cursor()
    for row in invalid_edges:
        rel_deleted += delete_relationship_by_element_id(session, row["rel_id"])
        if row["extracted_by"]:
            cur.execute(
                """
                DELETE FROM extraction_provenance
                WHERE chunk_id=%s AND predicate=%s AND object_id=%s AND extracted_by=%s
                """,
                (row["chunk_id"], row["predicate"], row["object_id"], row["extracted_by"]),
            )
        else:
            cur.execute(
                """
                DELETE FROM extraction_provenance
                WHERE chunk_id=%s AND predicate=%s AND object_id=%s AND extracted_by IS NULL
                """,
                (row["chunk_id"], row["predicate"], row["object_id"]),
            )
        prov_deleted += cur.rowcount
    cur.close()
    return {"neo4j_relationships": rel_deleted, "mariadb_provenance": prov_deleted}


def ensure_target_entity(session, label: str, name: str) -> str:
    kf = key_field(label)
    target_id = entity_key(label, name)
    session.run(
        f"""
        MERGE (n:{label} {{{kf}:$target_id}})
        SET n.name=$name, n.canonical=toLower($name)
        """,
        target_id=target_id,
        name=name,
    )
    return target_id


def merge_entity_pair(session, source_label: str, source_name: str, target_label: str, target_name: str) -> dict[str, Any]:
    source_kf = key_field(source_label)
    target_kf = key_field(target_label)
    target_id = ensure_target_entity(session, target_label, target_name)
    source_rows = session.run(
        f"""
        MATCH (s:{source_label} {{name:$source_name}})
        RETURN elementId(s) AS source_element_id, s.{source_kf} AS source_id
        """,
        source_name=source_name,
    ).data()

    merged = 0
    removed_cross_label = 0
    for row in source_rows:
        source_id = row["source_id"]
        if not source_id:
            continue
        result = session.run(
            f"""
            MATCH (target:{target_label} {{{target_kf}:$target_id}})
            MATCH (source:{source_label} {{{source_kf}:$source_id}})
            WHERE elementId(target) <> elementId(source)
            CALL apoc.refactor.mergeNodes([target, source],
                {{properties:'discard', mergeRels:true}}) YIELD node
            REMOVE node:{'Technology' if target_label == 'Product' else 'Product'}
            SET node:{target_label}, node.name=$target_name, node.canonical=toLower($target_name)
            RETURN count(node) AS c
            """,
            target_id=target_id,
            source_id=source_id,
            target_name=target_name,
        ).single()
        merge_count = int(result["c"] if result else 0)
        merged += merge_count

        # If Product and Technology labels live on one physical node, the merge
        # query above correctly skips it via elementId(target) <> elementId(source).
        # Remove the non-target label in that case.
        if merge_count == 0:
            result = session.run(
                f"""
                MATCH (n:{source_label}:{target_label})
                WHERE n.{source_kf}=$source_id OR n.{target_kf}=$target_id
                REMOVE n:{source_label}
                SET n:{target_label}, n.name=$target_name, n.canonical=toLower($target_name)
                RETURN count(n) AS c
                """,
                source_id=source_id,
                target_id=target_id,
                target_name=target_name,
            ).single()
            removed_cross_label += int(result["c"] if result else 0)

    return {
        "source_label": source_label,
        "source_name": source_name,
        "target_label": target_label,
        "target_name": target_name,
        "target_id": target_id,
        "source_ids": [r.get("source_id") for r in source_rows if r.get("source_id")],
        "merged": merged,
        "cross_label_removed": removed_cross_label,
    }


def plan_entity_merges(session) -> list[dict[str, Any]]:
    existing = fetch_entity_merge_sources(session)
    by_key = {(tuple(row["labels"]), row["name"]): row for row in existing}
    out = []
    for source_label, source_name, target_label, target_name, reason in ENTITY_MERGES:
        source_rows = [
            row for row in existing
            if source_label in row["labels"] and row["name"] == source_name
        ]
        target_rows = [
            row for row in existing
            if target_label in row["labels"] and row["name"] == target_name
        ]
        out.append(
            {
                "reason": reason,
                "source_label": source_label,
                "source_name": source_name,
                "target_label": target_label,
                "target_name": target_name,
                "source_nodes": source_rows,
                "target_nodes": target_rows,
                "source_ids": [
                    row.get("product_id") or row.get("tech_id") for row in source_rows
                ],
                "target_id": entity_key(target_label, target_name),
            }
        )
    return out


def fetch_wuxi_sources(session) -> list[dict[str, Any]]:
    return session.run(
        """
        MATCH (o:Organization)
        WHERE o.corp_code IS NULL
          AND (o.name IN $names OR o.er_name IN $ers)
        OPTIONAL MATCH (o)-[r]-()
        RETURN elementId(o) AS element_id, o.name AS name, o.er_name AS er_name,
               o.corp_code AS corp_code, count(r) AS rel_count
        ORDER BY o.name, o.er_name
        """,
        names=sorted(WUXI_FST_SOURCE_NAMES),
        ers=["wuixifst", "wuixifstco.l,td.", "wuxifst."],
    ).data()


def merge_wuxi_fst(session) -> dict[str, Any]:
    sources = fetch_wuxi_sources(session)
    session.run(
        """
        MERGE (target:Organization {er_name:$target_er, has_corp_code:false})
        SET target.name=$target_name, target.needs_er=true, target.has_corp_code=false
        """,
        target_er=WUXI_FST_TARGET["er_name"],
        target_name=WUXI_FST_TARGET["name"],
    )
    merged = 0
    for row in sources:
        er = row.get("er_name")
        if not er or er == WUXI_FST_TARGET["er_name"]:
            continue
        result = session.run(
            """
            MATCH (target:Organization {er_name:$target_er, has_corp_code:false})
            MATCH (source:Organization {er_name:$source_er, has_corp_code:false})
            WHERE elementId(target) <> elementId(source)
            CALL apoc.refactor.mergeNodes([target, source],
                {properties:'discard', mergeRels:true}) YIELD node
            SET node.name=$target_name, node.needs_er=true, node.has_corp_code=false
            RETURN count(node) AS c
            """,
            target_er=WUXI_FST_TARGET["er_name"],
            source_er=er,
            target_name=WUXI_FST_TARGET["name"],
        ).single()
        merged += int(result["c"] if result else 0)
    return {
        "target": WUXI_FST_TARGET,
        "sources": sources,
        "merged": merged,
    }


def fetch_cvd_ald_report(session) -> list[dict[str, Any]]:
    return session.run(
        """
        MATCH (n)
        WHERE (n:Product OR n:Technology)
          AND (
            toLower(coalesce(n.name,'')) CONTAINS 'cvd'
            OR toLower(coalesce(n.name,'')) CONTAINS 'ald'
            OR toLower(coalesce(n.name,'')) CONTAINS 'hollow silica'
          )
        OPTIONAL MATCH (o:Organization)-[r:PRODUCES|USES_TECH]->(n)
        RETURN labels(n) AS labels, n.name AS name, coalesce(n.product_id,n.tech_id) AS id,
               count(r) AS rel_count, collect(DISTINCT coalesce(r.extracted_by,''))[0..5] AS extracted_by
        ORDER BY name
        """
    ).data()


def build_plan() -> dict[str, Any]:
    driver = neo4j_driver()
    conn = mariadb_conn()
    try:
        with driver.session() as session:
            block_nodes = fetch_entity_nodes(session, BLOCK_ENTITY_NAMES)
            block_ids = {
                value for row in block_nodes
                for value in (row.get("product_id"), row.get("tech_id"))
                if value
            }
            invalid_hybrid = fetch_invalid_hybrid_edges(session)
            entity_merges = plan_entity_merges(session)
            wuxi_sources = fetch_wuxi_sources(session)
            cvd_ald_report = fetch_cvd_ald_report(session)
            summary = {
                "block_nodes": block_nodes,
                "block_provenance_rows": count_provenance_for_ids(conn, block_ids),
                "invalid_hybrid_edges": invalid_hybrid,
                "entity_merges": entity_merges,
                "wuxi_fst": {
                    "target": WUXI_FST_TARGET,
                    "sources": wuxi_sources,
                    "source_er_names": [r.get("er_name") for r in wuxi_sources if r.get("er_name")],
                },
                "cvd_ald_hollow_report": cvd_ald_report,
            }
            return summary
    finally:
        conn.close()
        driver.close()


def apply_cleanup(plan: dict[str, Any]) -> dict[str, Any]:
    driver = neo4j_driver()
    conn = mariadb_conn()
    applied: dict[str, Any] = {}
    try:
        with driver.session() as session:
            block_ids = {
                value for row in plan["block_nodes"]
                for value in (row.get("product_id"), row.get("tech_id"))
                if value
            }
            applied["block_provenance_deleted"] = delete_provenance_for_ids(conn, block_ids)
            result = session.run(
                """
                MATCH (n)
                WHERE (n:Product OR n:Technology) AND n.name IN $names
                WITH collect(n) AS nodes, count(n) AS c
                FOREACH (n IN nodes | DETACH DELETE n)
                RETURN c
                """,
                names=sorted(BLOCK_ENTITY_NAMES),
            ).single()
            applied["block_nodes_deleted"] = int(result["c"] if result else 0)

            invalid = plan["invalid_hybrid_edges"]
            applied["invalid_hybrid_deleted"] = delete_invalid_hybrid_edges(session, conn, invalid)

            id_map: dict[str, str] = {}
            merge_results = []
            for source_label, source_name, target_label, target_name, reason in ENTITY_MERGES:
                target_id = entity_key(target_label, target_name)
                for row in plan["entity_merges"]:
                    if (
                        row["source_label"] == source_label
                        and row["source_name"] == source_name
                        and row["target_label"] == target_label
                        and row["target_name"] == target_name
                    ):
                        for source_id in row["source_ids"]:
                            if source_id and source_id != target_id:
                                id_map[source_id] = target_id
                merge_result = merge_entity_pair(session, source_label, source_name, target_label, target_name)
                merge_result["reason"] = reason
                merge_results.append(merge_result)
            applied["entity_merges"] = merge_results
            applied["entity_provenance_rewritten"] = rewrite_provenance_ids(conn, id_map)

            org_sources = plan["wuxi_fst"]["source_er_names"]
            org_id_map = {
                er: WUXI_FST_TARGET["er_name"]
                for er in org_sources
                if er and er != WUXI_FST_TARGET["er_name"]
            }
            applied["wuxi_fst_merge"] = merge_wuxi_fst(session)
            applied["wuxi_fst_provenance_rewritten"] = rewrite_provenance_ids(conn, org_id_map)

        conn.commit()
        return applied
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
        driver.close()


def compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "block_nodes": {
            "count": len(plan["block_nodes"]),
            "names": [row["name"] for row in plan["block_nodes"]],
            "provenance_rows": plan["block_provenance_rows"],
        },
        "invalid_hybrid_edges": {
            "count": len(plan["invalid_hybrid_edges"]),
            "sample": plan["invalid_hybrid_edges"][:10],
        },
        "entity_merges": [
            {
                "reason": row["reason"],
                "source": f"{row['source_label']}:{row['source_name']}",
                "target": f"{row['target_label']}:{row['target_name']}",
                "source_node_count": len(row["source_nodes"]),
                "target_node_count": len(row["target_nodes"]),
                "source_ids": row["source_ids"],
                "target_id": row["target_id"],
            }
            for row in plan["entity_merges"]
        ],
        "wuxi_fst": {
            "source_count": len(plan["wuxi_fst"]["sources"]),
            "sources": plan["wuxi_fst"]["sources"],
            "target": plan["wuxi_fst"]["target"],
        },
        "cvd_ald_hollow_report_count": len(plan["cvd_ald_hollow_report"]),
        "cvd_ald_hollow_report": plan["cvd_ald_hollow_report"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="apply cleanup changes")
    parser.add_argument(
        "--full",
        action="store_true",
        help="print full plan instead of compact summary",
    )
    args = parser.parse_args()

    plan = build_plan()
    output: dict[str, Any] = {
        "mode": "apply" if args.apply else "dry_run",
        "plan": plan if args.full else compact_plan(plan),
    }
    if args.apply:
        output["applied"] = apply_cleanup(plan)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
