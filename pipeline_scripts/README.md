# pipeline_scripts/ — DART 수집·적재 스크립트 스냅샷

관리자 콘솔(`routers/admin.py` → `services/pipeline_runner.py`)이 `subprocess` 로 호출하는 스크립트들. **SSOT 는 솔로레포 `mnnk525/db/`** — 여기는 스냅샷 사본이다. 변경은 항상 솔로레포 → 여기 단방향.

## 동기 방법
```
# mnnk525/db/ 의 코드만(데이터 제외) 다시 가져오기
tar --exclude='__pycache__' --exclude='chunk/output' --exclude='raw' \
    --exclude='graph/_auto' --exclude='graph/ledger' --exclude='extra_finance/raw' \
    --exclude='*.jsonl' \
    -C <mnnk525>/db -cf - fetch_dart.py chunk load graph events extra28 extra_finance \
  | tar -C pipeline_scripts -xf -
```

## 런타임 데이터 (gitignore)
- `raw/` — fetch 가 다운로드하는 DART 원본
- `chunk/output/` — chunk 가 만드는 청크 JSONL
- `graph/ledger/`, `graph/_auto/` — extract 멱등성 원장·중간산출
- 이 디렉터리들은 `.gitignore` 처리됨. 빈 폴더만 유지.

## 단계 ↔ 스크립트
| step | script |
|---|---|
| fetch | `fetch_dart.py` |
| chunk | `chunk/chunk.py` |
| mariadb | `load/load_mariadb.py` |
| qdrant | `load/embed_qdrant.py` |
| neo4j_struct | `graph/load_structured.py` |
| extract | `graph/auto_runner.py` |

매핑 정의: `services/pipeline_steps.py`.

## 진행률 마커
`_polaris_marker.py` 의 `marker()` 를 스크립트에서 import 해 호출하면 콘솔에 진행률이 뜬다.
적용 가이드: `이식/db-pipeline-admin/scripts-patch/01-add-json-markers.md`.
"""
