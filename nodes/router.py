import json
import os
from functools import lru_cache
from pathlib import Path
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from core.state import AgentState
from config.llm import llm, json_llm

# DB 스키마/단어집 JSON. 파일 통째로 읽어 재구성 프롬프트에 그대로 주입한다.
# (POLARIS_GLOSSARY_PATH 로 경로 override, 기본 config/glossary.json)
_GLOSSARY_PATH = Path(
    os.environ.get("POLARIS_GLOSSARY_PATH")
    or Path(__file__).resolve().parent.parent / "config" / "glossary.json"
)

# 동적 엔티티 리스트(organization/person/product/technology)는 수천 개라 통째 주입하면
# 재구성 프롬프트가 75만 자가 되어 LLM 호출이 타임아웃한다(provider 180s 초과).
# 정적 용어 사전(fin_accounts·관계술어·보고서코드 등 — 프롬프트 규칙이 실제로 쓰는 것)은
# 유지하고, 엔티티 리스트는 degree 내림차순(이미 정렬됨) 상위 N개만 남긴다.
# 주요 기업·인물 위주라 대명사 치환·별칭 매핑에 충분하다.
_GLOSSARY_ENTITY_CAP = int(os.environ.get("POLARIS_GLOSSARY_ENTITY_CAP", "150"))
_GLOSSARY_ENTITY_KEYS = ("organization", "person", "product", "technology")


@lru_cache(maxsize=1)
def _load_glossary_text() -> str:
    """glossary.json 을 읽되 동적 엔티티 리스트는 상위 N개로 잘라 반환(프로세스 1회 캐시).

    파일이 없거나 읽기 실패하면 빈 문자열 — 프롬프트에서 단어집만 빠지고
    파이프라인은 정상 동작한다. JSON·CAP 갱신 시 반영하려면 프로세스 재시작.
    """
    try:
        raw = _GLOSSARY_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""
    try:
        data = json.loads(raw)
    except Exception:
        return raw  # 파싱 실패 시 원문(차선) — 최소한 정적 용어는 들어간다
    for key in _GLOSSARY_ENTITY_KEYS:
        v = data.get(key)
        if isinstance(v, list) and len(v) > _GLOSSARY_ENTITY_CAP:
            data[key] = v[:_GLOSSARY_ENTITY_CAP]
    return json.dumps(data, ensure_ascii=False, indent=2)


# 2-2. JSON 자동 파서 생성
parser = JsonOutputParser()

# 2-3. 문자열 기반 프롬프트 템플릿 생성
router_prompt = PromptTemplate(
    template="""당신은 공시 분석 서비스의 라우터입니다.
사용자의 메시지와 대화 맥락을 분석하여 의도를 'direct' / 'ctx' / 'global' 중 하나로 분류하세요.

- direct: 단순한 인사, 감사 표시, 검색이 필요 없는 잡담.
- global: 특정 회사 하나가 아니라 업계·시장 전체의 구조를 묻는 매크로/주제형 질문.
  예) "반도체 업계 전체 구조 요약", "전반적으로 어떤 그룹·계열로 나뉘나", "밸류체인/생태계가 어떻게 연결돼 있나",
      "가장 큰 군집(계열)은 무엇인가", "산업 전체적으로 기업들이 어떻게 연결돼 있나".
- ctx: 위에 해당하지 않으면서 특정 기업의 공시 정보, 재무제표, 관계사 등 데이터 검색이 필요한 질문.

설명이나 마크다운 없이 반드시 아래의 JSON 형식으로만 응답해야 합니다.
{{"intent": "ctx"}} 또는 {{"intent": "direct"}} 또는 {{"intent": "global"}}

[대화 기록]
{chat_history}

결과 JSON:""",
    input_variables=["chat_history"]
)

# 라우터가 허용하는 의도 — 그 외 값은 ctx 로 폴백.
_VALID_INTENTS = {"direct", "ctx", "global"}

def _run_router(chat_history: str) -> dict:
    prompt_text = router_prompt.format(chat_history=chat_history)
    return parser.invoke(json_llm.invoke(prompt_text))


# ==========================================
# 3. 라우터 노드 함수 적용
# ==========================================
def router_node(state: dict):
    """Route: 의도 분류"""
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "direct"}

    chat_history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in messages if hasattr(msg, 'type') and hasattr(msg, 'content')]
    )

    try:
        parsed_response = _run_router(chat_history_text)
        intent = parsed_response.get("intent", "ctx")
        # 모델이 엉뚱한 값을 주면 기존 동작(ctx)으로 폴백.
        if intent not in _VALID_INTENTS:
            intent = "ctx"

    except Exception as e:
        print(f"⚠️ [Router Node] 파싱 실패. 에러: {e}")
        print("기본값(ctx)으로 폴백합니다.")
        intent = "ctx"

    _label = {"ctx": "RAG", "global": "GLOBAL", "direct": "DIRECT"}.get(intent, "RAG")
    print(f"🧭 [Router Node] 의도 분류: {_label}")
    return {"intent": intent}

def direct_response_node(state: AgentState):
   
    response = "공시 관련 질문만 답변할 수 있습니다. 특정 기업의 공시나 재무 정보가 궁금하시다면 언제든 물어보세요!"
    
    return {"messages": [response]}

str_parser = StrOutputParser()

# 2-2. 문자열 기반 프롬프트 템플릿 생성
reconstruct_prompt = PromptTemplate(
    template="""당신은 공시 분석 시스템의 질의 재구성(Query Reconstruction) 전문가입니다.
이 시스템은 RDB / Vector / Graph(Neo4j) 검색기를 함께 사용합니다. 사용자의 구어체 질문을 검색기가 잘 이해하도록, 아래 [DB 스키마 및 단어집]을 참고해 표준 용어가 드러나는 독립적인 질문으로 다시 작성하세요.

[작성 규칙]
1. '그 회사', '거기', '이전 내용' 같은 대명사·생략된 주어는 [대화 기록]을 추적해 정확한 기업명·키워드로 치환하세요.
2. 사용자의 일상어·구어체는 단어집의 표준 용어로 정규화하되, 원문이 묻는 '범위'는 바꾸지 마세요. 한 가지를 물으면 표준어 하나로 바꾸고(예: "돈 얼마 벌었어?" → "매출액"), 전체·여러 측면을 묻는 넓은 질문(관계도/전체 관계/기업분석/개요/현황 등)이면 그 넓은 범위를 그대로 보존하고 한 측면(예: 특수관계자, 재무지표)으로 좁히지 마세요. 단, 사용자가 명시하지 않은 연도·항목·기준은 임의로 지어내지 마세요.
3. 이전 맥락을 모르는 사람이나 DB 검색기라도 완벽하게 이해할 수 있는 하나의 독립적인 질문으로 작성하세요.
4. 부연 설명이나 인사말 없이 오직 '재구성된 질문' 자체만 텍스트로 출력하세요.
5. 사용자가 말하지 않은 항목으로 범위를 임의로 넓히지 마세요('등·기타·예를 들어'로 없는 항목·지표를 덧붙이지 말 것). 반대로 사용자가 넓게 물은 범위를 임의로 한 측면으로 좁히지도 마세요.

[예시]
원본: "sk하이닉스 협력사 중에 잘나가는 데 어디야?"
O: "SK하이닉스에 제품을 공급하는 협력사 중 매출액이 가장 높은 회사는 어디입니까?"
X: "SK하이닉스 협력사 중 매출액, 영업이익 등이 가장 높은 회사는 어디입니까?"
원본: "동진쎄미켐 관계도 보여줘"
O: "동진쎄미켐의 전체 기업 관계 구조는 어떻게 구성되어 있습니까?"
X: "동진쎄미켐의 특수관계자는 누구입니까?"
원본: "동진쎄미켐 기업분석해줘"
O: "동진쎄미켐의 기업 개요와 주요 현황을 분석해 주십시오."
X: "동진쎄미켐의 재무지표를 분석해 주십시오."

[DB 스키마 및 단어집 (JSON)]
{glossary}

[대화 기록]
{chat_history}

재구성된 질문:""",
    input_variables=["chat_history", "glossary"]
)

def _run_reconstruct(chat_history: str) -> str:
    prompt_text = reconstruct_prompt.format(
        chat_history=chat_history,
        glossary=_load_glossary_text(),
    )
    return str_parser.invoke(llm.invoke(prompt_text))


def context_reconstruct_node(state: AgentState):
    """Ctx: 질문 문맥 재구성 (메모리 활용)"""
    messages = state.get("messages", [])

    print("✏️ [Context Node] 문장 재구성 중...")

    chat_history_text = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in messages if hasattr(msg, 'type') and hasattr(msg, 'content')]
    )

    reconstructed_query = _run_reconstruct(chat_history_text).strip()
    
    # 원본 질의 추출 (안전하게 처리)
    original_query = messages[-1].content if messages else "None"
    
    print(f"   -> 원본 질의: {original_query}")
    print(f"   -> 재구성됨: {reconstructed_query}")

    return {"reconstructed_query": reconstructed_query}


#======================================================================
# Result Check 노드 — 규칙 기반(LLM 미사용) 검색 결과 충분성 체크
#======================================================================
# 디버그 표시용 — 모든 검색 소스의 State 키 → 사용자에게 보여줄 한국어 명칭.
_SOURCE_LABELS: dict[str, str] = {
    "rdb_results": "정형 데이터(재무·공시 수치)",
    "vec_results": "문서 본문(공시 원문)",
    "graph_facts": "관계망 데이터(임원·주주·계열사 등)",
}

# gen 으로 통과하려면 결과가 비어 있으면 안 되는(필수) 소스.
#    "vec_results" 줄을 주석 처리하면 vec 가 비어도 통과한다.
_REQUIRED_SOURCES: list[str] = [
    "rdb_results",
    "vec_results",   #  각 디비 불안정 시 주석 하여 테스트 
    "graph_facts",
]


def empty_sources(state: AgentState) -> list[str]:
    """필수 소스 중 결과가 비어 있는 것들의 사용자용 명칭 목록을 반환한다.

    _REQUIRED_SOURCES 에 든 키만 검사한다(주석으로 vec 를 빼면 검사 제외).
    하나라도 비어 있으면 result_check 가 사용자에게 재질문을 유도한다.
    """
    return [
        _SOURCE_LABELS.get(key, key)
        for key in _REQUIRED_SOURCES
        if not state.get(key)
    ]


def _fmt_cell(value, width: int) -> str:
    """셀 값을 한 줄 문자열로 만들고 width 에 맞춰 자르거나 채운다."""
    if isinstance(value, float):  # relevance 등 소수는 보기 좋게 반올림.
        value = round(value, 4)
    text = "" if value is None else str(value).replace("\n", " ").replace("\r", " ")
    if len(text) > width:
        text = text[: width - 1] + "…"
    return text.ljust(width)


def _flatten_row(row: dict) -> dict:
    """중첩 dict 를 한 단계 펼친다. {'extra': {'relevance': 1.0}} → {'extra.relevance': 1.0}.

    extra/메타 같은 dict 컬럼이 표에 통짜 문자열로 박히는 걸 막아 한눈에 보이게 한다.
    """
    flat: dict = {}
    for k, v in row.items():
        if isinstance(v, dict):
            if not v:
                continue  # 빈 dict 는 컬럼 만들 게 없으니 스킵.
            for sub_k, sub_v in v.items():
                flat[f"{k}.{sub_k}"] = sub_v
        else:
            flat[k] = v
    return flat


def _prepare_table(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """row 들을 평탄화하고, 표시할 컬럼 목록(등장순 키 합집합 - 중복컬럼 제거)을 만든다.

    터미널 표와 HTML 덤프가 같은 컬럼/순서를 쓰도록 공유한다.
    """
    rows = [_flatten_row(r) if isinstance(r, dict) else r for r in rows]

    cols: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k in row:
            if k not in cols:
                cols.append(k)

    # 내용이 모든 row 에서 앞 컬럼과 똑같은(중복) 컬럼은 제거 — 예: graph 의 value==name.
    deduped: list[str] = []
    for c in cols:
        if not any(all(r.get(d) == r.get(c) for r in rows) for d in deduped):
            deduped.append(c)
    return rows, deduped


def _print_rows_table(rows: list[dict], indent: str = "        ", max_col: int = 28) -> None:
    """row(dict) 리스트를 정렬된 표로 출력한다. 컬럼은 키 합집합, 폭은 내용에 맞춰 자동."""
    rows, cols = _prepare_table(rows)
    if not cols:  # dict 가 아닌 row 면 그냥 줄줄이 출력.
        for i, row in enumerate(rows):
            print(f"{indent}[{i:>3}] {json.dumps(row, ensure_ascii=False, default=str)}")
        return

    # 각 컬럼 폭 = min(헤더/내용 최대 길이, max_col). '#' 는 인덱스 컬럼.
    widths = {c: min(max(len(c), *(len(str(r.get(c, ""))) for r in rows)), max_col) for c in cols}
    idx_w = max(1, len(str(len(rows) - 1)))

    header = f"{indent}{'#'.rjust(idx_w)} │ " + " │ ".join(_fmt_cell(c, widths[c]) for c in cols)
    print(header)
    print(f"{indent}{'─' * idx_w}─┼─" + "─┼─".join("─" * widths[c] for c in cols))
    for i, row in enumerate(rows):
        line = f"{indent}{str(i).rjust(idx_w)} │ " + " │ ".join(_fmt_cell(row.get(c), widths[c]) for c in cols)
        print(line)


# ResultCheck 상세 덤프 저장 위치. POLARIS_RESULTCHECK_DIR 로 override.
_RESULTCHECK_DUMP_DIR = Path(
    os.environ.get("POLARIS_RESULTCHECK_DIR")
    or Path(__file__).resolve().parent.parent / "logs" / "resultcheck"
)
# 보관할 최신 덤프 개수. 이보다 오래된 건 매 실행마다 정리. (POLARIS_RESULTCHECK_KEEP 로 override)
_RESULTCHECK_KEEP = int(os.environ.get("POLARIS_RESULTCHECK_KEEP") or 5)


def _prune_resultcheck_dumps(keep: int = _RESULTCHECK_KEEP) -> None:
    """덤프 폴더에서 최신 keep 개만 남기고 오래된 resultcheck_*.html 을 삭제한다."""
    files = sorted(
        _RESULTCHECK_DUMP_DIR.glob("resultcheck_*.html"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass  # 잠긴 파일 등은 다음 실행에서 다시 시도.


def _html_table(rows: list[dict]) -> str:
    """row 리스트를 정렬·검색 가능한 HTML <table> 문자열로 만든다."""
    from html import escape

    rows, cols = _prepare_table(rows)
    if not rows:
        return '<p class="empty">결과 없음</p>'
    if not cols:  # dict 가 아닌 row — JSON 으로.
        items = "".join(
            f"<tr><td>{i}</td><td>{escape(json.dumps(r, ensure_ascii=False, default=str))}</td></tr>"
            for i, r in enumerate(rows)
        )
        return f"<table><thead><tr><th>#</th><th>value</th></tr></thead><tbody>{items}</tbody></table>"

    def cell(v):
        if isinstance(v, float):
            v = round(v, 4)
        return escape("" if v is None else str(v))

    head = "<th>#</th>" + "".join(f"<th>{escape(c)}</th>" for c in cols)
    body = "".join(
        "<tr><td>" + str(i) + "</td>"
        + "".join(f"<td>{cell(r.get(c))}</td>" for c in cols)
        + "</tr>"
        for i, r in enumerate(rows)
    )
    return f"<table class='sortable'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _resultcheck_question(state: AgentState) -> str:
    """덤프 제목용 질문 텍스트 — 원본 사용자 질문(마지막 HumanMessage)을 우선."""
    for msg in reversed(state.get("messages", []) or []):
        if getattr(msg, "type", "") == "human" or msg.__class__.__name__ == "HumanMessage":
            content = getattr(msg, "content", None)
            if content:
                return str(content)
    return state.get("reconstructed_query") or "(질문 미상)"


def _timing_table_html(state: AgentState, total_elapsed: float | None) -> str:
    """노드별 소요시간 + 총 응답시간을 표로. 느린 순 정렬."""
    from html import escape

    timings: dict = state.get("node_timings") or {}
    if not timings and total_elapsed is None:
        return ""
    label = {
        "route": "의도 분류(route)", "ctx": "질의 재구성(ctx)",
        "rdb": "RDB 검색(rdb)", "vec": "벡터 검색(vec)", "graph": "그래프 검색(graph)",
        "result_check": "결과 점검(result_check)", "gen": "보고서 생성(gen)",
        "direct": "단순 응답(direct)",
    }
    rows = sorted(timings.items(), key=lambda kv: kv[1], reverse=True)
    body = "".join(
        f"<tr><td>{escape(label.get(k, k))}</td><td>{v:.3f}s</td></tr>" for k, v in rows
    )
    total_row = (
        f"<tr class='total'><td>⏱ 총 응답시간</td><td>{total_elapsed:.3f}s</td></tr>"
        if total_elapsed is not None else ""
    )
    note = "<span class='note'>※ rdb·vec·graph 는 병렬 실행이라 합≠총시간</span>"
    return (
        "<h2>⏱ 소요시간</h2>"
        f"<table class='timing'><thead><tr><th>노드</th><th>소요</th></tr></thead>"
        f"<tbody>{total_row}{body}</tbody></table>{note}"
    )


def _dump_resultcheck_html(state: AgentState, total_elapsed: float | None = None) -> Path:
    """세 소스의 전체 결과 + 질문 + 노드별/총 소요시간을 단일 HTML 로 떨군다. 경로 반환."""
    from datetime import datetime
    from html import escape

    counts = {label: len(state.get(key) or []) for key, label in _SOURCE_LABELS.items()}
    question = _resultcheck_question(state)
    q_short = question if len(question) <= 40 else question[:40] + "…"
    reconstructed = state.get("reconstructed_query") or ""

    _RESULTCHECK_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    # 밀리초까지 붙여 같은 초에 여러 번 실행돼도 파일이 서로 덮어쓰지 않게.
    ts = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
    path = _RESULTCHECK_DUMP_DIR / f"resultcheck_{ts}.html"

    sections = []
    for key, label in _SOURCE_LABELS.items():
        rows = state.get(key) or []
        sections.append(
            f"<h2>{label} <span class='cnt'>{len(rows)}건</span></h2>{_html_table(rows)}"
        )

    # global 경로는 rdb/vec/graph 가 비고 커뮤니티 요약으로 답하므로, 있으면 별도 섹션·집계.
    community = state.get("community_results") or []
    if community:
        label = "업계/그룹 종합(커뮤니티)"
        counts[label] = len(community)
        sections.append(
            f"<h2>{label} <span class='cnt'>{len(community)}건</span></h2>{_html_table(community)}"
        )

    summary = " · ".join(f"{label} {n}건" for label, n in counts.items())
    total_txt = f" · 총 {total_elapsed:.3f}s" if total_elapsed is not None else ""
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>[{q_short}] ResultCheck {ts}</title>
<style>
 body{{font-family:'Segoe UI',sans-serif;margin:24px;color:#222;background:#fafafa}}
 h1{{font-size:18px}} h2{{font-size:15px;margin-top:28px;border-bottom:2px solid #ddd;padding-bottom:4px}}
 .cnt{{color:#888;font-weight:normal;font-size:13px}}
 .summary{{background:#eef;padding:10px 14px;border-radius:8px;font-size:14px}}
 .q{{background:#fff8e1;padding:10px 14px;border-radius:8px;font-size:14px;margin:8px 0;border-left:4px solid #fbc02d}}
 .rq{{background:#e8f0fe;padding:10px 14px;border-radius:8px;font-size:14px;margin:8px 0;border-left:4px solid #4285f4}}
 .note{{color:#999;font-size:12px}}
 input{{margin:10px 0;padding:6px 10px;width:320px;font-size:14px}}
 table{{border-collapse:collapse;font-size:13px;background:#fff;width:100%}}
 table.timing{{width:auto;min-width:320px}}
 th,td{{border:1px solid #ddd;padding:4px 8px;text-align:left;max-width:420px;
   overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
 th{{background:#f0f0f0;position:sticky;top:0;cursor:pointer;user-select:none}}
 tr:nth-child(even){{background:#f8f8f8}}
 tr.total td{{font-weight:bold;background:#e8f5e9}}
 td:hover{{white-space:normal;overflow:visible}}
 .empty{{color:#999}}
</style></head><body>
<h1>🔎 ResultCheck — {ts}</h1>
<div class="q"><b>원본 질문</b> &nbsp;{escape(question)}</div>
{f'<div class="rq"><b>재구성된 질문</b> &nbsp;{escape(reconstructed)}</div>' if reconstructed else ''}
<div class="summary">{summary} · 합계 {sum(counts.values())}건{total_txt}</div>
{_timing_table_html(state, total_elapsed)}
<input id="q" placeholder="전체 검색 (행 필터)…" oninput="filt(this.value)">
{''.join(sections)}
<script>
 function filt(v){{v=v.toLowerCase();document.querySelectorAll('table.sortable tbody tr').forEach(function(r){{
   r.style.display=r.innerText.toLowerCase().includes(v)?'':'none';}});}}
 document.querySelectorAll('table.sortable th').forEach(function(th){{th.onclick=function(){{
   var t=th.closest('table'),i=[].indexOf.call(th.parentNode.children,th),
   rs=[].slice.call(t.tBodies[0].rows),asc=th.dataset.asc=th.dataset.asc==='1'?'':'1';
   rs.sort(function(a,b){{var x=a.cells[i].innerText,y=b.cells[i].innerText,
     n=parseFloat(x)-parseFloat(y);var c=isNaN(n)?x.localeCompare(y):n;return asc?c:-c;}});
   rs.forEach(function(r){{t.tBodies[0].appendChild(r);}});}};}});
</script>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    _prune_resultcheck_dumps()  # 최신 _RESULTCHECK_KEEP 개만 남기고 정리.
    return path


def route_result_check(state: AgentState) -> str:
    """result_check 분기.

    - global(매크로/업계): 세 소스(rdb/vec/graph)는 비기 마련이므로 검사하지 않고,
      커뮤니티 요약(community_results)이 하나라도 있으면 'gen'.
    - 그 외: 필수 소스가 모두 있으면 'gen', 하나라도 비면 'end'.
    """
    if state.get("intent") == "global":
        return "gen" if state.get("community_results") else "end"
    return "end" if empty_sources(state) else "gen"


def result_check_node(state: AgentState):
    """Result Check: LLM 없이 AgentState 의 검색 결과 유무만 규칙 기반으로 점검한다.

    필수 소스(_REQUIRED_SOURCES)가 모두 채워져 있으면 그대로 통과시켜 gen 노드가
    포매팅하게 하고, 하나라도 비어 있으면 어떤 검색에서 결과가 없었는지 명시하며
    더 구체적인 질문을 요청하는 답변을 만들어 END 로 종료한다.
    """
    print("🔎 [ResultCheck Node] 검색 결과 충분성 점검 중...")

    # global(매크로/업계)은 rdb/vec/graph 가 비기 마련이라 3소스 검사 대신
    # 커뮤니티 요약 유무만 본다. route_result_check 가 동일 기준으로 분기한다.
    if state.get("intent") == "global":
        n = len(state.get("community_results") or [])
        print(f"   -> 글로벌 경로: 커뮤니티 요약 {n}건 → {'통과✅' if n else '불충분❌'}")
        return {}

    # 디버깅: 터미널엔 소스별 건수 요약만 찍는다. 전체 row 상세 + 질문 + 노드별/총
    # 소요시간은 파이프라인 종착점(save 노드)에서 정렬·검색 가능한 HTML 로 떨군다.
    counts = {label: len(state.get(key) or []) for key, label in _SOURCE_LABELS.items()}
    total = sum(counts.values())
    print("   ┌─ 검색 결과 요약 " + "─" * 40)
    for label, n in counts.items():
        print(f"   │  {label}: {n}건")
    print(f"   └─ 합계: {total}건 " + "─" * 40)

    empties = empty_sources(state)

    # 필수 소스가 모두 있으면 통과 — gen 으로.
    if not empties:
        print("   -> 통과✅ (필수 검색 결과 모두 존재)")
        return {}

    # 하나라도 비어 있으면 어떤 소스가 비었는지 명시하고 재질문을 유도한다.
    print(f"   -> 불충분❌ (결과 없음: {', '.join(empties)})")
    empty_text = ", ".join(empties)
    guidance = (
        f"{empty_text}에서 검색 결과를 찾지 못했습니다. "
        "기업명, 연도, 항목(예: 매출액, 자회사 등)을 포함해 좀 더 구체적으로 "
        "질문해 주시면 더 정확한 답변을 드릴 수 있습니다."
    )
    return {
        "messages": [AIMessage(content=guidance)],
        "final_draft": guidance,
    }