# 뉴스 분석 탭 설계 — 우측 패널 4번째 탭

> 작성일: 2026-06-22
> 담당 영역: RDB·Vector 에이전트(내 담당) + 프론트 우측 패널
> 상태: 설계 승인 대기

---

## 1. 목표와 범위

### 무엇을 만드나
사용자 질문에 **언급된 기업의 최근 뉴스**를 실시간으로 가져와, LLM이 **우리 DART 데이터(공시·그래프)와 연관지어** 기업별로 요약·분석해 우측 패널 탭으로 보여준다.

### 발표 스토리
"공시 분석(RAG) + 기업 관계 그래프 + **최신 뉴스 맥락**을 한 화면에서" — 세 데이터가 한 질문에 연결되는 걸 우측 패널 탭으로 직접 보여준다. 이는 모노레포 상위 `CLAUDE.md`가 말하는 핵심 용도("지금 이 뉴스로 현재 상황을 분석")와 정렬된다.

### 명시적 비범위 (YAGNI)
- **뉴스 코퍼스 구축 안 함.** 상위 `CLAUDE.md`의 수집→저장→적재 파이프라인(PostgreSQL/Chroma/Neo4j)은 **이 기능과 무관**. 여기서는 질문 시점에 **실시간 fetch + 즉시 분석**만 한다. DB에 뉴스를 저장하지 않는다(분석 결과는 기존 패널 JSON에 캐시만).
- **뉴스를 검색 코어(LangGraph 노드)에 넣지 않음.** 검색 3종(rdb/vec/graph)은 그대로. 뉴스는 답변 생성 **이후** 별도 경로로 붙는다.
- **클러스터링/감성 추출 LLM 파이프라인 없음.** 감성 태그는 표시하되 LLM 요약에 포함시켜 한 번에 받는다(별도 호출 X).

---

## 2. 핵심 설계 결정

### 결정 1: digest 패턴을 그대로 재사용 (지연 로딩)
뉴스 fetch + LLM 분석은 외부 API 왕복 + LLM 1회라 느리다. 이걸 `/api/chat` 동기 응답에 넣으면 답변이 그만큼 늦는다.

→ **기존 `digest` 엔드포인트와 동일한 지연 패턴**을 따른다:
1. `/api/chat`는 답변을 먼저 반환 (뉴스 없이, `news_loading` 힌트만)
2. 프론트가 답변 렌더 후 `message_id`로 `POST /api/chat/news` 별도 호출
3. 결과를 패널 JSON(`search_plan` 컬럼)에 캐시 → 세션 재진입 시 복원

이 패턴은 이미 `core/digest.py` + `/api/chat/digest` + `chat_store.set_message_digest`로 검증돼 있다. 뉴스는 그 쌍둥이다.

### 결정 2: 네이버 검색 API + 인링크 본문 fetch
- 네이버 검색 API(`openapi.naver.com/v1/search/news.json`)는 제목 + ~100자 `description`만 준다 → 분석 근거로 부족.
- API가 주는 `link`(인링크 `n.news.naver.com`)로 **본문을 따로 fetch**한다. 인링크는 HTML 구조가 일정해 파싱이 안정적(상위 `CLAUDE.md` §collect/fetch 근거).
- 본문 fetch는 `C:\DART\news_fetcher.py`의 추출 로직을 재활용하되 `requests`→`httpx`, BS4 셀렉터→`trafilatura` 우선으로 현대화.
- **graceful degrade**: 본문 fetch 실패 시 제목+`description`만으로 분석 진행(0건 처리하지 않음).

### 결정 3: 기업별 그룹 + 기업별 벡터 검색으로 DART 연결
- 질문에서 추출한 기업(들) 각각에 대해:
  1. 네이버 API로 최근 뉴스 N건(기본 5) fetch
  2. 그 기업의 뉴스 제목/본문을 쿼리로 `search_vector_db(corp_codes=[code])` 1회 → 관련 DART 청크 2~3건
  3. 뉴스 + DART 청크를 함께 LLM에 넣어 "최근 동향 요약 + 공시 근거 연결" 생성
- 추가 비용은 **기업당 벡터 검색 1회 + LLM 1회**. 벡터 검색은 회사 pre-filter라 빠르다(오늘 검증: 회사 필터 쿼리 < 1초).

### 결정 4: 기존 검색 코어·기존 3탭 무변경
- LangGraph 노드, `tool/vector_store.py`·`tool/rdb_client.py`의 검색 로직, 기존 패널 직렬화(`serialize_state`)는 **건드리지 않는다**.
- 뉴스 탭은 순수 **추가**. 기존 관계도·재무·원본문서 탭은 지금과 동일하게 동작.

---

## 3. 아키텍처 / 데이터 흐름

```
[기존 경로 — 변경 없음]
POST /api/chat
  → LangGraph invoke → 답변 + 패널(graph/documents/financials)
  → message_id 반환 (panel JSON 에 news_loading 힌트 포함)

[신규 경로 — 지연 로딩]
프론트: 답변 렌더 후 message_id 로 호출
POST /api/chat/news { message_id }
  → chat_store.get_message_panel(message_id)         # 저장된 패널 복원
       → 분석 대상 기업(corp_codes/names) + reconstructed_query 추출
  → core/news.build_news_analysis(companies, query)
       │
       ├─ tool/news_client.fetch_company_news(corp_name, n=5)   # 네이버 API
       │     → [{title, link, description, pub_date}, ...]
       │     → 인링크 본문 fetch (trafilatura, 병렬, 실패 무시)
       │
       ├─ tool/vector_store.search_vector_db(뉴스텍스트, corp_codes=[code])  # DART 연결
       │     → 관련 공시 청크 2~3건
       │
       └─ LLM 1회: 뉴스 + DART 청크 → 기업별 분석 카드(JSON)
  → chat_store.set_message_news(message_id, news)    # 패널 JSON 캐시
  → NewsResponse { news: [...] } 반환
```

### 모듈 경계 (단일 책임)

| 모듈 | 책임 | 의존 |
|---|---|---|
| `tool/news_client.py` (신규) | 네이버 API 호출 + 인링크 본문 fetch. 순수 데이터 수집, LLM 없음 | `httpx`, `trafilatura` |
| `core/news.py` (신규) | 뉴스 + DART 청크 → LLM 연관 분석(JSON). `digest.py`의 쌍둥이 | `news_client`, `vector_store`, `config.llm` |
| `main.py` `/api/chat/news` (신규 엔드포인트) | 지연 로딩 진입점. `digest` 엔드포인트와 동형 | `core.news`, `chat_store` |
| `tool/chat_store.py` `set_message_news` (신규 함수) | 패널 JSON 의 `news` 필드 영속화. `set_message_digest`와 동형 | 기존 |
| `Chatbot.tsx` 뉴스 탭 (프론트 추가) | 4번째 탭 UI + 지연 fetch 트리거 | 기존 패널 구조 |

각 모듈은 독립 테스트 가능: `news_client`는 API 키만 있으면 단독 실행, `core/news`는 가짜 뉴스/청크 주입으로 검증.

---

## 4. 데이터 스키마

### 4-1. `news_client.fetch_company_news` 반환
```python
class NewsItem(TypedDict):
    title: str           # 네이버 API title (HTML 태그 제거)
    url: str             # n.news.naver.com 인링크
    description: str     # 네이버 API description (~100자)
    body: str            # 인링크 본문 (fetch 성공 시, 실패 시 "")
    press: str           # 언론사명 (URL oid 또는 본문에서, best-effort)
    pub_date: str        # YYYY-MM-DD
```

### 4-2. `core/news.build_news_analysis` 반환 (= 프론트 `news` 필드)
```python
class NewsAnalysisCompany(TypedDict):
    corp_name: str                  # "삼성전자"
    corp_code: str                  # 8자리
    summary: str                    # LLM 생성 — 최근 동향 2~3문장 (마크다운)
    relevance: str                  # LLM 생성 — DART 공시와의 연관 설명 (마크다운, 근거 포함)
    sentiment: str                  # "positive" | "negative" | "neutral" — 전반 논조
    articles: list[NewsArticleCard] # 개별 기사 카드

class NewsArticleCard(TypedDict):
    title: str
    url: str
    press: str
    pub_date: str
    sentiment: str                  # 기사별 논조
    evidence: list[str]             # 연결된 DART 근거 라벨 (예: "사업보고서 §II")
```

프론트 `news` 페이로드 = `{ companies: list[NewsAnalysisCompany] }`.

### 4-3. 패널 JSON 확장 (기존 `search_plan` 컬럼)
기존 패널 JSON에 `news` 필드만 추가. 스키마 마이그레이션 없음(digest와 동일하게 longtext JSON 안에서 확장).
```json
{ "graph": {...}, "documents": [...], "financials": [...],
  "digest": "...", "news": { "companies": [...] } }
```

---

## 5. 분석 대상 기업 결정

`POST /api/chat/news`는 `message_id`만 받으므로, 저장된 패널 JSON에서 대상 기업을 복원한다:

1. **1순위**: 패널 JSON의 `reconstructed_query`로 `extract_filter_signals()` 재실행 → `corp_codes`.
   - 이미 `tool/vector_store.py`에 있는 함수. 회사명→코드 매핑 + 비교형 판별 로직 재사용.
2. **2순위**: `documents`에 등장한 `corp_name`들(공시가 조회된 회사).
3. 둘 다 비면 → 빈 `news` 반환(뉴스 탭 미표시). 매크로/업계 질문은 특정 기업이 없으므로 정상적으로 0건.

**상한**: 분석 기업은 최대 3개로 제한(`NEWS_MAX_COMPANIES=3`). 비교형 질문이 4사 이상이면 상위 3개만(스코어/등장순). API 호출·LLM 비용 통제.

---

## 6. 네이버 API 세부

- 엔드포인트: `GET https://openapi.naver.com/v1/search/news.json`
- 파라미터: `query={corp_name}`, `sort=date`, `display={NEWS_PER_COMPANY:5}`
- 헤더: `X-Naver-Client-Id`, `X-Naver-Client-Secret` (`.env`)
- **인링크 필터**: `link`이 `n.news.naver.com` 패턴인 것만 통과(상위 `CLAUDE.md` §5-1). 아웃링크는 본문 파싱 불안정 → 제목+description만 사용하거나 폐기.
- **rate limit 방어**: 호출 간 짧은 딜레이 + 타임아웃 + 실패 시 빈 리스트(graceful). 일 25,000회 한도는 데모 규모에서 무관.
- 본문 fetch는 인링크 N건 **병렬**(`httpx.AsyncClient` 또는 스레드). 동기↔비동기 브리지는 기존 `config/llm.py` 패턴 참고(노드는 sync 호출).

### `.env` 추가
```
NAVER_CLIENT_ID=...
NAVER_CLIENT_SECRET=...
NEWS_PER_COMPANY=5
NEWS_MAX_COMPANIES=3
NEWS_ENABLED=true        # 키 없거나 false 면 기능 자체를 끄고 빈 결과
```
키가 없으면(`NEWS_ENABLED!=true` 또는 키 공백) `/api/chat/news`는 항상 빈 `news`를 반환 → 프론트는 탭을 숨김. 발표 환경에서만 켠다.

---

## 7. LLM 프롬프트 (core/news.py)

`digest.py`처럼 **JSON only** 강제, 실패 시 빈 결과로 격리.

입력 구성(기업별):
```
[기업] 삼성전자
[최근 뉴스]
1. (2026-06-21 한국경제) "삼성전자 HBM3E 수출규제 추가 검토"
   본문: ...(500자)...
2. ...
[관련 공시 발췌 (우리 DB)]
- 사업보고서(2025) §II 사업의 내용: ...(400자)...
- 분기보고서(2025) §IV MD&A: ...
```

출력(JSON): 기업별 `summary`(최근 동향) + `relevance`(공시 연관) + 기사별 `sentiment`/`evidence`.

프롬프트 규칙:
- 공시 발췌에 **실제로 있는** 사실만 근거로 연결(`digest.py` 규칙 0/1 차용 — 환각·"없음" 평가 금지).
- 연관이 약하면 `relevance`를 비우고 `summary`만 — 억지 연결 금지.
- `evidence`는 입력에 제공된 공시 라벨만 인용(지어내지 않음).

---

## 8. 프론트 (Chatbot.tsx)

### 추가 사항
- `PanelKey` 타입에 `'news'` 추가.
- `PANEL_META`에 `news: { label: '뉴스 분석', icon: Newspaper }` 추가.
- `Message` 인터페이스에 `news?: NewsData`, `newsLoading?: boolean` 추가.
- `availableTabs`: `news`에 데이터(또는 로딩 중)면 탭 노출.
- 버튼: 기존 `원본 문서 N건`·`관계도` 옆에 `뉴스 분석 N건` 버튼(amber 톤).
- 지연 fetch: `handleSend`에서 답변 수신 후 `digest`와 **동일한 비차단 패턴**으로 `POST /api/chat/news` 호출 → `news` 채움.
- 세션 복원(`loadSession`): 저장된 패널 JSON의 `news`를 그대로 복원(추가 fetch 없음).

### 탭 내용 (목업 `news-tab.html` 기준)
- 상단: AI 종합 요약(`summary`) — amber 톤 카드.
- 기업별 섹션: 기업명 + 논조 점 + `relevance`(공시 근거 칩 포함).
- 기사 카드: 제목(링크) + 언론사·날짜 + 감성 배지 + evidence 칩.

기존 3탭 UI·로직은 **변경 없음**.

---

## 9. 불변식 / 주의 (기존 CLAUDE.md 준수)

1. **`/api/chat/news`는 sync `def` 핸들러** — `/api/chat`·`/api/chat/digest`와 동일 이유(이벤트 루프 보호, 불변식 §9-1).
2. **읽기 전용** — 뉴스 기능은 3DB에 쓰지 않는다. 분석 결과는 기존 `chat_messages.search_plan` JSON에만 캐시(대화기록 영속화 경로, 신규 테이블 없음).
3. **실패는 빈 결과로 degrade** — 네이버 API 다운·키 없음·본문 fetch 실패·LLM 실패 모두 빈 `news` 반환. 챗봇 본체와 기존 패널은 영향 없음.
4. **검색 코어 무변경** — `tool/vector_store.py`는 `search_vector_db(corp_codes=...)`를 **호출만** 한다(이미 있는 공개 API). 내부 수정 없음.
5. **비밀키는 `.env`** — `NAVER_CLIENT_SECRET` 커밋 금지.
6. **저작권/약관** — 뉴스 원문은 분석 입력으로만 LLM에 전달하고 **DB 영속화·재배포 안 함**. 프론트는 제목·요약·링크만 노출(원문 본문 전체를 화면에 띄우지 않음). 상위 `CLAUDE.md` §12 준수.

---

## 10. 구현 순서 (개략)

1. `tool/news_client.py` — 네이버 API + 인링크 본문 fetch (단독 실행 테스트).
2. `core/news.py` — 뉴스+청크 → LLM 분석 JSON (가짜 입력 테스트).
3. `tool/chat_store.py` — `set_message_news` 추가.
4. `main.py` — `POST /api/chat/news` 엔드포인트 + `/api/chat` 패널에 `news_loading` 힌트.
5. `Chatbot.tsx` — 뉴스 탭 + 지연 fetch + 세션 복원.
6. `.env.example` — 네이버 키·플래그 항목 추가(값 없이).
7. eval/수동 점검 — 실제 기업 질문으로 뉴스 fetch → 분석 품질 확인.

세부 단계는 구현 계획(writing-plans)에서 확정한다.
