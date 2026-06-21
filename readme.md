# 🤖 LangGraph Multi-Agent RAG API

LangGraph와 FastAPI를 활용하여 구축한 **멀티 에이전트 기반 맞춤형 RAG(검색 증강 생성) 서비스**입니다. 
사용자의 의도를 분석하여 동적으로 라우팅하고, 장/단기 메모리를 활용하여 개인화된 응답(말투, 난이도 등)을 제공합니다.

## ✨ 주요 기능 (Key Features)

- **지능형 라우팅 (Intelligent Routing)**: 단순 질문/잡담과 정보성 RAG 질문을 구분하여 처리 경로를 최적화합니다.
- **복합 RAG 파이프라인 (Hybrid RAG Core)**: 정형 데이터(RDB), 비정형 문서(Vector), 관계망(Graph)을 병렬로 검색하고 병합합니다.
- **자기 반성 (Reflection)**: LLM이 스스로 검색 결과의 충분성을 검증하고, 정보가 부족하면 검색 계획을 다시 수립합니다.
- **개인화 메모리 시스템 (Dual Memory System)**:
  - `Checkpointer`: 세션(`thread_id`) 단위의 단기 대화 기록을 유지하여 문맥을 파악합니다. SQLite를 활용하여 영구 저장됩니다.
  - `Store`: 사용자(`user_id`) 단위의 장기 선호도(선호하는 말투, 설명 레벨 등)를 영구 저장하고 렌더링에 반영합니다.
- **FastAPI 서빙**: 직관적인 Swagger UI 문서와 비동기 처리를 지원하는 빠른 RESTful API를 제공합니다.

---

## 🛠️ 기술 스택 (Tech Stack)

- **Language**: Python 3.10+
- **AI/LLM Framework**: LangGraph, LangChain Core, LangChain OpenAI
- **Web Framework**: FastAPI, Uvicorn, Pydantic
- **Environment Management**: python-dotenv


---

## 📁 프로젝트 구조 (Directory Structure)

```text
polaris/
├── .env                    # API 키 및 환경변수 (Git 업로드 제외)
├── requirements.txt        # 패키지 의존성 목록
├── main.py                 # FastAPI 서버 실행 진입점 및 엔드포인트 정의
│
├── core/                   # LangGraph 핵심 엔진
│   ├── __init__.py
│   ├── state.py            # 상태 객체 (AgentState) 타입 및 리듀서 정의
│   └── graph.py            # StateGraph 노드/엣지 조립 및 컴파일
│
├── nodes/                  # 각 에이전트 및 기능 노드 구현부 (Controller/Service)
│   ├── __init__.py
│   ├── memory.py           # 단기/장기 메모리 로드 및 저장 노드
│   ├── router.py           # 의도 분류 및 분기 노드
│   ├── rag.py              # 검색 플랜, 검색 실행, 결과 합성 및 반성 노드
│   └── render.py           # 최종 응답의 톤 앤 매너 렌더링 노드
│
├── tools/                  # 외부 시스템 및 DB 통신 도구 (Repository/DAO)
│   ├── __init__.py
│   ├── rdb_client.py       # SQL 쿼리 실행기
│   ├── vector_store.py     # 임베딩 생성 및 벡터 DB 검색기
│   └── graph_client.py     # Graph DB (Cypher) 통신기
│
└── config/                 # 공통 설정 및 템플릿
    ├── __init__.py
    └── llm.py              # LangChain LLM 객체 초기화
```

---

# 🌿 Git 브랜치 전략 및 워크플로우 (Git Flow)

본 프로젝트는 안정적인 운영 환경 유지와 효율적인 기능 개발을 위해, 실무에서 가장 널리 쓰이는 **Git Flow**를 바탕으로 간소화된 브랜치 전략을 사용합니다.

---

## 1. 브랜치 구조 및 역할 (Branch Types)

프로젝트의 모든 브랜치는 목적에 따라 아래와 같이 엄격하게 구분하여 사용합니다.

| 브랜치 명 | 생성 위치 | 병합 대상 | 역할 및 설명 |
| :--- | :--- | :--- | :--- |
| **`main`** | - | - | **[운영/프로덕션 배포]** 언제든 사용자에게 배포할 수 있는 가장 안정적인 코드가 모이는 곳입니다. 직접 커밋은 엄격히 금지됩니다. |
| **`develop`** | `main` | `main` | **[개발 통합/테스트]** 다음 버전 배포를 위해 개발된 기능들이 모이는 중심 브랜치입니다. 모든 기능(feat)과 수정(fix)은 이곳으로 병합됩니다. |
| **`feat/*`** | `develop` | `develop` | **[새로운 기능 개발]** 새로운 기능이나 모듈을 개발할 때 생성합니다. (예: `feat/add-vector-db`, `feat/user-login`) |
| **`fix/*`** | `develop` | `develop` | **[버그 수정]** 개발 중 발견된 일반적인 버그를 수정할 때 생성합니다. (예: `fix/memory-leak`) |
| **`hotfix/*`**| `main` | `main`, `develop` | **[긴급 운영 수정]** `main` 브랜치(운영 환경)에서 발생한 치명적인 버그를 긴급하게 수정할 때만 예외적으로 사용합니다. |

---

## 2. 브랜치 네이밍 규칙 (Naming Conventions)

브랜치 이름은 작업의 의도와 내용을 직관적으로 파악할 수 있도록 작성합니다.
* 형식: `유형/작업내용` (단어 사이는 하이픈 `-` 사용)
* 예시:
  * `feat/rag-vector-search` (RAG 벡터 검색 기능 추가)
  * `fix/api-timeout-error` (API 타임아웃 에러 수정)
  * `docs/update-readme` (README 문서 업데이트)

---

## 3. 표준 개발 워크플로우 (Step-by-Step Workflow)

새로운 기능을 개발하거나 버그를 수정할 때 팀원 모두가 준수해야 할 표준 절차입니다.

### Step 1. 최신 상태 동기화 및 브랜치 생성
항상 `dev` 브랜치를 최신 상태로 유지한 후, 새로운 작업 브랜치를 생성합니다.
```bash
# 최신 develop 브랜치로 이동 및 동기화
git checkout dev
git pull origin dev

# 새로운 기능 브랜치 생성 및 이동
git checkout -b feat/add-new-feature

---

## 🚀 시작하기 (Getting Started)

### 1. 패키지 설치
Python 가상환경을 생성하고 활성화한 뒤, 필요한 의존성을 설치합니다.
pip install -r requirements.txt


### 2. 환경 변수 설정
프로젝트 최상위 경로에 `.env` 파일을 생성하고 OpenAI API 키를 입력합니다.

OPENAI_API_KEY="sk-your-api-key-here"


### 3. 서버 실행
(db 연결 시 docker-compose up -d )

FastAPI 서버를 실행합니다. 코드가 수정될 때마다 자동으로 재시작(`reload`) 됩니다.

python main.py

> `uvicorn main:api --reload` 로 직접 띄우지 마세요. 이 CLI 형태는 `.venv` 까지 감시 대상에
> 포함해, in-process LLM 호출이 `.venv` 패키지 파일을 건드리는 순간 요청 도중 리로드가
> 일어나 `/api/chat` 이 500 으로 끊깁니다. `python main.py` 는 감시 대상을 앱 소스
> 디렉터리로만 한정하므로 안전합니다.


---

## 🧪 API 테스트 (Usage)

서버가 실행되면 웹 브라우저에서 아래 주소로 접속하여 **Swagger UI**를 통해 API를 직접 테스트할 수 있습니다.

👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

### API 엔드포인트: `POST /api/chat`

**Request Body (JSON)**
```json
{
  "user_id": "dev_user_01",
  "message": "LangGraph로 시스템을 구축하는 방법을 알려줘.",
  "thread_id": "session_999"
}
```
* `user_id`: 장기 기억(말투, 선호도 등)을 식별하는 키 값입니다.
* `thread_id`: 단기 기억(대화 맥락)을 유지하기 위한 세션 식별자입니다.

**Response (JSON)**
```json
{
  "response": "LangGraph 시스템 구축 방법에 대한 (사용자 톤에 맞춘) 맞춤형 답변입니다...",
  "intent": "rag"
}
```