# Claude Agent SDK 영속 세션 적용 가이드 (2026)

매번 `claude` CLI 프로세스를 새로 띄우는 대신, **하나의 세션을 유지하며 반복 대화**하는 방법.
API 키 없이 **Claude Max/Pro 구독 인증**으로 동작한다. (개인/개발 MVP 용도 기준)

이 문서는 특정 프로젝트가 아니라 **임의의 파이썬 프로젝트에 이 방식을 적용하는 절차**를 다룬다.

> 이 가이드의 코드는 실제 실행·검증되었다. (claude-agent-sdk 0.2.94, Python 3.11, Windows 11)

---

## 0. 어떤 경우에 적용하나 (적용 판단)

아래 중 하나라도 해당하면 이 방식이 맞다.

- `subprocess.run(["claude", ...])` 처럼 **호출마다 CLI 프로세스를 새로 띄우고 있다.**
- 그래서 호출마다 부팅 + (있다면) **MCP 서버 재초기화**로 수 초씩 느리다.
- 직전 호출의 맥락이 다음 호출에 **안 이어진다** (예: 검색 → 상세조회가 서로 모름).

반대로, "단발성 1회 호출 후 종료"가 전부라면 굳이 영속 세션이 필요 없다. 기존 단발 호출로 충분하다.

---

## 1. 동작 원리 (왜 이게 빠르고 맥락이 유지되나)

| 구분 | 매번 새 프로세스 (`subprocess.run`) | 영속 세션 (`ClaudeSDKClient`) |
|---|---|---|
| 프로세스 | 호출마다 기동/종료 | **1개를 살려두고 재사용** |
| MCP 로드 | 호출마다 재초기화 | **세션당 1회** |
| 대화 맥락 | 매번 초기화 | 세션 내내 유지 |
| 인증 | CLI 구독 인증 | CLI 구독 인증 (동일) |

`ClaudeSDKClient`는 내부적으로 `claude` CLI 프로세스를 띄우되 **그 프로세스를 끝까지 살려두고**,
그 위로 여러 번 메시지를 보낸다. CLI가 들고 있는 구독 OAuth 인증을 그대로 쓰므로 API 키가 필요 없다.

---

## 2. 사전 조건

| 항목 | 확인 방법 | 비고 |
|---|---|---|
| claude CLI 설치 + 로그인 | `claude --version` | 구독(Max/Pro) 로그인 상태여야 함 |
| `ANTHROPIC_API_KEY` **미설정** | 환경변수 비어 있어야 함 | 설정돼 있으면 그걸 우선 써서 **과금**됨 |
| Python ≥ 3.11 | `python --version` | SDK 요구사항 |

> 인증 원리: Agent SDK는 내부적으로 `claude` CLI를 구동하고, CLI는 `claude login`으로 받은
> 구독 OAuth 인증을 그대로 쓴다. 그래서 API 키가 필요 없다.
> 단, `ANTHROPIC_API_KEY`가 환경에 있으면 그게 우선되므로 **코드에서 명시적으로 제거**한다 (4단계).

---

## 3. 설치

```bash
uv add claude-agent-sdk          # uv 프로젝트
# 또는
pip install claude-agent-sdk     # 일반 venv
```

함께 설치되는 핵심 패키지: `claude-agent-sdk`, `mcp`, `anyio` 등.

---

## 4. 적용 절차 (단계별)

### 4-1. 환경 가드 (모든 프로젝트 공통)

```python
import os, sys

# API 키가 있으면 구독 인증 대신 그걸 써버리므로 제거 (과금 방지)
os.environ.pop("ANTHROPIC_API_KEY", None)

# (선택) Windows 콘솔(cp949)에서 유니코드 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
```

### 4-2. 세션 열고 반복 대화

```python
import anyio
from claude_agent_sdk import (
    ClaudeSDKClient, ClaudeAgentOptions,
    AssistantMessage, TextBlock, ResultMessage,
)

def text_of(msg) -> str:
    """AssistantMessage에서 텍스트만 추출"""
    if isinstance(msg, AssistantMessage):
        return "".join(b.text for b in msg.content if isinstance(b, TextBlock))
    return ""

async def main():
    options = ClaudeAgentOptions(permission_mode="bypassPermissions")

    # async with → 세션(프로세스) 1개를 열고 블록 끝까지 유지
    async with ClaudeSDKClient(options=options) as client:
        for prompt in ["첫 번째 질문", "이전 맥락을 이어받는 두 번째 질문"]:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    print(text_of(msg))

anyio.run(main)
```

### 4-3. 핵심 API

| 요소 | 역할 |
|---|---|
| `ClaudeSDKClient(options=...)` | 영속 세션 클라이언트 |
| `async with client:` | 세션 1개 열고 블록 끝까지 유지 (`connect`/`disconnect` 자동) |
| `await client.query(prompt)` | 같은 세션에 메시지 전송 (여러 번 호출 가능) |
| `async for msg in client.receive_response()` | 한 턴의 응답 스트림 수신 |
| `AssistantMessage.content` → `TextBlock` | 실제 답변 텍스트 |
| `ResultMessage` | 턴 종료 신호 (`total_cost_usd`, 소요시간 등 메타) |

### 4-4. 기존 `subprocess` 호출을 교체할 때

동기 코드 안에 `run_claude_cli(prompt)` 같은 단발 함수가 흩어져 있다면, 보통 두 갈래로 옮긴다.

- **흐름 전체가 한 세션이어야 할 때** (검색→상세→분석처럼 맥락 공유가 핵심): 그 흐름 전체를
  하나의 `async with ClaudeSDKClient(...)` 블록으로 감싸고, 각 단계를 `query`/`receive_response`로 바꾼다.
- **호출부가 동기라 당장 못 바꿀 때**: 클라이언트를 한 번 열어두고, 동기 함수에서는
  `anyio.from_thread` / 별도 이벤트 루프로 위임하는 식으로 감싼다. (리팩터링 비용이 크면 흐름 단위로 점진 적용 권장)

---

## 5. ClaudeAgentOptions 주요 필드

| 필드 | 용도 |
|---|---|
| `permission_mode` | `"bypassPermissions"`(무인 자동) / `"default"`(승인 요구) 등 |
| `model` | 모델 지정 (미지정 시 CLI 기본) |
| `cwd` | 작업 디렉터리 (파일/도구 기준 경로) |
| `system_prompt` | 세션 전체에 적용할 시스템 프롬프트 |
| `mcp_servers` | 붙일 MCP 서버 정의 (6단계) |
| `allowed_tools` | 허용할 도구 이름 화이트리스트 |
| `setting_sources` | 설정 출처 (프로젝트/유저 설정 로드 여부) |
| `continue_conversation` / `resume` | 이전 대화 이어가기 / 특정 세션 재개 |
| `max_turns` | 한 호출에서 도는 최대 턴 수 제한 |

---

## 6. MCP(도구) 붙이기 — 일반형

세션에 MCP 서버를 한 번만 로드해 두면, 그 세션 안의 모든 턴이 같은 도구를 공유한다.
MCP가 세션당 1회만 초기화되므로 "도구를 쓰는 다단계 흐름"에서 특히 빠르다.

```python
options = ClaudeAgentOptions(
    permission_mode="bypassPermissions",
    mcp_servers={
        # 로컬 stdio 서버 예시
        "my-server": {
            "command": "npx",
            "args": ["-y", "@some/mcp-server"],
            "env": {"SOME_TOKEN": os.environ["SOME_TOKEN"]},
        },
        # URL(원격) 서버라면 {"url": "..."} 형태
    },
    allowed_tools=["mcp__my-server__some_tool"],   # 허용 도구만 화이트리스트
)
```

> `mcp_servers` 스키마는 등록할 서버 종류(로컬 stdio vs 원격 URL)에 따라 다르다.
> 도구 이름은 보통 `mcp__<서버키>__<도구명>` 규칙을 따른다 — 실제 서버 정의를 확정한 뒤 채운다.

---

## 7. 약관 주의 (중요)

- **개인 / 개발 MVP / 본인 데이터 자동화** → Max/Pro 구독 + Agent SDK 정상 사용 범위. 문제없음.
- 단, **남에게 서비스로 제공·재판매하거나 무인 대량 운영** → 구독 약관 위반.
  그 단계에서는 종량제 **API 키**로 전환해야 한다. (인증 부분만 교체하면 됨)
- 구독은 개인 인터랙티브 사용 기준이라 **rate limit**이 API보다 빡빡하다.
  MVP 단계에선 루프 처리량(배치 크기 등)을 작게 둘 것.

---

## 8. 검증 방법

적용 후 "세션이 정말 유지되는지"를 확인하려면, **1턴에서 준 정보를 2턴이 기억하는지** 본다.

```
1턴: "내가 좋아하는 숫자는 42야. 알겠다고만 답해."
2턴: "그 숫자에 8을 더하면? 숫자만."   → 50 이 나오면 세션 유지 성공
```

실측 예 (claude-agent-sdk 0.2.94 / Python 3.11 / Windows 11):

```
ANTHROPIC_API_KEY set? NO (구독 인증 사용)
[1턴] 네, 당신이 가장 좋아하는 숫자는 42라는 것을 알겠습니다!   (소요 2.5s, 비용 $0.0062)
[2턴] 50                                                       (소요 2.1s — 프로세스 재기동 없음)
✅ 세션 유지 성공: 2턴이 1턴의 '42'를 기억함 (42+8=50)
```

- **API 키 없이** 구독 인증으로 동작
- 2턴이 1턴의 "42"를 기억 → **세션 유지 확인**
- 턴당 ~2초, **프로세스 재기동 없음**

> 참고: `total_cost_usd`는 구독 환산 추정치로 표시될 뿐, 구독 사용 시 실제 카드 과금이 아니다.
> 이 검증 스크립트의 전체 구현은 저장소의 `test.py` 참고.

---

## 9. Codex도 같은 방식으로 가능한가

결론부터 말하면 **가능하다.** 다만 Claude의 `ClaudeSDKClient`와 1:1로 같은 API는 아니고,
Codex는 공식적으로 아래 세 가지 경로를 제공한다.

| 방식 | 언제 쓰나 | 세션 유지 방식 |
|---|---|---|
| `openai-codex` Python SDK / `@openai/codex-sdk` TypeScript SDK | 앱/스크립트 안에서 Codex를 직접 제어 | `thread`를 만들고 같은 thread에 `run()`을 반복 호출 |
| `codex exec` / `codex exec resume` | CI, 배치, 셸 파이프라인 자동화 | 저장된 세션 ID 또는 `--last`로 이전 실행 재개 |
| `codex mcp-server` | 다른 에이전트나 OpenAI Agents SDK에서 Codex를 도구처럼 호출 | `codex` 도구가 `threadId`를 반환하고 `codex-reply`가 같은 thread를 이어감 |

공식 문서 기준으로 Codex CLI/IDE는 **ChatGPT 로그인**과 **API key 로그인**을 모두 지원한다.
CLI는 유효한 세션이 없을 때 ChatGPT 로그인이 기본 경로이고, 로그인 정보는 로컬에 캐시되어 재사용된다.
단, 자동화/CI 같은 프로그램 실행에는 API key 인증이 기본 권장이다. ChatGPT 계정으로 반복 자동화를 돌리는
경로는 신뢰된 로컬 환경이나 Enterprise access token 같은 제한된 용도에 맞다.

공식 참고:

- Codex SDK: https://developers.openai.com/codex/sdk
- Codex authentication: https://developers.openai.com/codex/auth
- Codex non-interactive mode: https://developers.openai.com/codex/noninteractive
- Codex MCP / Agents SDK: https://developers.openai.com/codex/guides/agents-sdk

### 9-1. Python 프로젝트에서 가장 가까운 대응: `openai-codex`

Claude 문서의 `ClaudeSDKClient`에 가장 가까운 Codex 방식은 Python SDK의 `Codex()` + thread API다.
프로세스 내부에서 Codex app-server를 제어하고, 같은 thread에 여러 번 `run()`을 호출해 대화 맥락을 이어간다.

```bash
pip install openai-codex
```

```python
from openai_codex import Codex, Sandbox

with Codex() as codex:
    thread = codex.thread_start(
        sandbox=Sandbox.workspace_write,
    )

    first = thread.run("내가 좋아하는 숫자는 42야. 알겠다고만 답해.")
    print(first.final_response)

    second = thread.run("그 숫자에 8을 더하면? 숫자만.")
    print(second.final_response)  # 50 이면 같은 thread 맥락 유지 성공
```

비동기 앱 안에서는 `AsyncCodex`를 쓴다.

```python
import asyncio

from openai_codex import AsyncCodex, Sandbox

async def main() -> None:
    async with AsyncCodex() as codex:
        thread = await codex.thread_start(sandbox=Sandbox.workspace_write)
        await thread.run("내가 좋아하는 숫자는 42야. 알겠다고만 답해.")
        result = await thread.run("그 숫자에 8을 더하면? 숫자만.")
        print(result.final_response)

asyncio.run(main())
```

### 9-2. CLI 자동화에서 세션을 이어가려면: `codex exec resume`

단순 셸 자동화는 SDK보다 `codex exec`가 더 간단하다. 첫 실행 후 이어서 작업해야 하면 `resume`을 사용한다.

```bash
codex exec "이 저장소의 테스트 실패 가능성을 검토해줘"
codex exec resume --last "방금 찾은 문제 중 가장 작은 수정안을 제안해줘"
```

특정 세션을 정확히 이어야 하면 `codex exec resume <SESSION_ID>`를 사용한다.
스크립트에서 이벤트를 파싱해야 하면 JSONL 출력이 더 안정적이다.

```bash
codex exec --json "프로젝트 구조를 요약해줘"
```

Codex는 안전을 위해 기본적으로 Git 저장소 안에서 실행하는 것을 요구한다. Git 저장소가 아닌 임시 디렉터리에서
검증해야 한다면, 환경이 안전하다는 전제하에 `--skip-git-repo-check`를 명시한다.

### 9-3. OpenAI Agents SDK에서 Codex를 도구처럼 쓰려면: MCP

다른 에이전트가 Codex를 호출해야 하는 구조라면 Codex CLI를 MCP 서버로 띄운다.

```bash
codex mcp-server
```

이 서버는 핵심적으로 두 도구를 제공한다.

| 도구 | 역할 |
|---|---|
| `codex` | 새 Codex session/thread 시작 |
| `codex-reply` | 반환된 `threadId`로 기존 Codex thread 이어가기 |

OpenAI Agents SDK와 연결하는 최소 형태는 아래와 같다.

```python
import asyncio

from agents.mcp import MCPServerStdio

async def main() -> None:
    async with MCPServerStdio(
        name="Codex CLI",
        params={
            "command": "codex",
            "args": ["mcp-server"],
        },
        client_session_timeout_seconds=360000,
    ) as codex_mcp_server:
        print("Codex MCP server started.")
        # 이후 Agent(..., mcp_servers=[codex_mcp_server]) 형태로 연결한다.

asyncio.run(main())
```

이 방식은 Codex를 직접 호출하는 단일 앱보다 무겁지만, 여러 에이전트가 역할을 나눠 Codex를 호출하거나
handoff/trace/guardrail 같은 Agents SDK 기능이 필요할 때 적합하다.

### 9-4. Claude 방식과 Codex 방식의 차이

| 항목 | Claude Agent SDK | Codex |
|---|---|---|
| 핵심 객체 | `ClaudeSDKClient` | `Codex()` / `AsyncCodex()` 또는 SDK thread |
| 대화 유지 | 같은 client 세션에 `query()` 반복 | 같은 thread에 `run()` 반복 또는 thread ID 재개 |
| CLI 기반 인증 | Claude CLI 구독 OAuth | ChatGPT 로그인 또는 API key 로그인 |
| 자동화 기본 권장 | 개인/MVP는 구독 인증 가능, 상용은 API 전환 | CI/프로그램 실행은 API key 권장, ChatGPT auth 자동화는 신뢰된 환경 중심 |
| MCP 통합 | SDK 옵션의 `mcp_servers` | `codex mcp-server`가 Codex 자체를 MCP 도구로 노출 |

### 9-5. Codex 적용 판단

아래 중 하나라면 Codex SDK/세션 유지 방식이 맞다.

- 파이썬/타입스크립트 앱에서 Codex에게 여러 번 이어서 작업을 맡겨야 한다.
- 1턴의 분석 결과를 2턴의 구현/검토가 그대로 이어받아야 한다.
- 매번 새 CLI 호출을 하는 대신 같은 thread ID를 유지하고 싶다.
- 다른 에이전트가 Codex를 도구처럼 호출하는 워크플로가 필요하다.

반대로 단발성 코드 검토, 릴리스 노트 생성, 로그 요약처럼 한 번 실행하고 끝나는 작업은 `codex exec` 단발 호출로 충분하다.

### 9-6. Codex 검증 절차

1. Codex CLI 설치 확인

```bash
codex --version
```

2. 로그인 확인

```bash
codex login
```

3. Python SDK 설치

```bash
pip install openai-codex
```

4. 같은 thread에서 2턴 기억 테스트

```python
from openai_codex import Codex

with Codex() as codex:
    thread = codex.thread_start()
    thread.run("내가 좋아하는 숫자는 42야. 알겠다고만 답해.")
    result = thread.run("그 숫자에 8을 더하면? 숫자만.")
    print(result.final_response)
```

출력이 `50`이면 Codex thread가 유지되고 있는 것이다.

5. CLI resume 경로도 같이 확인

```bash
codex exec "내가 좋아하는 숫자는 42야. 알겠다고만 답해."
codex exec resume --last "그 숫자에 8을 더하면? 숫자만."
```

### 9-7. 운영 주의

- 개인 로컬 개발이나 MVP 검증은 ChatGPT 로그인 기반으로도 시작할 수 있다.
- CI/CD, 서버 자동화, 다중 사용자 서비스는 API key 기반 설계를 기본값으로 둔다.
- `~/.codex/auth.json`은 access token을 담을 수 있으므로 비밀번호처럼 취급한다.
- `CODEX_API_KEY`는 `codex exec`에서만 단일 실행용으로 지원된다. job 전체 환경변수로 넓게 노출하지 않는다.
- 외부 사용자에게 Codex 실행 기능을 직접 제공하는 구조는 구독형 개인 사용 범위를 벗어날 수 있으므로, OpenAI Platform API 과금/권한 모델로 전환하는 것을 전제로 설계한다.
