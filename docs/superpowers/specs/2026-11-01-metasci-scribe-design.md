# Metasci_scribe 초기 버전 설계

작성일: 2026-11-01
기준 코드: EvoScientist `main` 27d3c38 (분석 시점, `pyproject` 버전 0.4.0)

## 1. 목적

Metasci_scribe는 연구자가 자기 ChatGPT 구독으로 돌리는 **터미널 논문 작성 에이전트**다. EvoScientist를 포크해 글쓰기에 필요한 부분만 남기고, 이후 메타 스킬과 기존 metasciscribe의 단락 작성 흐름을 얹을 기반을 만든다.

EvoScientist를 고른 이유는 여러 번의 작업에서 배우는 구조(기억, 스킬 제안)를 나중에 쓰기 위해서다. 초기 버전의 목표는 그 기반을 **가볍고 동작하는 상태**로 만드는 것이다.

## 2. 사용자 흐름

```text
cd 내논문
metasci-scribe          앱 실행, 대화 화면 진입
/login                  ChatGPT 계정 연결 (처음 한 번)
/init                   현재 폴더에 논문 작업 틀 생성
(대화)                   에이전트와 대화하며 작성
/run                    자율 작성 시작
```

터미널에서 바로 실행하는 `metasci-scribe run`은 `/run`과 같은 기능이다.

### 완료 기준

연구자가 Windows 터미널에서 다음을 끝까지 수행할 수 있으면 초기 버전이 완성된 것이다.

1. 앱을 설치하고 실행한다.
2. `/login`으로 ChatGPT에 연결한다.
3. `/init`으로 논문 폴더 틀을 만든다.
4. 대화로 `outline.md`를 고치고 문단을 쓴다. 에이전트가 파일을 읽고 쓴다.
5. `/run`으로 자율 작성을 돌려 `manuscript.md`를 얻는다.
6. 앱을 다시 열어 같은 작업을 이어 간다.
7. 사용 한도에 걸리면 자동 과금 전환 없이 멈추고, 나중에 이어 쓸 수 있다.

## 3. 범위

### 목표

- 터미널 대화 모드와 자율 모드
- ChatGPT 플랜 연결, 대체 경로로 사용자 본인의 API 키
- 논문 폴더 틀과 프로젝트별 세션 분리 (로컬 디스크)
- `Knowledge/`의 PDF 텍스트 추출
- 스킬을 넣을 수 있는 자리 (전역, 프로젝트)
- 오픈소스 공개 가능한 라이선스·고지 정리

### 하지 않는 것

- 웹 화면, 전체 화면 TUI
- 메타 스킬 탑재, 단락별 하네스 이식
- 작업 간 학습 (기억 워커, AutoSkills) 활성화
- 메신저 채널, 음성 입력, Docker, 예약 실행, 능동 알림
- Claude 구독 연결
- 유료 또는 원격 호스팅 배포
- 웹 검색. 기존 metasciscribe와 같이 끄고, 근거는 프로젝트 폴더로 한정한다. `tavily` 의존성을 뺀다.

### 절대 하면 안 되는 것

- `chatgpt.com/backend-api` 호출. 공식 문서가 공개 Responses API만 쓰라고 정한다.
- `ccproxy` 등으로 Claude 구독 토큰 재사용. Anthropic 문서가 금지한다.
- ChatGPT 플랜 오류 시 API 키 등 다른 과금 경로로 자동 전환.
- 토큰을 로그나 저장소에 남기는 것.
- `/init`이 사용자의 기존 파일을 덮어쓰는 것.

## 4. 모델 연결

### 경로

| 인증 | 사용 조건 | 초기 버전 |
|---|---|---|
| Sign in with ChatGPT 플랜 사용 | 오픈소스 프로젝트는 현재 사용 가능 | 기본 경로 |
| 사용자 본인의 OpenAI API 키 | 제한 없음 | 대체 경로 |
| 사용자 본인의 Anthropic API 키 | 제한 없음 | 대체 경로 |

나머지 공급자(DeepSeek, Google, NVIDIA, Ollama, OpenRouter 등)는 초기 버전에서 뺀다.

### 로그인 (`/login`)

공식 문서의 직접 등록 흐름을 따른다.

- 시스템 브라우저로 `https://auth.openai.com/api/accounts/authorize`를 연다.
- 콜백은 `http://127.0.0.1:<port>/auth/callback`이다. `localhost`로 바꾸지 않는다.
- 처음 등록할 때 `client_id=dynamic_agent_client`, `agent_name_hint=Metasci_scribe`, `ext_agent_host_id`를 보낸다.
- 범위는 `openid profile email offline_access resource.invoke chatgpt.tokens.use.direct`, `resource`는 `https://api.openai.com/v1`이다.
- `state`, `nonce`, PKCE(S256)를 시도마다 새로 만든다. ID 토큰을 검증한다.
- 콜백이 돌려준 `client_id`를 저장해 재로그인에 쓴다. `dynamic_agent_client`는 저장하지 않는다.
- 부여된 범위에 `chatgpt.tokens.use.direct`가 없으면 플랜 사용을 꺼진 상태로 표시하고 추론하지 않는다.

토큰은 `~/.metasci-scribe/auth/chatgpt.json`에 사용자 전용 권한으로 저장하고, EvoScientist의 파일 잠금 패턴을 재사용한다. `/logout`은 저장된 토큰을 지운다. 초기 버전은 계정 하나만 지원한다.

### 호출

`llm/chatgpt_plan.py`를 새로 만들고 `chatgpt_oauth.py`는 삭제한다. 새 모듈은 `langchain-openai`의 공개 API만 쓴다. `ChatOpenAI`를 상속해 요청 본문을 다음 규칙에 맞춘다.

- `POST https://api.openai.com/v1/responses`, `Authorization: Bearer <access_token>`
- `store: false`, `stream: true`
- `previous_response_id` 없이 대화 기록을 `input`에 담는다.
- 시스템 프롬프트는 `instructions` 또는 developer 메시지로 보낸다.
- `temperature`, `top_p`, `max_output_tokens`, `metadata`, `user` 등 미지원 필드를 제거한다.
- 함수 도구는 namespace로 묶거나 `additional_tools` 입력으로 보낸다. 어느 쪽이 맞는지는 M0에서 정한다.
- 비스트림 호출도 내부적으로 스트림을 읽는다.
- `response.completed`를 받아야 성공으로 처리한다.
- `GET https://api.openai.com/v1/models`로 사용 가능한 모델을 가져온다.

### 오류 처리

| 상황 | 동작 |
|---|---|
| `subscription_sharing_usage_limit_exceeded` | 멈추고 ChatGPT 설정의 사용량 화면을 안내. 작업은 이어 쓸 수 있는 상태로 보존 |
| `subscription_sharing_user_not_eligible` | 플랜 사용 불가를 설명하고 API 키 경로를 안내. 재시도하지 않음 |
| `subscription_sharing_unsupported_capability` | `error.param`을 표시. 같은 요청을 재시도하지 않음 |
| `subscription_sharing_usage_unavailable`, 503 | 제한된 횟수로 지연 재시도 |
| 401, 복구 불가능한 갱신 오류 | 토큰을 정리하고 `/login` 안내 |
| 일시적 네트워크 오류 | 토큰을 지우지 않음 |

요청 ID와 오류 코드는 로그에 남긴다.

### 대안

M0에서 엔진과 Responses API를 맞출 수 없으면 Codex app-server를 대안으로 검토한다. app-server는 공식 문서상 ChatGPT 플랜 토큰을 받을 수 있지만, 실험 단계이고 운영 환경 지원 대상이 아니다.

## 5. 코드 구조

### 포크 방식

- 위치: `C:\dev\Metasci_scribe`
- EvoScientist의 최신 릴리스 태그를 시작점으로 삼고, 커밋 이력 없이 새 저장소로 시작한다. 시작 커밋 메시지에 원본 태그와 커밋을 기록한다.
- 패키지 `metasci_scribe`, 명령 `metasci-scribe`, 환경변수 접두사 `METASCI_SCRIBE_`, 데이터 폴더 `~/.metasci-scribe`
- `LICENSE` 유지, `NOTICE` 추가, 수정 사실을 README에 표시. EvoScientist 로고와 배지는 제거한다.

### 유지·제거

Python 약 9.8만 줄이고 모듈 간 결합이 강하다. 정적 import 분석 기준으로 나눈다.

| 구분 | 모듈 | 처리 |
|---|---|---|
| 유지 | `EvoScientist.py`, `backends`, `paths`, `runtime`, `sessions`, `stream`, `gateway`(local), `tools`, `prompts`, `utils`, `skills`, `mcp` | 이름만 변경 |
| 축소 | `cli`, `commands`, `config`, `llm`, `middleware`, `subagents` | 아래 참고 |
| 제거 | `channels`, `cli/widgets`, `cli/tui_*`, `ccproxy_manager`, `llm/chatgpt_oauth`, `stt`, `proactive`, `updater`, `update_helper`, `update_check`, `uv_tool`, Docker 파일, WebUI 설치 코드 | 삭제 |
| 유지하되 끔 | `memory`, `cron`, `langgraph_dev` | 코드는 남기고 기본값 끔 |
| 판단 후 처리 | `deploy`, `setup` | 결합이 약하면 삭제, 강하면 유지 |

**축소 세부**

- `cli`: 일반 CLI(`interactive.py`)만 남긴다. 기본값 `ui_backend="cli"`.
- `commands`: `general`, `session`, `skills`, `model`, `mcp`, `mcp_install`, `rubric`, `goal`을 남기고 `channel`, `proactive`, `schedule`, `experts`, `autoskills`, `model_fallback`을 뺀다. `/login`, `/logout`, `/init`, `/run`을 추가한다.
- `llm`: `openai`, `anthropic`, ChatGPT 플랜만 남긴다.
- `middleware`: `proactive_mode`, `scheduler`, `active_team`, `expert_async_subagent`, `model_fallback`을 뺀다.
- `subagents`: `planner`, `research`, `writing`만 남긴다. `research`는 웹 검색 대신 프로젝트 자료를 읽어 정리하는 역할로 바꾼다. `code`, `debug`, `data_analysis`, `scheduler`, `expert_*`를 뺀다.

**채널 제거 시 옮길 코드**

`stream/display.py`는 `channels/interaction.py`, `channels/hitl_budget.py`를, `cli/interactive.py`는 `channels/consumer.py`의 `_format_todo_list`와 `ChannelRuntime`을 쓴다. 승인 처리 코드는 `stream` 쪽으로 옮기고 채널 전용 분기는 지운다.

**제거 순서**

한 번에 지우지 않는다. 모듈 단위로 삭제하고, 그때마다 import 검사와 남은 테스트를 통과시킨다. 제거된 기능의 테스트는 함께 지운다.

### 설치 전제

초기 버전은 Python, uv, Git을 미리 설치하도록 요구하고 `uv tool install`로 설치한다. 한 줄 설치기는 만들지 않는다. 스킬의 스크립트 실행에 필요하므로 에이전트의 셸 실행(`execute`)은 유지하며, Windows에서는 Git Bash를 쓴다. `setup`은 Git Bash 탐지와 에이전트 Python 환경 구성만 남기는 방향으로 줄인다.

### 기억

코드는 남기고 `memory_profile_enabled`, `memory_observations_enabled`, `memory_workers_enabled`, `enable_async_subagents`의 기본값을 끈다. 워커를 켜면 사용자의 플랜 사용량이 백그라운드에서 소모되고 `langgraph dev` 서버가 필요하기 때문이다.

초기 버전의 기억은 세션 재개로 충분하다. `sessions`가 대화를 프로젝트 단위로 저장하므로, 앱을 다시 열어 이어 갈 수 있다. 학습 기능은 글쓰기용 신호를 설계한 뒤 다음 단계에서 켠다.

## 6. 글쓰기 기능

### `/init`

현재 폴더에 다음을 만든다. 이미 있는 파일과 폴더는 건너뛰고, 건너뛴 항목을 알려 준다.

```text
내논문/
├── program.md          논문 설정 (제목, 학술지, 핵심 메시지, 저자 메모)
├── outline.md          구성 지도 (섹션과 단락별 주장, 근거)
├── manuscript.md       원고
├── Knowledge/          참고 자료
├── Figures/            그림
├── Tables/             표
└── .metasci-scribe/    앱 상태, 프로젝트 스킬
```

템플릿은 기존 metasciscribe의 `program.md`, `outline.md`를 단순화해 패키지에 넣는다.

네트워크 경로는 지원하지 않는다. `Workspace`가 `Z:`를 UNC 경로로 해석하고 Windows 경로 처리와 셸이 이를 다루지 못하기 때문이다. `/init`과 `run`은 UNC 또는 네트워크 드라이브를 감지하면 로컬 폴더를 쓰도록 안내하고 멈춘다.

### PDF 읽기

`Knowledge/`의 PDF는 `pypdf`로 텍스트를 추출해 `.metasci-scribe/extracted/<파일명>.md`에 저장한다. 라이선스가 BSD이고 추가 시스템 의존성이 없어 선택했다.

- 앱 시작과 `run` 시작 때 새 파일이나 변경된 파일만 추출한다.
- 쪽마다 `[p. N]` 표시를 넣어 인용 위치를 확인할 수 있게 한다.
- 원본 PDF는 수정하지 않고, 에이전트는 추출본을 읽는다. 플랜 경로는 매 요청에 기록을 다시 보내므로 PDF 파일 자체를 대화에 넣지 않는다.
- 텍스트가 없는 스캔 PDF는 OCR하지 않고 경고로 알린다.

### 프롬프트

`prompts.py`의 실험 절차를 글쓰기 절차로 바꾼다.

- 근거는 프로젝트 폴더의 자료를 우선한다.
- 자료에 없는 수치와 인용을 지어내지 않는다. 근거가 없으면 `[evidence needed]`로 표시한다.
- 산출물은 `manuscript.md`에 쓴다. `outline.md` 수정은 대화 모드에서 사용자와 합의한 경우에만 한다.
- 언어는 사용자가 정한다. 기본은 사용자의 입력 언어를 따른다.

전문 작성 규칙은 메타 스킬이 맡으므로 넣지 않는다. 내장 스킬은 `find-skills`, `skill-creator`만 남기고 EvoScientist의 연구 스킬은 넣지 않는다.

### `/run`, `metasci-scribe run`

자율 모드는 EvoScientist의 `auto_mode`를 쓴다. 승인 요청과 `ask_user`가 꺼지고, 기존 명령 차단 규칙은 유지한다.

1. `program.md`와 `outline.md`가 있는지 확인한다. 없으면 `/init`을 안내하고, `outline.md`에 미결 표시(◇)가 남아 있으면 시작하지 않는다.
2. `outline.md`에서 단락 ID를 읽어 작성 순서를 정한다. 형식은 기존 metasciscribe의 `### R1 · ①` 표기를 유지한다.
3. 실행마다 대화와 분리된 자율 작업 스레드를 만든다. 두 모드를 잇는 기준은 대화 기록이 아니라 파일이다.
4. 고정 지시문으로 실행한다. 내용은 "outline.md 순서대로 초안을 작성해 manuscript.md에 저장하고, 단락마다 run.json을 갱신하라"이다.
5. 단락별 상태를 `.metasci-scribe/run.json`에 `pending`, `done`, `blocked`로 기록한다. 완료 여부는 모델의 말이 아니라 `outline.md`의 모든 단락이 `done`인지로 판정한다.
6. 다시 실행하면 같은 자율 스레드에서 남은 단락부터 이어 간다. `outline.md`가 바뀌었으면 새 스레드로 시작하되 `done` 단락은 유지한다.
7. 근거가 없는 곳은 `[evidence needed]`를 남기고 계속 쓴다. 이 표시는 실패가 아니며, 종료할 때 개수를 보고한다.
8. 자료가 단락의 주장을 뒷받침하지 않거나 구성을 바꿔야 하면 그 단락을 `blocked`로 표시하고 멈춘다. 자율 모드는 `outline.md`를 고치지 않는다.
9. 종료 코드: 0 전체 완료, 2 실패, 3 사용 한도, 4 사람의 결정 필요(`blocked`).

앱 안의 `/run`은 `metasci-scribe run`을 별도 프로세스로 실행하고 진행 상황을 표시한다. `auto_mode`는 에이전트를 만들 때 읽히므로, 대화 세션의 에이전트를 바꾸지 않기 위해서다.

실행 본문은 `metasci_scribe/run.py`의 함수 하나로 둔다. 나중에 metasciscribe의 단락 작성 흐름으로 이 함수만 교체한다.

### 스킬

전역 `~/.metasci-scribe/skills/`와 프로젝트 `.metasci-scribe/skills/`를 읽는다. EvoScientist의 스킬 로딩, `/install-skill`, 다음 메시지부터 적용되는 재로딩을 그대로 쓴다.

## 7. 개발 순서

| 단계 | 내용 | 통과 조건 |
|---|---|---|
| M0 | 시험 구현: 로그인 + deepagents 도구 호출 1회 | Plus 또는 Pro 계정으로 추출한 PDF 텍스트를 읽는 도구 호출과 긴 도구 결과를 포함한 한 턴 완료 |
| M1 | 포크, 개명, 라이선스 정리 | 기존 테스트 기준선 기록, 앱 실행 |
| M2 | 불필요 모듈 제거 | 제거 범위 외 테스트 통과, 의존성 축소 |
| M3 | 정식 ChatGPT 플랜 연결 | 오류 표의 동작을 단위 테스트로 검증 |
| M4 | `/init`, 프롬프트, 서브에이전트 | 대화로 파일 읽기·쓰기 동작 |
| M5 | `/run`, 멈춤과 이어 쓰기 | 가짜 모델로 정상·한도 초과·재개 시험 통과 |
| M6 | 설치, README, 고지 | 깨끗한 Windows 환경에서 완료 기준 수행 |

M0가 실패하면 M1 이후를 진행하지 않고 대안을 다시 판단한다.

## 8. 시험

- 네트워크 없는 단위 테스트: 요청 본문 변환, 토큰 저장과 갱신, 오류 처리, `/init`, `run` 상태 전환
- 가짜 모델 통합 테스트: 대화 한 턴, 자율 실행 완료, 사용 한도 중단 후 재개
- 수동 실계정 시험: 로그인, 도구 호출, 긴 작업. 사용자가 브라우저에서 직접 로그인한다.
- 기본 개발 환경은 Windows다. EvoScientist 테스트는 약 260개 파일이고 먼저 기준선을 측정한다.

## 9. 알려진 위험

- ChatGPT 플랜 사용은 미리보기 단계이고 조건이 바뀔 수 있다.
- 문서에서 명시된 플랜은 Plus와 Pro뿐이다. Edu와 Business 지원 여부는 불명확하다.
- 한도 계산 방식이 문서에 없어, 자율 모드가 도중에 멈출 수 있다.
- 문서가 오픈소스의 조건을 정의하지 않는다. 앱과 메타 스킬을 모두 공개하기로 했으므로 가장 안전한 경우에 해당하지만, 공개 전에 현재 약관을 다시 확인한다. 공개는 늦어도 M6에서 한다.
- deepagents와 langchain 의존이 남아 설치 크기는 여전히 크다.
- 개발 저장소, 가상환경, 시험용 논문 폴더는 모두 로컬 디스크에 둔다.
- UI/UX 가이드라인은 아직 읽지 않았다. 터미널의 "Continue with ChatGPT" 표기와 브랜딩을 M3에서 확인한다.

## 10. 근거 문서

- [Sign in with ChatGPT](https://developers.openai.com/siwc)
- [Registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Errors and recovery](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
- [Claude Code legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)
- [EvoScientist](https://github.com/EvoScientist/EvoScientist)

## 11. 구현 중 변경

구현과 보안 검토에서 다음이 바뀌었다. 이 절이 앞 절보다 우선한다.

- 저장소 위치는 `C:\dev\Metasci_scribe`다.
- 자율 실행에는 셸, MCP 도구, 삭제, 스킬 설치가 없다. 메타 스킬이 자율 실행에서 스크립트를 쓰려면 별도의 안전한 실행 방식을 설계해야 한다.
- 대화의 셸 명령은 "모두 승인"을 눌러도 건별로 승인받는다.
- 에이전트 파일 도구는 토큰 폴더, 시작 설정 파일, `.metasci-scribe/` 쓰기, `.env` 읽기를 거부한다.
- 작업 폴더 `.env`는 공급자 API 키만 반영한다. `*_BASE_URL`과 앱 설정은 전역 설정에서만 읽는다.
- `--dangerous`, `--mode`, `--name`, `--host`를 제거했다.
- 데이터 폴더를 포함하는 작업 폴더는 거부한다.
- API 401은 토큰을 유지하고 `/login`을 안내한다. 확인된 취소나 복구 불가능한 갱신 오류만 토큰을 지운다.
- 도구는 `additional_tools` 입력 항목으로 보낸다.
- 단락 상태는 `mark_paragraph` 도구로 받아 앱이 기록한다.
- `/logout`은 OpenAI에 토큰 폐기를 요청하고 계정 등록을 지운다.
- 자율 실행은 패스마다 원고를 백업하고, 기존 줄이 사라지면 멈춘다.
- 원고는 개요의 `##` 절마다 `Manuscript/<절 이름>.md` 파일 하나에 쓴다. `manuscript.md`는 앱이 이 파일들을 개요 순서로 합쳐 만들고(대화 응답마다, 자율 실행이 끝날 때마다), 앱이 쓰지 않은 내용이 있으면 덮어쓰지 않는다. `/init`은 새로 만든 개요의 절마다 빈 파일을 만든다.
- `/run <절>`(`metasci-scribe run --section <절>`)은 그 절의 단락만 쓴다. 절마다 자기 스레드를 쓰고, 다른 절의 미결 표시(◇)는 실행을 막지 않는다. 원고 백업과 줄 사라짐 검사는 `Manuscript/` 전체를 대상으로 한다.
- 드라이브 문자가 있는 네트워크 드라이브(`Z:`)를 지원한다. RaiDrive 같은 도구는 Windows가 경로를 해석할 때 Windows 자신도 열지 못하는 UNC 이름(`\\RaiDrive-KEI\Synology\...`)을 돌려준다. 그래서 앱은 시작할 때 `os.path.realpath`를 감싸, 매핑된 드라이브의 UNC 루트를 드라이브 문자로 되돌린다(`drive_paths.py`). 드라이브 문자 없이 UNC로만 지정한 공유 폴더는 계속 거부한다. `cmd.exe`가 그런 경로를 작업 폴더로 쓰지 못하기 때문이다.
