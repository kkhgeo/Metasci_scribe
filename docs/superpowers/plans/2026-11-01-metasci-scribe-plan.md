# Metasci_scribe 초기 버전 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EvoScientist v0.3.8을 포크해, 사용자의 ChatGPT Plus/Pro 구독으로 돌아가는 터미널 논문 작성 에이전트 Metasci_scribe를 만든다.

**Architecture:** deepagents 엔진을 유지하고 공개 Responses API용 모델 연결을 새로 만든다. 메신저, TUI, 웹, 실험 기능을 제거하고 `/login`, `/init`, `/run`, PDF 추출을 추가한다. 자율 실행 본문은 `run.py` 하나로 두어 나중에 교체한다.

**Tech Stack:** Python 3.11~3.13, deepagents, langchain-openai, langchain-anthropic, typer, rich, prompt-toolkit, httpx, pypdf, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-11-01-metasci-scribe-design.md`

## Global Constraints

- 시작 코드: EvoScientist 태그 `v0.3.8` (커밋 `e2eef14`). 커밋 이력은 가져오지 않는다.
- 패키지 `metasci_scribe`, 명령 `metasci-scribe`, 환경변수 접두사 `METASCI_SCRIBE_`, 데이터 폴더 `~/.metasci-scribe`.
- 저장소 `C:\dev\Metasci_scribe`, 시험 구현 폴더 `C:\dev\Metasci_scribe\spikes\m0_chatgpt_plan`. 쓰기가 막히면 멈추고 사용자에게 묻는다.
- 가상환경: `UV_PROJECT_ENVIRONMENT=C:\dev\venvs\metasci-scribe`.
- git 전역 설정을 바꾸지 않는다.
- `chatgpt.com/backend-api` 호출 금지. 추론은 `POST https://api.openai.com/v1/responses`만 쓴다.
- ccproxy 등 Claude 구독 토큰 재사용 코드 금지.
- ChatGPT 플랜 오류 시 다른 과금 경로로 자동 전환 금지.
- 토큰을 로그, 저장소, 오류 메시지에 남기지 않는다.
- `/init`은 기존 파일을 덮어쓰지 않는다.
- 웹 검색 기능을 넣지 않는다.
- `LICENSE`를 유지한다. 수정한 파일 기록은 `NOTICE`로 고지한다.
- 커밋 메시지 끝에 `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`을 넣는다.
- 원격 저장소 생성과 push는 사용자가 지시할 때만 한다.
- 표기 "Continue with ChatGPT"는 T9에서 UI/UX 가이드라인을 읽고 확정한다.

## Review Focus

1. 로그인 중 브라우저를 닫거나, 동의를 거부하거나, 포트가 사용 중인 경우: 앱이 멈추지 않고 이유를 알려 준다. (T7)
2. `chat`과 `run`을 동시에 띄운 상태에서 토큰이 만료되는 경우: 갱신 토큰 재사용 오류 없이 둘 다 계속 동작한다. (T7)
3. 자율 실행 중 Ctrl+C나 프로세스 종료: `run.json`이 깨지지 않고 다음 실행이 남은 단락부터 이어 간다. (T13)
4. 손상·암호 PDF, 한글·공백 파일명: 해당 파일만 경고하고 나머지는 처리한다. (T11)
5. `outline.md` 형식 오류나 단락 ID 중복: 비용을 쓰기 전에 줄 번호와 함께 알린다. (T13)

---

## M0: 기술 시험

### Task 1: ChatGPT 플랜으로 도구 호출 검증

사용자가 브라우저에서 직접 로그인한다. 에이전트는 자격 증명을 입력하지 않는다.

**Files:**
- Create: `spikes/m0_chatgpt_plan/{spike_login.py, spike_turn.py, RESULT.md, sample.txt}`

**Interfaces:**
- Produces: `RESULT.md`. 도구 전달 방식(namespace 또는 `additional_tools`), 재생해야 하는 응답 항목, 사용 모델 slug, 통과 여부. T8이 이를 따른다.

- [ ] **Step 1: `spike_login.py` 작성.** 설계 4장 로그인 흐름 그대로. 표준 라이브러리와 httpx만 쓴다. 토큰은 저장소 밖 `~/.metasci-scribe/spike_token.json`에 저장한다.
- [ ] **Step 2: 사용자 로그인 요청 후 확인.** 기대: 부여 범위에 `chatgpt.tokens.use.direct` 포함, `GET /v1/models`가 모델 목록 반환.
- [ ] **Step 3: `spike_turn.py` 작성.** deepagents `create_deep_agent`에 파일 읽기 도구를 주고, 약 4만 자의 추출 PDF 텍스트 `sample.txt`를 읽어 요약하게 한다. 요청 본문 규칙은 설계 4장을 따른다.
- [ ] **Step 4: 실행.** 기대: 도구 호출 후 `response.completed`까지 한 턴 완료. 2턴 이상 이어지는 대화도 시험한다.
- [ ] **Step 5: 판정 기록.** `RESULT.md`에 결과를 적는다. 실패 시 이후 Task를 진행하지 않고 사용자에게 Codex app-server 대안을 보고한다.

---

## M1: 포크

### Task 2: 코드 가져오기, 개명, 고지

**Files:**
- Create: `Metasci_scribe/` (v0.3.8 내용), `NOTICE`, `tools/rename_map.py`
- Modify: `pyproject.toml`, 모든 `*.py`의 import와 `EVOSCIENTIST_` 접두사
- Delete: `README.zh-CN.md`, `.github/assets/`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- Test: `tests/test_identity.py`

**Interfaces:**
- Produces: 패키지 `metasci_scribe`, `metasci_scribe.paths.DATA_DIR == ~/.metasci-scribe`, 진입점 `metasci-scribe = metasci_scribe.cli:main`.

- [ ] **Step 1: 내려받기.** `git clone --depth 1 --branch v0.3.8`를 임시 폴더에 받아 `.git`을 빼고 저장소 최상위로 복사한다. `docs/superpowers/`, `spikes/`, `.superpowers/`는 건드리지 않는다. 커밋: `chore: import EvoScientist v0.3.8 (e2eef14)`.
- [ ] **Step 2: 기준선 측정.** `uv sync --dev`, `uv run pytest -n auto -q`. 통과·실패 수를 `docs/baseline.md`에 기록한다.
- [ ] **Step 3: 실패 테스트 작성.**

```python
def test_package_identity():
    import metasci_scribe, metasci_scribe.paths as p
    assert p.DATA_DIR.name == ".metasci-scribe"
    assert metasci_scribe.__name__ == "metasci_scribe"

def test_env_prefix_replaced():
    from metasci_scribe.config import settings
    assert all(v.startswith("METASCI_SCRIBE_") for v in settings.ENV_MAPPING.values())
```

`ENV_MAPPING`의 실제 이름은 `config/settings.py:1222` 근처 사전 이름을 따른다.

- [ ] **Step 4: 개명.** `tools/rename_map.py`로 `EvoScientist/` → `metasci_scribe/`, `EvoScientist.py` → `agent.py`, `EvoScientist` import, `EVOSCIENTIST_`, `~/.evoscientist`를 바꾼다. 명령 이름은 `metasci-scribe`만 남긴다. 기존 사용자 데이터 이전 코드(`migrate_legacy_sessions_db`)는 지운다.
- [ ] **Step 5: 테스트 통과 확인.** `uv run pytest tests/test_identity.py -v` PASS, 전체 결과가 기준선과 같다.
- [ ] **Step 6: `NOTICE` 작성, README 교체.** 원본 프로젝트와 커밋, 수정 사실, Apache-2.0을 적는다. EvoScientist 로고와 배지는 쓰지 않는다.
- [ ] **Step 7: 커밋.** `chore: rename to metasci_scribe and add notices`

---

## M2: 제거

공통 절차: 삭제 → `uv run python -c "import metasci_scribe.cli"` → 관련 테스트 삭제 → 전체 테스트 → 커밋. 기준선보다 실패가 늘면 다음으로 넘어가지 않는다.

### Task 3: 메신저 채널 제거

**Files:**
- Create: `metasci_scribe/stream/interaction.py`, `metasci_scribe/stream/hitl_budget.py`
- Modify: `stream/display.py`, `cli/interactive.py`, `cli/commands.py`, `commands/__init__.py`, `config/settings.py`, `gateway/`, `langgraph_dev/`
- Delete: `metasci_scribe/channels/`, `cli/channel.py`, `cli/channel_sends.py`, `commands/channel_ui.py`, `commands/implementation/channel.py`
- Test: 기존 `tests/`에서 채널 시험 삭제, `tests/test_no_channels.py`

**Interfaces:**
- Produces: `metasci_scribe.stream.interaction.ApprovalOutcome`, `stream.hitl_budget.hitl_budget_stop`, `channel_response_with_budget_stop` 이름 유지. `_format_todo_list`는 `stream/formatter.py`로 이동.

- [ ] **Step 1: 실패 테스트 작성.** `test_no_channels`: `importlib.util.find_spec("metasci_scribe.channels") is None`, `metasci_scribe/` 아래 어떤 파일도 `.channels` import를 포함하지 않는다.
- [ ] **Step 2: 실패 확인.** `uv run pytest tests/test_no_channels.py -v` FAIL.
- [ ] **Step 3: 공유 코드 이동.** 위 Produces대로 옮기고 호출부를 고친다. 채널 전용 분기(`ChannelRuntime`, `_channels_is_running`, 전송 관련 인자)는 지운다.
- [ ] **Step 4: 채널 삭제.** 공통 절차 수행.
- [ ] **Step 5: 의존성 삭제.** `pyproject.toml`의 채널 extra 전부와 `stt`, `oauth` extra를 뺀다.
- [ ] **Step 6: 커밋.** `refactor: remove messaging channels`

### Task 4: 화면·운영 기능 제거

**Files:**
- Delete: `cli/tui_*.py`, `cli/widgets/`, `ccproxy_manager.py`, `stt.py`, `proactive/`, `updater.py`, `update_helper.py`, `update_check.py`, `uv_tool.py`, `middleware/proactive_mode.py`, `cli/service_cmd.py`, WebUI 설치 코드
- Modify: `cli/commands.py`, `cli/_app.py`, `langgraph_dev/`, `config/settings.py`, `setup/`
- Test: `tests/test_removed_surfaces.py`

**Interfaces:**
- Produces: `EvoScientistConfig`를 개명한 설정의 `ui_backend` 타입 `Literal["cli"]`, 기본값 `"cli"`.

- [ ] **Step 1: 실패 테스트 작성.** 위 모듈 각각 `find_spec(...) is None`, `Config().ui_backend == "cli"`, `textual`이 `pyproject.toml` 의존성에 없음.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 모듈 단위로 삭제.** 항목마다 공통 절차를 수행한다. `cli/interactive.py`가 유일한 대화 화면이 되도록 분기를 지운다.
- [ ] **Step 4: `deploy`, `setup` 판단.** import를 끊는 수정이 100줄 이하면 `deploy`를 삭제한다. 초과면 유지하고 사유를 `docs/baseline.md`에 적는다. `setup`은 Git Bash 탐지와 에이전트 Python 환경 구성만 남긴다.
- [ ] **Step 5: 통과 확인 후 커밋.** `refactor: remove TUI, web, and operations surfaces`

### Task 5: 명령, 미들웨어, 서브에이전트, 공급자 축소

**Files:**
- Delete: `commands/implementation/{proactive,schedule,experts,autoskills,model_fallback}.py`, `middleware/{scheduler,active_team,expert_async_subagent,model_fallback}.py`, `subagents/{code,debug,data_analysis,scheduler}.yaml`, `subagents/expert_container*.py`, `llm/chatgpt_oauth.py`, `tools`의 `tavily_search`
- Modify: `agent.py`, `config/settings.py`, `llm/models.py`, `pyproject.toml`
- Test: `tests/test_reduced_surface.py`

**Interfaces:**
- Produces: 설정 기본값 `memory_profile_enabled=False`, `memory_observations_enabled=False`, `memory_workers_enabled=False`, `enable_async_subagents=False`, `provider="openai"`.

- [ ] **Step 1: 실패 테스트 작성.**

```python
def test_memory_and_async_off_by_default():
    c = Config()
    assert not any([c.memory_profile_enabled, c.memory_observations_enabled,
                    c.memory_workers_enabled, c.enable_async_subagents])

def test_only_three_subagents():
    names = {p.stem for p in SUBAGENTS_CONFIG.glob("*.yaml")}
    assert names == {"planner", "research", "writing"}

def test_only_supported_providers():
    assert set(PROVIDERS) == {"openai", "anthropic"}
```

`PROVIDERS`는 `llm/models.py`의 공급자 목록 이름을 따른다.

- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 삭제와 기본값 변경.** 공통 절차. `_fold_expert_subagents`, `_route_async_specs_through_evo_middleware`는 제거하고 호출부를 정리한다.
- [ ] **Step 4: 의존성 삭제.** `langchain-deepseek`, `langchain-nvidia-ai-endpoints`, `langchain-google-genai`, `langchain-ollama`, `langchain-openrouter`, `openrouter`, `tavily-python`, `textual`을 뺀다. `langgraph-api`, `langgraph-cli`는 `memory` 코드 때문에 유지한다.
- [ ] **Step 5: 메모리 꺼진 상태 확인.** 시작 시 `langgraph dev` 프로세스가 뜨지 않음을 테스트로 고정한다.
- [ ] **Step 6: 통과 확인 후 커밋.** `refactor: reduce to writing-focused surface`

---

## M3: ChatGPT 플랜 연결

### Task 6: 스킬 위치와 프로젝트 상태 폴더

**Files:**
- Modify: `metasci_scribe/paths.py:121`
- Test: `tests/test_workspace_paths.py`

**Interfaces:**
- Produces: `Workspace.state_dir -> Path` (`root/.metasci-scribe`), `Workspace.skills_dir -> Path` (`state_dir/skills`). `GLOBAL_SKILLS_DIR == DATA_DIR/"skills"` 유지.

- [ ] **Step 1: 실패 테스트 작성.** `Workspace(root=tmp).skills_dir == tmp/".metasci-scribe"/"skills"`.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.** 프로젝트 스킬이 있으면 다음 메시지부터 적용되는 기존 재로딩이 동작하는지 기존 테스트로 확인한다.
- [ ] **Step 4: 통과 확인 후 커밋.** `feat: keep project state under .metasci-scribe`

### Task 7: 로그인과 토큰 관리

**Files:**
- Create: `metasci_scribe/auth/__init__.py`, `metasci_scribe/auth/chatgpt_plan.py`
- Test: `tests/test_chatgpt_auth.py`

**Interfaces:**
- Produces:
  - `@dataclass Credentials(client_id: str, access_token: str, refresh_token: str, id_token: str, expires_at: float, scopes: frozenset[str], email: str | None)`, 속성 `plan_enabled -> bool`
  - `build_authorize_url(*, client_id: str | None, redirect_uri: str, host_id: str, state: str, nonce: str, code_challenge: str, id_token_hint: str | None = None) -> str`
  - `login(*, open_browser: Callable[[str], bool] = webbrowser.open, timeout: float = 300.0) -> Credentials`
  - `load() -> Credentials | None`, `save(c: Credentials) -> None`, `clear() -> None`
  - `get_access_token() -> str`
  - 예외 `NotSignedIn`, `LoginCancelled`, `LoginTimeout`, `RefreshRejected`

- [ ] **Step 1: 실패 테스트 작성.**

```python
def test_first_login_uses_dynamic_client():
    q = parse_qs(urlparse(build_authorize_url(client_id=None, redirect_uri=R, host_id="h",
                                              state="s", nonce="n", code_challenge="c")).query)
    assert q["client_id"] == ["dynamic_agent_client"]
    assert q["agent_name_hint"] == ["Metasci_scribe"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["resource"] == ["https://api.openai.com/v1"]
    assert set(q["scope"][0].split()) == {"openid", "profile", "email", "offline_access",
                                          "resource.invoke", "chatgpt.tokens.use.direct"}

def test_relogin_omits_agent_name_hint(): ...   # 저장된 client_id 사용, agent_name_hint 없음
def test_redirect_uses_127_0_0_1(): ...         # localhost 금지, 경로 /auth/callback
def test_rejects_state_mismatch(): ...
def test_never_saves_dynamic_client_id(): ...
def test_missing_issued_client_id_is_incomplete(): ...
def test_plan_disabled_without_direct_scope(): ...
def test_access_denied_raises_login_cancelled(): ...
def test_login_times_out_when_browser_closed(): ...
def test_port_in_use_falls_back_to_another_port(): ...
def test_token_file_excludes_from_repr(): ...       # repr(Credentials)에 토큰 없음
def test_concurrent_refresh_uses_token_once(): ...  # 두 스레드, 갱신 호출 1회
def test_temporary_network_failure_keeps_tokens(): ...
def test_terminal_refresh_error_clears_tokens(): ...  # invalid_grant, refresh_token_reused 등
```

- [ ] **Step 2: 실패 확인.** `uv run pytest tests/test_chatgpt_auth.py -v`
- [ ] **Step 3: 구현.** httpx 가짜 전송으로 시험한다. 잠금은 `filelock`, 저장은 임시 파일 후 교체. 만료 60초 전에 갱신하고, 잠금 안에서 파일을 다시 읽어 다른 프로세스의 갱신을 재사용한다. `host_id`는 `DATA_DIR/auth/host_id`에 영구 저장한다.
- [ ] **Step 4: 통과 확인 후 커밋.** `feat: add ChatGPT plan sign-in`

### Task 8: Responses API 모델 연결과 오류 분류

**Files:**
- Create: `metasci_scribe/llm/chatgpt_plan.py`
- Modify: `metasci_scribe/llm/models.py:707-717`, `config/settings.py:219`
- Test: `tests/test_chatgpt_plan_model.py`

**Interfaces:**
- Consumes: `get_access_token()` (T7), T1 `RESULT.md`
- Produces:
  - `normalize_payload(payload: dict) -> dict`
  - `class PlanAction(Enum): STOP_USAGE, STOP_INELIGIBLE, STOP_BAD_REQUEST, RETRY, RELOGIN`
  - `classify_error(status: int, body: dict | str) -> PlanAction`
  - `class PlanStop(RuntimeError)` 속성 `action`, `code`, `request_id`, `param`
  - `build_model(model_id: str, **kwargs) -> BaseChatModel`
  - `list_models() -> list[tuple[str, str]]` (slug, display_name)

- [ ] **Step 1: 실패 테스트 작성.**

```python
def test_normalize_sets_required_fields():
    out = normalize_payload({"model": "m", "input": [], "temperature": 0.2, "top_p": 1,
                             "max_output_tokens": 5, "metadata": {}, "user": "u",
                             "previous_response_id": "r"})
    assert out["store"] is False and out["stream"] is True
    assert not {"temperature", "top_p", "max_output_tokens", "metadata", "user",
                "previous_response_id"} & out.keys()

def test_system_item_becomes_instructions(): ...
def test_tools_follow_spike_format(): ...           # RESULT.md 방식

@pytest.mark.parametrize("code,status,action", [
    ("subscription_sharing_usage_limit_exceeded", 429, PlanAction.STOP_USAGE),
    ("subscription_sharing_user_not_eligible", 403, PlanAction.STOP_INELIGIBLE),
    ("subscription_sharing_unsupported_capability", 400, PlanAction.STOP_BAD_REQUEST),
    ("subscription_sharing_usage_unavailable", 503, PlanAction.RETRY),
    ("subscription_sharing_invalid_user", 401, PlanAction.RELOGIN),
])
def test_classify(code, status, action): ...

def test_detail_body_without_error_object(): ...    # {"detail": "..."}, 401/403/503
def test_failed_event_after_stream_started_raises_plan_stop(): ...
def test_stream_without_completed_raises(): ...
def test_no_fallback_on_plan_stop(): ...            # 다른 모델·과금 경로 호출 0회
def test_token_not_in_error_text(): ...
def test_request_never_targets_backend_api(): ...
```

- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.** `ChatOpenAI(use_responses_api=True, base_url="https://api.openai.com/v1", ...)`에 공개 인자 `http_client`, `http_async_client`로 httpx 전송 래퍼를 넣는다. 래퍼가 요청마다 `normalize_payload`를 적용하고 `get_access_token()`을 Bearer로 넣는다. `_generate`, `_agenerate`는 기존 EvoScientist처럼 스트림을 읽는다. `openai_auth_mode="chatgpt_plan"`일 때 `models.py`가 `build_model`을 쓴다. `PlanStop`은 `model_retry` 미들웨어의 재시도 대상에서 뺀다.
- [ ] **Step 4: 통과 확인 후 커밋.** `feat: connect to Responses API with ChatGPT plan`

### Task 9: `/login`, `/logout`, 모델 선택, 첫 실행 안내

**Files:**
- Create: `metasci_scribe/commands/implementation/auth.py`
- Modify: `commands/implementation/model.py`, `commands/implementation/__init__.py`, `config/onboard/`
- Test: `tests/test_auth_commands.py`

**Interfaces:**
- Consumes: `login`, `clear`, `load` (T7), `list_models` (T8)
- Produces: `LoginCommand`, `LogoutCommand` (`Command` 상속, `name="/login"`, `"/logout"`)

- [ ] **Step 1: UI/UX 가이드라인 읽기.** `https://developers.openai.com/siwc/token-sharing-open-source/ui-ux-guidelines`. 터미널 표기와 브랜딩을 결정해 `docs/baseline.md`에 적는다.
- [ ] **Step 2: 실패 테스트 작성.** `/login` 성공 시 계정과 플랜 상태 표시, `plan_enabled=False`면 API 키 안내, `/logout` 후 `load() is None`, 로그인 없이 메시지를 보내면 `/login` 안내.
- [ ] **Step 3: 실패 확인.**
- [ ] **Step 4: 구현.** `Command` 패턴을 따른다. 로그인 대기 중 Ctrl+C로 취소할 수 있게 한다. 온보딩은 ChatGPT 로그인을 첫 선택지로 두고, API 키를 대체 경로로 둔다.
- [ ] **Step 5: 통과 확인 후 커밋.** `feat: add login and logout commands`

---

## M4: 글쓰기 기능

### Task 10: `/init`과 네트워크 경로 감지

**Files:**
- Create: `metasci_scribe/project.py`, `metasci_scribe/templates/{program.md,outline.md}`, `metasci_scribe/commands/implementation/init.py`
- Modify: `pyproject.toml` (패키지 데이터)
- Test: `tests/test_project_init.py`

**Interfaces:**
- Produces:
  - `is_network_path(path: Path) -> bool`
  - `class NetworkPathError(RuntimeError)`
  - `@dataclass InitResult(created: list[Path], skipped: list[Path])`
  - `init_project(root: Path) -> InitResult`
  - `extract_dir(root: Path) -> Path` (`root/.metasci-scribe/extracted`)

- [ ] **Step 1: 실패 테스트 작성.**

```python
def test_init_creates_layout(tmp_path):
    r = init_project(tmp_path)
    for n in ["program.md", "outline.md", "manuscript.md", "Knowledge", "Figures", "Tables",
              ".metasci-scribe"]:
        assert (tmp_path / n).exists()

def test_init_never_overwrites(tmp_path):
    (tmp_path / "outline.md").write_text("mine", encoding="utf-8")
    r = init_project(tmp_path)
    assert (tmp_path / "outline.md").read_text(encoding="utf-8") == "mine"
    assert tmp_path / "outline.md" in r.skipped

def test_unc_path_is_network(): assert is_network_path(PureWindowsPath(r"\\host\share\p"))
def test_init_refuses_network_path(monkeypatch, tmp_path): ...
def test_init_is_idempotent(tmp_path): ...
```

- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.** Windows는 `GetDriveTypeW`로 `DRIVE_REMOTE`를 확인하고 UNC 경로도 원격으로 본다. 다른 OS는 False. 템플릿은 기존 metasciscribe 형식을 단순화하되 `### R1 · ①` 단락 표기와 ◇ 미결 표시를 유지한다.
- [ ] **Step 4: 통과 확인 후 커밋.** `feat: add project init`

### Task 11: PDF 텍스트 추출

**Files:**
- Create: `metasci_scribe/knowledge.py`
- Modify: `pyproject.toml` (`pypdf`), `cli/interactive.py`
- Test: `tests/test_knowledge.py`, `tests/fixtures/{text.pdf,scan.pdf,broken.pdf}`

**Interfaces:**
- Consumes: `extract_dir` (T10)
- Produces: `@dataclass SyncReport(extracted: list[Path], unchanged: int, scanned: list[Path], failed: list[tuple[Path, str]])`, `sync_knowledge(root: Path) -> SyncReport`

- [ ] **Step 1: 실패 테스트 작성.**

```python
def test_extract_marks_pages(project): ...          # "[p. 1]", "[p. 2]" 포함
def test_unchanged_file_is_skipped(project): ...    # 두 번째 실행 unchanged == 1
def test_changed_file_is_reextracted(project): ...
def test_scan_pdf_reported_not_failed(project): ... # 텍스트 없으면 scanned
def test_broken_and_encrypted_do_not_stop_others(project): ...
def test_korean_filename_with_spaces(project): ...
def test_original_pdf_unchanged(project): ...
```

- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.** 변경 판정은 파일 크기와 수정 시각. 출력은 임시 파일 후 교체. 앱 시작 시 호출하고 scanned, failed를 화면에 알린다.
- [ ] **Step 4: 통과 확인 후 커밋.** `feat: extract PDF text from Knowledge`

### Task 12: 글쓰기 프롬프트와 서브에이전트

**Files:**
- Modify: `metasci_scribe/prompts.py`, `subagents/{planner,research,writing}.yaml`, `skills/` (`find-skills`, `skill-creator`만 유지)
- Test: `tests/test_writing_prompts.py`

**Interfaces:**
- Produces: `prompts.WRITING_WORKFLOW: str`. `EXPERIMENT_WORKFLOW`, `REPORT_TEMPLATE`는 삭제.

- [ ] **Step 1: 실패 테스트 작성.** 시스템 프롬프트가 `[evidence needed]`, `manuscript.md`, `.metasci-scribe/extracted`를 포함하고 `experiment`를 포함하지 않는다. `research`의 도구에 웹 검색이 없다. 내장 스킬이 두 개다.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.** 설계 6장 프롬프트 규칙을 그대로 쓴다. `planner`는 구성 지도 정리, `research`는 프로젝트 자료 정리, `writing`은 단락 초안을 맡는다. 전문 작성 규칙은 넣지 않는다.
- [ ] **Step 4: 가짜 모델로 대화 시험.** 파일 읽기와 쓰기가 포함된 한 턴이 끝나는 통합 테스트를 추가한다.
- [ ] **Step 5: 통과 확인 후 커밋.** `feat: replace experiment workflow with writing workflow`

---

## M5: 자율 실행

### Task 13: 구성 지도 읽기와 실행 상태

**Files:**
- Create: `metasci_scribe/run.py`
- Modify: `metasci_scribe/tools/__init__.py`
- Test: `tests/test_run_state.py`

**Interfaces:**
- Produces:
  - `@dataclass Paragraph(id: str, section: str, title: str)`
  - `class OutlineError(ValueError)` 속성 `line`
  - `parse_paragraphs(text: str) -> list[Paragraph]`
  - `open_markers(text: str) -> list[int]` (◇가 있는 줄 번호)
  - `class RunExit(IntEnum): OK=0, FAILED=2, USAGE_LIMIT=3, NEEDS_DECISION=4`
  - `@dataclass RunState(thread_id: str, outline_digest: str, paragraphs: dict[str, str], reasons: dict[str, str])`, `RunState.load(root) -> RunState | None`, `save(root) -> None`
  - `AgentInvoke = Callable[[str, str], None]` (`thread_id`, `prompt`)
  - `run_project(root: Path, invoke: AgentInvoke) -> RunResult`
  - `@dataclass RunResult(exit: RunExit, done: int, total: int, evidence_needed: int, message: str)`
  - 도구 `mark_paragraph(paragraph_id: str, status: Literal["done", "blocked"], reason: str = "")`

- [ ] **Step 1: 실패 테스트 작성.**

```python
def test_parse_paragraph_ids(): ...                 # "### R1 · ①" → Paragraph("R1", ...)
def test_duplicate_id_reports_line(): ...           # OutlineError.line
def test_open_marker_blocks_start(project):
    assert run_project(project, boom).exit == RunExit.NEEDS_DECISION   # invoke 0회
def test_missing_files_suggest_init(project): ...
def test_complete_only_when_all_done(project): ...  # 모델이 끝났다고 해도 pending 있으면 재호출
def test_gives_up_after_three_passes(project): ...  # RunExit.FAILED
def test_blocked_stops_with_4(project): ...
def test_usage_limit_exits_3_and_keeps_state(project): ...
def test_resume_uses_same_thread_when_outline_same(project): ...
def test_outline_change_new_thread_keeps_done(project): ...
def test_evidence_needed_counted_not_failure(project): ...
def test_save_is_atomic_on_crash(project): ...      # 교체 전 예외 → 이전 파일 유효
def test_keyboard_interrupt_preserves_state(project): ...
def test_mark_paragraph_rejects_unknown_id(project): ...
```

`boom`은 호출되면 실패하는 `AgentInvoke`다.

- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.** 설계 6장의 규칙을 따른다. 상태 기록은 모델이 JSON을 직접 쓰지 않고 `mark_paragraph` 도구로 받아 ID를 검증한다(내 결정). `invoke` 재호출은 최대 3회. `PlanStop(STOP_USAGE)`는 `USAGE_LIMIT`, 나머지 `PlanStop`은 `FAILED`로 바꾼다. 네트워크 경로와 `sync_knowledge`는 시작 시 처리한다. 모델이 없는 순수 로직으로 시험한다.
- [ ] **Step 4: 통과 확인 후 커밋.** `feat: add run state and resume logic`

### Task 14: `metasci-scribe run`과 `/run`

**Files:**
- Create: `metasci_scribe/commands/implementation/run.py`, `metasci_scribe/run_agent.py`
- Modify: `cli/commands.py`
- Test: `tests/test_run_command.py`

**Interfaces:**
- Consumes: `run_project`, `RunResult` (T13), `create_cli_agent`
- Produces: `real_invoke(root: Path) -> AgentInvoke`, `RunCommand` (`name="/run"`), CLI `metasci-scribe run`

- [ ] **Step 1: 실패 테스트 작성.** 가짜 모델로 전체 완료(0), 사용 한도(3), `blocked`(4), 재실행 재개를 CLI 수준에서 확인한다. `/run`이 별도 프로세스를 실행하고 대화 스레드를 바꾸지 않는다. 자율 실행에서 `ask_user`가 꺼져 있고 명령 차단 규칙은 켜져 있다.
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.** `real_invoke`는 설정을 복사해 `auto_mode=True`로 `create_cli_agent`를 만들고, 기존 단발 실행 경로의 local gateway로 한 번 실행한다. `/run`은 `subprocess`로 `metasci-scribe run`을 실행해 출력을 대화 화면에 표시한다. 종료 코드는 `RunExit` 값 그대로 쓴다.
- [ ] **Step 4: 통과 확인 후 커밋.** `feat: add autonomous run command`

---

## M6: 배포 준비

### Task 15: 설치, 문서, 완료 기준 시험

**Files:**
- Modify: `README.md`, `pyproject.toml`, `.github/workflows/`
- Create: `docs/acceptance.md`
- Test: 전체 테스트, 수동 시험

- [ ] **Step 1: 워크플로 정리.** Docker, 릴리스, 설치 스모크 워크플로를 지우고 lint와 test만 남긴다. 저장소 내 `EvoScientist` 문자열은 `NOTICE`와 README 출처 표기에만 남는지 검색으로 확인한다.
- [ ] **Step 2: README 작성.** 설치 전제(Python, uv, Git), 설치 명령, 사용 흐름, 로컬 디스크 조건, 지원 플랜과 미리보기 경고, 사용량 안내.
- [ ] **Step 3: 전체 시험.** `uv run pytest -n auto -q`, `uv run ruff check .` 통과.
- [ ] **Step 4: 약관 재확인.** 공개 전에 Sign in with ChatGPT 문서의 현재 조건을 다시 읽고 `docs/acceptance.md`에 확인일과 내용을 적는다.
- [ ] **Step 5: 수동 완료 기준 시험.** 깨끗한 Windows 사용자 환경에서 설계 2장의 7단계를 수행하고 결과를 `docs/acceptance.md`에 적는다. 사용자가 브라우저에서 로그인한다.
- [ ] **Step 6: 커밋.** `docs: add install guide and acceptance record`

공개 저장소 생성과 push는 사용자의 지시를 받은 뒤 진행한다.
