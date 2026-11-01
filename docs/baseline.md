# Test baseline

Measured on the unmodified EvoScientist v0.3.8 import (commit aa55c60, upstream e2eef14),
before any rename. Windows 11, Python 3.13.14 (uv-managed), default locale cp949.

Command: `uv run pytest -n auto -q` (with `UV_PROJECT_ENVIRONMENT=C:\dev\venvs\metasci-scribe`)

Result: **5 failed, 7435 passed, 76 skipped** in 139 s.

## Failing tests (pre-existing)

All five fail from the Windows cp949 default locale, not from the code under test
(`Path.read_text()` / `open()` without `encoding="utf-8"` in the tests, hitting a UTF-8 byte or an em dash).

| Test | Error |
| --- | --- |
| `tests/test_config.py::TestDotenvIsolation::test_copied_template_does_not_shadow_config_keys` | UnicodeDecodeError cp949 |
| `tests/test_hitl_loop.py::test_tui_unresolved_pause_exits_without_budget_notice` | UnicodeDecodeError cp949 |
| `tests/test_hitl_loop.py::test_tui_budget_stop_folds_notice_into_channel_response` | UnicodeDecodeError cp949 |
| `tests/test_hitl_loop.py::test_tui_hitl_loop_checks_completed_round_cap_before_clearing_pending` | UnicodeDecodeError cp949 |
| `tests/test_langgraph_manager.py::TestReadTunnelUrl::test_returns_none_on_timeout` | UnicodeEncodeError cp949 |

Rule for later tasks: the failure count must not exceed 5 (and the failing set must not grow).

## Task 4: TUI, WebUI and operations surfaces removed

Result after the removal: **2 failed, 5434 passed, 34 skipped** in 86 s. The test count drops because the
TUI, WebUI, updater, proactive, install-script and ccproxy tests went with their code.

Failing set after the removal (same command): the two cp949 failures that still exist,
`tests/test_config.py::TestDotenvIsolation::test_copied_template_does_not_shadow_config_keys`
and `tests/test_langgraph_manager.py::TestReadTunnelUrl::test_returns_none_on_timeout`.
The two `tests/test_hitl_loop.py::test_tui_*` failures went away with the tests: they read
`cli/tui_interactive.py` as source text, and that module is gone.

Decisions:

- **`deploy` was deleted.** The rule is: delete it when cutting its imports takes 100 lines or fewer.
  The imports ran through `cli/__init__.py` (1 line), `config/onboard/helpers.py` (`no_display`,
  inlined, about 10 lines) and `langgraph_dev/manager.py` (the `job` parameter of
  `start_langgraph_dev`, about 12 lines). That is about 25 lines, so it went, with `deploy/busy.py`,
  `backend_job.py`, `json_mode.py`, `launcher.py`, `server.py` and `webui.py`.
- **`setup` keeps the `node`, `git` and `research-env` stages.** Git Bash detection and the agent
  Python environment stay as required. The `webui` stage is gone. Node stays because MCP servers
  started through `npx` and `activate_runtime()` at CLI start-up still rely on it; the brief's
  "only Git Bash and agent Python" would also drop that, so this is flagged for the lead.
- **The Claude subscription route is gone.** `ccproxy_manager`, `anthropic_auth_mode`, `ccproxy_port`,
  the `ccproxy-oauth` placeholder key handling in `config/settings.py`, the onboarding step and the
  `metasci-scribe[oauth]` install hints are deleted. `openai_auth_mode: oauth` (ChatGPT sign-in) stays.
- **`ui_backend` is `Literal["cli"]`, default `"cli"`.** A config file or `ScribeConfig(...)` that holds
  `tui`, `webui` or `textual` loads as `cli` with the existing "Invalid ui_backend" warning; `rich`
  is still an alias for `cli`. `config set ui_backend tui` is a clear error.
- **`GatewaySurface.TUI` and `gateway_backend_tui` are gone** (the TUI was the only user). Old config
  files that carry the key are filtered out on load.
- **Shared code moved, not deleted:** `cli/tui_runtime.py` became `cli/stream_runtime.py` (streaming
  worker and cancellation helpers; the backend selection and TUI fallback are gone), and the picker
  style and thread grouping from `cli/widgets/thread_selector.py` became `cli/picker.py`.
- **`mcp/registry.py` lost its uv-tool-environment installer** (it needed `uv_tool`). Libraries and
  CLI tools install with `uv pip install` or `pip install`, or `uv tool install` for CLI tools.
- **Dead media-folder plumbing removed:** `agent._run_media_dir`, the `media_dir` arguments of the
  sandbox backend, `prepare_sandbox_command` and the background-process tool, `_MediaFolderView`,
  and `Workspace.media_dir`. Only channel attachments used them.
- **Removed alongside:** `/proactive`, the proactive graph and its `langgraph.json` entry,
  `ProactiveModeMiddleware`, the unused `sessions.latest_origin_checkpoints`, the `update` command,
  `scripts/install.*`, `.github/workflows/install-smoke.yml` and `.github/scripts/install_smoke_*`
  (they could only fail), the `textual` and `packaging` dependencies, and the stale pins in
  `constraints.txt` (regenerated from `uv.lock`).

Not done here (left for later tasks): the langgraph dev server's WebUI-facing HTTP routes
(`langgraph_dev/http.py`, `settings_api.py`, the `boundary.py` CORS rules) are unchanged, and prose
that still mentions the TUI or the WebUI in comments and in `docs/` and the README.

## Task 5: commands, middleware, sub-agents and providers reduced

Result after the reduction (same command): **2 failed, 4551 passed, 25 skipped** in 75 s. The count
drops again because the tests of the removed commands, middleware, sub-agents, providers and
WebUI routes went with them. The two failures are the cp949 ones from Task 4
(`tests/test_config.py::TestDotenvIsolation::test_copied_template_does_not_shadow_config_keys` and
`tests/test_langgraph_manager.py::TestReadTunnelUrl::test_returns_none_on_timeout`).

Decisions:

- **Defaults.** `provider="openai"`, `model="gpt-6.1-sol"`; `memory_profile_enabled`,
  `memory_observations_enabled`, `memory_workers_enabled`, `memory_skill_synthesis_enabled`,
  `enable_async_subagents` and `enable_scheduler` are all `False`. These are the only triggers of
  `needs_langgraph_dev`, so a plain start spawns no `langgraph dev` server and prints no
  "Starting background agent server" line. `tests/test_reduced_surface.py` pins that with the spawn
  function patched.
- **Providers: OpenAI and Anthropic only.** `config.provider_keys.PROVIDERS`, `VALID_PROVIDERS` and the
  model registry hold just those two. All routed providers (MiniMax, DeepSeek, Moonshot, Zhipu, ...,
  `custom-openai`, `custom-anthropic`), Google, NVIDIA, Ollama and OpenRouter are gone with their
  settings, key checks, onboarding steps and the OpenRouter referer. `OPENAI_BASE_URL` and
  `ANTHROPIC_BASE_URL` still point the two providers at a compatible gateway.
- **`openai_auth_mode`** is `Literal["api_key", "chatgpt_plan"]`, default `"api_key"`. `"chatgpt_plan"` is
  reserved for the ChatGPT plan connector (T8): `get_chat_model` refuses it with a clear message. The
  old `"oauth"` value reads as `"api_key"` with the usual invalid-value warning; `llm/chatgpt_oauth.py`,
  the `--auth-mode` flag, the sign-in onboarding step and `_patch_openai_responses_whole_tool_args`
  (Codex-only) are deleted.
- **Tavily is gone completely**: the tool (`tools/search.py`, with the unused `fetch_webpage_content`),
  the `tavily_api_key` setting, its onboarding step, `configure tavily`, `--tavily-key`, the key check, and
  the `tavily-python` and `markdownify` dependencies. `research.yaml` now lists only `think_tool`.
- **`model_fallback` is gone**, so `model_retry` took over the two helpers it used from it
  (`_is_deterministic_request_error`, and a single failure instead of a chain). `UnknownModelError`
  stays and is what an unsupported provider raises.
- **Server HTTP surface.** `langgraph_dev/http.py` stays, with only `/api/bg_process_status` (the CLI's
  `gateway/local.py` and `gateway/server.py` poll it) and the `BrowserBoundary` guard in front of the
  whole server. `settings_api.py`, `config_baseline.py`, `local_secret.py`, the other routes, the
  `cors` block of `langgraph.json` and `boundary.cors_config` / `CORS_CONFIG` are removed. The guard is kept on
  purpose: it is what stops a web page from driving the loopback server.
- **agent_shell job objects stay.** They are not dead: `ShellLaunch.started` / `terminate_job` are used by
  the agent's `execute` tool (`backends.py`) and by background processes (`background.py`).
- **Async sub-agents** are off by default but not deleted: `writing-agent` keeps `async: true` and its
  graph. With `EvoAsyncSubAgentMiddleware` gone they use deepagents' own `AsyncSubAgentMiddleware`, so
  they no longer forward the session folders or a `/model` override to the child run.
- **cron** keeps only what AutoSkills uses (`create_cron`, `delete_cron`, search and ownership helpers);
  the scheduling API, the startup re-tagging of scheduled tasks and the "autoskill proposals" notice
  of the deleted `/autoskills` command are removed.
- **Sub-agents**: only `planner`, `research` and `writing`. The system prompt no longer sends work to
  `code-agent`, `debug-agent` or `data-analysis-agent`, and the code-generation mode question is gone.

Not done here: `config/settings.py` still carries the env-origin record (`ENV_ORIGINS_VAR`,
`record_env_origin`, `_server_env_origins` in the manager) that only the deleted settings API read;
`middleware/events.py` and `stream/sink.py` still carry the fallback-notice event; `llm/patches.py` keeps
the unused message-flattening patches and client proxies; `tools/skills_manager.py` still classifies expert
skills. Prose in `docs/` and the README is for T15.
