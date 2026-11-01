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
