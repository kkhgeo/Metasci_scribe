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

## Task 9: `/login`, `/logout`, model choice and first-run guidance

OpenAI's UI/UX page for ChatGPT plan usage is `https://developers.openai.com/siwc/ui-ux-guidelines`; the
path the brief gave (under `token-sharing-open-source/`) returns 404. It prescribes the label
"Continue with ChatGPT" for a control that starts the sign-in, a one-time notice after the first sign-in
with plan usage on ("You're using your ChatGPT plan."), "Using ChatGPT plan" with a "Manage usage" link
(`chatgpt.com/settings/usage`) wherever requests use the plan, and "Manage usage" as the first action on a
usage-limit error. It has no rules for terminals, and none on logos, sign-out wording or consent.

Terminal decision, in `auth/notices.py`:

- **Label and branding.** Plain text only. No logo, mark, colour or ASCII art stands for ChatGPT or OpenAI;
  the approved button formats are for web pages. The label is exactly `Continue with ChatGPT`, the first
  choice of the onboarding provider step (API keys follow as the other way in) and the words at the top of
  `/login` ("Opening your browser to continue with ChatGPT"). The command names are `/login` and `/logout`.
- **After sign-in.** Account (email, when OpenAI sent one), "ChatGPT plan usage is on." and the
  "Manage usage" URL. The notice above is shown once per machine (`auth/plan_welcome` marker, kept by every
  sign-out) and not on later sign-ins. A sign-in without plan usage says so, switches nothing, and names
  the API key.
- **Not done:** a "Using ChatGPT plan" tag in the status line or the model picker. The CLI status line and
  the picker table have no per-route field, so it is left for a later task.

Other decisions:

- **`/logout` revokes, best effort, and forgets the account.** `auth.sign_out()` posts the refresh token to
  the `revocation_endpoint` of the discovery document (form body `token`, `token_type_hint=refresh_token`,
  `client_id`), up to three tries on a network error, 5xx, 408 or 429, and refuses an endpoint off the
  issuer's host. Whatever happens it then deletes `chatgpt.json` whole: tokens, `subject` and the issued
  `client_id`. The result carries a fixed problem text, which `/logout` shows with "disconnect Metasci
  Scribe in ChatGPT Settings". `host_id` stays. This deliberately departs from OpenAI's advice to reuse the
  client after sign-out: with the registration kept, a different ChatGPT account fails with
  `account_mismatch`, so `/logout` is how to switch accounts and the next sign-in registers a new client.
  `clear()` is unchanged and still keeps the registration. The revocation runs with the sign-in lock held
  (no new try starts after 60 s, so a waiting process, which gives up after 120 s, is not starved).
- **`/login` can be cancelled.** `login(cancel=Event)` ends the wait, closes the loopback listener and raises
  `LoginCancelled`. The command runs the sign-in in a thread and, on the main thread, routes the first
  Ctrl+C to that event instead of unwinding the session; a second Ctrl+C is an ordinary interrupt.
- **`/login` on success** saves `openai_auth_mode=chatgpt_plan` and `provider=openai` (and the default model
  when the session was on another provider), then rebuilds the session through `/model`'s transactional
  switch (`_apply_model(..., auth_mode=...)`). The route is set in the environment for the trial build and
  put back if the build or the agent load fails. Not applied to a running langgraph server (the command says
  to restart). The model the session keeps is checked against the account's `list_models()`: one the account
  does not list gives way to the first listed (said, and saved); an unreadable catalog keeps the model. A
  co-pilot (auxiliary) model setting is cleared, in the config and the session, so the main instance serves
  background work (said; restored in the session if the switch fails). The browser-could-not-open message
  goes through the command UI (`login(show_url=...)`). Ctrl+C returns within about a second even inside the
  token exchange, and `login` re-reads the cancel event before saving, so a sign-in finished after a cancel
  saves nothing and the session is not half-switched.
- **`/logout` and the route.** Under `openai_auth_mode=chatgpt_plan` it saves `api_key` and the session
  follows: with `OPENAI_API_KEY` set an OpenAI session is rebuilt on the key and told so ("billed to your API
  account, not your ChatGPT plan"); without one it says the session cannot send requests until `/login` or a
  key plus restart. `ensure_model_configured` never lets a key serve the plan route: under a persisted plan
  mode with no sign-in the message is "Not signed in to ChatGPT. Run /login, or ... config set
  openai_auth_mode api_key". A record that cannot be deleted after the revocation is reported beside it
  (`SignOutResult.removal_problem`) and leaves the route alone. The auxiliary build paths in `agent.py` now
  go through `ensure_model_configured` too.
- **`/model` on the plan** lists the account's models (`list_models()`, run in a thread) and accepts a slug
  only if the account lists it; Claude names still go by their own key. A `PlanStop` is shown with its own
  message (code and request ID included), yellow for `RETRY`, red otherwise.
- **`ModelNotConfigured`** (`llm/credentials.py`) replaces both text-matching "API key not configured"
  branches. `agent._build_chat_model` raises it through `ensure_model_configured` when OpenAI has no key and
  no saved sign-in (or the plan route has no sign-in), or Anthropic has neither key nor base URL. It is
  checked there and not in `get_chat_model`, which the plan connector's tests and the lazy sign-in rely on.
  Single-shot `-p` prints the message and exits 1 instead of a traceback; the prompt prints it at start-up
  and on a message, and stays open so `/login` can fix it (`/login` adopts the rebuilt agent). A provider
  error that merely mentions an API key is now a plain "Error: ..." line.
- **Onboarding** offers `Continue with ChatGPT` first (preselected for a fresh or plan config), signs in
  inside the wizard, skips the API key step, and lists the account's models. A declined, timed-out or
  plan-off sign-in returns to the choice. Choosing an OpenAI key (or `--provider openai`) sets
  `openai_auth_mode=api_key`. The co-pilot picker has no ChatGPT entry, reuses the plan sign-in for
  OpenAI and lists the account's models on the plan route. If the folder the CLI would start in holds
  the data folder (the home folder, say), a full run stops before asking anything: it prints the
  `require_workspace_root` message with `mkdir paper`, `cd paper`, `metasci-scribe onboard`, says nothing
  was changed, never says the setup succeeded, and returns `False`. Re-running a single section
  (`configure <section>`) only warns. A run that saves a setup with no key and no saved sign-in prints
  "Not ready yet" and points to `/login` (Continue with ChatGPT) or `configure provider`.
- The `/login` warning, two lines, before the browser opens: the agent can read the paper folder; commands
  always need approval in chat and autonomous runs get no shell.

## Task 7: what a workspace `.env` can no longer set

A workspace `.env` (the one `find_dotenv` finds at or above the start folder) is a file the agent is
steered to write and a file that arrives with a cloned paper folder, so `get_effective_config` now merges
only provider credentials from it: an uppercase name ending in `_API_KEY`, `_KEY`, `_KEY_ID`, `_TOKEN`,
`_SECRET`, `_API_VERSION`, `_ORG_ID`, `_ORGANIZATION`, `_PROJECT`, `_PROJECT_ID` or `_REGION`, plus
`GOOGLE_APPLICATION_CREDENTIALS` (`metasci_scribe/policy.py`, `dotenv_key_allowed`). Everything else in the
file is ignored, with no warning yet. In particular these must now come from the shell or the config file:

- every `METASCI_SCRIBE_*` setting (`METASCI_SCRIBE_DANGEROUS_MODE`, `METASCI_SCRIBE_WORKSPACE_DIR`,
  `METASCI_SCRIBE_MEMORIES_DIR`, the langgraph dev port and host, ...);
- every endpoint: `*_BASE_URL`, `*_API_BASE`, `*_API_URL`, `*_ENDPOINT` (`OPENAI_BASE_URL`,
  `ANTHROPIC_BASE_URL`, `MINIMAX_BASE_URL`, `CUSTOM_*_BASE_URL`, `OLLAMA_BASE_URL`, `AZURE_OPENAI_ENDPOINT`);
  `.env.example` still lists the `*_BASE_URL` names, which T15 should move to the config-file section;
- anything that changes how a program runs: `GIT_*`, `GCM_*`, `SSH_*`, `PYTHON*`, `NODE_*`, `NPM_*`, `LD_*`,
  `DYLD_*`, `UV_*`, `PIP_*`, `CONDA_*`, `JAVA_*`, `PERL*`, `RUBY*`, `BASH_*`, `XDG_*`, and any name
  containing `PATH`, `PROXY`, `SSL`, `CERT`, `CA_BUNDLE` or `TRACING` (so `HTTPS_PROXY`,
  `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`);
- LangSmith tracing (`LANGCHAIN_*`, `LANGSMITH_*`, `LANGGRAPH_*`);
- lowercase or mixed-case names (Windows would map `Git_Ssh_Command` onto `GIT_SSH_COMMAND`).

The same round made the agent's file tools refuse, in every mode, to write `.env`, anything under the
project's `.metasci-scribe/` (its skills tier included: writing a skill is an install, which needs a person),
the data and config folders, `.git`, `.ssh`, shell and Python startup files and the Python environment the
app runs in; an agent `skill_manager install` pauses for approval and is unavailable in autonomous runs.
