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
