"""Tests for TUI command-completion state sync."""

from types import SimpleNamespace

import pytest

from metasci_scribe.commands.base import CommandContext, SessionRuntime
from metasci_scribe.paths import SessionDirs
from tests.fakes.common import TEST_WORKSPACE

pytest.importorskip("textual")


class _Loader:
    def __init__(self) -> None:
        self.adopt_calls: list[object] = []
        self.agent: object | None = None

    def adopt(self, agent: object) -> None:
        self.adopt_calls.append(agent)
        self.agent = agent


class _StubApp:
    def __init__(self) -> None:
        self._agent_loader = _Loader()
        self._conversation_tid = "thread-1"
        self._session_runtime = SessionRuntime()
        self._exiting = False
        self.model_updates: list[tuple[str, str | None]] = []
        self.refresh_calls: list[bool] = []

    def update_status_after_model_change(
        self,
        new_model: str,
        new_provider: str | None = None,
    ) -> None:
        self.model_updates.append((new_model, new_provider))

    async def _refresh_status_snapshot(
        self,
        *,
        reset_streaming_text: bool = True,
    ) -> None:
        self.refresh_calls.append(reset_streaming_text)


async def test_sync_tui_command_completion_adopts_agent_swap(monkeypatch):
    import metasci_scribe.cli.tui_interactive as tui_mod
    from metasci_scribe import agent as scribe_mod

    app = _StubApp()
    ctx = CommandContext(
        dirs=SessionDirs(TEST_WORKSPACE),
        agent="new-agent",
        thread_id="thread-1",
        ui=SimpleNamespace(),
    )
    cmd = SimpleNamespace(name="/model")

    monkeypatch.setattr(
        scribe_mod,
        "_ensure_config",
        lambda: SimpleNamespace(model="gpt-5.5", provider="openai"),
    )

    await tui_mod._sync_tui_command_completion(app, ctx, "old-agent", cmd)

    assert app._agent_loader.adopt_calls == ["new-agent"]
    assert app.model_updates == [("gpt-5.5", "openai")]
    assert app.refresh_calls == [True]


async def test_sync_tui_command_completion_refreshes_without_agent_swap(monkeypatch):
    import metasci_scribe.cli.tui_interactive as tui_mod

    app = _StubApp()
    ctx = CommandContext(
        dirs=SessionDirs(TEST_WORKSPACE),
        agent="same-agent",
        thread_id="thread-1",
        ui=SimpleNamespace(),
    )
    cmd = SimpleNamespace(name="/compact")

    await tui_mod._sync_tui_command_completion(app, ctx, "same-agent", cmd)

    assert app._agent_loader.adopt_calls == []
    assert app.model_updates == []
    assert app.refresh_calls == [True]


async def test_sync_tui_command_completion_skips_unmounted_app(monkeypatch):
    import metasci_scribe.cli.tui_interactive as tui_mod

    app = _StubApp()
    app._exiting = True
    ctx = CommandContext(
        dirs=SessionDirs(TEST_WORKSPACE),
        agent="new-agent",
        thread_id="thread-1",
        ui=SimpleNamespace(),
    )
    cmd = SimpleNamespace(name="/exit")

    await tui_mod._sync_tui_command_completion(app, ctx, "old-agent", cmd)

    assert app._agent_loader.adopt_calls == []
    assert app.model_updates == []
    assert app.refresh_calls == []
