"""Turns the TUI's queue poll starts on its own (background task
notifications) take the app only if it is still free when they start.

The poll runs on its timer and schedules the turn as a callback on the app's
message pump, so input handled in between can take the app first. These tests
call the poll themselves, as the timer would, and count the runs that are going
at once. They wait with ``asyncio.sleep``: a turn started that way holds
the message pump, which ``pilot.pause`` would wait for."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("textual")

from metasci_scribe.cli import async_notifier
from metasci_scribe.cli.async_notifier import (
    AsyncTaskNotification,
    drain_notifications,
    enqueue_task_notification,
    has_pending_notifications,
)
from metasci_scribe.paths import SessionDirs
from metasci_scribe.stream import display as display_mod
from tests.fakes.common import FakeGraphGateway, FakeThreadStore
from tests.fakes.tui import NoHistory, capture_app, submit

NOTICE = "notice"


def _label(message: str) -> str:
    return NOTICE if message.startswith("[Async tasks update]") else message


class _Runs:
    """A fake gateway whose runs each wait until the test finishes them. It
    records the runs in the order they started and the most that were ever
    going at once."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.running: list[tuple[str, asyncio.Event]] = []
        self.peak = 0
        self.free = False  # once set, a run finishes as soon as it starts
        outer = self

        class _Gateway(FakeGraphGateway):
            async def stream_events(self, request):
                self.requests.append(request)
                entry = (_label(str(request.message)), asyncio.Event())
                outer.started.append(entry[0])
                outer.running.append(entry)
                outer.peak = max(outer.peak, len(outer.running))
                try:
                    if not outer.free:
                        await entry[1].wait()
                    yield {"type": "text", "content": "ok"}
                    yield {"type": "done", "response": "ok"}
                finally:
                    outer.running.remove(entry)

        self.gateway = _Gateway(thread_store=FakeThreadStore(messages=[]))

    def finish(self, name: str | None = None) -> None:
        """Let the oldest running run (named *name*, if given) finish."""
        for entry_name, done in self.running:
            if name is None or entry_name == name:
                done.set()
                return

    def finish_all(self) -> None:
        self.free = True
        for _, done in self.running:
            done.set()


def _empty_queues() -> None:
    drain_notifications()


@pytest.fixture(autouse=True)
def _fresh_queues(monkeypatch):
    """The notification queue is module-wide."""
    _empty_queues()
    yield
    _empty_queues()
    display_mod.discard_stream_cancel()


async def _app(monkeypatch, workspace):
    app = await capture_app(monkeypatch, workspace)
    # No poll timer: each test runs the poll itself.
    app._start_notification_poll = lambda: None
    return app


def _wire(app, workspace, runs: _Runs) -> None:
    app._dirs = SessionDirs(workspace)
    app._runtime_gateways = SimpleNamespace(graph_gateway=runs.gateway)
    app._await_agent_ready = AsyncMock(return_value=object())
    app._history_suggester = NoHistory()


def _notification(app, task_id: str) -> None:
    enqueue_task_notification(
        AsyncTaskNotification(
            task_id=task_id,
            agent_name="worker",
            status="success",
            received_at="2026-10-18T00:00:00Z",
            origin_cli_thread_id=app._conversation_tid,
        )
    )


def _idle(app, runs: _Runs) -> bool:
    return (
        not app._busy
        and app._run_task is None
        and not app._queued_messages
        and not app._notification_consuming
        and not runs.running
        and not has_pending_notifications(app._conversation_tid)
    )


async def _until(condition, *, seconds: float = 10.0) -> None:
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("the expected point was never reached")


async def _poll_until(app, condition, *, seconds: float = 10.0) -> None:
    """Run the poll in place of its timer, every 50 ms, until *condition* holds."""
    for _ in range(int(seconds / 0.05)):
        if condition():
            return
        app._poll_notification_queue()
        await asyncio.sleep(0.05)
    raise AssertionError("the expected point was never reached")


async def test_a_notifier_turn_waits_for_input_typed_before_it_starts(
    monkeypatch, workspace
):
    """Started anyway, the notifier turn took the typed turn's handle: Ctrl+C
    then stopped the notifier turn and lost the notification, and the typed
    turn ran on with the app marked free."""
    runs = _Runs()
    app = await _app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        try:
            await pilot.pause()
            _wire(app, workspace, runs)
            _notification(app, "task-1")
            app._poll_notification_queue()  # schedules the notifier
            await submit(app, "typed")  # Enter, handled before the notifier runs
            typed = app._run_task
            await _until(lambda: runs.started)
            # Past the notifier's batch wait.
            await asyncio.sleep(async_notifier.NOTIFICATION_BATCH_GRACE_SECONDS + 0.3)
            assert runs.started == ["typed"]
            assert app._run_task is typed
            assert has_pending_notifications(app._conversation_tid)
            app.action_request_quit()  # stops the typed turn
            await _until(lambda: not app._busy)
            assert typed.done()
            await _poll_until(app, lambda: len(runs.started) == 2)
            assert runs.started == ["typed", NOTICE]
            runs.finish_all()
            await _poll_until(app, lambda: _idle(app, runs))
            assert runs.peak == 1
        finally:
            runs.finish_all()


async def test_the_notification_poll_starts_once_the_agent_has_loaded(
    monkeypatch, workspace
):
    """The 100 ms poll is what drains notifications: it must run without
    anything else (it used to start only with the channel startup)."""
    app = await capture_app(monkeypatch, workspace)
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        await _until(lambda: app._notification_timer is not None)
        timer = app._notification_timer
        app._start_notification_poll()  # a second start keeps the one timer
        assert app._notification_timer is timer
