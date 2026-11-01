"""Tests for the streaming runtime shared by the interactive and single-shot CLI."""

import asyncio
import threading
import time

import pytest

from metasci_scribe.cli.stream_runtime import (
    StreamCancellationTimeout,
    run_streaming,
    run_streaming_async,
)
from metasci_scribe.runtime import AsyncRuntimeError
from tests.fakes.common import FakeGraphGateway


def test_run_streaming_passes_the_starting_grant_to_the_loop(monkeypatch):
    """The Rich loop gets ``approve_all_at_start``."""
    seen = {}

    def _loop(**kwargs):
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr("metasci_scribe.cli.stream_runtime._run_streaming", _loop)
    result = run_streaming(
        agent=object(),
        message="hello",
        thread_id="t1",
        show_thinking=False,
        interactive=True,
        gateway=FakeGraphGateway(),
        approve_all_at_start=True,
    )
    assert result == "ok"
    assert seen["approve_all_at_start"] is True


def test_run_streaming_lets_an_owned_runtime_error_through(monkeypatch):
    def _loop(**kwargs):
        raise AsyncRuntimeError("owned runtime failed")

    monkeypatch.setattr("metasci_scribe.cli.stream_runtime._run_streaming", _loop)

    with pytest.raises(AsyncRuntimeError, match="owned runtime failed"):
        run_streaming(
            agent=object(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            gateway=FakeGraphGateway(),
        )


async def test_async_streaming_cancellation_stops_and_joins_worker(monkeypatch):
    from metasci_scribe.stream.display import (
        discard_stream_cancel,
        is_stream_cancel_requested,
    )

    scope = "test:async-renderer-cancel"
    started = threading.Event()
    finished = threading.Event()

    def fake_run_streaming(**kwargs):
        assert kwargs["cancel_scope"] == scope
        started.set()
        while not is_stream_cancel_requested(scope):
            time.sleep(0.001)
        finished.set()
        return "stopped"

    monkeypatch.setattr(
        "metasci_scribe.cli.stream_runtime.run_streaming", fake_run_streaming
    )

    task = asyncio.create_task(run_streaming_async(cancel_scope=scope))
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
    discard_stream_cancel(scope)


async def test_async_streaming_can_recover_foreground_task_after_cancel(monkeypatch):
    from metasci_scribe.stream.display import (
        discard_stream_cancel,
        is_stream_cancel_requested,
    )

    scope = "test:async-renderer-recover"
    started = threading.Event()

    def fake_run_streaming(**kwargs):
        started.set()
        while not is_stream_cancel_requested(scope):
            time.sleep(0.001)
        return "[Stopped.]"

    cleanup_called = False

    async def fake_cleanup() -> None:
        nonlocal cleanup_called
        cleanup_called = True

    monkeypatch.setattr(
        "metasci_scribe.cli.stream_runtime.run_streaming", fake_run_streaming
    )
    monkeypatch.setattr(
        "metasci_scribe.middleware.code_interpreter.aclose_code_interpreters",
        fake_cleanup,
    )

    task = asyncio.create_task(
        run_streaming_async(cancel_scope=scope, recover_on_cancel=True)
    )
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    assert await task == "[Stopped.]"
    assert cleanup_called
    assert not task.cancelled()
    discard_stream_cancel(scope)


async def test_noncooperative_worker_reports_settlement_timeout(monkeypatch):
    """Cancellation timeout is an ordinary lifecycle error, not BaseException."""
    from metasci_scribe.cli import stream_runtime
    from metasci_scribe.stream.display import discard_stream_cancel

    scope = "test:async-renderer-timeout"
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def fake_run_streaming(**_kwargs):
        started.set()
        release.wait()
        finished.set()
        return "late"

    async def fake_cleanup() -> None:
        return None

    monkeypatch.setattr(stream_runtime, "run_streaming", fake_run_streaming)
    monkeypatch.setattr(stream_runtime, "STREAM_CANCEL_SETTLE_TIMEOUT", 0.05)
    monkeypatch.setattr(
        "metasci_scribe.middleware.code_interpreter.aclose_code_interpreters",
        fake_cleanup,
    )

    task = asyncio.create_task(
        run_streaming_async(cancel_scope=scope, recover_on_cancel=True)
    )
    assert await asyncio.to_thread(started.wait, 1)
    task.cancel()

    with pytest.raises(StreamCancellationTimeout, match="did not stop"):
        await task

    assert not finished.is_set()
    release.set()
    assert await asyncio.to_thread(finished.wait, 1)
    await asyncio.sleep(0)
    discard_stream_cancel(scope)
