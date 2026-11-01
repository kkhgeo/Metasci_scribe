"""Unit tests for helpers in ``metasci_scribe.commands.base``."""

from __future__ import annotations

from metasci_scribe.commands.base import SessionRuntime


class TestSessionRuntimeClear:
    """``SessionRuntime.clear`` drops the agent binding."""

    def test_clear_drops_the_agent_and_thread(self):
        runtime = SessionRuntime()
        runtime.agent = object()
        runtime.thread_id = "t-42"
        runtime.clear()
        assert runtime.agent is None
        assert runtime.thread_id is None
