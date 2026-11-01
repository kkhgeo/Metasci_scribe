"""Tests for sub-agent text fallback (fix/subagent-summarize).

Covers:
    1. StreamEventEmitter.subagent_text — event construction
    2. stream_agent_events — subagent_text emission for sub-agent text chunks
    3. Prompt — DELEGATION_STRATEGY contains summarize guidance
"""

from __future__ import annotations

from metasci_scribe.stream.emitter import StreamEvent, StreamEventEmitter
from tests.fakes.stream_v3 import (
    FakeSubagent,
    FakeV3Agent,
    collect_events,
    message_delta,
)

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# 1. StreamEventEmitter.subagent_text
# ═══════════════════════════════════════════════════════════════════


class TestSubagentTextEmitter:
    def test_creates_correct_event_type(self):
        ev = StreamEventEmitter.subagent_text(
            "research-agent", "Found 3 papers.", "task:research"
        )
        assert isinstance(ev, StreamEvent)
        assert ev.type == "subagent_text"

    def test_data_contains_subagent_and_content(self):
        ev = StreamEventEmitter.subagent_text(
            "analyst", "Result summary", "task:analyst"
        )
        assert ev.data["subagent"] == "analyst"
        assert ev.data["content"] == "Result summary"

    def test_data_contains_type_key(self):
        """Event data dict should include 'type' matching event type (project convention)."""
        ev = StreamEventEmitter.subagent_text("a", "b", "task:a")
        assert ev.data["type"] == "subagent_text"

    def test_empty_content(self):
        ev = StreamEventEmitter.subagent_text("agent", "", "task:agent")
        assert ev.data["content"] == ""

    def test_instance_id_included(self):
        ev = StreamEventEmitter.subagent_text(
            "agent", "content", instance_id="tracker:abc"
        )
        assert ev.data["instance_id"] == "tracker:abc"

    def test_included_in_all_events_type_check(self):
        """subagent_text must pass the same invariant as other emitters."""
        ev = StreamEventEmitter.subagent_text("s", "c", "task:s")
        assert "type" in ev.data
        assert ev.data["type"] == ev.type


# ═══════════════════════════════════════════════════════════════════
# 2. stream_agent_events — subagent_text emission
# ═══════════════════════════════════════════════════════════════════


class TestStreamAgentEventsSubagentText:
    """Verify sub-agent text chunks yield subagent_text events."""

    async def test_subagent_text_emitted_for_subagent_chunks(self):
        """When a sub-agent produces text, subagent_text events should appear."""
        namespace = ("sub", "research")
        agent = FakeV3Agent(
            [
                message_delta(
                    "Sub-agent finding: X is significant.", namespace=namespace
                )
            ],
            subagents=[FakeSubagent(namespace, "research-agent")],
        )
        events = await collect_events(agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        assert len(sa_text) == 1
        assert "Sub-agent finding" in sa_text[0]["content"]
        # instance_id must be present and non-empty
        assert sa_text[0].get("instance_id"), "instance_id must be a non-empty string"

    async def test_subagent_text_not_emitted_for_main_agent(self):
        """Main agent text should produce 'text' events, not 'subagent_text'."""
        agent = FakeV3Agent([message_delta("Main agent reply.")])
        events = await collect_events(agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        text_events = [e for e in events if e.get("type") == "text"]
        assert len(sa_text) == 0
        assert len(text_events) == 1

    async def test_multiple_subagent_text_chunks_all_emitted(self):
        """Multiple text chunks from a sub-agent all yield subagent_text events."""
        namespace = ("sub", "a")
        agent = FakeV3Agent(
            [
                message_delta("Part 1.", namespace=namespace),
                message_delta("Part 2.", namespace=namespace),
                message_delta("Part 3.", namespace=namespace),
            ],
            subagents=[FakeSubagent(namespace, "research-agent")],
        )
        events = await collect_events(agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        assert len(sa_text) == 3
        combined = "".join(e["content"] for e in sa_text)
        assert "Part 1." in combined
        assert "Part 2." in combined
        assert "Part 3." in combined
        # All chunks from the same namespace must share the same instance_id
        ids = {e["instance_id"] for e in sa_text}
        assert len(ids) == 1, f"Expected 1 unique instance_id, got {ids}"
        assert all(e.get("instance_id") for e in sa_text)

    async def test_parallel_same_name_agents_get_distinct_instance_ids(self):
        """Two sub-agents with the same display name but different namespaces
        produce subagent_text events with different instance_id values.

        This is the core of the interleaving fix: the consumer can tell
        the two instances apart even though their 'subagent' field is
        identical.
        """
        # Two different v3 namespaces, same projected subagent display name.
        ns1 = ("ns", "task", "id1", "agent")
        ns2 = ("ns", "task", "id2", "agent")
        agent = FakeV3Agent(
            [
                message_delta("Instance-1 text.", namespace=ns1),
                message_delta("Instance-2 text.", namespace=ns2),
                message_delta(" More from 1.", namespace=ns1),
            ],
            subagents=[
                FakeSubagent(ns1, "research-agent"),
                FakeSubagent(ns2, "research-agent"),
            ],
        )
        events = await collect_events(agent)
        sa_text = [e for e in events if e.get("type") == "subagent_text"]
        assert len(sa_text) == 3

        # All events have the same subagent display name
        assert all(e["subagent"] == "research-agent" for e in sa_text)

        # But instance_ids differ between the two namespaces
        id1_events = [
            e
            for e in sa_text
            if "Instance-1" in e["content"] or "More from 1" in e["content"]
        ]
        id2_events = [e for e in sa_text if "Instance-2" in e["content"]]
        assert len(id1_events) == 2
        assert len(id2_events) == 1

        # Same namespace → same instance_id
        assert id1_events[0]["instance_id"] == id1_events[1]["instance_id"]
        # Different namespace → different instance_id
        assert id1_events[0]["instance_id"] != id2_events[0]["instance_id"]


# ═══════════════════════════════════════════════════════════════════
# 3. Prompt guidance
# ═══════════════════════════════════════════════════════════════════


class TestDelegationPromptSummarize:
    def test_framework_task_tool_contains_summarize_guidance(self):
        """The upstream TASK_TOOL_DESCRIPTION already instructs the LLM to summarize."""
        from deepagents.middleware.subagents import TASK_TOOL_DESCRIPTION

        assert "not shown to the user" in TASK_TOOL_DESCRIPTION
        assert "relay a summary yourself" in TASK_TOOL_DESCRIPTION
