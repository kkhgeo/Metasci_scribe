"""Whitespace that a model streams as chunks of its own reaches the answer."""

from __future__ import annotations

from metasci_scribe.stream.emitter import StreamEventEmitter
from metasci_scribe.stream.events import (
    _SubagentInfo,
    _SubagentRegistry,
    _V3EventProcessor,
)


def _processor() -> _V3EventProcessor:
    return _V3EventProcessor(StreamEventEmitter(), _SubagentRegistry(), None)


def _texts(events: list[dict]) -> str:
    return "".join(e.get("content", "") for e in events if e.get("type") == "text")


# How an OpenAI model streamed a table: line breaks and spaces arrive alone.
CHUNKS = [
    "간략 구성표",
    "\n\n",
    "| 번호 | 역할 |",
    "\n",
    "|---|---|",
    "\n",
    "| R1",
    " ",
    "·",
    " ",
    "①",
]


def test_line_breaks_and_spaces_streamed_alone_are_kept():
    p = _processor()
    events = [e for chunk in CHUNKS for e in p._emit_text(chunk, None)]
    expected = "간략 구성표\n\n| 번호 | 역할 |\n|---|---|\n| R1 · ①"
    assert p.full_response == expected
    assert _texts(events) == expected


def test_whitespace_with_no_text_after_it_is_not_emitted():
    p = _processor()
    assert p._emit_text("Done.", None)
    assert p._emit_text("\n\n", None) == []
    assert p.full_response == "Done."


def test_a_subagents_whitespace_stays_with_that_subagent():
    p = _processor()
    helper = _SubagentInfo(path=("tools:1",), name="planner-agent")
    p._emit_text("\n", helper)
    p._emit_text("Root", None)
    events = p._emit_text("text", helper)
    assert p.full_response == "Root"
    assert events[0]["content"] == "\ntext"
