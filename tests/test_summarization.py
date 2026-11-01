"""Tests for the summarization event pipeline and its display."""

from langchain_core.messages import HumanMessage

from metasci_scribe.stream.emitter import StreamEventEmitter
from metasci_scribe.stream.state import StreamState
from metasci_scribe.stream.summarization import _extract_summarization_text

# ---------------------------------------------------------------------------
# Emitter
# ---------------------------------------------------------------------------


class TestSummarizationEmitter:
    """StreamEventEmitter.summarization()."""

    def test_event_type(self):
        ev = StreamEventEmitter.summarization("hello")
        assert ev.type == "summarization"

    def test_event_data(self):
        ev = StreamEventEmitter.summarization("ctx compressed")
        assert ev.data["type"] == "summarization"
        assert ev.data["content"] == "ctx compressed"

    def test_empty_content(self):
        ev = StreamEventEmitter.summarization("")
        assert ev.data["content"] == ""

    def test_start_event(self):
        ev = StreamEventEmitter.summarization_start()
        assert ev.type == "summarization_start"
        assert ev.data["type"] == "summarization_start"


# ---------------------------------------------------------------------------
# StreamState
# ---------------------------------------------------------------------------


class TestSummarizationState:
    """StreamState handling of summarization events."""

    def test_initial_state(self):
        state = StreamState()
        assert state.summarization_text == ""
        assert state.is_summarizing is False

    def test_handle_summarization_start(self):
        state = StreamState()
        etype = state.handle_event({"type": "summarization_start"})
        assert etype == "summarization_start"
        assert state.is_summarizing is True

    def test_handle_summarization(self):
        state = StreamState()
        etype = state.handle_event({"type": "summarization", "content": "summary"})
        assert etype == "summarization"
        assert state.summarization_text == "summary"
        assert state.is_summarizing is True

    def test_accumulates_chunks(self):
        """Summarization chunks are accumulated (streaming)."""
        state = StreamState()
        state.handle_event({"type": "summarization", "content": "first"})
        state.handle_event({"type": "summarization", "content": "second"})
        assert state.summarization_text == "firstsecond"

    def test_get_display_args_includes_field(self):
        state = StreamState()
        state.handle_event({"type": "summarization", "content": "ctx"})
        args = state.get_display_args()
        assert "summarization_text" in args
        assert args["summarization_text"] == "ctx"
        assert args["is_summarizing"] is True

    def test_does_not_affect_thinking(self):
        state = StreamState()
        state.handle_event({"type": "thinking", "content": "think"})
        state.handle_event({"type": "summarization", "content": "sum"})
        assert state.thinking_text == "think"
        assert state.summarization_text == "sum"

    def test_does_not_affect_response(self):
        state = StreamState()
        state.handle_event({"type": "text", "content": "hello"})
        state.handle_event({"type": "summarization", "content": "sum"})
        assert state.response_text == "hello"
        assert state.summarization_text == "sum"

    def test_text_ends_summarizing_state(self):
        state = StreamState()
        state.handle_event({"type": "summarization_start"})
        state.handle_event({"type": "summarization", "content": "sum"})
        state.handle_event({"type": "text", "content": "hello"})
        assert state.is_summarizing is False


# ---------------------------------------------------------------------------
# Rich CLI display
# ---------------------------------------------------------------------------


def _render_group(group) -> str:
    """Render a Rich Group to plain text for assertion checks."""
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    console = Console(file=buf, width=120, force_terminal=True)
    console.print(group)
    return buf.getvalue()


class TestSummarizationRichDisplay:
    """create_streaming_display() with summarization_text."""

    def test_no_panel_when_empty(self):
        from metasci_scribe.stream.display import create_streaming_display

        group = create_streaming_display(summarization_text="")
        rendered = _render_group(group)
        assert "Context Summarized" not in rendered

    def test_panel_rendered(self):
        from metasci_scribe.stream.display import create_streaming_display

        group = create_streaming_display(
            summarization_text="The conversation was about ML.",
            response_text="ok",
        )
        rendered = _render_group(group)
        assert "Context Summarized" in rendered

    def test_panel_rendered_while_summarizing(self):
        from metasci_scribe.stream.display import create_streaming_display

        group = create_streaming_display(
            summarization_text="The conversation was about ML.",
            is_summarizing=True,
            response_text="ok",
        )
        rendered = _render_group(group)
        assert "Context Summarizing..." in rendered

    def test_panel_rendered_start_placeholder(self):
        from metasci_scribe.stream.display import create_streaming_display

        group = create_streaming_display(
            summarization_text="",
            is_summarizing=True,
            response_text="ok",
        )
        rendered = _render_group(group)
        assert "Context Summarizing..." in rendered

    def test_long_text_truncated(self):
        from metasci_scribe.stream.display import create_streaming_display

        long_text = "x" * 500
        group = create_streaming_display(
            summarization_text=long_text,
            response_text="ok",
        )
        rendered = _render_group(group)
        assert "..." in rendered


# ---------------------------------------------------------------------------
# _extract_summarization_text helper
# ---------------------------------------------------------------------------


class TestExtractSummarizationText:
    """Content extraction from summarization chunks."""

    def test_string_content(self):
        assert (
            _extract_summarization_text(HumanMessage(content="hello world"))
            == "hello world"
        )

    def test_content_blocks(self):
        assert (
            _extract_summarization_text(
                HumanMessage(
                    content=[
                        {"type": "text", "text": "part1"},
                        {"type": "text", "text": "part2"},
                    ]
                )
            )
            == "part1part2"
        )

    def test_content_blocks_with_index(self):
        """Content blocks may include 'index' field — should still extract text."""

        assert (
            _extract_summarization_text(
                HumanMessage(content=[{"type": "text", "text": " vs", "index": 1}])
            )
            == " vs"
        )

    def test_empty_list(self):
        assert _extract_summarization_text(HumanMessage(content=[])) == ""

    def test_mixed_block_types(self):
        assert (
            _extract_summarization_text(
                HumanMessage(
                    content=[
                        {"type": "text", "text": "hello"},
                        {"type": "image", "url": "..."},
                    ]
                )
            )
            == "hello"
        )

    def test_string_blocks_in_list(self):
        assert (
            _extract_summarization_text(HumanMessage(content=["hello", "world"]))
            == "helloworld"
        )
