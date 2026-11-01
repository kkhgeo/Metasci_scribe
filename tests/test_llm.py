"""Tests for Metasci Scribe LLM module."""

from unittest.mock import MagicMock, patch

import pytest

# Side-effect import: applies module-level monkey-patches (e.g.,
# _patch_openai_capture_reasoning_content) before tests reference patched
# functions from langchain_openai.
import metasci_scribe.llm.patches  # noqa: F401
from metasci_scribe.llm import (
    DEFAULT_MODEL,
    MODELS,
    get_chat_model,
    get_model_info,
    get_models_for_provider,
    list_models,
)
from metasci_scribe.llm.models import _MODEL_ENTRIES

# =============================================================================
# Test MODELS registry
# =============================================================================


class TestModelsRegistry:
    def test_models_is_dict(self):
        """Test that MODELS is a dictionary."""
        assert isinstance(MODELS, dict)

    @pytest.mark.parametrize(
        ("custom", "native"),
        [("custom-openai", "openai"), ("custom-anthropic", "anthropic")],
    )
    def test_custom_endpoint_defaults_are_names_the_native_provider_lists(
        self, custom, native
    ):
        """Otherwise a bare name would move a native user onto a custom endpoint."""
        native_names = {name for name, _ in get_models_for_provider(native)}

        assert {name for name, _ in get_models_for_provider(custom)} <= native_names

    def test_entries_has_all_providers(self):
        """The registry lists exactly the two supported providers."""
        providers = {p for _, _, p in _MODEL_ENTRIES}
        assert providers == {"anthropic", "openai"}

    def test_entries_are_valid_tuples(self):
        """Test that _MODEL_ENTRIES contains valid (name, model_id, provider) tuples."""
        valid_providers = {"anthropic", "openai"}
        for entry in _MODEL_ENTRIES:
            assert len(entry) == 3, f"Entry {entry} doesn't have 3 elements"
            name, model_id, provider = entry
            assert isinstance(name, str)
            assert isinstance(model_id, str)
            assert provider in valid_providers, (
                f"Unknown provider '{provider}' for '{name}'"
            )

    def test_get_models_for_provider(self):
        """Test that get_models_for_provider returns correct models."""
        for provider in ("anthropic", "openai"):
            models = get_models_for_provider(provider)
            assert len(models) > 0
            for name, model_id in models:
                assert isinstance(name, str)
                assert isinstance(model_id, str)

        assert get_models_for_provider("openrouter") == []


# =============================================================================
# Test DEFAULT_MODEL
# =============================================================================


class TestDefaultModel:
    def test_default_model_exists_in_registry(self):
        """Test that DEFAULT_MODEL is a valid model in MODELS."""
        assert DEFAULT_MODEL in MODELS

    def test_default_model_is_openai(self):
        """Test that default model uses OpenAI."""
        _, provider = MODELS[DEFAULT_MODEL]
        assert provider == "openai"


# =============================================================================
# Test list_models
# =============================================================================


class TestListModels:
    def test_returns_list(self):
        """Test that list_models returns a list."""
        result = list_models()
        assert isinstance(result, list)

    def test_returns_all_model_names(self):
        """Test that list_models returns all model names."""
        result = list_models()
        assert set(result) == set(MODELS.keys())

    def test_list_is_not_empty(self):
        """Test that the list is not empty."""
        assert len(list_models()) > 0


# =============================================================================
# Test get_model_info
# =============================================================================


class TestGetModelInfo:
    def test_returns_tuple_for_valid_model(self):
        """Test that get_model_info returns tuple for valid model."""
        result = get_model_info("claude-sonnet-5-5")
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_none_for_invalid_model(self):
        """Test that get_model_info returns None for invalid model."""
        result = get_model_info("nonexistent-model")
        assert result is None

    def test_returns_correct_info(self):
        """Test that get_model_info returns correct info."""
        model_id, provider = get_model_info("claude-haiku-5-5")
        assert model_id == "claude-haiku-5-5"
        assert provider == "anthropic"


# =============================================================================
# Test get_chat_model
# =============================================================================


class TestGetChatModel:
    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_uses_default_model_when_none(self, mock_init):
        """Test that get_chat_model uses default model when model=None."""
        mock_init.return_value = "mock_model"

        get_chat_model()

        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        # Default model should be resolved from MODELS
        expected_model_id, expected_provider = MODELS[DEFAULT_MODEL]
        assert call_kwargs["model"] == expected_model_id
        assert call_kwargs["model_provider"] == expected_provider

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_resolves_short_name(self, mock_init):
        """Test that get_chat_model resolves short names correctly."""
        mock_init.return_value = "mock_model"

        get_chat_model("claude-opus-5-5")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "claude-opus-5-5"
        assert call_kwargs["model_provider"] == "anthropic"

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_resolves_openai_short_name(self, mock_init):
        """Test that get_chat_model resolves OpenAI short names."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-6-luna", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "gpt-6-luna"
        assert call_kwargs["model_provider"] == "openai"

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_uses_full_model_id(self, mock_init):
        """Test that get_chat_model accepts full model IDs."""
        mock_init.return_value = "mock_model"

        get_chat_model("claude-3-opus-20240229")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "claude-3-opus-20240229"
        # Should infer anthropic from the model prefix
        assert call_kwargs["model_provider"] == "anthropic"

    def test_unsupported_provider_is_refused(self):
        """Only openai and anthropic can build a chat model."""
        for provider in ("custom_provider", "google-genai", "openrouter"):
            with pytest.raises(ValueError, match="Unsupported provider"):
                get_chat_model("claude-sonnet-4-6", provider=provider)

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_passes_kwargs(self, mock_init):
        """Test that additional kwargs are passed through."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-5-nano", temperature=0.7, max_tokens=1000)

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_infers_openai_from_gpt_prefix(self, mock_init):
        """Test that OpenAI is inferred from gpt- prefix."""
        mock_init.return_value = "mock_model"

        get_chat_model("gpt-4-turbo-preview")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_infers_openai_from_o1_prefix(self, mock_init):
        """Test that OpenAI is inferred from o1 prefix."""
        mock_init.return_value = "mock_model"

        get_chat_model("o1-preview")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_defaults_to_openai_for_unknown(self, mock_init):
        """Test that openai is the default for unknown model prefixes."""
        mock_init.return_value = "mock_model"

        get_chat_model("some-unknown-model")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"


# =============================================================================
# Test slash model ID no longer routes to nvidia
# =============================================================================


class TestSlashModelIdFallback:
    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_slash_model_id_defaults_to_openai(self, mock_init):
        """Unregistered model IDs containing '/' fall through to the default
        'openai' provider, like all other unknown model IDs."""
        mock_init.return_value = "mock_model"

        get_chat_model("some-org/some-model")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model"] == "some-org/some-model"
        assert call_kwargs["model_provider"] == "openai"


# =============================================================================
# Test _flatten_message_content
# =============================================================================


class TestFlattenMessageContent:
    """Tests for the content-flattening utility used by OpenAI-compatible providers."""

    def test_string_passthrough(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        assert _flatten_message_content("hello") == "hello"

    def test_non_list_passthrough(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        assert _flatten_message_content(42) == 42
        assert _flatten_message_content(None) is None

    def test_text_blocks(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        content = [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": "World"},
        ]
        assert _flatten_message_content(content) == "Hello\n\nWorld"

    def test_skips_thinking_blocks(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        content = [
            {"type": "thinking", "text": "Let me think..."},
            {"type": "text", "text": "The answer is 42"},
            {"type": "reasoning", "text": "internal reasoning"},
            {"type": "reasoning_content", "text": "more reasoning"},
        ]
        assert _flatten_message_content(content) == "The answer is 42"

    def test_string_blocks(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        content = ["hello", "world"]
        assert _flatten_message_content(content) == "hello\n\nworld"

    def test_mixed_blocks(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        content = [
            {"type": "thinking", "text": "skip me"},
            "plain string",
            {"type": "text", "text": "dict text"},
        ]
        assert _flatten_message_content(content) == "plain string\n\ndict text"

    def test_empty_list(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        assert _flatten_message_content([]) == ""

    def test_only_thinking_blocks(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        content = [{"type": "thinking", "text": "thought"}]
        assert _flatten_message_content(content) == ""

    def test_preserves_image_block(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        assert _flatten_message_content([img]) == [img]

    def test_preserves_image_url_block(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
        assert _flatten_message_content([img]) == [img]

    def test_preserves_file_block(self):
        # PDF/document files are preserved (capable models read them).
        from metasci_scribe.llm.patches import _flatten_message_content

        f = {"type": "file", "base64": "FFF", "mime_type": "application/pdf"}
        assert _flatten_message_content([f]) == [f]

    def test_unsupported_media_dropped(self):
        # video/audio are NOT in the allowlist -> dropped, not crashing
        # (langchain-openai raises ValueError on `video`).
        from metasci_scribe.llm.patches import _flatten_message_content

        for block in (
            {"type": "video", "base64": "VVV", "mime_type": "video/mp4"},
            {"type": "audio", "base64": "ZZZ", "mime_type": "audio/wav"},
        ):
            assert _flatten_message_content([block]) == ""

    def test_non_image_media_dropped_keeps_text(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        content = [
            {"type": "text", "text": "hi"},
            {"type": "video", "base64": "VVV", "mime_type": "video/mp4"},
        ]
        # Video dropped, text kept -> plain string (no media list).
        assert _flatten_message_content(content) == "hi"

    def test_consolidates_text_and_image(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        content = [{"type": "text", "text": "a photo"}, img]
        assert _flatten_message_content(content) == [
            {"type": "text", "text": "a photo"},
            img,
        ]

    def test_multiple_text_blocks_with_image(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        content = [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
            img,
        ]
        assert _flatten_message_content(content) == [
            {"type": "text", "text": "a\n\nb"},
            img,
        ]

    def test_preserves_text_media_ordering(self):
        # Text after an image must stay AFTER it (not consolidated to the front).
        from metasci_scribe.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        content = [
            {"type": "text", "text": "before"},
            img,
            {"type": "text", "text": "after"},
        ]
        assert _flatten_message_content(content) == [
            {"type": "text", "text": "before"},
            img,
            {"type": "text", "text": "after"},
        ]

    def test_thinking_dropped_image_kept(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        content = [{"type": "thinking", "text": "hmm"}, img]
        assert _flatten_message_content(content) == [img]

    def test_pure_text_still_returns_string(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        content = [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]
        result = _flatten_message_content(content)
        assert result == "x\n\ny"
        assert isinstance(result, str)

    def test_unknown_nontext_block_still_dropped(self):
        from metasci_scribe.llm.patches import _flatten_message_content

        content = [{"type": "tool_use", "id": "1", "name": "foo"}]
        assert _flatten_message_content(content) == ""


# =============================================================================
# Test _patch_openai_compat_content (all 4 paths)
# =============================================================================


class TestPatchOpenAICompatContent:
    """Verify content flattening covers _generate, _agenerate, _stream, _astream."""

    def _make_model(self):
        """Create a minimal mock model with all 4 methods."""
        from unittest.mock import AsyncMock

        model = MagicMock()
        model._generate = MagicMock(return_value="gen_result")
        model._agenerate = AsyncMock(return_value="agen_result")
        model._stream = MagicMock(return_value=iter(["chunk1"]))
        model._astream = AsyncMock()
        return model

    def test_generate_flattened(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        model._generate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == "hello"

    async def test_agenerate_flattened(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._agenerate
        _patch_openai_compat_content(model)

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        await model._agenerate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == "hello"

    def test_stream_flattened(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._stream
        _patch_openai_compat_content(model)

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        list(model._stream([msg]))

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == "hello"

    async def test_astream_flattened(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        received_msgs = []

        async def _fake_astream(messages, *args, **kwargs):
            received_msgs.extend(messages)
            for chunk in ["c1", "c2"]:
                yield chunk

        model._astream = _fake_astream
        _patch_openai_compat_content(model)

        msg = HumanMessage(content=[{"type": "text", "text": "hello"}])
        chunks = []
        async for c in model._astream([msg]):
            chunks.append(c)

        assert chunks == ["c1", "c2"]
        assert received_msgs[0].content == "hello"

    def test_generate_preserves_media(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img])
        model._generate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == [{"type": "text", "text": "see"}, img]

    async def test_agenerate_preserves_media(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._agenerate
        _patch_openai_compat_content(model)

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img])
        await model._agenerate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == [{"type": "text", "text": "see"}, img]

    def test_stream_preserves_media(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._stream
        _patch_openai_compat_content(model)

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img])
        list(model._stream([msg]))

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == [{"type": "text", "text": "see"}, img]

    async def test_astream_preserves_media(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        received_msgs = []

        async def _fake_astream(messages, *args, **kwargs):
            received_msgs.extend(messages)
            for chunk in ["c1", "c2"]:
                yield chunk

        model._astream = _fake_astream
        _patch_openai_compat_content(model)

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img])
        chunks = []
        async for c in model._astream([msg]):
            chunks.append(c)

        assert chunks == ["c1", "c2"]
        assert received_msgs[0].content == [{"type": "text", "text": "see"}, img]

    def test_toolmessage_image_hoisted_to_human(self):
        from langchain_core.messages import ToolMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)  # hoist_tool_media=True (OpenAI-compat)

        # deepagents read_file emits this exact shape for an image file.
        tm = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        model._generate([tm])

        called_msgs = orig.call_args[0][0]
        # Tool content becomes a string placeholder (OpenAI-compat requirement) ...
        assert isinstance(called_msgs[0].content, str)
        # ... and the image is hoisted into a following HumanMessage.
        assert len(called_msgs) == 2
        hoisted = called_msgs[1]
        assert hoisted.type == "human"
        assert any(
            isinstance(b, dict) and b.get("type") == "image" for b in hoisted.content
        )

    def test_toolmessage_image_kept_inline_when_no_hoist(self):
        from langchain_core.messages import ToolMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model, hoist_tool_media=False)  # Anthropic-routed

        tm = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        model._generate([tm])

        called_msgs = orig.call_args[0][0]
        # No hoisting: image stays inline in the tool message content.
        assert len(called_msgs) == 1
        content = called_msgs[0].content
        assert isinstance(content, list)
        assert any(isinstance(b, dict) and b.get("type") == "image" for b in content)

    def test_parallel_tool_images_hoisted_after_tools(self):
        from langchain_core.messages import AIMessage, ToolMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        ai = AIMessage(
            content="",
            tool_calls=[
                {"id": "c1", "name": "read_file", "args": {}},
                {"id": "c2", "name": "read_file", "args": {}},
            ],
        )
        t1 = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="c1",
            name="read_file",
        )
        t2 = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "BBB", "mime_type": "image/png"}
            ],
            tool_call_id="c2",
            name="read_file",
        )
        model._generate([ai, t1, t2])

        called_msgs = orig.call_args[0][0]
        # Tool results stay consecutive; one hoisted HumanMessage follows them.
        assert [m.type for m in called_msgs] == ["ai", "tool", "tool", "human"]
        assert isinstance(called_msgs[1].content, str)
        assert isinstance(called_msgs[2].content, str)
        imgs = [b for b in called_msgs[3].content if b.get("type") == "image"]
        assert len(imgs) == 2

    def test_assistant_text_still_flattened_to_string(self):
        from langchain_core.messages import AIMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        msg = AIMessage(
            content=[
                {"type": "text", "text": "hi"},
                {"type": "thinking", "text": "t"},
            ]
        )
        model._generate([msg])

        called_msgs = orig.call_args[0][0]
        assert called_msgs[0].content == "hi"

    def test_tool_media_flushed_before_next_human(self):
        from langchain_core.messages import HumanMessage, ToolMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        tm = ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        nxt = HumanMessage(content="thanks")
        model._generate([tm, nxt])

        called = orig.call_args[0][0]
        # tool(placeholder), hoisted image (human), then the original human msg
        assert [m.type for m in called] == ["tool", "human", "human"]
        assert isinstance(called[0].content, str)
        assert any(b.get("type") == "image" for b in called[1].content)
        assert called[2].content == "thanks"

    def test_tool_message_text_and_image_split(self):
        from langchain_core.messages import ToolMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        tm = ToolMessage(
            content=[
                {"type": "text", "text": "chart description"},
                {"type": "image", "base64": "AAA", "mime_type": "image/png"},
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        model._generate([tm])

        called = orig.call_args[0][0]
        # Tool keeps the text as its string content; image hoisted to a human msg.
        assert called[0].content == "chart description"
        assert any(b.get("type") == "image" for b in called[1].content)

    def test_tool_message_interleaved_text_not_lost(self):
        # Interleaved [text, image, text] in a tool result: BOTH text runs must
        # survive the hoisting split (not just the first).
        from langchain_core.messages import ToolMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        orig = model._generate
        _patch_openai_compat_content(model)

        tm = ToolMessage(
            content=[
                {"type": "text", "text": "before"},
                {"type": "image", "base64": "AAA", "mime_type": "image/png"},
                {"type": "text", "text": "after"},
            ],
            tool_call_id="tc1",
            name="read_file",
        )
        model._generate([tm])

        called = orig.call_args[0][0]
        # both text runs preserved in the tool placeholder; image hoisted
        assert "before" in called[0].content
        assert "after" in called[0].content
        assert any(b.get("type") == "image" for b in called[1].content)


# =============================================================================
# Test no-vision fallback (models that reject image input)
# =============================================================================


class TestNoVisionFallback:
    """Verify image-rejecting models fall back to a text placeholder."""

    def _img_tool(self):
        from langchain_core.messages import ToolMessage

        return ToolMessage(
            content_blocks=[
                {"type": "image", "base64": "AAA", "mime_type": "image/png"}
            ],
            tool_call_id="t1",
            name="read_file",
        )

    def _make_model(self):

        model = MagicMock()
        model._agenerate = None
        model._stream = None
        model._astream = None
        return model

    def test_media_error_types(self):
        from metasci_scribe.llm.patches import (
            _FILE_CONTENT_TYPES,
            _IMAGE_CONTENT_TYPES,
            _is_http_400,
            _media_error_types,
        )

        # marker identifies the specific modality
        assert (
            _media_error_types(Exception("No endpoints found that support image input"))
            >= _IMAGE_CONTENT_TYPES
        )
        assert (
            _media_error_types(Exception("file input is not supported"))
            == _FILE_CONTENT_TYPES
        )
        # DeepSeek-style maps to all media (generic "expected text")
        assert (
            _media_error_types(
                Exception("unknown variant `image_url`, expected `text`")
            )
            >= _IMAGE_CONTENT_TYPES
        )
        # non-media errors implicate nothing
        assert _media_error_types(Exception("rate limit exceeded")) == set()
        assert (
            _media_error_types(Exception("No endpoints found for some/model")) == set()
        )
        # bare "expected text" (non-media schema error) must NOT match
        assert (
            _media_error_types(
                Exception("tool schema validation failed: expected text")
            )
            == set()
        )

        class _E(Exception):
            status_code = 400

        assert _is_http_400(_E("bad request"))
        assert not _is_http_400(Exception("rate limit exceeded"))

    def test_media_types_in(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _media_types_in

        img = {"type": "image", "base64": "A", "mime_type": "image/png"}
        f = {"type": "file", "base64": "F", "mime_type": "application/pdf"}
        assert _media_types_in([HumanMessage(content=[img, f])]) == {"image", "file"}
        assert _media_types_in([HumanMessage(content="hi")]) == set()

    def test_strip_media_types_replaces_only_given(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _strip_media_types

        img = {"type": "image", "base64": "AAA", "mime_type": "image/png"}
        f = {"type": "file", "base64": "FFF", "mime_type": "application/pdf"}
        msg = HumanMessage(content=[{"type": "text", "text": "see"}, img, f])
        # Strip only files -> image survives, file becomes a placeholder block.
        out = _strip_media_types([msg], {"file"})
        types = [b.get("type") for b in out[0].content if isinstance(b, dict)]
        assert "image" in types  # image preserved
        assert "file" not in types  # file stripped
        assert any(
            b.get("type") == "text" and "omitted" in b.get("text", "").lower()
            for b in out[0].content
        )

    def test_strip_media_types_preserves_position(self):
        # Stripped block is replaced IN PLACE; surrounding text/kept media keep
        # their order (placeholder where the image was, file stays last).
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _strip_media_types

        img = {"type": "image", "base64": "A", "mime_type": "image/png"}
        f = {"type": "file", "base64": "F", "mime_type": "application/pdf"}
        msg = HumanMessage(
            content=[
                {"type": "text", "text": "t1"},
                img,
                {"type": "text", "text": "t2"},
                f,
            ]
        )
        out = _strip_media_types([msg], {"image"})  # block only image
        content = out[0].content
        assert all(b.get("type") != "image" for b in content)  # image gone
        # order preserved: t1, placeholder (where image was), t2, file
        assert content[0]["text"] == "t1"
        assert content[1]["type"] == "text"
        assert "omitted" in content[1]["text"].lower()
        assert content[2]["text"] == "t2"
        assert content[3]["type"] == "file"  # file kept at its original position

    def test_strip_media_types_dedups_consecutive(self):
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _strip_media_types

        a = {"type": "image", "base64": "A", "mime_type": "image/png"}
        b = {"type": "image", "base64": "B", "mime_type": "image/png"}
        msg = HumanMessage(content=[a, b])
        out = _strip_media_types([msg], {"image"})
        # two adjacent stripped blocks collapse into ONE placeholder
        assert len(out[0].content) == 1
        assert "omitted" in out[0].content[0]["text"].lower()

    def test_profile_no_vision_strips_upfront(self):
        # Proactive: profile says image_inputs is False -> strip from the start,
        # no failing first request.
        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model.profile = {"image_inputs": False}
        calls = []

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        assert model._generate([self._img_tool()]) == "ok"
        assert len(calls) == 1  # no failed attempt
        assert all(isinstance(m.content, str) for m in calls[0])
        assert any("omitted" in m.content.lower() for m in calls[0])

    def test_profile_with_vision_does_not_strip(self):
        # Profile says image_inputs is True -> normal preserve path (no upfront strip).
        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model.profile = {"image_inputs": True}
        calls = []

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        assert model._generate([self._img_tool()]) == "ok"
        # Image preserved (hoisted), not replaced by a placeholder.
        assert any(
            isinstance(m.content, list)
            and any(b.get("type") == "image" for b in m.content)
            for m in calls[0]
        )

    def test_generate_falls_back_and_caches(self):
        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        calls = []
        state = {"raised": False}

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            if not state["raised"]:  # fail exactly once, ever
                state["raised"] = True
                raise Exception("unknown variant `image_url`, expected `text`")
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        tm = self._img_tool()
        # 1st turn: preserve attempt fails once -> strip -> ok
        assert model._generate([tm]) == "ok"
        assert len(calls) == 2
        retry = calls[1]
        assert all(isinstance(m.content, str) for m in retry)
        assert any("omitted" in m.content.lower() for m in retry)

        # 2nd turn: cached no-vision -> straight to stripped, single call (no failure)
        calls.clear()
        assert model._generate([tm]) == "ok"
        assert len(calls) == 1
        assert all(isinstance(m.content, str) for m in calls[0])

    def test_non_image_error_not_retried(self):
        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        calls = []

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            raise Exception("rate limit exceeded")

        model._generate = _gen
        _patch_openai_compat_content(model)

        with pytest.raises(Exception, match="rate limit"):
            model._generate([self._img_tool()])
        assert len(calls) == 1

    def test_stream_falls_back(self):

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model._generate = MagicMock(return_value="g")
        calls = []

        def _stream(msgs, *a, **k):
            calls.append(msgs)
            if len(calls) == 1:
                raise Exception("No endpoints found that support image input")
            yield from ["x", "y"]

        model._stream = _stream
        _patch_openai_compat_content(model)

        out = list(model._stream([self._img_tool()]))
        assert out == ["x", "y"]
        assert len(calls) == 2

    async def test_astream_falls_back(self):

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model._generate = MagicMock(return_value="g")
        calls = []

        async def _astream(msgs, *a, **k):
            calls.append(msgs)
            if len(calls) == 1:
                raise Exception("No endpoints found that support image input")
            for c in ["x", "y"]:
                yield c

        model._astream = _astream
        _patch_openai_compat_content(model)

        out = [c async for c in model._astream([self._img_tool()])]
        assert out == ["x", "y"]
        assert len(calls) == 2

    def test_unrelated_400_retry_fails_not_cached(self):
        # A non-media 400 (e.g. tool schema) whose stripped retry ALSO fails must
        # surface the original error and must NOT permanently flip to no-media.
        from metasci_scribe.llm.patches import _patch_openai_compat_content

        class _E(Exception):
            status_code = 400

        model = self._make_model()
        calls = []

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            raise _E("invalid tool schema")  # 400, not media; fails every time

        model._generate = _gen
        _patch_openai_compat_content(model)

        tm = self._img_tool()
        with pytest.raises(_E):
            model._generate([tm])
        assert len(calls) == 2  # preserve attempt + stripped retry (both fail)

        # Not cached: the next call attempts preserve again (not straight-to-stripped)
        calls.clear()
        with pytest.raises(_E):
            model._generate([tm])
        assert len(calls) == 2

    def test_pdf_rejection_does_not_disable_images(self):
        # Per-modality: a PDF/file rejection caches only file types; a later
        # image must still be preserved (not stripped).
        from langchain_core.messages import HumanMessage, ToolMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        calls = []
        state = {"raised": False}

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            has_file = any(
                isinstance(m.content, list)
                and any(
                    isinstance(b, dict) and b.get("type") == "file" for b in m.content
                )
                for m in msgs
            )
            if has_file and not state["raised"]:
                state["raised"] = True
                raise Exception("file input is not supported")
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        pdf_tm = ToolMessage(
            content_blocks=[
                {"type": "file", "base64": "F", "mime_type": "application/pdf"}
            ],
            tool_call_id="t1",
            name="read_file",
        )
        assert model._generate([pdf_tm]) == "ok"  # file rejected -> stripped -> ok

        # Now an image: must still be preserved (images not blocked by a PDF reject)
        calls.clear()
        img_msg = HumanMessage(
            content=[{"type": "image", "base64": "A", "mime_type": "image/png"}]
        )
        assert model._generate([img_msg]) == "ok"
        assert len(calls) == 1  # single attempt, no failure
        assert any(
            isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") == "image" for b in m.content)
            for m in calls[0]
        )

    def test_bare_400_recovers_but_not_cached(self):
        # A bare 400 with NO media marker recovers this request (stripped retry)
        # but must NOT cache (no permanent degradation) — High #1.
        from metasci_scribe.llm.patches import _patch_openai_compat_content

        class _E(Exception):
            status_code = 400

        model = self._make_model()
        calls = []
        state = {"raised": False}

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            if not state["raised"]:
                state["raised"] = True
                raise _E("transient bad request")  # 400, no media marker
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        tm = self._img_tool()
        assert model._generate([tm]) == "ok"  # bare 400 -> stripped retry -> ok
        assert len(calls) == 2

        # NOT cached: the next call still attempts preserve (image kept, not stripped)
        calls.clear()
        assert model._generate([tm]) == "ok"
        assert len(calls) == 1
        assert any(
            isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") == "image" for b in m.content)
            for m in calls[0]
        )

    def test_mixed_modality_caches_only_culprit(self):
        # image+file message; provider rejects only the file -> cache file only,
        # images stay preserved on later turns — High #2.
        from langchain_core.messages import HumanMessage

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        calls = []
        state = {"raised": False}

        def _gen(msgs, *a, **k):
            calls.append(msgs)
            if not state["raised"]:
                state["raised"] = True
                raise Exception("file input is not supported")
            return "ok"

        model._generate = _gen
        _patch_openai_compat_content(model)

        mixed = HumanMessage(
            content=[
                {"type": "image", "base64": "A", "mime_type": "image/png"},
                {"type": "file", "base64": "F", "mime_type": "application/pdf"},
            ]
        )
        assert model._generate([mixed]) == "ok"  # file rejected -> retry -> cache file

        # later image-only request: image must still be preserved
        calls.clear()
        img = HumanMessage(
            content=[{"type": "image", "base64": "A", "mime_type": "image/png"}]
        )
        assert model._generate([img]) == "ok"
        assert len(calls) == 1
        assert any(
            isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") == "image" for b in m.content)
            for m in calls[0]
        )

    def test_stream_empty_retry_raises_original(self):
        # If the stripped streaming retry yields ZERO chunks, surface the
        # original error instead of silently returning an empty stream.

        from metasci_scribe.llm.patches import _patch_openai_compat_content

        model = self._make_model()
        model._generate = MagicMock(return_value="g")
        calls = []

        def _stream(msgs, *a, **k):
            calls.append(msgs)
            if len(calls) == 1:
                raise Exception("No endpoints found that support image input")
            return  # retry yields nothing
            yield  # pragma: no cover  (makes this a generator)

        model._stream = _stream
        _patch_openai_compat_content(model)

        with pytest.raises(Exception, match="support image"):
            list(model._stream([self._img_tool()]))
        assert len(calls) == 2


# =============================================================================
# Test _patch_openai_capture_reasoning_content (module-level monkey-patch)
# =============================================================================


class TestPatchOpenAICaptureReasoningContent:
    """Verify reasoning_content is captured into AIMessage.additional_kwargs.

    This patch is applied at import time and globally affects langchain-openai's
    _convert_dict_to_message and _convert_delta_to_message_chunk.
    """

    def test_capture_from_non_streaming_response(self):
        """reasoning_content in OpenAI response dict → AIMessage.additional_kwargs."""
        from langchain_openai.chat_models.base import _convert_dict_to_message

        msg = _convert_dict_to_message(
            {
                "role": "assistant",
                "content": "hi",
                "reasoning_content": "let me think...",
            }
        )
        assert msg.additional_kwargs.get("reasoning_content") == "let me think..."

    def test_capture_absent_when_field_missing(self):
        """No reasoning_content in response → not added to additional_kwargs."""
        from langchain_openai.chat_models.base import _convert_dict_to_message

        msg = _convert_dict_to_message({"role": "assistant", "content": "hi"})
        assert "reasoning_content" not in msg.additional_kwargs

    def test_capture_from_streaming_chunk(self):
        """reasoning_content delta is captured onto the chunk's additional_kwargs."""
        from langchain_core.messages import AIMessageChunk
        from langchain_openai.chat_models.base import (
            _convert_delta_to_message_chunk,
        )

        chunk = _convert_delta_to_message_chunk(
            {"role": "assistant", "content": "", "reasoning_content": "thinking"},
            AIMessageChunk,
        )
        assert chunk.additional_kwargs.get("reasoning_content") == "thinking"

    def test_capture_reasoning_alias_from_non_streaming_response(self):
        """A `reasoning` string (vLLM-style) is captured under reasoning_content."""
        from langchain_openai.chat_models.base import _convert_dict_to_message

        msg = _convert_dict_to_message(
            {"role": "assistant", "content": "No.", "reasoning": "91 = 7 x 13"}
        )
        assert msg.additional_kwargs.get("reasoning_content") == "91 = 7 x 13"

    def test_capture_reasoning_alias_from_streaming_chunk(self):
        """A `reasoning` delta is captured onto the chunk's additional_kwargs."""
        from langchain_core.messages import AIMessageChunk
        from langchain_openai.chat_models.base import (
            _convert_delta_to_message_chunk,
        )

        chunk = _convert_delta_to_message_chunk(
            {"role": "assistant", "content": "", "reasoning": "91 = "},
            AIMessageChunk,
        )
        assert chunk.additional_kwargs.get("reasoning_content") == "91 = "

    def test_capture_does_not_affect_other_fields(self):
        """Existing tool_calls / function_call extraction unaffected."""
        from langchain_openai.chat_models.base import _convert_dict_to_message

        msg = _convert_dict_to_message(
            {
                "role": "assistant",
                "content": "calling tool",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
                "reasoning_content": "use the tool",
            }
        )
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.additional_kwargs.get("reasoning_content") == "use the tool"


# =============================================================================
# Test _patch_anthropic_strip_foreign_reasoning
# =============================================================================


def _anthropic_httpx():
    """Return the httpx flavour the installed anthropic SDK accepts as ``http_client``.

    anthropic >= 1.0 is built on ``httpx2`` and rejects an ``httpx.Client``.
    """
    import anthropic
    from packaging.version import Version

    if Version(anthropic.__version__) >= Version("1"):
        import httpx2 as httpx
    else:
        import httpx
    return httpx


class TestAnthropicStripForeignReasoning:
    def test_strip_removes_reasoning_content_blocks(self):
        """reasoning_content blocks are dropped; text and thinking survive."""
        from langchain_core.messages import AIMessage, HumanMessage

        from metasci_scribe.llm.patches import _normalize_anthropic_replay_messages

        messages = [
            HumanMessage("hello"),
            AIMessage(
                content=[
                    {"type": "reasoning_content", "reasoning_content": {"text": "hm"}},
                    {"type": "thinking", "thinking": "hm", "signature": ""},
                    {"type": "text", "text": "hi"},
                ]
            ),
        ]

        result = _normalize_anthropic_replay_messages(messages)

        types = [b["type"] for b in result[1].content]
        assert types == ["thinking", "text"]

    def test_missing_thinking_signature_defaulted(self):
        """Streamed thinking blocks without a signature key get signature ''."""
        from langchain_core.messages import AIMessage

        from metasci_scribe.llm.patches import _normalize_anthropic_replay_messages

        messages = [
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "hm", "index": 0},
                    {"type": "text", "text": "hi", "index": 1},
                ]
            ),
        ]

        result = _normalize_anthropic_replay_messages(messages)

        assert result[0].content[0]["signature"] == ""
        assert "signature" not in result[0].content[1]

    def test_strip_no_change_returns_same_object(self):
        """Clean histories pass through without copying."""
        from langchain_core.messages import AIMessage, HumanMessage

        from metasci_scribe.llm.patches import _normalize_anthropic_replay_messages

        messages = [
            HumanMessage("hello"),
            AIMessage(content=[{"type": "text", "text": "hi"}]),
            AIMessage(content="plain string content"),
        ]

        assert _normalize_anthropic_replay_messages(messages) is messages

    def test_reasoning_content_stripped_on_the_wire(self, monkeypatch):
        """End-to-end: foreign reasoning blocks never reach the Anthropic wire."""
        import json

        import anthropic
        from langchain_core.messages import AIMessage, HumanMessage

        httpx = _anthropic_httpx()

        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://compat.example.com")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        model = get_chat_model(
            "claude-sonnet-5-5", provider="anthropic", max_tokens=1024
        )

        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content.decode()))
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "claude-sonnet-5-5",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        model._client = anthropic.Anthropic(
            api_key="test-key",
            base_url="https://compat.example.com",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        history = [
            HumanMessage("hello"),
            AIMessage(
                content=[
                    {"type": "reasoning_content", "reasoning_content": {"text": "hm"}},
                    {"type": "thinking", "thinking": "hm", "index": 0},
                    {"type": "text", "text": "hi there"},
                ]
            ),
            HumanMessage("say ok"),
        ]
        result = model.invoke(history)

        sent_blocks = [
            block
            for message in captured["messages"]
            for block in (
                message["content"] if isinstance(message["content"], list) else []
            )
        ]
        sent_types = [block["type"] for block in sent_blocks]
        assert "reasoning_content" not in sent_types
        assert "text" in sent_types
        thinking_blocks = [b for b in sent_blocks if b["type"] == "thinking"]
        assert thinking_blocks
        assert thinking_blocks[0]["signature"] == ""
        assert result.content == "ok"


# =============================================================================
# Test _patch_deepagents_bound_reasoning_claude_only
# =============================================================================


class TestSummaryKeepsNonClaudeThinking:
    """Only Claude binds thinking to the history a summary replaces; Kimi backends
    reject a tool-call turn whose thinking is missing."""

    @pytest.fixture(autouse=True)
    def _agent_summarization(self, tmp_path):
        from deepagents.backends import FilesystemBackend
        from langchain_core.language_models import FakeListChatModel

        from metasci_scribe.middleware import create_per_run_summarization_middleware

        # Building the agent's summarization middleware installs the patch.
        create_per_run_summarization_middleware(
            FakeListChatModel(responses=["unused"]),
            FilesystemBackend(root_dir=tmp_path),
        )

    @staticmethod
    def _events(model: str) -> list[dict]:
        return [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "plan"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "ls",
                    "input": {},
                },
            },
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]

    @pytest.mark.parametrize("stream", [False, True], ids=["invoke", "stream"])
    @pytest.mark.parametrize(
        ("provider", "model", "key_env", "kept"),
        [
            ("anthropic", "claude-sonnet-5-5", "ANTHROPIC_API_KEY", ["tool_use"]),
        ],
    )
    def test_reply_kept_across_a_summary(
        self, monkeypatch, provider, model, key_env, kept, stream
    ):
        import json

        import anthropic
        from deepagents.middleware.summarization import _drop_bound_reasoning
        from langchain_core.messages import HumanMessage

        httpx = _anthropic_httpx()
        monkeypatch.setenv(key_env, "test-key")
        chat_model = get_chat_model(model, provider=provider)
        events = self._events(model)

        def respond(request):
            if json.loads(request.content).get("stream"):
                body = "".join(
                    f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
                )
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body.encode(),
                )
            message = {
                **events[0]["message"],
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": ""},
                    events[4]["content_block"],
                ],
                "stop_reason": "tool_use",
            }
            return httpx.Response(200, json=message)

        chat_model._client = anthropic.Anthropic(
            api_key=chat_model.anthropic_api_key.get_secret_value(),
            base_url=chat_model.anthropic_api_url,
            timeout=None,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )
        prompt = [HumanMessage("list files")]
        if stream:
            reply = None
            for chunk in chat_model.stream(prompt):
                reply = chunk if reply is None else reply + chunk
        else:
            reply = chat_model.invoke(prompt)

        assert reply.response_metadata["model_name"] == model
        assert [b["type"] for b in _drop_bound_reasoning(reply).content] == kept

    def test_unnamed_reply_keeps_upstreams_drop(self):
        from deepagents.middleware.summarization import _drop_bound_reasoning
        from langchain_core.messages import AIMessage

        reply = AIMessage(
            content=[
                {"type": "thinking", "thinking": "plan", "signature": "sig"},
                {"type": "text", "text": "done"},
            ],
            response_metadata={"model_provider": "anthropic"},
        )

        assert [b["type"] for b in _drop_bound_reasoning(reply).content] == ["text"]


# =============================================================================
# Test _apply_auto_config
# =============================================================================


class TestAutoConfig:
    @pytest.fixture(autouse=True)
    def _clear_reasoning_effort_env(self, monkeypatch):
        """Keep auto-config tests independent of the developer environment."""
        monkeypatch.delenv("METASCI_SCRIBE_REASONING_EFFORT", raising=False)

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_anthropic_4_5_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-5 models get enabled thinking with budget."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-haiku-4-5")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_anthropic_4_6_adaptive_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-6 models get adaptive thinking with max effort."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-sonnet-4-6")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call_kwargs["effort"] == "max"

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_anthropic_4_8_adaptive_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-8 models get adaptive thinking with max effort."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-opus-4-8")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call_kwargs["effort"] == "max"

    @pytest.mark.parametrize(
        ("model", "max_tokens"),
        [
            ("claude-opus-5", 128000),
            ("claude-opus-5-5", 128000),
            ("claude-sonnet-5", 128000),
            ("claude-sonnet-5-5", 128000),
            ("claude-opus-6", 128000),  # unregistered: new ids need no code change
        ],
    )
    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_anthropic_5_series_adaptive_thinking(
        self, mock_init, model, max_tokens, monkeypatch
    ):
        """Current Claude models get adaptive thinking (budget_tokens would 400)."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model(model, provider="anthropic")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call_kwargs["effort"] == "max"
        assert call_kwargs.get("max_tokens") == max_tokens

    @pytest.mark.parametrize(
        ("kwargs", "base_url"),
        [({"thinking": {"type": "adaptive"}}, None), ({}, "http://localhost:8000")],
    )
    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_opus_5_5_max_tokens_outside_thinking_branch(
        self, mock_init, kwargs, base_url, monkeypatch
    ):
        """Explicit thinking or a local proxy must still get 128000."""
        mock_init.return_value = "mock_model"
        if base_url:
            monkeypatch.setenv("ANTHROPIC_BASE_URL", base_url)
        else:
            monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        get_chat_model("claude-opus-5-5", provider="anthropic", **kwargs)

        assert mock_init.call_args[1]["max_tokens"] == 128000

    def test_claude_structured_output_filled_only_when_profile_lacks_it(
        self, monkeypatch
    ):
        """Profile-less Claude ids get native structured output; others untouched."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

        new = get_chat_model("claude-sonnet-5-5", provider="anthropic").profile
        known = get_chat_model("claude-haiku-4-5", provider="anthropic").profile

        assert new["structured_output"] is True
        assert known["max_output_tokens"] == 64000  # upstream profile kept whole

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_anthropic_4_6_proxy_no_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-6 models via proxy skip thinking (history round-trip 422)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "local-proxy-key")

        get_chat_model("claude-sonnet-4-6")

        call_kwargs = mock_init.call_args[1]
        assert "thinking" not in call_kwargs
        assert "effort" not in call_kwargs

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_anthropic_4_5_proxy_no_thinking(self, mock_init, monkeypatch):
        """Anthropic 4-5 models via proxy also skip thinking (history round-trip 422)."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8000")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "local-proxy-key")

        get_chat_model("claude-haiku-4-5")

        call_kwargs = mock_init.call_args[1]
        assert "thinking" not in call_kwargs

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_anthropic_4_6_no_proxy_no_downgrade(self, mock_init, monkeypatch):
        """Anthropic 4-6 models without proxy still get adaptive thinking."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-real")

        get_chat_model("claude-sonnet-4-6")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call_kwargs["effort"] == "max"

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_anthropic_thinking_not_overridden(self, mock_init):
        """User-supplied thinking config should not be overridden."""
        mock_init.return_value = "mock_model"
        custom_thinking = {"type": "enabled", "budget_tokens": 500}

        get_chat_model("claude-sonnet-4-6", thinking=custom_thinking)

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["thinking"] == custom_thinking

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_openai_reasoning_xhigh(self, mock_init, monkeypatch):
        """gpt-5.4+ and codex models get xhigh reasoning."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("METASCI_SCRIBE_REASONING_EFFORT", raising=False)

        get_chat_model("gpt-5.4", provider="openai")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "xhigh",
            "summary": "auto",
        }

        get_chat_model("gpt-5.3-codex", provider="openai")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "xhigh",
            "summary": "auto",
        }

        get_chat_model("gpt-5.5", provider="openai")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "xhigh",
            "summary": "auto",
        }

        get_chat_model("gpt-5.6-sol", provider="openai")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "xhigh",
            "summary": "auto",
        }

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_openai_reasoning_effort_from_env(self, mock_init, monkeypatch):
        """Native OpenAI reasoning effort should be configurable via env var."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("METASCI_SCRIBE_REASONING_EFFORT", "high")

        get_chat_model("gpt-5.5", provider="openai")

        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "high",
            "summary": "auto",
        }

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_openai_reasoning_high_fallback(self, mock_init, monkeypatch):
        """Other OpenAI models get high reasoning effort."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        get_chat_model("gpt-5-nano")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "high",
            "summary": "auto",
        }

        get_chat_model("gpt-5.2", provider="openai")
        assert mock_init.call_args[1]["reasoning"] == {
            "effort": "high",
            "summary": "auto",
        }

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_openai_base_url_override(self, mock_init, monkeypatch):
        """A user-set OPENAI_BASE_URL (vLLM, etc.) passes through with the usual defaults."""
        mock_init.return_value = "mock_model"
        monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-local-key")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert call_kwargs["base_url"] == "http://127.0.0.1:8080/v1"
        assert call_kwargs["api_key"] == "sk-local-key"
        # Reasoning is applied; Chat Completions is not forced
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}
        assert "use_responses_api" not in call_kwargs
        assert "streaming" not in call_kwargs

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_openai_no_base_url_when_unset(self, mock_init, monkeypatch):
        """OpenAI provider should not set base_url when env var is empty."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["model_provider"] == "openai"
        assert "base_url" not in call_kwargs

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_use_responses_api_false(self, mock_init, monkeypatch):
        """use_responses_api=false forces Chat Completions and drops reasoning."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("METASCI_SCRIBE_USE_RESPONSES_API", "false")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["use_responses_api"] is False
        assert "reasoning" not in call_kwargs

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_use_responses_api_true(self, mock_init, monkeypatch):
        """use_responses_api=true explicitly enables the Responses API."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("METASCI_SCRIBE_USE_RESPONSES_API", "true")

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["use_responses_api"] is True
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}

    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_use_responses_api_default_unchanged(self, mock_init, monkeypatch):
        """Empty use_responses_api preserves default behavior (no kwarg set)."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("METASCI_SCRIBE_USE_RESPONSES_API", raising=False)

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert "use_responses_api" not in call_kwargs
        assert call_kwargs["reasoning"] == {"effort": "high", "summary": "auto"}

    @pytest.mark.parametrize("env_value", ["FALSE", " false ", "False"])
    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_use_responses_api_false_normalization(
        self, mock_init, monkeypatch, env_value
    ):
        """Case/whitespace variants of 'false' are normalized correctly."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("METASCI_SCRIBE_USE_RESPONSES_API", env_value)

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["use_responses_api"] is False
        assert "reasoning" not in call_kwargs

    @pytest.mark.parametrize("env_value", ["TRUE", " true ", "True"])
    @patch("metasci_scribe.llm.models.init_chat_model")
    def test_use_responses_api_true_normalization(
        self, mock_init, monkeypatch, env_value
    ):
        """Case/whitespace variants of 'true' are normalized correctly."""
        mock_init.return_value = "mock_model"
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("METASCI_SCRIBE_USE_RESPONSES_API", env_value)

        get_chat_model("gpt-5-nano", provider="openai")

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["use_responses_api"] is True


class TestResolveProvider:
    """Short-name resolution keeps the caller's provider when it serves the name."""

    @pytest.fixture(autouse=True)
    def _registry(self, monkeypatch):
        from metasci_scribe.llm import registry

        entries = [
            ("m", "m-plan", "vendor-plan"),
            ("m", "m", "vendor"),
            ("solo", "solo", "vendor"),
        ]
        monkeypatch.setattr(registry, "_MODEL_ENTRIES", entries)
        monkeypatch.setattr(registry, "MODELS", {n: (mid, p) for n, mid, p in entries})

    def test_keeps_preferred_provider_that_serves_the_name(self):
        from metasci_scribe.llm.registry import resolve_provider

        assert resolve_provider("m", "vendor-plan") == "vendor-plan"

    def test_falls_back_to_last_entry_otherwise(self):
        from metasci_scribe.llm.registry import resolve_provider

        assert resolve_provider("m") == "vendor"
        assert resolve_provider("solo", "vendor-plan") == "vendor"

    def test_unknown_name(self):
        from metasci_scribe.llm.registry import resolve_provider

        assert resolve_provider("nope", "vendor") is None
