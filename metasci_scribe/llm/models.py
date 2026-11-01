"""LLM model configuration based on LangChain init_chat_model.

This module provides a unified interface for creating chat model instances
for the two supported providers, OpenAI and Anthropic, each through its own API
key (and an optional base URL for a compatible gateway), with convenient short
names for common models.
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain.chat_models import init_chat_model

from .context_window import apply_known_context_window
from .patches import _patch_anthropic_strip_foreign_reasoning
from .registry import (
    _MODEL_ENTRIES,
    DEFAULT_MODEL,
    MODELS,
    UnknownModelError,
    get_model_info,  # noqa: F401 — re-exported for existing import sites
    get_models_for_provider,  # noqa: F401 — re-exported for existing import sites
    list_model_picker_entries,  # noqa: F401 — re-exported for existing import sites
    list_models,  # noqa: F401 — re-exported for existing import sites
    list_models_by_provider,  # noqa: F401 — re-exported for existing import sites
)


def _reasoning_effort(default: str) -> str:
    """The configured reasoning effort, or ``default`` when none is set."""
    return os.environ.get("METASCI_SCRIBE_REASONING_EFFORT", "").strip() or default


# Claude models from before adaptive thinking (4.6). A closed set, so any newer id,
# including one langchain-anthropic has no profile for yet, counts as current.
_PRE_ADAPTIVE_CLAUDE = re.compile(
    r"claude-(3|(haiku|sonnet|opus)-4-[015](-|$)|(sonnet|opus)-4-\d{8})"
)


def _is_current_claude(model_id: str) -> bool:
    return model_id.startswith("claude-") and not _PRE_ADAPTIVE_CLAUDE.match(model_id)


def _fill_claude_structured_output(chat_model: Any, model_id: str) -> None:
    """Declare native structured output for a current Claude id that
    langchain-anthropic has no profile for yet; without it agents fall back to
    forced tool calling, which newer Claude models reject."""
    if not _is_current_claude(model_id):
        return
    profile = getattr(chat_model, "profile", None)
    if isinstance(profile, dict) and "structured_output" in profile:
        return
    base = profile if isinstance(profile, dict) else {}
    try:
        chat_model.profile = {**base, "structured_output": True}
    except Exception:
        pass


def _apply_auto_config(
    provider: str,
    model_id: str,
    kwargs: dict[str, Any],
) -> None:
    """Auto-enable provider-specific features (thinking, reasoning, etc.).

    Mutates *kwargs* in place.  Only sets keys that the caller hasn't already
    provided, so explicit user settings are never overridden.
    """
    # Current Claude models all output up to 128K (their profiles agree); without a
    # profile langchain-anthropic falls back to 4096.
    if provider == "anthropic" and _is_current_claude(model_id):
        kwargs.setdefault("max_tokens", 128000)

    # Anthropic: extended thinking
    if provider == "anthropic" and "thinking" not in kwargs:
        # Detect a local proxy: thinking blocks in conversation history cause
        # 422 errors because the proxy doesn't accept 'thinking' as a valid
        # content block type on round-trip.
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        is_proxy = "127.0.0.1" in base_url or "localhost" in base_url
        if is_proxy:
            pass
        elif _is_current_claude(model_id):
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
            kwargs.setdefault("effort", "max")
        else:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

    # OpenAI: reasoning
    if provider == "openai" and "reasoning" not in kwargs:
        default_effort = (
            "xhigh"
            if (
                "5.4" in model_id
                or "5.5" in model_id
                or "5.6" in model_id
                or "codex" in model_id
            )
            else "high"
        )
        kwargs["reasoning"] = {
            "effort": _reasoning_effort(default_effort),
            "summary": "auto",
        }


SUPPORTED_PROVIDERS = ("openai", "anthropic")


def get_chat_model(
    model: str | None = None,
    provider: str | None = None,
    **kwargs: Any,
) -> Any:
    """Get a chat model instance.

    Args:
        model: Model name (short name like 'gpt-6.1-sol' or a full model ID).
               Defaults to DEFAULT_MODEL.
        provider: Override the provider (``openai`` or ``anthropic``). If not
                  specified, inferred from the model name or defaults to
                  ``openai``.
        **kwargs: Additional arguments passed to init_chat_model (e.g., temperature).

    Returns:
        A LangChain chat model instance.

    Raises:
        UnknownModelError: The provider is not supported.
        ValueError: ``openai_auth_mode`` asks for a sign-in route that is not
            available yet.

    Examples:
        >>> model = get_chat_model()  # Uses default (gpt-6.1-sol)
        >>> model = get_chat_model("claude-opus-5-5")  # Use short name
        >>> model = get_chat_model("gpt-4o")  # OpenAI model
        >>> model = get_chat_model("claude-3-opus-20240229", provider="anthropic")  # Full ID
    """
    model = model or DEFAULT_MODEL

    # Look up short name in registry (provider-aware)
    model_id = None
    if provider:
        # Try exact match with provider first
        for name, mid, p in _MODEL_ENTRIES:
            if name == model and p == provider:
                model_id = mid
                break
    if model_id is None and model in MODELS:
        model_id, default_provider = MODELS[model]
        provider = provider or default_provider

    if model_id is None:
        # Assume it's a full model ID
        model_id = model
        # Try to infer provider from model ID prefix
        if provider is None:
            if model_id.startswith(("claude-", "anthropic")):
                provider = "anthropic"
            else:
                provider = "openai"

    if provider not in SUPPORTED_PROVIDERS:
        raise UnknownModelError(
            f"Unsupported provider {provider!r}. Metasci Scribe supports: "
            f"{', '.join(SUPPORTED_PROVIDERS)}."
        )

    if provider == "anthropic":
        # Optional base_url override (e.g. a local Anthropic-compatible proxy)
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key

    else:
        auth_mode = os.environ.get("METASCI_SCRIBE_OPENAI_AUTH_MODE", "api_key")
        if auth_mode == "chatgpt_plan":
            raise ValueError(
                'openai_auth_mode "chatgpt_plan" (ChatGPT plan sign-in) is not '
                "available yet. Use an API key: metasci-scribe config set "
                "openai_auth_mode api_key"
            )
        # Optional base_url override (e.g. a local OpenAI-compatible server)
        base_url = os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            kwargs["api_key"] = api_key

    _apply_auto_config(provider, model_id, kwargs)

    if provider == "openai":
        # User-level override for the OpenAI Responses API vs Chat Completions.
        # When "false", force Chat Completions and drop reasoning (which triggers
        # the Responses API path in langchain-openai).
        responses_api_setting = (
            os.environ.get("METASCI_SCRIBE_USE_RESPONSES_API", "").strip().lower()
        )
        if responses_api_setting == "false":
            kwargs["use_responses_api"] = False
            kwargs.pop("reasoning", None)
        elif responses_api_setting == "true":
            kwargs["use_responses_api"] = True

    chat_model = init_chat_model(model=model_id, model_provider=provider, **kwargs)

    if provider == "anthropic":
        _patch_anthropic_strip_foreign_reasoning()
        _fill_claude_structured_output(chat_model, model_id)

    apply_known_context_window(chat_model)

    return chat_model
