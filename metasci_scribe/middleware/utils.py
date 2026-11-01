"""Shared utilities for Metasci Scribe middleware.

Functions here are used by multiple middleware modules (memory, tool_selector)
and should not depend on any specific middleware class.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage


def disable_thinking(model: BaseChatModel) -> BaseChatModel:
    """Return a copy of the model with thinking/reasoning disabled.

    Anthropic's API does not allow extended thinking when ``tool_choice``
    forces tool use (as ``with_structured_output`` does).  Similarly,
    OpenAI reasoning can conflict.  Strip these settings so structured
    output calls work reliably.

    Uses ``model_copy()`` to produce a real new instance — ``bind()`` only
    wraps the model in a ``RunnableBinding`` whose kwargs do NOT override
    first-class Pydantic fields like ``thinking`` on ``ChatAnthropic``.
    """
    updates: dict[str, Any] = {}
    model_kwargs = getattr(model, "model_kwargs", {}) or {}

    if getattr(model, "thinking", None) or "thinking" in model_kwargs:
        updates["thinking"] = None
    if getattr(model, "reasoning", None) or "reasoning" in model_kwargs:
        updates["reasoning"] = None

    if not updates:
        return model

    # Prefer Pydantic model_copy (creates a true new instance with the
    # field cleared) over bind() which only adds invocation kwargs.
    try:
        return model.model_copy(update=updates)
    except Exception:
        # Fallback for non-Pydantic or unusual model classes
        # Note: bind() may not effectively override first-class Pydantic fields
        return model.bind(**updates)


def disable_streaming(model: BaseChatModel) -> BaseChatModel:
    """Return a copy of the model with ``disable_streaming=True``.

    ``BaseChatModel._streaming_disabled()`` (langchain_core
    ``chat_models.py:513``) reads only the instance's Pydantic
    ``disable_streaming`` field — the single gate before langchain routes
    to ``_stream`` / ``_astream`` when a streaming-aware callback handler
    is attached (which langgraph's ``astream_events(v3)`` always does).

    Alternatives that don't work:

    - ``model.bind(disable_streaming=True)``: puts kwargs on a
      ``RunnableBinding``, which ``_streaming_disabled()`` doesn't read.
      Silent no-op.
    - ``model.streaming = False``: only honored when ``streaming`` is
      explicitly in ``model_fields_set``. Provider defaults defeat it
      (``ChatOpenAI.streaming=False`` is already-False and not
      explicit-set).

    Uses ``model_copy`` to leave the caller's reference untouched.
    """
    return model.model_copy(update={"disable_streaming": True})


def append_to_system_message(
    system_message: SystemMessage | None, text: str
) -> SystemMessage:
    """Append a text block to a system message, preserving its metadata.

    Used by the memory middleware. Unlike building a fresh
    ``SystemMessage``, ``model_copy`` keeps ``additional_kwargs`` (e.g.
    ``cache_control`` prompt-cache breakpoints), ``id``, ``name`` and
    ``response_metadata`` from the original message.
    """
    existing_blocks = list(system_message.content_blocks) if system_message else []
    new_blocks = [*existing_blocks, {"type": "text", "text": text}]
    if system_message is None:
        return SystemMessage(content=new_blocks)
    return system_message.model_copy(update={"content": new_blocks})


def replace_block_by_sentinel(
    system_message: SystemMessage | None,
    sentinel: str,
    replacement: str,
) -> SystemMessage | None:
    """Swap the block containing ``sentinel`` for ``replacement`` text.

    Iterates ``system_message.content_blocks`` and returns a new
    ``SystemMessage`` whose block-list has the first block containing
    ``sentinel`` replaced by ``{"type": "text", "text": replacement}``.
    Other blocks and metadata (``additional_kwargs``, ``id``, ``name``,
    ``response_metadata``) are preserved via ``model_copy``.

    Returns ``None`` when no block carries the sentinel — the caller
    decides fallback policy (typically log-and-append rather than
    hard-fail, so a deepagents base-stack refactor degrades gracefully
    instead of killing the graph).
    """
    if system_message is None:
        return None
    blocks = list(system_message.content_blocks)
    for i, block in enumerate(blocks):
        if isinstance(block, dict) and sentinel in block.get("text", ""):
            new_blocks = [
                *blocks[:i],
                {"type": "text", "text": replacement},
                *blocks[i + 1 :],
            ]
            return system_message.model_copy(update={"content": new_blocks})
    return None


def request_without_tool(request: Any, name: str) -> Any:
    """*request* without the tool called *name*; *request* itself when absent."""
    tools = [tool for tool in request.tools if _tool_name(tool) != name]
    if len(tools) == len(request.tools):
        return request
    return request.override(tools=tools)


def _tool_name(candidate: Any) -> Any:
    if isinstance(candidate, Mapping):
        # A provider schema names its tool at the top or under "function".
        nested = candidate.get("function")
        inner = nested.get("name") if isinstance(nested, Mapping) else None
        return candidate.get("name") or inner
    return getattr(candidate, "name", None)
