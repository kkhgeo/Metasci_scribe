"""Helpers for resolving model context windows across LangChain providers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_CONTEXT_WINDOW_FALLBACK = 200_000

# Patch table for new models that providers haven't registered profile data
# for yet. Real ``model.profile`` data always wins, so entries only fill gaps;
# remove one only once the provider ships it at its pyproject floor, not just the lock.
# Keys are matched against ``model.model_name`` (or ``model.model`` /
# ``model.name``); lookup tries exact match first, then ``split('/')[-1]``
# to also accept ``vendor/model`` IDs served by a gateway.
_KNOWN_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    # Claude Haiku 4.5 — exception to the ``claude-`` family (200K, not 1M).
    "claude-haiku-4-5": 200_000,
}

# Family-level fallbacks: tried only after exact-name lookup misses.
# Each entry is (substring_to_match_in_lowercased_model_id, window).
# Order matters — first match wins; put more specific patterns first.
_KNOWN_MODEL_FAMILIES: list[tuple[str, int]] = [
    # All Claude — 1M via the ``context-1m-2025-08-07`` beta header.
    ("claude-", 1_000_000),
    # OpenAI GPT-6 family — astra, astra-pro, future variants
    ("gpt-6", 1_050_000),
    # OpenAI GPT-5.5 family — base, pro, future variants
    ("gpt-5.5", 1_050_000),
]

_DIRECT_WINDOW_ATTRS = (
    "context_window",
    "context_length",
    "num_ctx",
    "max_input_tokens",
)
_CONTAINER_ATTRS = (
    "profile",
    "context_management",
    "model_kwargs",
    "metadata",
)
_NAME_ATTRS = ("model_name", "model", "name")


def _coerce_positive_int(value: Any) -> int | None:
    """Best-effort coercion for positive integer-like values."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value > 0 and value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().replace(",", "").replace("_", "")
        if normalized.isdigit():
            parsed = int(normalized)
            return parsed if parsed > 0 else None
    return None


def _resolve_from_mapping(mapping: Mapping[str, Any]) -> int | None:
    """Resolve a context window from a metadata mapping."""
    for key in _DIRECT_WINDOW_ATTRS:
        if key in mapping:
            resolved = _coerce_positive_int(mapping.get(key))
            if resolved is not None:
                return resolved
    return None


def _lookup_by_model_name(model_obj: Any) -> int | None:
    """Return a patched context window by model name, if registered.

    Lookup is case-insensitive. Matching order:
        1. exact lowercased ``model_name`` value;
        2. last ``/``-segment of the lowercased value (handles
           gateway-style ``vendor/model`` IDs);
        3. family-level substring patterns (e.g. all ``claude-*``).
    """
    for attr in _NAME_ATTRS:
        value = getattr(model_obj, attr, None)
        if not isinstance(value, str) or not value:
            continue
        lowered = value.lower()
        if lowered in _KNOWN_MODEL_CONTEXT_WINDOWS:
            return _KNOWN_MODEL_CONTEXT_WINDOWS[lowered]
        short = lowered.split("/")[-1]
        if short != lowered and short in _KNOWN_MODEL_CONTEXT_WINDOWS:
            return _KNOWN_MODEL_CONTEXT_WINDOWS[short]
        for pattern, window in _KNOWN_MODEL_FAMILIES:
            if pattern in lowered:
                return window
    return None


def get_context_window(model_obj: Any | None) -> int | None:
    """Return the best available context-window value from a model object."""
    if model_obj is None:
        return None

    for attr in _DIRECT_WINDOW_ATTRS:
        resolved = _coerce_positive_int(getattr(model_obj, attr, None))
        if resolved is not None:
            return resolved

    for attr in _CONTAINER_ATTRS:
        candidate = getattr(model_obj, attr, None)
        if isinstance(candidate, Mapping):
            resolved = _resolve_from_mapping(candidate)
            if resolved is not None:
                return resolved

    return _lookup_by_model_name(model_obj)


def resolve_context_window(
    model_obj: Any | None,
    *,
    fallback: int = DEFAULT_CONTEXT_WINDOW_FALLBACK,
) -> int:
    """Resolve a usable context window with a stable fallback."""
    resolved = get_context_window(model_obj)
    if resolved is not None:
        return resolved
    return fallback


def apply_known_context_window(model: Any) -> None:
    """Inject patched ``max_input_tokens`` into ``model.profile`` for new
    models that providers haven't registered profile data for yet.

    Without this, deepagents' ``SummarizationMiddleware`` falls back to a
    hardcoded 170K trigger, ignoring the true (e.g. 1M) context window.
    Real provider-supplied profile data always wins — we only fill gaps.
    """
    patched = _lookup_by_model_name(model)
    if patched is None:
        return
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict) and "max_input_tokens" in profile:
        return
    base = profile if isinstance(profile, dict) else {}
    new_profile = {**base, "max_input_tokens": patched}
    try:
        model.profile = new_profile
    except Exception:
        pass
