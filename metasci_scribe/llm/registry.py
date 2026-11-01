"""Model registry data — short names, model ids and providers.

Pure data with no langchain/provider-SDK imports: the onboard wizard, the
``/model`` pickers, and provider validation read this registry without paying
for the chat-model construction stack in :mod:`.models` (~2000 modules).
"""

from __future__ import annotations


class UnknownModelError(ValueError):
    """A model or provider that Metasci Scribe cannot build a chat model for."""


# Model registry: list of (short_name, model_id, provider)
# Metasci Scribe supports two providers, OpenAI and Anthropic, each through
# its own API key (and an optional base URL for a compatible gateway).
_MODEL_ENTRIES: list[tuple[str, str, str]] = [
    # OpenAI (the default provider; the first entry is the default model)
    ("gpt-6.1-sol", "gpt-6.1-sol", "openai"),
    ("gpt-6-sol", "gpt-6-sol", "openai"),
    ("gpt-6-luna", "gpt-6-luna", "openai"),
    ("gpt-6-astra", "gpt-6-astra", "openai"),
    ("gpt-5.6-sol", "gpt-5.6-sol", "openai"),
    ("gpt-5.6-terra", "gpt-5.6-terra", "openai"),
    ("gpt-5.6-luna", "gpt-5.6-luna", "openai"),
    # Anthropic (current generation)
    ("claude-fable-5-1", "claude-fable-5-1", "anthropic"),
    ("claude-opus-5-5", "claude-opus-5-5", "anthropic"),
    ("claude-sonnet-5-5", "claude-sonnet-5-5", "anthropic"),
    ("claude-haiku-5-5", "claude-haiku-5-5", "anthropic"),
]

# Public dict for simple lookups (last entry wins for duplicate names).
# Use get_models_for_provider() for provider-aware lookups.
MODELS: dict[str, tuple[str, str]] = {
    name: (model_id, provider) for name, model_id, provider in _MODEL_ENTRIES
}

DEFAULT_MODEL = "gpt-6.1-sol"


def get_models_for_provider(provider: str) -> list[tuple[str, str]]:
    """Get all models for a specific provider.

    Args:
        provider: Provider name (``openai`` or ``anthropic``).

    Returns:
        List of (short_name, model_id) tuples for the provider.
    """
    return [(name, model_id) for name, model_id, p in _MODEL_ENTRIES if p == provider]


def list_models() -> list[str]:
    """List all available model short names.

    Returns:
        List of unique model short names that can be passed to get_chat_model().
    """
    seen = set()
    result = []
    for name, _, _ in _MODEL_ENTRIES:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def list_models_by_provider() -> list[tuple[str, str, str]]:
    """List all unique (short_name, model_id, provider) entries.

    Returns:
        De-duplicated list of model entries preserving registry order.
    """
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str, str]] = []
    for name, model_id, provider in _MODEL_ENTRIES:
        key = (name, provider)
        if key not in seen:
            seen.add(key)
            result.append((name, model_id, provider))
    return result


def list_model_picker_entries() -> list[tuple[str, str, str]]:
    """Return the entries the ``/model`` picker offers."""
    return list_models_by_provider()


def resolve_provider(model: str, preferred: str | None = None) -> str | None:
    """Provider for a short name, staying on ``preferred`` when it serves it.

    ``MODELS`` keeps only the last entry per short name, which would move a
    user off their provider when two providers list the same short name.
    """
    if preferred and any(n == model and p == preferred for n, _, p in _MODEL_ENTRIES):
        return preferred
    info = MODELS.get(model)
    return info[1] if info else None


def get_model_info(model: str) -> tuple[str, str] | None:
    """Get the (model_id, provider) tuple for a short name.

    Args:
        model: Short model name.

    Returns:
        Tuple of (model_id, provider) or None if not found.
    """
    return MODELS.get(model)
