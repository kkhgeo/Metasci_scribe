"""API key checks for the model providers (OpenAI and Anthropic).

Kept free of UI imports so callers can use them without loading the
onboarding wizard. :data:`PROVIDERS` maps each provider to its
key field, its base URL field and its check; :func:`check_provider_key` runs
a check and returns a :class:`KeyCheck`.

Every check makes one live request with a 10 s timeout and no retries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

CheckStatus = Literal["valid", "invalid", "inconclusive", "unavailable", "empty"]

_TIMEOUT = 10


class KeyCheck(NamedTuple):
    """Result of a key check.

    ``valid`` and ``invalid`` are conclusive. ``inconclusive`` covers a
    transient or unexpected failure (offline, timeout, 5xx), ``unavailable``
    a provider or endpoint that cannot be checked, ``empty`` a missing key.
    """

    status: CheckStatus
    message: str
    models: tuple[str, ...] = ()

    @property
    def ok(self) -> bool | None:
        """``True`` or ``False`` for a conclusive result, else ``None``."""
        if self.status == "valid":
            return True
        if self.status == "invalid":
            return False
        return None


_SKIPPED = KeyCheck("empty", "Skipped (no key provided)")
_INVALID = KeyCheck("invalid", "Invalid API key")

# =============================================================================
# API Key Validation
# =============================================================================


# Avoids bare "invalid" — collides with "Invalid request: model X not found".
_AUTH_FAILURE_HINTS = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid api key",
    "invalid_api_key",
    "incorrect api key",
    "incorrect_api_key",
    "api key not valid",
    "invalid token",
)

_TRANSIENT_HINTS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "500",
    "502",
    "503",
    "504",
    "timeout",
    "timed out",
    "connection",
    "service unavailable",
    "temporarily unavailable",
    "upstream",
)


def _classify_validation_error(error: BaseException) -> KeyCheck:
    """Classify a check's exception as an auth failure or an inconclusive result.

    An exception that matches neither hint list is inconclusive too, with the
    error text as its message.
    """
    s = str(error).lower()
    if any(h in s for h in _AUTH_FAILURE_HINTS):
        return _INVALID
    if any(h in s for h in _TRANSIENT_HINTS):
        return KeyCheck(
            "inconclusive",
            "Validation inconclusive — transient error, try again later",
        )
    return KeyCheck("inconclusive", f"Error: {error}")


def _status_result(status_code: int) -> KeyCheck:
    """Result for an HTTP status other than the ones a check accepts."""
    # Only 401/403 mean the key is actually rejected. 429 (rate-limit) and 5xx
    # (provider incident) leave the key validity unknown — surface the real
    # status so the user doesn't go re-roll a good key during an outage.
    if status_code in (401, 403):
        return _INVALID
    return KeyCheck("inconclusive", f"Validation inconclusive (HTTP {status_code})")


def check_anthropic_key(api_key: str) -> KeyCheck:
    """Check an Anthropic API key by listing models."""
    if not api_key:
        return _SKIPPED

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=_TIMEOUT, max_retries=0)
        client.models.list()
        return KeyCheck("valid", "Valid")
    except anthropic.AuthenticationError:
        return _INVALID
    except Exception as e:
        return _classify_validation_error(e)


def check_openai_key(api_key: str) -> KeyCheck:
    """Check an OpenAI API key by listing models."""
    if not api_key:
        return _SKIPPED

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, timeout=_TIMEOUT, max_retries=0)
        client.models.list()
        return KeyCheck("valid", "Valid")
    except openai.AuthenticationError:
        return _INVALID
    except Exception as e:
        return _classify_validation_error(e)


# =============================================================================
# Provider table
# =============================================================================


@dataclass(frozen=True)
class KeySpec:
    """How one provider is configured and checked.

    ``check`` names a function in this module, looked up when the check runs,
    so tests can patch it. A check with a ``base_url_field`` takes
    ``(key, base_url)``, the others ``(key)``.
    """

    label: str
    key_field: str | None
    key_env: str | None
    check: str | None
    base_url_field: str | None = None
    base_url_env: str | None = None
    default_base_url: str | None = None
    needs_base_url: bool = False
    lists_models: bool = False

    @property
    def needs_key(self) -> bool:
        return self.key_field is not None

    @property
    def has_check(self) -> bool:
        return self.check is not None


def _key(label: str, field: str, env: str, check: str) -> KeySpec:
    return KeySpec(label=label, key_field=field, key_env=env, check=check)


PROVIDERS: dict[str, KeySpec] = {
    "openai": _key("OpenAI", "openai_api_key", "OPENAI_API_KEY", "check_openai_key"),
    "anthropic": _key(
        "Anthropic", "anthropic_api_key", "ANTHROPIC_API_KEY", "check_anthropic_key"
    ),
}


def provider_spec(provider: str) -> KeySpec:
    """The table entry for ``provider``; ``KeyError`` for an unknown one."""
    return PROVIDERS[provider]


def configured_base_url(spec: KeySpec, config: Any) -> str:
    """The base URL a check uses: the config's, else the environment's, else
    the provider's default (blank values count as unset)."""
    if spec.base_url_field is None:
        return ""
    return (
        str(getattr(config, spec.base_url_field, "") or "").strip()
        or os.environ.get(spec.base_url_env or "", "").strip()
        or (spec.default_base_url or "")
    )


def run_check(spec: KeySpec, key: str, base_url: str = "") -> KeyCheck:
    """Run ``spec``'s check on ``key`` (and ``base_url`` where it takes one)."""
    if spec.check is None:
        return KeyCheck("unavailable", f"No check is available for {spec.label}")
    check = globals()[spec.check]
    if spec.base_url_field is not None:
        return check(key, base_url=base_url or spec.default_base_url or "")
    return check(key)


def check_provider_key(name: str, key: str, base_url: str = "") -> KeyCheck:
    """Run the check for a provider; ``KeyError`` if unknown."""
    return run_check(PROVIDERS[name], key, base_url)
