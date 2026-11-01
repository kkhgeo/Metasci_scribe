"""Provider-error surface for langgraph SSE frames.

Provides :class:`ProviderStreamError` — a normalized, non-dataclass
exception raised by ``ErrorNormalizationMiddleware`` in place of the
provider SDK exception that a chat model call raised. Non-dataclass on
purpose: since orjson 3.0, dataclass instances are serialized natively
via their field enumeration, skipping the ``default=`` hook that
would otherwise build our SSE envelope. A provider SDK may decorate its
exceptions with ``@dataclass``, in which case its errors emerge on the
wire as raw dataclass fields — no envelope, no way for a client to
distinguish quota / auth / rate-limit. Wrapping them in
a plain ``Exception`` subclass here keeps orjson on the ``default=``
path, which then calls :meth:`ProviderStreamError.model_dump`
(upstream ``langgraph_api.serde.default`` checks that hook before its
``BaseException`` branch) — no serde monkey-patch needed.

Also lives here: the pure-function helpers the middleware uses to
build the envelope (provider tag from ``ModelRequest.model``, SDK
field extractors, env-driven API-key redaction). They stay next to
:class:`ProviderStreamError` because the middleware is their only
consumer.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any

from langchain_core.exceptions import ModelError

# ---------------------------------------------------------------------------
# ChatGPT plan failures
# ---------------------------------------------------------------------------


class PlanAction(Enum):
    """What a ChatGPT plan failure asks of the caller."""

    STOP_USAGE = "stop_usage"
    STOP_INELIGIBLE = "stop_ineligible"
    STOP_BAD_REQUEST = "stop_bad_request"
    RETRY = "retry"
    RELOGIN = "relogin"


class PlanStop(RuntimeError, ModelError):
    """A ChatGPT plan request ended without ``response.completed``.

    Inference stops here: nothing switches to another model or billing path.
    Only :attr:`PlanAction.RETRY` is worth another attempt, and ``model_retry``
    bounds those. The type is kept all the way to the caller, which tells a
    usage limit from other failures by ``action``.

    A ``ModelError`` so that langchain's own ``with_retry`` layers, such as the
    one around the summary call, leave it alone. ``is_retryable`` is what they
    read, so it is false for every action: ``model_retry`` goes by ``action``.

    ``headers`` holds only what sets the retry wait. No attribute may hold a
    token or a raw response body.
    """

    is_retryable = False

    def __init__(
        self,
        action: PlanAction,
        message: str = "",
        *,
        code: str | None = None,
        request_id: str | None = None,
        param: str | None = None,
        status_code: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(action, message)
        self.action = action
        self.message = message or action.name
        self.code = code
        self.request_id = request_id
        self.param = param
        self.status_code = status_code
        self.headers = dict(headers or {})

    def __str__(self) -> str:
        return self.message


# ---------------------------------------------------------------------------
# ProviderStreamError
# ---------------------------------------------------------------------------


class ProviderStreamError(Exception):
    """Envelope-shaped wrapper for a provider SDK exception raised
    inside a chat model call.

    Attributes mirror the SSE envelope one-for-one:

    - ``provider`` — concrete provider tag (``openai`` / ``anthropic``,
      or ``openai_compat`` / ``anthropic_compat`` for a gateway)
    - ``class_qualname`` — fully qualified name of the underlying
      exception's class (e.g. ``openai.RateLimitError``)
    - ``message`` — API-key-redacted ``str(exc)``
    - ``status_code`` — HTTP status if the SDK exposed one
    - ``code`` — provider error code (``insufficient_quota``, …)
    - ``err_type`` — provider error type label (openai's ``.type``)
    - ``request_id`` — SDK-provided correlation id

    The underlying exception is available via ``__cause__`` (set by
    ``raise ProviderStreamError(...) from exc`` in the middleware).
    """

    def __init__(
        self,
        provider: str,
        class_qualname: str,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        err_type: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.class_qualname = class_qualname
        self.message = message
        self.status_code = status_code
        self.code = code
        self.err_type = err_type
        self.request_id = request_id

    def as_envelope(self) -> dict[str, Any]:
        """Return the SSE envelope dict — the shape the WebUI consumes."""
        payload: dict[str, Any] = {
            "error": self.class_qualname.rsplit(".", 1)[-1],
            "class": self.class_qualname,
            "message": self.message,
            "provider": self.provider,
        }
        if self.status_code is not None:
            payload["status_code"] = self.status_code
        if self.code is not None:
            payload["code"] = self.code
        if self.err_type is not None:
            payload["type"] = self.err_type
        if self.request_id:
            payload["request_id"] = self.request_id
        return payload

    def model_dump(self) -> dict[str, Any]:
        """Serialization hook consumed by ``langgraph_api.serde.default``.

        Upstream's dispatch checks ``hasattr(obj, 'model_dump')`` BEFORE
        the ``isinstance(obj, BaseException)`` branch, so exposing this
        method lets upstream emit our envelope with no monkey-patch on
        its ``default`` callable. The name matches Pydantic's
        convention deliberately — it's the hook upstream is looking
        for.
        """
        return self.as_envelope()


class ModelErrorFinishError(RuntimeError):
    """The provider ended the response with ``finish_reason == "error"`` instead of raising."""

    def __init__(self, ai_message: Any) -> None:
        self.ai_message = ai_message
        super().__init__(
            "The model ended its response with finish_reason=error "
            "(the provider reported a failure)"
        )


# ---------------------------------------------------------------------------
# API-key redaction — env-driven, prefix-only
# ---------------------------------------------------------------------------
#
# Redaction is built from credentials actually deployed via env vars,
# not from generic key shapes. Rationale: (a) zero false positives —
# we only scrub strings we know are secrets, (b) defense-in-depth —
# the compiled regex holds only the first 8 chars of each key, so a
# leak of the regex object itself (traceback locals, process dump)
# can't expose the secret. Suffix-greedy match consumes the rest of
# the key shape at runtime. The table is rebuilt on every
# ``_redact_api_keys`` call so credentials loaded after import
# (typically ``load_dotenv`` in a main entry point) still get
# scrubbed. ``re.compile`` caches by source string internally, so an
# unchanged env costs a dict lookup.

_API_KEY_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")
_API_KEY_MIN_LEN = 12
_API_KEY_PREFIX_LEN = 8


def _build_env_key_redaction_re() -> re.Pattern[str] | None:
    prefixes: list[str] = []
    for k, v in os.environ.items():
        if not k.endswith(_API_KEY_ENV_SUFFIXES):
            continue
        if not isinstance(v, str) or len(v) < _API_KEY_MIN_LEN:
            continue
        prefixes.append(re.escape(v[:_API_KEY_PREFIX_LEN]))
    if not prefixes:
        return None
    alternation = "|".join(f"{p}[A-Za-z0-9_+/=.-]*" for p in prefixes)
    return re.compile(alternation)


def _redact_api_keys(message: str) -> str:
    """Replace any deployed key prefix in *message* with ``<redacted>``.

    Defensive; provider error messages occasionally echo the
    authorization header back. Rebuilt per call so credentials loaded
    after import (typical ``load_dotenv`` pattern) are still redacted.
    """
    pattern = _build_env_key_redaction_re()
    if pattern is None:
        return message
    return pattern.sub("<redacted>", message)


# ---------------------------------------------------------------------------
# Provider inference from ModelRequest.model
# ---------------------------------------------------------------------------
#
# Host → concrete provider. Kept here rather than imported from ``models.py``
# to keep the import surface of ``errors.py`` minimal — importing ``models.py``
# would pull in every langchain chat-model client at first middleware access.
# Consumed by ``_lookup_host_or_compat``; unknown hosts fall back to
# ``<module>_compat`` so a client knows "openai/anthropic SDK, but not native"
# instead of getting a misleading concrete tag.
#
# Related sibling: ``_PROVIDER_EXC_MODULE_PREFIXES`` in
# ``middleware/error_normalization.py`` — the exception-side
# provider allow-list. Adding a whole new provider SDK (not just a
# new base_url routed through an existing one) means updating that
# list too.

_HOST_TO_PROVIDER: dict[str, str] = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
}


def _provider_from_model(model: Any) -> str | None:
    """Derive the concrete provider tag from a chat model instance.

    ``openai_api_base`` / ``anthropic_api_url`` looked up in
    ``_HOST_TO_PROVIDER`` for openai/anthropic-shape clients (native +
    gateway). Returns ``None``
    when the model isn't from a recognized provider SDK — the caller
    (``ErrorNormalizationMiddleware``) then passes the exception
    through unchanged.
    """
    cls_modules = {cls.__module__ for cls in type(model).__mro__}

    def _uses_sdk(module_prefix: str) -> bool:
        return any(module.startswith(module_prefix) for module in cls_modules)

    if _uses_sdk("langchain_openai"):
        return _lookup_host_or_compat(
            getattr(model, "openai_api_base", None), module_tag="openai"
        )
    if _uses_sdk("langchain_anthropic"):
        return _lookup_host_or_compat(
            getattr(model, "anthropic_api_url", None), module_tag="anthropic"
        )
    return None


def _lookup_host_or_compat(base_url: str | None, module_tag: str) -> str:
    """Extract host from *base_url* and look up in ``_HOST_TO_PROVIDER``.

    Falls back to *module_tag* when no ``base_url`` is set (native SDK
    default endpoint) or ``<module_tag>_compat`` for an unrecognized
    host — the honest "openai SDK shape but unknown upstream" tag.
    """
    if not base_url:
        return module_tag
    try:
        from urllib.parse import urlparse

        host = urlparse(base_url).hostname
    except Exception:
        host = None
    if not host:
        return module_tag
    return _HOST_TO_PROVIDER.get(host.lower(), f"{module_tag}_compat")


# ---------------------------------------------------------------------------
# SDK-field extractors — populate the envelope's optional fields
# ---------------------------------------------------------------------------


def _extract_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status code from a provider SDK exception.

    Order matters: openai/anthropic store it on ``.status_code``;
    httpx-wrappers expose it via ``.response.status_code``; an SDK that
    stores it as an integer ``.code`` is type-disambiguated from
    openai/anthropic's string ``.code`` (provider error code, surfaced
    separately).
    """
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        rsc = getattr(response, "status_code", None)
        if isinstance(rsc, int):
            return rsc
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        return code
    return None


def _extract_provider_code(exc: BaseException) -> str | None:
    """Provider error code (e.g. ``insufficient_quota``,
    ``invalid_api_key``). Distinct from HTTP status; higher signal for
    a WebUI toast than the integer alone.
    """
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    return None


def _extract_error_type(exc: BaseException) -> str | None:
    """Provider error type label.

    - openai exposes this as ``.type`` (``rate_limit_error`` etc.)
    - an SDK that stores a string label at ``.status``
      (``"NOT_FOUND"``, ``"RESOURCE_EXHAUSTED"``, …) is a good fit for the
      same field.

    ``.type`` takes precedence when both are set.
    """
    err_type = getattr(exc, "type", None)
    if isinstance(err_type, str) and err_type:
        return err_type
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status:
        return status
    return None
