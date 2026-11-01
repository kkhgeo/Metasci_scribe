"""Inference on the user's ChatGPT plan, over the public Responses API.

https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference

``ChatOpenAI`` builds the request as usual. A wrapper httpx transport, plugged
in through the public ``http_client`` and ``http_async_client`` arguments, then
does everything specific to this route on every request:

- refuses any address but ``POST /v1/responses`` and ``GET /v1/models``,
- rewrites the body with :func:`normalize_payload`,
- swaps the placeholder API key for the plan's access token,
- reads the event stream on its way to the SDK and turns every ending other
  than ``response.completed`` into :class:`PlanStop`.

The real token lives only on the transport's own request, never on the one the
SDK holds, so SDK and httpx exceptions cannot carry it. Text from an error body
passes through :func:`_scrub`, since a server can echo the credential.

A failure stops inference; nothing here switches to another model or billing
path. ``model_retry`` is the one retry layer, so a retried call also drops what
the failed one had streamed. A network failure leaves the sign-in untouched, and
tokens are erased only by the sign-in module, after a terminal refresh refusal.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, NamedTuple
from urllib.request import getproxies, proxy_bypass

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from ..auth.chatgpt_plan import (
    RESOURCE,
    ChatGPTAuthError,
    CredentialStoreError,
    PlanUsageDisabled,
    RefreshUnavailable,
    get_access_token,
)
from .errors import PlanAction, PlanStop, _redact_api_keys

__all__ = [
    "ChatGPTPlanModel",
    "PlanAction",
    "PlanStop",
    "build_model",
    "classify_error",
    "list_models",
    "normalize_payload",
]

logger = logging.getLogger(__name__)

API_HOST = "api.openai.com"
_ROUTES = frozenset({("POST", "/v1/responses"), ("GET", "/v1/models")})
# ChatOpenAI requires a key; the transport replaces it on every request.
_PLACEHOLDER_KEY = "replaced-by-the-plan-transport"
_TIMEOUT = httpx.Timeout(300.0, connect=30.0)

# Preview limitations, plus what langchain-openai can add from the caller's
# arguments and the service-tier override the error reference names.
_UNSUPPORTED_FIELDS = (
    "background",
    "conversation",
    "max_output_tokens",
    "max_tool_calls",
    "metadata",
    "moderation",
    "multi_agent",
    "prompt",
    "prompt_cache_retention",
    "safety_identifier",
    "temperature",
    "top_logprobs",
    "top_p",
    "truncation",
    "user",
    "previous_response_id",
    "service_tier",
    "stream_options",
    "stop",
    "max_tokens",
    "max_completion_tokens",
)
_TOOL_TYPES = frozenset({"function", "custom"})
# What M0 showed a conversation needs, and the tools item this module adds,
# which has to survive a second pass.
_INPUT_ITEM_TYPES = frozenset(
    {"message", "function_call", "function_call_output", "additional_tools"}
)
# Sent only by the API key route; this bearer belongs to a ChatGPT account.
_ACCOUNT_HEADERS = ("openai-organization", "openai-project")

_CODE_ACTIONS = {
    "subscription_sharing_usage_limit_exceeded": PlanAction.STOP_USAGE,
    "subscription_sharing_user_not_eligible": PlanAction.STOP_INELIGIBLE,
    "subscription_sharing_unsupported_capability": PlanAction.STOP_BAD_REQUEST,
    "subscription_sharing_route_not_supported": PlanAction.STOP_BAD_REQUEST,
    "subscription_sharing_usage_unavailable": PlanAction.RETRY,
    "subscription_sharing_user_unavailable": PlanAction.RETRY,
    "subscription_sharing_invalid_user": PlanAction.RELOGIN,
    "chatpass_v2_scope_not_authorized": PlanAction.STOP_INELIGIBLE,
    "chatpass_v2_invalid_authorization_context": PlanAction.STOP_INELIGIBLE,
}
_RETRY_STATUSES = frozenset({408, 409, 425, 429})
# A failed event arrives on a 200, so its code is all there is to go on.
_RETRY_EVENT_CODES = frozenset({"server_error", "rate_limit_exceeded"})
_CONTEXT_OVERFLOW_CODE = "context_length_exceeded"

_GUIDANCE = {
    PlanAction.STOP_USAGE: (
        "ChatGPT plan usage limit reached. See ChatGPT settings > Usage, then continue."
    ),
    PlanAction.STOP_INELIGIBLE: (
        "ChatGPT plan usage is not available for this account, workspace or "
        "policy. To use your own API key instead: metasci-scribe config set "
        "openai_auth_mode api_key"
    ),
    PlanAction.STOP_BAD_REQUEST: (
        "The ChatGPT plan route refused this request; it will not be sent again."
    ),
    PlanAction.RETRY: (
        "ChatGPT plan usage is temporarily unavailable. Your sign-in is kept."
    ),
    PlanAction.RELOGIN: "ChatGPT did not accept this sign-in. Run /login.",
}

_CODE_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_PARAM_PATTERN = re.compile(r"[A-Za-z0-9_.\[\]-]{1,128}")
_SECRET_SHAPES = re.compile(r"(?i)\bbearer\s+\S+|\beyJ[\w.*-]{8,}|\bsk-[\w*-]{8,}")
_DETAIL_MAX = 300


# ---------------------------------------------------------------- request body


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = [
        block["text"]
        for block in content or []
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts)


def _is_system_like(item: Any) -> bool:
    return (
        isinstance(item, dict)
        and item.get("role") in ("system", "developer")
        and item.get("type", "message") == "message"
    )


def _split_system(items: list[Any]) -> tuple[list[str], list[Any]]:
    """Leading system and developer messages become instructions. The route
    rejects explicit system items, so a later one becomes a developer message."""
    instructions: list[str] = []
    rest: list[Any] = []
    leading = True
    for item in items:
        if leading and _is_system_like(item):
            instructions.append(_text_of(item.get("content")))
            continue
        leading = False
        if isinstance(item, dict) and item.get("role") == "system":
            item = {**item, "role": "developer"}
        rest.append(item)
    return instructions, rest


def _replayable(items: list[Any]) -> list[Any]:
    """Only the item types M0 proved, without IDs: with ``store: false`` the
    server cannot resolve its own. An item with no type is a message."""
    kept: list[Any] = []
    dropped: set[str] = set()
    for item in items:
        kind = item.get("type", "message") if isinstance(item, dict) else None
        if isinstance(kind, str) and kind in _INPUT_ITEM_TYPES:
            kept.append({k: v for k, v in item.items() if k not in ("id", "status")})
        elif kind != "reasoning":  # expected: never needed, so not worth a warning
            dropped.add(str(kind))
    if dropped:
        logger.warning(
            "ChatGPT plan route: left items of types %s out of the history",
            sorted(dropped),
        )
    return kept


def normalize_payload(payload: dict) -> dict:
    """``payload``, a langchain-built Responses body, under this route's rules.

    Pure: returns a new dict. Function tools travel as an ``additional_tools``
    input item, so the result has no top-level ``tools``.
    """
    body = copy.deepcopy(payload)
    for name in _UNSUPPORTED_FIELDS:
        body.pop(name, None)
    body["store"] = False
    body["stream"] = True

    raw = body.get("input") or []
    if isinstance(raw, str):
        raw = [{"type": "message", "role": "user", "content": raw}]
    instructions, items = _split_system(raw)
    if body.get("instructions"):
        instructions.insert(0, body["instructions"])
    if instructions:
        body["instructions"] = "\n\n".join(part for part in instructions if part)
    items = _replayable(items)

    tools = body.pop("tools", None) or []
    body.pop("tool_choice", None)
    body.pop("parallel_tool_calls", None)
    usable = [tool for tool in tools if tool.get("type") in _TOOL_TYPES]
    if len(usable) != len(tools):
        logger.warning(
            "ChatGPT plan route: dropped tools of types %s",
            sorted({str(t.get("type")) for t in tools if t not in usable}),
        )
    if usable:
        items.insert(
            0, {"type": "additional_tools", "role": "developer", "tools": usable}
        )

    body["input"] = items
    return body


# ---------------------------------------------------------------- error bodies


class _ErrorFields(NamedTuple):
    code: str | None = None
    param: str | None = None
    kind: str | None = None
    message: str | None = None


def _text_or_none(value: Any) -> str | None:
    """An error body is whatever the server sent; only text can be a field."""
    return value if isinstance(value, str) else None


def _error_fields(body: dict | str) -> _ErrorFields:
    """Read ``{"error": {...}}``, a flat error event, ``{"detail": "..."}`` or text."""
    parsed: Any = body
    if isinstance(body, str):
        try:
            parsed = json.loads(body)
        except ValueError:
            return _ErrorFields(message=body)
    if not isinstance(parsed, dict):
        return _ErrorFields(message=body if isinstance(body, str) else None)
    error = parsed.get("error")
    fields = error if isinstance(error, dict) else parsed
    message = fields.get("message") or parsed.get("detail") or error
    return _ErrorFields(
        code=_text_or_none(fields.get("code")),
        param=_text_or_none(fields.get("param")),
        kind=_text_or_none(fields.get("type")),
        message=_text_or_none(message),
    )


def classify_error(status: int, body: dict | str) -> PlanAction:
    """The action for a failure. A listed code decides; else the status does.

    ``status`` is 200 for a failure reported as an event in an open stream.
    """
    fields = _error_fields(body)
    action = _CODE_ACTIONS.get(fields.code)
    if action is not None:
        return action
    if status == 401:
        return PlanAction.RELOGIN
    if status == 403:
        return PlanAction.STOP_INELIGIBLE
    if status in _RETRY_STATUSES or status >= 500:
        return PlanAction.RETRY
    if status >= 400:
        return PlanAction.STOP_BAD_REQUEST
    if fields.code in _RETRY_EVENT_CODES or fields.kind == "server_error":
        return PlanAction.RETRY
    return PlanAction.STOP_BAD_REQUEST


def _safe_field(value: Any, pattern: re.Pattern[str], token: str | None) -> str | None:
    """``value`` when it has the shape of its field and is not the credential."""
    if not isinstance(value, str) or not pattern.fullmatch(value):
        return None
    return None if token and token in value else value


def _scrub(text: str, token: str | None) -> str:
    if token:
        text = text.replace(token, "<redacted>")
    text = " ".join(_redact_api_keys(_SECRET_SHAPES.sub("<redacted>", text)).split())
    return text[:_DETAIL_MAX]


def _stop(
    action: PlanAction,
    *,
    guidance: str | None = None,
    detail: str | None = None,
    code: str | None = None,
    param: str | None = None,
    request_id: str | None = None,
    status_code: int | None = None,
    headers: Mapping[str, str] | None = None,
) -> PlanStop:
    """Build a :class:`PlanStop` and log what a support request needs."""
    facts = [
        f"HTTP {status_code}" if status_code else None,
        f"code {code}" if code else None,
        f"param {param}" if param else None,
        f"request ID {request_id}" if request_id else None,
    ]
    known = ", ".join(fact for fact in facts if fact)
    message = guidance or _GUIDANCE[action]
    if detail:
        message = f"{message} {detail}"
    if known:
        message = f"{message} ({known})"
    logger.warning(
        "ChatGPT plan request stopped: action=%s status=%s code=%s param=%s "
        "request_id=%s",
        action.name,
        status_code,
        code,
        param,
        request_id,
    )
    return PlanStop(
        action,
        message,
        code=code,
        param=param,
        request_id=request_id,
        status_code=status_code,
        headers=headers,
    )


def _reported_failure(
    status: int,
    body: dict | str,
    *,
    token: str | None,
    request_id: str | None,
    headers: Mapping[str, str] | None = None,
) -> PlanStop:
    """A failure the server described, in a response body or a stream event."""
    fields = _error_fields(body)
    return _stop(
        classify_error(status, body),
        detail=_scrub(fields.message, token) if fields.message else None,
        code=_safe_field(fields.code, _CODE_PATTERN, token),
        param=_safe_field(fields.param, _PARAM_PATTERN, token),
        request_id=request_id,
        status_code=status if status >= 400 else None,
        headers=headers,
    )


def _sign_in_stop(exc: ChatGPTAuthError) -> PlanStop:
    """The sign-in module's text is fixed and free of secrets, so it is kept."""
    if isinstance(exc, RefreshUnavailable):
        return _stop(
            PlanAction.RETRY,
            guidance=f"{exc} Your sign-in is kept.",
            code=exc.code or "refresh_unavailable",
        )
    if isinstance(exc, PlanUsageDisabled):
        return _stop(
            PlanAction.STOP_INELIGIBLE,
            guidance=(
                f"{exc} Run /login and approve plan usage, or use your own API "
                "key: metasci-scribe config set openai_auth_mode api_key"
            ),
            code="plan_usage_disabled",
        )
    if isinstance(exc, CredentialStoreError):
        return _stop(
            PlanAction.STOP_BAD_REQUEST,
            guidance=str(exc),
            code="credential_store_error",
        )
    return _stop(
        PlanAction.RELOGIN,
        guidance=f"{exc} Run /login.",
        code=exc.code or "not_signed_in",
    )


def _token_or_stop() -> str:
    try:
        return get_access_token()
    except ChatGPTAuthError as exc:
        stop = _sign_in_stop(exc)
    raise stop


# ---------------------------------------------------------------- the transports


def _check_route(request: httpx.Request) -> None:
    url = request.url
    route = (request.method, url.path.rstrip("/"))
    if url.scheme != "https" or url.host != API_HOST or route not in _ROUTES:
        raise _stop(
            PlanAction.STOP_BAD_REQUEST,
            guidance=(
                f"Refused {request.method} {url.scheme}://{url.host}{url.path}: "
                "the ChatGPT plan route only uses the public Responses API."
            ),
            code="route_refused",
        )


def _authorized(request: httpx.Request, token: str) -> httpx.Request:
    headers = httpx.Headers(request.headers)
    for name in _ACCOUNT_HEADERS:
        headers.pop(name, None)
    headers["authorization"] = f"Bearer {token}"
    headers["accept-encoding"] = "identity"  # the watcher reads the bytes as they are
    content = b""
    if request.method == "POST":
        content = json.dumps(normalize_payload(json.loads(request.content))).encode()
        headers["content-length"] = str(len(content))
        headers["content-type"] = "application/json"
    return httpx.Request(
        request.method,
        request.url,
        headers=headers,
        content=content,
        extensions=request.extensions,
    )


def _request_id(headers: httpx.Headers) -> str | None:
    value = headers.get("x-request-id") or headers.get("openai-request-id")
    return _safe_field(value, _CODE_PATTERN, None)


def _network_stop(exc: httpx.RequestError) -> PlanStop:
    return _stop(
        PlanAction.RETRY,
        guidance=(
            f"Could not reach OpenAI ({type(exc).__name__}). Check the network or "
            "proxy settings. Your sign-in is kept."
        ),
        code="network_error",
    )


def _read_error_body(response: httpx.Response) -> bytes:
    """The body, or nothing if it was cut off: the status still says what happened."""
    try:
        return b"".join(response.stream)
    except httpx.TransportError:
        return b""
    finally:
        response.close()


async def _aread_error_body(response: httpx.Response) -> bytes:
    try:
        return b"".join([chunk async for chunk in response.stream])
    except httpx.TransportError:
        return b""
    finally:
        await response.aclose()


def _http_failure(response: httpx.Response, body: bytes, token: str) -> PlanStop:
    wait = {
        name: response.headers[name]
        for name in ("retry-after", "retry-after-ms")
        if name in response.headers
    }
    return _reported_failure(
        response.status_code,
        body.decode("utf-8", "replace"),
        token=token,
        request_id=_request_id(response.headers),
        headers=wait,
    )


class _EventWatch:
    """Reads the event stream beside the SDK and refuses every ending but one."""

    def __init__(self, token: str, request_id: str | None) -> None:
        self._token = token
        self._request_id = request_id
        self._tail = b""
        self._completed = False

    def feed(self, chunk: bytes) -> None:
        lines = (self._tail + chunk).split(b"\n")
        self._tail = lines.pop()
        for line in lines:
            self._read_line(line)

    def finish(self) -> None:
        tail, self._tail = self._tail, b""
        self._read_line(tail)
        if not self._completed:
            raise self._cut_off()

    def _cut_off(self) -> PlanStop:
        return _stop(
            PlanAction.RETRY,
            guidance=(
                "The response stream ended before response.completed. "
                "Your sign-in is kept."
            ),
            code="stream_incomplete",
            request_id=self._request_id,
        )

    def _read_line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data == b"[DONE]":
            if not self._completed:
                raise self._cut_off()
            return
        try:
            event = json.loads(data)
        except ValueError:
            return
        if isinstance(event, dict):
            self._read_event(event)

    def _read_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        response = event.get("response")
        response = response if isinstance(response, dict) else {}
        if kind == "response.completed":
            self._completed = True
        elif kind == "response.failed":
            body = {"error": response.get("error") or {}}
            raise self._event_failure(body)
        elif kind == "error" or (kind is None and event.get("error")):
            raise self._event_failure(event)
        elif kind == "response.incomplete":
            details = response.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            raise _stop(
                PlanAction.STOP_BAD_REQUEST,
                guidance="The model ended its response before finishing it.",
                code="response_incomplete",
                param=_safe_field(reason, _PARAM_PATTERN, self._token),
                request_id=self._request_id,
            )

    def _event_failure(self, body: dict[str, Any]) -> PlanStop:
        return _reported_failure(
            200, body, token=self._token, request_id=self._request_id
        )


class _WatchedStream(httpx.SyncByteStream):
    def __init__(self, inner: httpx.SyncByteStream, watch: _EventWatch) -> None:
        self._inner = inner
        self._watch = watch

    def __iter__(self) -> Iterator[bytes]:
        try:
            for chunk in self._inner:
                self._watch.feed(chunk)
                yield chunk
        except httpx.TransportError:
            pass  # ``finish`` decides: only a stream cut short before the end fails
        self._watch.finish()

    def close(self) -> None:
        self._inner.close()


class _AsyncWatchedStream(httpx.AsyncByteStream):
    def __init__(self, inner: httpx.AsyncByteStream, watch: _EventWatch) -> None:
        self._inner = inner
        self._watch = watch

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._inner:
                self._watch.feed(chunk)
                yield chunk
        except httpx.TransportError:
            pass  # ``finish`` decides: only a stream cut short before the end fails
        self._watch.finish()

    async def aclose(self) -> None:
        await self._inner.aclose()


def _watched(
    request: httpx.Request, response: httpx.Response, stream: Any, token: str
) -> httpx.Response:
    request_id = _request_id(response.headers)
    logger.debug(
        "ChatGPT plan %s %s: HTTP %s, request ID %s",
        request.method,
        request.url.path,
        response.status_code,
        request_id,
    )
    if request.method != "POST":
        return response
    return httpx.Response(
        response.status_code,
        headers=response.headers,
        stream=stream(response.stream, _EventWatch(token, request_id)),
        extensions=response.extensions,
    )


class _PlanTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport) -> None:
        self._inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        _check_route(request)
        token = _token_or_stop()
        outgoing = _authorized(request, token)
        try:
            response = self._inner.handle_request(outgoing)
        except httpx.TransportError as exc:
            stop = _network_stop(exc)
        else:
            if response.status_code < 400:
                return _watched(outgoing, response, _WatchedStream, token)
            stop = _http_failure(response, _read_error_body(response), token)
        raise stop

    def close(self) -> None:
        self._inner.close()


class _AsyncPlanTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        _check_route(request)
        # Can block on the sign-in lock and on a refresh.
        token = await asyncio.to_thread(_token_or_stop)
        outgoing = _authorized(request, token)
        try:
            response = await self._inner.handle_async_request(outgoing)
        except httpx.TransportError as exc:
            stop = _network_stop(exc)
        else:
            if response.status_code < 400:
                return _watched(outgoing, response, _AsyncWatchedStream, token)
            body = await _aread_error_body(response)
            stop = _http_failure(response, body, token)
        raise stop

    async def aclose(self) -> None:
        await self._inner.aclose()


def _proxy_url() -> str | None:
    """The proxy for the API host, from the environment or the system settings.

    httpx reads these only when it builds its own transport, and it ignores them
    once it is handed one, as it is here.
    """
    proxies = getproxies()
    proxy = proxies.get("https") or proxies.get("all")
    if not proxy or proxy_bypass(API_HOST):
        return None
    return proxy if "://" in proxy else f"http://{proxy}"


def _base_transport() -> httpx.BaseTransport:
    return httpx.HTTPTransport(proxy=_proxy_url())


def _base_async_transport() -> httpx.AsyncBaseTransport:
    return httpx.AsyncHTTPTransport(proxy=_proxy_url())


# ---------------------------------------------------------------- the model


def _plan_failure(exc: BaseException) -> Exception | None:
    """What to raise for ``exc``, or ``None`` when it is not a plan failure.

    The SDK reports a transport's exception as its own connection error, so the
    stop is found through ``__cause__``. A context overflow stays the type the
    compaction layer catches: shortening history on the same model is no fallback.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, PlanStop):
            if current.code == _CONTEXT_OVERFLOW_CODE:
                return ContextOverflowError(str(current))
            return current
        seen.add(id(current))
        current = current.__cause__
    return None


class ChatGPTPlanModel(ChatOpenAI):
    """``ChatOpenAI`` on the plan route: always streamed, failing as PlanStop."""

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        failure: Exception | None = None
        try:
            yield from super()._stream(*args, **kwargs)
        except Exception as exc:
            failure = _plan_failure(exc)
            if failure is None:
                raise
        # Out here, so the SDK exception does not travel along as context.
        if failure is not None:
            raise failure

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        failure: Exception | None = None
        try:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk
        except Exception as exc:
            failure = _plan_failure(exc)
            if failure is None:
                raise
        if failure is not None:
            raise failure

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return generate_from_stream(
            self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await agenerate_from_stream(
            self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )


def build_model(model_id: str, **kwargs: Any) -> BaseChatModel:
    """The chat model for ``model_id`` on the signed-in ChatGPT plan.

    Signs in lazily: a sign-in problem surfaces as :class:`PlanStop` on the
    first request. ``kwargs`` go to ``ChatOpenAI``, except that the route's own
    settings cannot be changed.
    """
    for alias in ("openai_api_key", "openai_api_base"):
        kwargs.pop(alias, None)
    settings = {
        **kwargs,
        "model": model_id,
        "base_url": RESOURCE,
        "api_key": _PLACEHOLDER_KEY,
        "use_responses_api": True,
        "use_previous_response_id": False,
        "streaming": True,
        "store": False,
        "max_retries": 0,
        "openai_proxy": None,
        "http_client": httpx.Client(
            transport=_PlanTransport(_base_transport()), timeout=_TIMEOUT
        ),
        "http_async_client": httpx.AsyncClient(
            transport=_AsyncPlanTransport(_base_async_transport()), timeout=_TIMEOUT
        ),
    }
    return ChatGPTPlanModel(**settings)


def list_models() -> list[tuple[str, str]]:
    """``(slug, display_name)`` for the account's listed models, in server order.

    Raises:
        PlanStop: The catalog could not be read.
    """
    transport = _PlanTransport(_base_transport())
    failure: PlanStop | None = None
    try:
        with httpx.Client(transport=transport, timeout=_TIMEOUT) as client:
            response = client.get(f"{RESOURCE}/models")
    except httpx.RequestError as exc:  # the client reads this body itself
        failure = _network_stop(exc)
    if failure is not None:
        raise failure
    try:
        document = response.json()
    except ValueError:
        document = None
    models = document.get("models") if isinstance(document, dict) else None
    if not isinstance(models, list):
        raise _stop(
            PlanAction.RETRY,
            guidance="OpenAI sent an unreadable model catalog.",
            code="invalid_model_catalog",
            request_id=_request_id(response.headers),
        )
    return [
        (model["slug"], _text_or_none(model.get("display_name")) or model["slug"])
        for model in models
        if isinstance(model, dict)
        and model.get("visibility") == "list"
        and isinstance(model.get("slug"), str)
    ]
