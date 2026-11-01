"""ChatGPT plan route: request rules, error classes and the stop-never-switch rule.

Everything runs against a scripted fake of api.openai.com plugged in under the
real wrapper transports, so the openai SDK and langchain-openai run for real.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
import itertools
import json
import logging
import pickle
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from deepagents import graph as deepagents_graph
from deepagents.backends import FilesystemBackend
from langchain.agents import create_agent
from langchain.agents.middleware._retry import default_retry_on
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

import metasci_scribe
from metasci_scribe.agent import create_cli_agent
from metasci_scribe.auth import chatgpt_plan as auth
from metasci_scribe.config import ScribeConfig
from metasci_scribe.llm import chatgpt_plan as plan
from metasci_scribe.llm.chatgpt_plan import (
    PlanAction,
    PlanStop,
    build_model,
    classify_error,
    list_models,
    normalize_payload,
)
from metasci_scribe.llm.models import get_chat_model
from metasci_scribe.middleware.context_overflow import ContextOverflowMapperMiddleware
from metasci_scribe.middleware.error_normalization import ErrorNormalizationMiddleware
from metasci_scribe.middleware.model_retry import ModelRetryMiddleware, is_transient
from metasci_scribe.middleware.summarization import (
    create_per_run_summarization_middleware,
)
from metasci_scribe.paths import Workspace

TOKEN = "plan-access-token-DO-NOT-LEAK"
SSE_HEADERS = {"content-type": "text/event-stream", "x-request-id": "req_stream"}
DOCUMENTED_UNSUPPORTED = (
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
)


def event(kind: str, **fields: Any) -> dict[str, Any]:
    return {"type": kind, "sequence_number": 0, **fields}


_response_ids = itertools.count(1)


def response_object(status: str = "completed", **fields: Any) -> dict[str, Any]:
    return {
        "id": f"resp_{next(_response_ids)}",
        "object": "response",
        "created_at": 1,
        "model": "gpt-6.1-sol",
        "status": status,
        "error": None,
        "incomplete_details": None,
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        **fields,
    }


CREATED = event("response.created", response=response_object("in_progress"))
COMPLETED = event("response.completed", response=response_object())


def text_events(reply: str = "done", **response_fields: Any) -> list[dict[str, Any]]:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": reply, "annotations": []}],
    }
    return [
        event("response.created", response=response_object("in_progress")),
        event(
            "response.output_item.added",
            output_index=0,
            item={**message, "status": "in_progress", "content": []},
        ),
        event(
            "response.content_part.added",
            item_id="msg_1",
            output_index=0,
            content_index=0,
            part={"type": "output_text", "text": "", "annotations": []},
        ),
        event(
            "response.output_text.delta",
            item_id="msg_1",
            output_index=0,
            content_index=0,
            delta=reply,
            logprobs=[],
        ),
        event("response.output_item.done", output_index=0, item=message),
        event(
            "response.completed",
            response=response_object(output=[message], **response_fields),
        ),
    ]


def tool_call_events(name: str, arguments: str) -> list[dict[str, Any]]:
    item = {
        "id": "fc_1",
        "type": "function_call",
        "status": "completed",
        "call_id": "call_1",
        "name": name,
        "arguments": arguments,
    }
    return [
        event("response.created", response=response_object("in_progress")),
        event(
            "response.output_item.added",
            output_index=0,
            item={**item, "status": "in_progress", "arguments": ""},
        ),
        event(
            "response.function_call_arguments.delta",
            item_id="fc_1",
            output_index=0,
            delta=arguments,
        ),
        event("response.output_item.done", output_index=0, item=item),
        event("response.completed", response=response_object(output=[item])),
    ]


def sse_bytes(events: list[dict[str, Any]]) -> bytes:
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )


Reply = Callable[[], httpx.Response]


def sse(*events: dict[str, Any], headers: dict[str, str] | None = None) -> Reply:
    body = sse_bytes(list(events))
    return lambda: httpx.Response(
        200, headers={**SSE_HEADERS, **(headers or {})}, content=body
    )


def http_error(
    status: int,
    body: dict[str, Any] | str,
    *,
    request_id: str = "req_http",
    headers: dict[str, str] | None = None,
) -> Reply:
    content = body if isinstance(body, str) else json.dumps(body)
    return lambda: httpx.Response(
        status,
        headers={
            "content-type": "application/json",
            "x-request-id": request_id,
            **(headers or {}),
        },
        content=content.encode(),
    )


def structured(code: str, *, param: str | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": "refused", "param": param}}


class _Dropped(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Sends ``chunks``, then loses the connection."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks
        raise httpx.ReadError("connection lost")

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        raise httpx.ReadError("connection lost")


def dropped(*events: dict[str, Any]) -> Reply:
    return lambda: httpx.Response(
        200, headers=SSE_HEADERS, stream=_Dropped([sse_bytes(list(events))])
    )


def connect_refused() -> httpx.Response:
    raise httpx.ConnectError("refused")


class FakeOpenAI:
    """Answers each request with the next reply; the last one repeats."""

    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply()

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if r.method == "POST"]


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setattr(plan, "get_access_token", lambda: TOKEN)


@pytest.fixture
def serve(monkeypatch, token):
    """Install a scripted server under the real wrapper transports."""

    def install(*replies: Reply, server: FakeOpenAI | None = None) -> FakeOpenAI:
        server = server or FakeOpenAI(*replies)
        transport = httpx.MockTransport(server)
        monkeypatch.setattr(plan, "_base_transport", lambda: transport)
        monkeypatch.setattr(plan, "_base_async_transport", lambda: transport)
        return server

    return install


@pytest.fixture
def signed_in(monkeypatch, tmp_path):
    """A saved sign-in made of test values, in a folder of this test's own."""
    monkeypatch.setattr("metasci_scribe.paths.DATA_DIR", tmp_path)
    auth.save(
        auth.Credentials(
            client_id="app_test",
            access_token="saved-test-access",
            refresh_token="saved-test-refresh",
            id_token="saved-test-id",
            expires_at=time.time() + 3600,
            scopes=frozenset({auth.DIRECT_SCOPE}),
            email=None,
        )
    )


def run(model: Any, mode: str, messages: Any = "hello", **kwargs: Any) -> Any:
    if mode == "async":
        return asyncio.run(model.ainvoke(messages, **kwargs))
    return model.invoke(messages, **kwargs)


both_modes = pytest.mark.parametrize("mode", ["sync", "async"])


# --------------------------------------------------------------- normalize_payload


def test_normalize_sets_required_fields():
    out = normalize_payload(
        {
            "model": "m",
            "input": [],
            "temperature": 0.2,
            "top_p": 1,
            "max_output_tokens": 5,
            "metadata": {},
            "user": "u",
            "previous_response_id": "r",
        }
    )
    assert out["store"] is False
    assert out["stream"] is True
    assert (
        not {
            "temperature",
            "top_p",
            "max_output_tokens",
            "metadata",
            "user",
            "previous_response_id",
        }
        & out.keys()
    )


@pytest.mark.parametrize(
    "field",
    [
        *DOCUMENTED_UNSUPPORTED,
        "previous_response_id",
        "service_tier",
        "stream_options",
        "stop",
        "max_tokens",
        "max_completion_tokens",
    ],
)
def test_every_unsupported_field_is_removed(field):
    assert field not in normalize_payload({"model": "m", "input": [], field: "x"})


def test_store_and_stream_overrule_the_caller():
    out = normalize_payload({"input": [], "store": True, "stream": False})
    assert out["store"] is False
    assert out["stream"] is True


def test_normalize_leaves_its_argument_alone():
    payload = {
        "input": [{"role": "system", "content": "s"}],
        "tools": [{"type": "function", "name": "f"}],
        "temperature": 1,
    }
    before = copy.deepcopy(payload)
    normalize_payload(payload)
    assert payload == before


def test_string_input_becomes_an_array():
    out = normalize_payload({"input": "hi"})
    assert out["input"] == [{"type": "message", "role": "user", "content": "hi"}]


def test_missing_input_becomes_an_empty_array():
    assert normalize_payload({"model": "m"})["input"] == []


def test_system_item_becomes_instructions():
    out = normalize_payload(
        {
            "instructions": "first",
            "input": [
                {"role": "system", "content": "second"},
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "third"}],
                },
                {"type": "message", "role": "user", "content": "hi"},
            ],
        }
    )
    assert out["instructions"] == "first\n\nsecond\n\nthird"
    assert out["input"] == [{"type": "message", "role": "user", "content": "hi"}]


def test_a_later_system_item_becomes_developer():
    out = normalize_payload(
        {
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "message", "role": "system", "content": "late"},
            ]
        }
    )
    assert "instructions" not in out
    assert [item["role"] for item in out["input"]] == ["user", "developer"]


def test_no_system_role_survives_anywhere():
    out = normalize_payload(
        {
            "input": [
                {"role": "system", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "system", "content": "c"},
                {"role": "assistant", "content": "d"},
                {"role": "system", "content": "e"},
            ]
        }
    )
    assert "system" not in {item.get("role") for item in out["input"]}


def test_tools_follow_spike_format():
    tools = [
        {"type": "function", "name": "read", "parameters": {}},
        {"type": "custom", "name": "edit"},
    ]
    out = normalize_payload(
        {
            "input": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
    )
    assert out["input"][0] == {
        "type": "additional_tools",
        "role": "developer",
        "tools": tools,
    }
    assert out["input"][1] == {"role": "user", "content": "hi"}
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & out.keys()
    assert "namespace" not in json.dumps(out)


def test_no_tools_means_no_additional_tools_item():
    out = normalize_payload(
        {"input": [{"role": "user", "content": "hi"}], "tools": [], "tool_choice": "x"}
    )
    assert len(out["input"]) == 1
    assert "tool_choice" not in out


def test_tools_this_route_cannot_run_are_dropped():
    out = normalize_payload(
        {
            "input": [],
            "tools": [
                {"type": "code_interpreter"},
                {"type": "function", "name": "read"},
            ],
        }
    )
    assert "tools" not in out
    assert out["input"][0]["tools"] == [{"type": "function", "name": "read"}]


def test_replay_keeps_the_three_item_kinds_without_server_ids():
    out = normalize_payload(
        {
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "id": "msg_1",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "t"}],
                },
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "reasoning",
                    "id": "rs_2",
                    "summary": [],
                    "encrypted_content": "blob",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "status": "completed",
                    "call_id": "c",
                    "name": "read",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "c", "output": "x"},
            ]
        }
    )
    assert [item["type"] for item in out["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    for item in out["input"]:
        assert not {"id", "status"} & item.keys()


def test_normalize_twice_changes_nothing():
    once = normalize_payload(
        {
            "input": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "hi"},
            ],
            "tools": [{"type": "function", "name": "read"}],
        }
    )
    assert normalize_payload(once) == once


# -------------------------------------------------------------------- classify_error


@pytest.mark.parametrize(
    ("code", "status", "action"),
    [
        ("subscription_sharing_usage_limit_exceeded", 429, PlanAction.STOP_USAGE),
        ("subscription_sharing_user_not_eligible", 403, PlanAction.STOP_INELIGIBLE),
        (
            "subscription_sharing_unsupported_capability",
            400,
            PlanAction.STOP_BAD_REQUEST,
        ),
        ("subscription_sharing_usage_unavailable", 503, PlanAction.RETRY),
        ("subscription_sharing_invalid_user", 401, PlanAction.RELOGIN),
        ("subscription_sharing_user_unavailable", 503, PlanAction.RETRY),
        ("subscription_sharing_route_not_supported", 403, PlanAction.STOP_BAD_REQUEST),
        ("chatpass_v2_scope_not_authorized", 403, PlanAction.STOP_INELIGIBLE),
        ("chatpass_v2_invalid_authorization_context", 403, PlanAction.STOP_INELIGIBLE),
    ],
)
def test_classify(code, status, action):
    assert classify_error(status, structured(code)) is action


@pytest.mark.parametrize(
    ("code", "action"),
    [
        ("subscription_sharing_usage_limit_exceeded", PlanAction.STOP_USAGE),
        ("subscription_sharing_usage_unavailable", PlanAction.RETRY),
    ],
)
def test_the_code_decides_on_a_failed_event(code, action):
    assert classify_error(200, {"error": {"code": code}}) is action


@pytest.mark.parametrize(
    ("status", "action"),
    [
        (401, PlanAction.RELOGIN),
        (403, PlanAction.STOP_INELIGIBLE),
        (503, PlanAction.RETRY),
    ],
)
def test_detail_body_without_error_object(status, action):
    assert classify_error(status, {"detail": "not admitted"}) is action


@pytest.mark.parametrize(
    ("status", "body", "action"),
    [
        (
            429,
            json.dumps(structured("subscription_sharing_usage_limit_exceeded")),
            PlanAction.STOP_USAGE,
        ),
        (503, "upstream connect error", PlanAction.RETRY),
        (401, "", PlanAction.RELOGIN),
        (400, "not json", PlanAction.STOP_BAD_REQUEST),
        (404, structured("model_not_found"), PlanAction.STOP_BAD_REQUEST),
        (422, {}, PlanAction.STOP_BAD_REQUEST),
        (429, structured("rate_limit_exceeded"), PlanAction.RETRY),
        (500, {"error": {"code": None, "type": "server_error"}}, PlanAction.RETRY),
        (502, {}, PlanAction.RETRY),
        (408, {}, PlanAction.RETRY),
        (200, structured("server_error"), PlanAction.RETRY),
        (200, structured("something_new"), PlanAction.STOP_BAD_REQUEST),
        (200, {"error": "plain string"}, PlanAction.STOP_BAD_REQUEST),
    ],
)
def test_classify_without_a_listed_code(status, body, action):
    assert classify_error(status, body) is action


def test_plan_stop_is_pickled_with_its_fields():
    stop = PlanStop(PlanAction.STOP_USAGE, "limit", code="c", request_id="r", param="p")
    copied = pickle.loads(pickle.dumps(stop))
    assert (copied.action, copied.code, copied.request_id, copied.param) == (
        PlanAction.STOP_USAGE,
        "c",
        "r",
        "p",
    )
    assert str(copied) == "limit"


# ----------------------------------------------------------------------- happy path


@both_modes
def test_a_turn_reaches_only_the_responses_endpoint(serve, mode, monkeypatch):
    monkeypatch.setenv("OPENAI_ORG_ID", "org-from-the-environment")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "proj-from-the-environment")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-environment-key")
    server = serve(sse(*text_events("hello back")))
    reply = run(build_model("gpt-6.1-sol", temperature=0.3), mode)
    assert reply.text == "hello back"

    request = server.requests[0]
    assert (request.method, str(request.url)) == (
        "POST",
        "https://api.openai.com/v1/responses",
    )
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert "openai-organization" not in request.headers
    assert "openai-project" not in request.headers
    body = server.bodies[0]
    assert body["store"] is False
    assert body["stream"] is True
    assert body["model"] == "gpt-6.1-sol"
    assert "temperature" not in body


def test_a_system_message_goes_to_instructions_on_the_wire(serve):
    server = serve(sse(*text_events()))
    build_model("gpt-6.1-sol").invoke(
        [SystemMessage(content="be brief"), HumanMessage(content="hi")]
    )
    body = server.bodies[0]
    assert body["instructions"] == "be brief"
    assert {item.get("role") for item in body["input"]} == {"user"}


@both_modes
def test_generate_reads_the_stream(serve, mode):
    """A copy with streaming off still asks for and reads a stream."""
    server = serve(sse(*text_events("read from the stream")))
    model = build_model("gpt-6.1-sol", disable_streaming=True)
    messages = [HumanMessage(content="hi")]
    if mode == "async":
        result = asyncio.run(model._agenerate(messages))
    else:
        result = model._generate(messages)
    assert result.generations[0].message.text == "read from the stream"
    assert server.bodies[0]["stream"] is True


def test_structured_output_from_a_pydantic_class_reads_the_stream(serve):
    class Answer(BaseModel):
        value: int

    json_schema = {"type": "json_schema", "name": "Answer", "schema": {}}
    server = serve(sse(*text_events('{"value": 7}', text={"format": json_schema})))
    model = build_model("gpt-6.1-sol", disable_streaming=True)
    answer = model.with_structured_output(Answer).invoke("seven")
    assert answer == Answer(value=7)
    assert server.bodies[0]["stream"] is True


def test_the_final_chunk_arrives_when_the_event_splits_across_reads(serve):
    body = sse_bytes(text_events("split"))
    cut = len(body) - 40

    class _Split(httpx.SyncByteStream):
        def __iter__(self):
            yield body[:cut]
            yield body[cut:].rstrip(b"\n")

    serve(lambda: httpx.Response(200, headers=SSE_HEADERS, stream=_Split()))
    assert build_model("gpt-6.1-sol").invoke("hi").text == "split"


def test_tool_history_replays_through_the_additional_tools_item(serve):
    @tool
    def read_note(path: str) -> str:
        """Read a note."""
        return f"contents of {path}"

    server = serve(
        sse(*tool_call_events("read_note", '{"path": "a.txt"}')),
        sse(*text_events("the note says hi")),
        sse(*text_events("still hi")),
    )
    agent = create_agent(
        build_model("gpt-6.1-sol"), tools=[read_note], system_prompt="be careful"
    )
    first = agent.invoke({"messages": [HumanMessage(content="read a.txt")]})
    assert first["messages"][-1].text == "the note says hi"
    agent.invoke({"messages": [*first["messages"], HumanMessage(content="again")]})

    for body in server.bodies:
        assert "tools" not in body
        assert body["input"][0]["type"] == "additional_tools"
        assert body["input"][0]["role"] == "developer"
        assert [t["name"] for t in body["input"][0]["tools"]] == ["read_note"]
        assert body["instructions"] == "be careful"
        assert "system" not in {item.get("role") for item in body["input"]}

    replayed = server.bodies[2]["input"]
    kinds = [item.get("type") for item in replayed]
    assert "function_call" in kinds
    assert "function_call_output" in kinds
    assert "reasoning" not in kinds
    call = next(item for item in replayed if item.get("type") == "function_call")
    assert call["call_id"] == "call_1"
    assert call["name"] == "read_note"
    assert not {"id", "status"} & call.keys()
    output = next(
        item for item in replayed if item.get("type") == "function_call_output"
    )
    assert output["call_id"] == "call_1"
    assert "contents of a.txt" in json.dumps(output)


# -------------------------------------------------------------------- failure routes


@both_modes
@pytest.mark.parametrize(
    ("code", "status", "action"),
    [
        ("subscription_sharing_usage_limit_exceeded", 429, PlanAction.STOP_USAGE),
        ("subscription_sharing_user_not_eligible", 403, PlanAction.STOP_INELIGIBLE),
        ("subscription_sharing_usage_unavailable", 503, PlanAction.RETRY),
        ("subscription_sharing_invalid_user", 401, PlanAction.RELOGIN),
    ],
)
def test_an_http_error_raises_plan_stop(serve, mode, code, status, action):
    server = serve(http_error(status, structured(code), request_id="req_abc"))
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    stop = caught.value
    assert (stop.action, stop.code, stop.request_id, stop.status_code) == (
        action,
        code,
        "req_abc",
        status,
    )
    assert len(server.requests) == 1


def test_an_unsupported_capability_names_the_param(serve):
    serve(
        http_error(
            400,
            structured(
                "subscription_sharing_unsupported_capability", param="tools[0].type"
            ),
        )
    )
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert caught.value.param == "tools[0].type"
    assert "tools[0].type" in str(caught.value)


def test_usage_limit_message_points_at_chatgpt_settings(serve):
    serve(http_error(429, structured("subscription_sharing_usage_limit_exceeded")))
    with pytest.raises(PlanStop, match="ChatGPT settings"):
        build_model("gpt-6.1-sol").invoke("hi")


def test_a_detail_body_keeps_its_text_as_diagnostics(serve):
    serve(http_error(403, {"detail": "Serving region not permitted"}))
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert caught.value.action is PlanAction.STOP_INELIGIBLE
    assert caught.value.code is None
    assert "Serving region not permitted" in str(caught.value)
    assert "req_http" in str(caught.value)


@both_modes
@pytest.mark.parametrize(
    ("code", "action"),
    [
        ("subscription_sharing_usage_limit_exceeded", PlanAction.STOP_USAGE),
        ("subscription_sharing_usage_unavailable", PlanAction.RETRY),
    ],
)
def test_failed_event_after_stream_started_raises_plan_stop(serve, mode, code, action):
    failed = response_object("failed", error={"code": code, "message": "no"})
    serve(
        sse(
            CREATED,
            event(
                "response.output_text.delta",
                item_id="m",
                output_index=0,
                content_index=0,
                delta="half an ans",
                logprobs=[],
            ),
            event("response.failed", response=failed),
            headers={"x-request-id": "req_mid"},
        )
    )
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    assert caught.value.action is action
    assert caught.value.code == code
    assert caught.value.request_id == "req_mid"


@pytest.mark.parametrize(
    "error_event",
    [
        event("error", code="subscription_sharing_usage_limit_exceeded", message="x"),
        event(
            "error",
            error={"code": "subscription_sharing_usage_limit_exceeded", "message": "x"},
        ),
    ],
)
def test_an_error_event_raises_plan_stop(serve, error_event):
    serve(sse(CREATED, error_event))
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert caught.value.action is PlanAction.STOP_USAGE


def test_an_incomplete_response_is_not_a_success(serve):
    details = {"reason": "content_filter"}
    serve(
        sse(
            CREATED,
            event(
                "response.incomplete",
                response=response_object("incomplete", incomplete_details=details),
            ),
        )
    )
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert caught.value.action is PlanAction.STOP_BAD_REQUEST
    assert caught.value.code == "response_incomplete"
    assert caught.value.param == "content_filter"


@both_modes
def test_stream_without_completed_raises(serve, mode):
    serve(sse(CREATED))
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    assert caught.value.action is PlanAction.RETRY
    assert caught.value.code == "stream_incomplete"


def test_a_done_marker_before_completed_is_not_a_success(serve):
    body = sse_bytes([CREATED]) + b"data: [DONE]\n\n"
    serve(lambda: httpx.Response(200, headers=SSE_HEADERS, content=body))
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert caught.value.code == "stream_incomplete"


def test_a_body_with_no_events_is_not_a_success(serve):
    serve(lambda: httpx.Response(200, json={"id": "resp_1"}))
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert caught.value.code == "stream_incomplete"


@both_modes
def test_a_lost_connection_mid_stream_is_a_retry(serve, mode):
    serve(dropped(CREATED))
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    assert caught.value.action is PlanAction.RETRY
    assert caught.value.request_id == "req_stream"


@both_modes
def test_a_refused_connection_is_a_retry_and_keeps_credentials(
    serve, mode, monkeypatch
):
    monkeypatch.setattr(
        auth, "clear", lambda: pytest.fail("network trouble cleared the sign-in")
    )
    serve(connect_refused)
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    assert caught.value.action is PlanAction.RETRY
    assert caught.value.code == "network_error"


def test_no_response_failure_clears_the_sign_in(serve, monkeypatch):
    """Only the sign-in flow erases tokens, and only after a terminal refusal."""
    monkeypatch.setattr(auth, "clear", lambda: pytest.fail("a 401 cleared the sign-in"))
    serve(http_error(401, structured("subscription_sharing_invalid_user")))
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert caught.value.action is PlanAction.RELOGIN


@pytest.mark.parametrize(
    ("failure", "action"),
    [
        (auth.NotSignedIn("Not signed in to ChatGPT."), PlanAction.RELOGIN),
        (
            auth.RefreshRejected("no longer valid", code="invalid_grant"),
            PlanAction.RELOGIN,
        ),
        (
            auth.RefreshUnavailable("Could not reach the sign-in server."),
            PlanAction.RETRY,
        ),
        (
            auth.PlanUsageDisabled("ChatGPT plan usage is not enabled."),
            PlanAction.STOP_INELIGIBLE,
        ),
        (
            auth.CredentialStoreError("Could not read chatgpt.json (OSError)."),
            PlanAction.STOP_BAD_REQUEST,
        ),
        (auth.LoginFailed("The sign-in failed."), PlanAction.RELOGIN),
    ],
    ids=lambda value: type(value).__name__,
)
@both_modes
def test_sign_in_trouble_maps_to_an_action(serve, monkeypatch, mode, failure, action):
    def refuse():
        raise failure

    server = serve(sse(*text_events()))
    monkeypatch.setattr(plan, "get_access_token", refuse)
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    assert caught.value.action is action
    assert str(failure) in str(caught.value)
    assert server.requests == []


def test_disabled_plan_usage_says_what_to_do(serve, monkeypatch):
    def refuse():
        raise auth.PlanUsageDisabled("ChatGPT plan usage is not enabled.")

    serve(sse(*text_events()))
    monkeypatch.setattr(plan, "get_access_token", refuse)
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert "/login" in str(caught.value)
    assert "openai_auth_mode api_key" in str(caught.value)


def test_an_async_turn_fetches_the_token_off_the_event_loop(serve, monkeypatch):
    loop_thread = threading.get_ident()
    seen: list[int] = []

    def fetch() -> str:
        seen.append(threading.get_ident())
        return TOKEN

    serve(sse(*text_events()))
    monkeypatch.setattr(plan, "get_access_token", fetch)

    async def turn() -> None:
        nonlocal loop_thread
        loop_thread = threading.get_ident()
        await build_model("gpt-6.1-sol").ainvoke("hi")

    asyncio.run(turn())
    assert seen
    assert all(ident != loop_thread for ident in seen)


@pytest.mark.parametrize(
    "reply",
    [
        http_error(400, structured("context_length_exceeded")),
        sse(
            CREATED,
            event(
                "response.failed",
                response=response_object(
                    "failed", error={"code": "context_length_exceeded", "message": "x"}
                ),
            ),
        ),
    ],
    ids=["http", "event"],
)
def test_a_context_overflow_still_reaches_compaction(serve, reply):
    """Compacting history on the same model is not a fallback."""
    serve(reply)
    with pytest.raises(ContextOverflowError):
        build_model("gpt-6.1-sol").invoke("hi")


def test_no_fallback_on_plan_stop(serve, monkeypatch, signed_in):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fallback-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback-key")
    monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
    server = serve(
        http_error(429, structured("subscription_sharing_usage_limit_exceeded"))
    )

    def other_route(*args: Any, **kwargs: Any) -> None:
        pytest.fail("a plan error reached another model or billing path")

    monkeypatch.setattr("metasci_scribe.llm.models.init_chat_model", other_route)
    model = get_chat_model("gpt-6.1-sol", provider="openai")
    with pytest.raises(PlanStop):
        model.invoke("hi")
    assert len(server.requests) == 1
    assert server.requests[0].headers["authorization"] == f"Bearer {TOKEN}"


# ------------------------------------------------------------------- token hygiene


def _chain(exc: BaseException | None) -> list[BaseException]:
    found: list[BaseException] = []
    pending = [exc]
    while pending:
        current = pending.pop()
        if current is None or current in found:
            continue
        found.append(current)
        pending += [current.__cause__, current.__context__]
    return found


def _assert_no_token(exc: BaseException) -> None:
    for item in _chain(exc):
        text = " ".join(
            [str(item), repr(item), repr(vars(item)), repr(getattr(item, "args", ""))]
        )
        assert TOKEN not in text


@both_modes
@pytest.mark.parametrize(
    "reply",
    [
        http_error(401, {"error": {"message": f"Bad token {TOKEN}", "code": None}}),
        http_error(403, {"detail": f"token {TOKEN} refused"}),
        http_error(503, f"upstream said Bearer {TOKEN}"),
        http_error(400, structured(TOKEN, param=TOKEN)),
        sse(
            CREATED,
            event(
                "response.failed",
                response=response_object(
                    "failed", error={"code": "x", "message": f"echo {TOKEN}"}
                ),
            ),
        ),
        dropped(CREATED),
        connect_refused,
    ],
    ids=[
        "error-object",
        "detail",
        "text",
        "code-field",
        "failed-event",
        "drop",
        "refused",
    ],
)
def test_token_not_in_error_text(serve, caplog, mode, reply):
    serve(reply)
    caplog.set_level(logging.DEBUG)
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    _assert_no_token(caught.value)
    assert TOKEN not in caplog.text


def test_a_bearer_or_jwt_the_transport_did_not_send_is_also_scrubbed(serve):
    jwt_like = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlLXZhbHVl"
    serve(http_error(403, {"detail": f"saw {jwt_like} and Bearer other-secret-value"}))
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert jwt_like not in str(caught.value)
    assert "other-secret-value" not in str(caught.value)


def test_the_model_repr_holds_no_token(serve):
    serve(sse(*text_events()))
    model = build_model("gpt-6.1-sol")
    model.invoke("hi")
    assert TOKEN not in repr(model)


def test_the_request_id_and_code_are_logged(serve, caplog):
    serve(
        http_error(
            429,
            structured("subscription_sharing_usage_limit_exceeded"),
            request_id="req_logged",
        )
    )
    caplog.set_level(logging.WARNING, logger=plan.__name__)
    with pytest.raises(PlanStop):
        build_model("gpt-6.1-sol").invoke("hi")
    assert "req_logged" in caplog.text
    assert "subscription_sharing_usage_limit_exceeded" in caplog.text


# ----------------------------------------------------------------- only the route


@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("POST", "https://chatgpt.com/backend-api/codex/responses"),
        ("POST", "https://chatgpt.com/backend-api/conversation"),
        ("POST", "https://api.openai.com/backend-api/responses"),
        ("POST", "http://api.openai.com/v1/responses"),
        ("POST", "https://api.openai.com/v1/chat/completions"),
        ("GET", "https://api.openai.com/v1/responses"),
        ("POST", "https://api.openai.com/v1/models"),
    ],
)
@both_modes
def test_request_never_targets_backend_api(serve, monkeypatch, mode, method, url):
    fetched: list[str] = []
    monkeypatch.setattr(
        plan, "get_access_token", lambda: fetched.append("token") or TOKEN
    )
    server = serve(sse(*text_events()))
    model = build_model("gpt-6.1-sol")
    request = httpx.Request(method, url, content=b"{}")

    def send() -> httpx.Response:
        if mode == "async":
            return asyncio.run(model.http_async_client.send(request))
        return model.http_client.send(request)

    with pytest.raises(PlanStop) as caught:
        send()
    assert caught.value.action is PlanAction.STOP_BAD_REQUEST
    assert server.requests == []
    assert fetched == []


def test_build_model_pins_the_route_against_its_kwargs(serve, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.test/v1")
    server = serve(sse(*text_events()))
    model = build_model(
        "gpt-6.1-sol",
        base_url="https://chatgpt.com/backend-api",
        openai_api_base="https://chatgpt.com/backend-api",
        api_key="sk-caller",
        openai_api_key="sk-caller",
        use_responses_api=False,
        max_retries=5,
        store=True,
        streaming=False,
    )
    model.invoke("hi")
    assert isinstance(model, ChatOpenAI)
    assert model.max_retries == 0
    assert server.requests[0].url.host == "api.openai.com"
    assert server.requests[0].url.path == "/v1/responses"


def test_a_proxy_setting_for_the_key_route_does_not_break_the_build(serve, monkeypatch):
    monkeypatch.setenv("OPENAI_PROXY", "http://proxy.example.test:3128")
    serve(sse(*text_events()))
    build_model("gpt-6.1-sol")


# -------------------------------------------------------------------- model catalog


def test_list_models_keeps_listed_models_in_server_order(serve):
    server = serve(
        lambda: httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "z-1", "display_name": "Z One", "visibility": "list"},
                    {"slug": "hidden", "display_name": "H", "visibility": "hide"},
                    {"slug": "a-1", "display_name": "A One", "visibility": "list"},
                    {"slug": "no-name", "visibility": "list"},
                    {"display_name": "No slug", "visibility": "list"},
                    "junk",
                ]
            },
        )
    )
    assert list_models() == [
        ("z-1", "Z One"),
        ("a-1", "A One"),
        ("no-name", "no-name"),
    ]
    request = server.requests[0]
    assert (request.method, str(request.url)) == (
        "GET",
        "https://api.openai.com/v1/models",
    )
    assert request.headers["authorization"] == f"Bearer {TOKEN}"


def test_list_models_stops_on_a_plan_error(serve):
    serve(http_error(403, structured("subscription_sharing_user_not_eligible")))
    with pytest.raises(PlanStop) as caught:
        list_models()
    assert caught.value.action is PlanAction.STOP_INELIGIBLE


def test_list_models_network_failure_is_a_retry(serve):
    serve(connect_refused)
    with pytest.raises(PlanStop) as caught:
        list_models()
    assert caught.value.action is PlanAction.RETRY
    _assert_no_token(caught.value)


@pytest.mark.parametrize("body", [{"data": []}, {"models": "nope"}, ["models"]])
def test_list_models_refuses_a_catalog_without_a_models_array(serve, body):
    serve(lambda: httpx.Response(200, json=body))
    with pytest.raises(PlanStop) as caught:
        list_models()
    assert caught.value.code == "invalid_model_catalog"


def test_list_models_refuses_a_body_that_is_not_json(serve):
    serve(lambda: httpx.Response(200, content=b"<html>"))
    with pytest.raises(PlanStop) as caught:
        list_models()
    assert caught.value.code == "invalid_model_catalog"


# ------------------------------------------------------------------- models.py wiring


def test_plan_mode_builds_the_plan_model(serve, monkeypatch, signed_in):
    monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unused")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.test/v1")
    monkeypatch.setenv("METASCI_SCRIBE_USE_RESPONSES_API", "false")
    server = serve(sse(*text_events()))
    model = get_chat_model("gpt-6.1-sol", provider="openai")
    assert isinstance(model, plan.ChatGPTPlanModel)
    assert model.use_responses_api is True
    assert model.profile["max_input_tokens"] > 0
    model.invoke("hi")
    assert server.requests[0].url.host == "api.openai.com"
    assert server.requests[0].headers["authorization"] == f"Bearer {TOKEN}"


def test_plan_mode_keeps_the_configured_reasoning_effort(serve, monkeypatch, signed_in):
    monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
    monkeypatch.setenv("METASCI_SCRIBE_REASONING_EFFORT", "medium")
    server = serve(sse(*text_events()))
    get_chat_model("gpt-6.1-sol", provider="openai").invoke("hi")
    assert server.bodies[0]["reasoning"]["effort"] == "medium"


def test_api_key_mode_stays_on_normal_behavior(monkeypatch):
    monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "api_key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("METASCI_SCRIBE_USE_RESPONSES_API", "false")
    model = get_chat_model("gpt-6-sol", provider="openai", temperature=0.5)
    assert type(model) is ChatOpenAI
    assert model.use_responses_api is False
    assert model.temperature == 0.5
    assert model.http_client is None


def test_anthropic_ignores_the_plan_mode(monkeypatch):
    monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    model = get_chat_model("claude-opus-5-5", provider="anthropic")
    assert not isinstance(model, plan.ChatGPTPlanModel)


# ---------------------------------------------------------------- retry and the caller


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    async def asleep(self, seconds: float) -> None:
        self.sleep(seconds)


def _agent(clock: _Clock, budget: float = 30.0) -> Any:
    """The middleware that sit around the model in ``create_cli_agent``, in order."""
    return create_agent(
        build_model("gpt-6.1-sol"),
        middleware=[
            ModelRetryMiddleware(
                wait_budget_seconds=budget,
                unattended=False,
                honor_run_key=False,
                clock=clock,
                sleep=clock.asleep,
                sync_sleep=clock.sleep,
                rng=lambda: 0.5,
            ),
            ErrorNormalizationMiddleware(),
            ContextOverflowMapperMiddleware(),
        ],
    )


def _turn(agent: Any, mode: str) -> dict[str, Any]:
    state = {"messages": [HumanMessage(content="hi")]}
    if mode == "async":
        return asyncio.run(agent.ainvoke(state))
    return agent.invoke(state)


@both_modes
@pytest.mark.parametrize(
    ("reply", "action"),
    [
        (
            http_error(429, structured("subscription_sharing_usage_limit_exceeded")),
            PlanAction.STOP_USAGE,
        ),
        (
            sse(
                CREATED,
                event(
                    "response.failed",
                    response=response_object(
                        "failed",
                        error={
                            "code": "subscription_sharing_usage_limit_exceeded",
                            "message": "too many tokens, 400 used",
                        },
                    ),
                ),
            ),
            PlanAction.STOP_USAGE,
        ),
        (
            http_error(403, structured("subscription_sharing_user_not_eligible")),
            PlanAction.STOP_INELIGIBLE,
        ),
        (
            http_error(400, structured("subscription_sharing_unsupported_capability")),
            PlanAction.STOP_BAD_REQUEST,
        ),
        (
            http_error(401, structured("subscription_sharing_invalid_user")),
            PlanAction.RELOGIN,
        ),
    ],
    ids=["usage-http", "usage-event", "ineligible", "bad-request", "relogin"],
)
def test_a_stop_is_never_retried_and_reaches_the_caller_typed(
    serve, mode, reply, action
):
    server = serve(reply)
    clock = _Clock()
    with pytest.raises(PlanStop) as caught:
        _turn(_agent(clock), mode)
    assert caught.value.action is action
    assert len(server.requests) == 1
    assert clock.waits == []


@both_modes
@pytest.mark.parametrize(
    "first",
    [
        http_error(503, structured("subscription_sharing_usage_unavailable")),
        http_error(503, {"detail": "routing unavailable"}),
        sse(
            CREATED,
            event(
                "response.failed",
                response=response_object(
                    "failed",
                    error={
                        "code": "subscription_sharing_usage_unavailable",
                        "message": "x",
                    },
                ),
            ),
        ),
        dropped(CREATED),
        connect_refused,
    ],
    ids=["usage-unavailable", "detail-503", "failed-event", "drop", "refused"],
)
def test_a_temporary_failure_is_retried(serve, mode, first):
    server = serve(first, sse(*text_events("recovered")))
    clock = _Clock()
    result = _turn(_agent(clock), mode)
    assert result["messages"][-1].text == "recovered"
    assert len(server.requests) == 2
    assert len(clock.waits) == 1


@both_modes
def test_retries_end_with_a_typed_error(serve, mode):
    server = serve(
        http_error(503, structured("subscription_sharing_usage_unavailable"))
    )
    clock = _Clock()
    with pytest.raises(PlanStop) as caught:
        _turn(_agent(clock, budget=10.0), mode)
    assert caught.value.action is PlanAction.RETRY
    assert 1 < len(server.requests) <= 5
    assert sum(clock.waits) <= 10.0


def test_a_refresh_outage_is_retried_and_keeps_the_sign_in(serve, monkeypatch):
    calls: list[str] = []

    def flaky() -> str:
        calls.append("try")
        if len(calls) == 1:
            raise auth.RefreshUnavailable("Could not reach the sign-in server.")
        return TOKEN

    monkeypatch.setattr(
        auth, "clear", lambda: pytest.fail("a refresh outage cleared the sign-in")
    )
    serve(sse(*text_events("back")))
    monkeypatch.setattr(plan, "get_access_token", flaky)
    clock = _Clock()
    result = _turn(_agent(clock), "sync")
    assert result["messages"][-1].text == "back"
    assert calls == ["try", "try"]


def test_retry_after_from_the_server_sets_the_wait(serve):
    serve(
        http_error(
            503,
            structured("subscription_sharing_usage_unavailable"),
            headers={"retry-after": "7"},
        ),
        sse(*text_events()),
    )
    clock = _Clock()
    _turn(_agent(clock), "sync")
    assert clock.waits == [7.0]


def test_a_plan_stop_with_a_budget_of_zero_still_reaches_the_caller(serve):
    server = serve(
        http_error(503, structured("subscription_sharing_usage_unavailable"))
    )
    with pytest.raises(PlanStop):
        _turn(_agent(_Clock(), budget=0.0), "sync")
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("action", "transient"),
    [
        (PlanAction.RETRY, True),
        (PlanAction.STOP_USAGE, False),
        (PlanAction.STOP_INELIGIBLE, False),
        (PlanAction.STOP_BAD_REQUEST, False),
        (PlanAction.RELOGIN, False),
    ],
)
def test_only_the_retry_action_counts_as_transient(action, transient):
    assert is_transient(PlanStop(action, "text", status_code=503)) is transient


def test_a_stop_keeps_its_type_with_tool_history_in_the_request(serve):
    serve(http_error(429, structured("subscription_sharing_usage_limit_exceeded")))
    history = [
        HumanMessage(content="hi"),
        AIMessage(
            content="",
            tool_calls=[{"name": "t", "args": {}, "id": "call_1", "type": "tool_call"}],
        ),
        ToolMessage(content="out", tool_call_id="call_1"),
    ]
    with pytest.raises(PlanStop):
        build_model("gpt-6.1-sol").invoke(history)


# ------------------------------------------------------------- the production stack


class MainAndSubAgent(FakeOpenAI):
    """Scripts the main agent and its sub-agent apart, by the tools they carry."""

    def __init__(self, main: list[Reply], sub: list[Reply]) -> None:
        super().__init__(sse(*text_events()))
        self.main, self.sub = main, sub
        self.main_calls = self.sub_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        tools = [
            tool["name"]
            for item in body["input"]
            if item.get("type") == "additional_tools"
            for tool in item["tools"]
        ]
        if "task" in tools:
            self.main_calls += 1
            return self.main[min(self.main_calls, len(self.main)) - 1]()
        self.sub_calls += 1
        return self.sub[min(self.sub_calls, len(self.sub)) - 1]()


def task_call() -> Reply:
    arguments = {
        "description": "draft the paragraph",
        "subagent_type": "general-purpose",
    }
    return sse(*tool_call_events("task", json.dumps(arguments)))


def production_agent(tmp_path: Path, monkeypatch: Any) -> Any:
    config = ScribeConfig(enable_async_subagents=False, model_retry_wait_seconds=0)
    monkeypatch.setattr("metasci_scribe.agent._config", config)
    return create_cli_agent(
        workspace=Workspace(tmp_path),
        config=config,
        chat_model=build_model("gpt-6.1-sol"),
        checkpointer=InMemorySaver(),
    )


def run_production(agent: Any, mode: str) -> dict[str, Any]:
    state = {"messages": [HumanMessage(content="write it")]}
    config = {"configurable": {"thread_id": "t"}}
    if mode == "async":
        return asyncio.run(agent.ainvoke(state, config))
    return agent.invoke(state, config)


@pytest.mark.usefixtures("no_mcp")
@both_modes
@pytest.mark.parametrize(
    ("reply", "action"),
    [
        (
            http_error(429, structured("subscription_sharing_usage_limit_exceeded")),
            PlanAction.STOP_USAGE,
        ),
        (
            sse(
                CREATED,
                event(
                    "response.failed",
                    response=response_object(
                        "failed",
                        error={
                            "code": "subscription_sharing_usage_limit_exceeded",
                            "message": "x",
                        },
                    ),
                ),
            ),
            PlanAction.STOP_USAGE,
        ),
        (
            http_error(403, structured("subscription_sharing_user_not_eligible")),
            PlanAction.STOP_INELIGIBLE,
        ),
        (
            http_error(401, structured("subscription_sharing_invalid_user")),
            PlanAction.RELOGIN,
        ),
    ],
    ids=["usage-http", "usage-event", "ineligible", "relogin"],
)
def test_a_sub_agent_stop_reaches_the_caller_and_ends_the_run(
    serve, tmp_path, monkeypatch, mode, reply, action
):
    server = MainAndSubAgent(main=[task_call(), sse(*text_events())], sub=[reply])
    serve(server=server)
    agent = production_agent(tmp_path, monkeypatch)
    with pytest.raises(PlanStop) as caught:
        run_production(agent, mode)
    assert caught.value.action is action
    assert (server.main_calls, server.sub_calls) == (1, 1)


@pytest.mark.usefixtures("no_mcp")
@both_modes
def test_a_sub_agent_retry_stays_a_tool_error_for_the_main_agent(
    serve, tmp_path, monkeypatch, mode
):
    server = MainAndSubAgent(
        main=[task_call(), sse(*text_events("carried on"))],
        sub=[http_error(503, structured("subscription_sharing_usage_unavailable"))],
    )
    serve(server=server)
    agent = production_agent(tmp_path, monkeypatch)
    result = run_production(agent, mode)
    assert result["messages"][-1].text == "carried on"
    tool_message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_message.status == "error"
    assert server.main_calls == 2


# ----------------------------------------------------------------- summarization


def long_history() -> list[Any]:
    history: list[Any] = []
    for turn in range(8):
        history.append(HumanMessage(content=f"question {turn} " + "word " * 200))
        history.append(AIMessage(content=f"answer {turn} " + "word " * 200))
    history.append(HumanMessage(content="and now?"))
    return history


def summarizing_agent(tmp_path: Path) -> Any:
    """The summarization middleware production mounts, on a small context window."""
    model = build_model("gpt-6.1-sol")
    model.profile = {"max_input_tokens": 2_000}
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    return create_agent(
        model, middleware=[create_per_run_summarization_middleware(model, backend)]
    )


@both_modes
@pytest.mark.parametrize(
    ("reply", "action"),
    [
        (
            http_error(429, structured("subscription_sharing_usage_limit_exceeded")),
            PlanAction.STOP_USAGE,
        ),
        (
            http_error(403, structured("subscription_sharing_user_not_eligible")),
            PlanAction.STOP_INELIGIBLE,
        ),
        (
            http_error(401, structured("subscription_sharing_invalid_user")),
            PlanAction.RELOGIN,
        ),
        (
            http_error(503, structured("subscription_sharing_usage_unavailable")),
            PlanAction.RETRY,
        ),
    ],
    ids=["usage", "ineligible", "relogin", "retry"],
)
def test_a_failed_summary_call_is_sent_once_and_stays_typed(
    serve, tmp_path, mode, reply, action
):
    """``model_retry`` is the one retry layer, here too."""
    server = serve(reply)
    agent = summarizing_agent(tmp_path)
    state = {"messages": long_history()}

    def turn() -> Any:
        if mode == "async":
            return asyncio.run(agent.ainvoke(state))
        return agent.invoke(state)

    with pytest.raises(PlanStop) as caught:
        turn()
    assert caught.value.action is action
    assert len(server.requests) == 1


@pytest.mark.parametrize("action", list(PlanAction))
def test_runnable_retries_leave_every_plan_failure_alone(action):
    assert default_retry_on(PlanStop(action, "text")) is False


def test_the_compact_tool_that_turns_errors_into_messages_is_not_mounted():
    """Its handlers turn any exception into a ToolMessage. Mounting it would hide a
    plan stop from the caller, so it must learn to re-raise one first."""
    name = "SummarizationToolMiddleware"
    assert name not in inspect.getsource(deepagents_graph)
    package = Path(metasci_scribe.__file__).parent
    mounts = [
        path
        for path in package.rglob("*.py")
        if name in path.read_text(encoding="utf-8")
    ]
    assert mounts == []


# ------------------------------------------------------------------------- proxies


class _ProxyListener:
    """A loopback proxy that records the first request line, then hangs up."""

    def __init__(self) -> None:
        self.server = socket.create_server(("127.0.0.1", 0))
        self.lines: list[str] = []
        self.url = f"http://127.0.0.1:{self.server.getsockname()[1]}"
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                connection, _ = self.server.accept()
            except OSError:
                return
            with connection:
                data = connection.recv(4096)
                self.lines.append(data.split(b"\r\n", 1)[0].decode())
                connection.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")

    def close(self) -> None:
        self.server.close()


@pytest.fixture
def proxy():
    listener = _ProxyListener()
    yield listener
    listener.close()


@both_modes
def test_a_turn_goes_through_the_configured_proxy(token, proxy, monkeypatch, mode):
    monkeypatch.setenv("HTTPS_PROXY", proxy.url)
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    assert proxy.lines == ["CONNECT api.openai.com:443 HTTP/1.1"]
    assert caught.value.action is PlanAction.RETRY
    assert "proxy" in str(caught.value)


def test_the_catalog_goes_through_the_configured_proxy(token, proxy, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", proxy.url)
    with pytest.raises(PlanStop):
        list_models()
    assert proxy.lines == ["CONNECT api.openai.com:443 HTTP/1.1"]


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"HTTPS_PROXY": "http://p.test:3128"}, "http://p.test:3128"),
        ({"https_proxy": "http://p.test:3128"}, "http://p.test:3128"),
        ({"ALL_PROXY": "http://p.test:3128"}, "http://p.test:3128"),
        ({"HTTPS_PROXY": "p.test:3128"}, "http://p.test:3128"),
        ({"HTTPS_PROXY": "http://p.test:3128", "NO_PROXY": "api.openai.com"}, None),
        ({"HTTPS_PROXY": "http://p.test:3128", "NO_PROXY": ".openai.com"}, None),
        ({"HTTPS_PROXY": "http://p.test:3128", "NO_PROXY": "*"}, None),
    ],
)
def test_the_proxy_comes_from_the_environment(monkeypatch, env, expected):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert plan._proxy_url() == expected


def test_a_proxy_for_plain_http_does_not_apply(monkeypatch):
    monkeypatch.setattr(plan, "getproxies", lambda: {"http": "http://p.test:3128"})
    assert plan._proxy_url() is None


def test_no_proxy_settings_mean_a_direct_connection(monkeypatch):
    monkeypatch.setattr(plan, "getproxies", dict)
    assert plan._proxy_url() is None


# ---------------------------------------------------------------- every odd ending


def test_an_error_event_without_a_type_raises_plan_stop(serve):
    untyped = {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}
    body = sse_bytes([CREATED]) + f"data: {json.dumps(untyped)}\n\n".encode()
    serve(lambda: httpx.Response(200, headers=SSE_HEADERS, content=body))
    with pytest.raises(PlanStop) as caught:
        build_model("gpt-6.1-sol").invoke("hi")
    assert caught.value.action is PlanAction.STOP_USAGE


@pytest.mark.parametrize("code", [["a", "list"], {"a": "dict"}, 503, True])
def test_a_code_that_is_not_text_is_no_code(code):
    body = {"error": {"code": code, "type": code}}
    assert classify_error(400, body) is PlanAction.STOP_BAD_REQUEST
    assert classify_error(200, body) is PlanAction.STOP_BAD_REQUEST
    assert classify_error(503, body) is PlanAction.RETRY


@both_modes
def test_a_code_that_is_not_text_still_ends_as_plan_stop(serve, mode):
    serve(http_error(400, {"error": {"code": ["x"], "message": "odd"}}))
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    assert caught.value.code is None


class _ClosableDrop(_Dropped):
    def __init__(self) -> None:
        super().__init__([b'{"error"'])
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


@both_modes
@pytest.mark.parametrize(
    ("status", "action"),
    [
        (429, PlanAction.RETRY),
        (401, PlanAction.RELOGIN),
        (403, PlanAction.STOP_INELIGIBLE),
    ],
)
def test_an_error_body_cut_off_keeps_the_status_and_closes_the_response(
    serve, mode, status, action
):
    stream = _ClosableDrop()
    serve(
        lambda: httpx.Response(
            status, headers={"x-request-id": "req_cut"}, stream=stream
        )
    )
    with pytest.raises(PlanStop) as caught:
        run(build_model("gpt-6.1-sol"), mode)
    assert caught.value.action is action
    assert caught.value.status_code == status
    assert caught.value.request_id == "req_cut"
    assert stream.closed


def test_a_catalog_cut_off_mid_body_is_a_typed_retry(serve):
    serve(lambda: httpx.Response(200, stream=_Dropped([b'{"models": ['])))
    with pytest.raises(PlanStop) as caught:
        list_models()
    assert caught.value.action is PlanAction.RETRY
    assert caught.value.code == "network_error"
    _assert_no_token(caught.value)


def test_a_display_name_that_is_not_text_falls_back_to_the_slug(serve):
    models = [{"slug": "a", "display_name": ["x"], "visibility": "list"}]
    serve(lambda: httpx.Response(200, json={"models": models}))
    assert list_models() == [("a", "a")]


# ------------------------------------------------------------------ replay whitelist


@pytest.mark.parametrize(
    "item",
    [
        {"type": "web_search_call", "id": "ws_1", "status": "completed"},
        {"type": "image_generation_call", "id": "ig_1", "result": "b64"},
        {"type": "item_reference", "id": "msg_1"},
        {"type": "compaction", "encrypted_content": "blob"},
        {"type": "mcp_call", "id": "mcp_1"},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        "a bare string",
    ],
    ids=lambda item: item["type"] if isinstance(item, dict) else "string",
)
def test_replay_drops_everything_outside_the_whitelist(item):
    user = {"type": "message", "role": "user", "content": "hi"}
    assert normalize_payload({"input": [item, user]})["input"] == [user]


def test_replay_keeps_a_message_that_names_no_type():
    item = {"role": "user", "content": "hi"}
    assert normalize_payload({"input": [item]})["input"] == [item]
