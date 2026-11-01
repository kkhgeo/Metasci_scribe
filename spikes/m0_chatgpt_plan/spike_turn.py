"""M0 spike: run a deepagents agent over the ChatGPT plan (Responses API).

    python spike_turn.py --tool-format namespace
    python spike_turn.py --tool-format additional

Needs a prior `python spike_login.py`. Runs two turns: turn 1 makes the agent
read sample.txt with a tool and report the sentinel code; turn 2 asks for the
code again from history, which forces replay of the tool-call items.

Requests go through a ChatOpenAI built on the public `http_client` /
`http_async_client` hooks. An httpx transport rewrites the request body to the
plan rules (design section 4), sets the Bearer token, and prints diagnostics:
request-body field names, SSE event types, tool calls and results, and on
failure the status, request ID, error code and full error body. Tokens and
prompt text are never printed.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import httpx

import spike_login

SPIKE_DIR = Path(__file__).resolve().parent
ALLOWED_HOST = "api.openai.com"
UNSUPPORTED_FIELDS = (
    "background", "conversation", "max_output_tokens", "max_tool_calls", "metadata",
    "moderation", "multi_agent", "prompt", "prompt_cache_retention", "safety_identifier",
    "temperature", "top_logprobs", "top_p", "truncation", "user",
    "previous_response_id", "max_tokens", "max_completion_tokens", "stop",
)
SENTINEL_CODE = "SENTINEL-KARST-7Q2X-ZEPHYR-4419"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- body rewrite


class Options:
    def __init__(
        self,
        tool_format: str = "namespace",
        namespace: str = "scribe",
        keep_item_ids: bool = False,
        include_reasoning: bool = False,
    ):
        if tool_format not in ("namespace", "additional"):
            raise ValueError("tool_format must be namespace or additional")
        self.tool_format = tool_format
        self.namespace = namespace
        self.keep_item_ids = keep_item_ids
        self.include_reasoning = include_reasoning


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for b in content or []:
        if isinstance(b, dict) and isinstance(b.get("text"), str):
            parts.append(b["text"])
    return "\n".join(parts)


def normalize_body(body: dict, opts: Options) -> dict:
    """Rewrite a langchain-built Responses body to the ChatGPT plan rules.

    Pure function: returns a new dict, leaves `body` untouched.
    """
    b = copy.deepcopy(body)
    for f in UNSUPPORTED_FIELDS:
        b.pop(f, None)
    b["store"] = False
    b["stream"] = True
    b.pop("stream_options", None)

    # input must be an array
    inp = b.get("input", [])
    if isinstance(inp, str):
        inp = [{"type": "message", "role": "user", "content": inp}]

    # system prompt: leading system/developer messages go to `instructions`;
    # explicit system items are rejected, so any later one becomes developer.
    instr_parts = [b["instructions"]] if b.get("instructions") else []
    items: list = []
    leading = True
    for it in inp:
        is_msg = isinstance(it, dict) and it.get("role") in ("system", "developer") and it.get("type", "message") == "message"
        if is_msg and leading:
            instr_parts.append(_text_of(it.get("content")))
            continue
        leading = False
        if is_msg or (isinstance(it, dict) and it.get("role") == "system"):
            it = {**it, "role": "developer"}
        items.append(it)
    if instr_parts:
        b["instructions"] = "\n\n".join(p for p in instr_parts if p)

    # replay hygiene: with store=false server-side item ids cannot be resolved
    cleaned: list = []
    for it in items:
        if isinstance(it, dict):
            t = it.get("type")
            if t == "reasoning":
                if not it.get("encrypted_content"):
                    continue  # unreplayable without storage
                it = {k: v for k, v in it.items() if k != "status"}
                if not opts.keep_item_ids:
                    it.pop("id", None)
            elif not opts.keep_item_ids:
                it = {k: v for k, v in it.items() if k != "id" or t in ("additional_tools",)}
                it.pop("status", None)
        cleaned.append(it)
    items = cleaned

    # tools
    tools = b.pop("tools", None) or []
    fn_tools = [t for t in tools if t.get("type") in ("function", "custom")]
    other = [t for t in tools if t.get("type") not in ("function", "custom", "namespace")]
    ready_namespaces = [t for t in tools if t.get("type") == "namespace"]
    names = {t["name"] for t in fn_tools if "name" in t}
    new_tools: list = list(other) + ready_namespaces
    if fn_tools:
        if opts.tool_format == "namespace":
            new_tools.append({
                "type": "namespace",
                "name": opts.namespace,
                "description": "Tools the Metasci_scribe agent can call.",
                "tools": fn_tools,
            })
            for it in items:
                if isinstance(it, dict) and it.get("type") in ("function_call", "custom_tool_call") \
                        and it.get("name") in names and not it.get("namespace"):
                    it["namespace"] = opts.namespace
        else:  # additional: tools travel as an input item, before the conversation
            items = [{"type": "additional_tools", "role": "developer", "tools": fn_tools}] + items
    if new_tools:
        b["tools"] = new_tools
    else:
        b.pop("tool_choice", None)
        b.pop("parallel_tool_calls", None)

    if opts.include_reasoning:
        inc = list(b.get("include") or [])
        if "reasoning.encrypted_content" not in inc:
            inc.append("reasoning.encrypted_content")
        b["include"] = inc

    b["input"] = items
    return b


def describe_body(b: dict) -> str:
    items = b.get("input", [])
    types = {}
    for it in items:
        k = it.get("type") or ("message:" + str(it.get("role"))) if isinstance(it, dict) else "?"
        types[k] = types.get(k, 0) + 1
    tool_kinds = [t.get("type") for t in b.get("tools", [])]
    return (
        f"fields={sorted(b)} input_items={types} top_level_tools={tool_kinds} "
        f"model={b.get('model')} instructions={'yes' if b.get('instructions') else 'no'}"
    )


# --------------------------------------------------------------------------- SSE recording


class Recorder:
    """Collects what happened on the wire; contains no tokens or prompt text."""

    def __init__(self, dump_dir: Path | None = None):
        self.requests = 0
        self.completed = 0
        self.failed: list[dict] = []
        self.http_errors: list[dict] = []
        self.event_types: list[str] = []
        self.output_items: list[dict] = []
        self.dump_dir = dump_dir

    def note_event(self, data: dict) -> None:
        t = data.get("type", "?")
        if not self.event_types or self.event_types[-1] != t:
            self.event_types.append(t)
        if t == "response.completed":
            self.completed += 1
            for it in (data.get("response") or {}).get("output", []) or []:
                self.output_items.append({k: it.get(k) for k in ("type", "name", "namespace", "role") if k in it})
            log(f"  [event] response.completed output={[o.get('type') for o in (data.get('response') or {}).get('output', []) or []]}")
        elif t in ("response.failed", "response.incomplete", "error"):
            resp = data.get("response") or {}
            err = resp.get("error") or data.get("error") or {k: data.get(k) for k in ("code", "message", "param")}
            self.failed.append({"type": t, "error": err, "incomplete": resp.get("incomplete_details")})
            log(f"  [event] {t} error={json.dumps(err)} incomplete={resp.get('incomplete_details')}")
        elif t == "response.output_item.done":
            it = data.get("item") or {}
            if it.get("type") in ("function_call", "custom_tool_call"):
                log(f"  [event] function_call item: name={it.get('name')} namespace={it.get('namespace')} fields={sorted(it)}")
            elif it.get("type") == "reasoning":
                log(f"  [event] reasoning item: has_encrypted_content={bool(it.get('encrypted_content'))} fields={sorted(it)}")


class _SSEParser:
    def __init__(self, rec: Recorder):
        self.rec = rec
        self.buf = b""
        self.counts: dict[str, int] = {}

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            if isinstance(data, dict):
                self.counts[data.get("type", "?")] = self.counts.get(data.get("type", "?"), 0) + 1
                self.rec.note_event(data)

    def finish(self) -> None:
        log(f"  [stream] event type counts: {json.dumps(self.counts, sort_keys=True)}")


class _TeeSync(httpx.SyncByteStream):
    def __init__(self, inner, parser: _SSEParser):
        self.inner, self.parser = inner, parser

    def __iter__(self):
        for chunk in self.inner:
            self.parser.feed(chunk)
            yield chunk
        self.parser.finish()

    def close(self):
        if hasattr(self.inner, "close"):
            self.inner.close()


class _TeeAsync(httpx.AsyncByteStream):
    def __init__(self, inner, parser: _SSEParser):
        self.inner, self.parser = inner, parser

    async def __aiter__(self):
        async for chunk in self.inner:
            self.parser.feed(chunk)
            yield chunk
        self.parser.finish()

    async def aclose(self):
        if hasattr(self.inner, "aclose"):
            await self.inner.aclose()


# --------------------------------------------------------------------------- transports


class _Core:
    def __init__(self, token_provider, opts: Options, rec: Recorder):
        self.token_provider, self.opts, self.rec = token_provider, opts, rec

    def prepare(self, request: httpx.Request) -> httpx.Request:
        url = request.url
        if url.host != ALLOWED_HOST or not url.path.startswith("/v1/"):
            raise RuntimeError(f"refusing request to {url.host}{url.path}; only {ALLOWED_HOST}/v1 is allowed")
        headers = httpx.Headers(request.headers)
        headers["authorization"] = f"Bearer {self.token_provider()}"
        headers["accept-encoding"] = "identity"  # lets the tee read the SSE bytes as-is
        content = request.content if request.method != "GET" else b""
        if request.method == "POST" and url.path.rstrip("/").endswith("/responses"):
            original = json.loads(content)
            new = normalize_body(original, self.opts)
            self.rec.requests += 1
            n = self.rec.requests
            log(f"[request {n}] POST {url.path}")
            log(f"  [body in ] fields={sorted(original)}")
            log(f"  [body out] {describe_body(new)}")
            if self.rec.dump_dir:
                self.rec.dump_dir.mkdir(parents=True, exist_ok=True)
                (self.rec.dump_dir / f"request_{n}.json").write_text(json.dumps(new, indent=2), encoding="utf-8")
            content = json.dumps(new).encode("utf-8")
            headers["content-length"] = str(len(content))
            headers["content-type"] = "application/json"
        return httpx.Request(request.method, url, headers=headers, content=content, extensions=request.extensions)

    def report_error(self, request: httpx.Request, status: int, headers: httpx.Headers, body: bytes) -> None:
        rid = headers.get("x-request-id") or headers.get("openai-request-id")
        text = body.decode("utf-8", "replace")
        code = None
        try:
            parsed = json.loads(text)
            err = parsed.get("error") if isinstance(parsed, dict) else None
            code = err.get("code") if isinstance(err, dict) else None
            param = err.get("param") if isinstance(err, dict) else None
        except ValueError:
            param = None
        self.rec.http_errors.append({"status": status, "code": code, "param": param, "request_id": rid})
        log(f"  [HTTP ERROR] status={status} code={code} param={param} request_id={rid}")
        log(f"  [HTTP ERROR] full body: {text}")

    def wrap_ok(self, response: httpx.Response, tee_cls, parser: _SSEParser) -> httpx.Response:
        rid = response.headers.get("x-request-id")
        log(f"  [HTTP {response.status_code}] request_id={rid} content-type={response.headers.get('content-type')}")
        return httpx.Response(
            response.status_code, headers=response.headers,
            stream=tee_cls(response.stream, parser), extensions=response.extensions,
        )

    def rebuild(self, response: httpx.Response, body: bytes) -> httpx.Response:
        headers = [(k, v) for k, v in response.headers.multi_items()
                   if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")]
        return httpx.Response(response.status_code, headers=headers, content=body, extensions=response.extensions)


class RewriteTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport, token_provider, opts: Options, rec: Recorder):
        self.inner, self.core = inner, _Core(token_provider, opts, rec)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        req = self.core.prepare(request)
        resp = self.inner.handle_request(req)
        if req.method == "POST" and req.url.path.rstrip("/").endswith("/responses"):
            if resp.status_code >= 400:
                body = b"".join(resp.stream)
                resp.close()
                self.core.report_error(req, resp.status_code, resp.headers, body)
                return self.core.rebuild(resp, body)
            return self.core.wrap_ok(resp, _TeeSync, _SSEParser(self.core.rec))
        return resp

    def close(self) -> None:
        self.inner.close()


class AsyncRewriteTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport, token_provider, opts: Options, rec: Recorder):
        self.inner, self.core = inner, _Core(token_provider, opts, rec)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        req = self.core.prepare(request)
        resp = await self.inner.handle_async_request(req)
        if req.method == "POST" and req.url.path.rstrip("/").endswith("/responses"):
            if resp.status_code >= 400:
                body = b"".join([c async for c in resp.stream])
                await resp.aclose()
                self.core.report_error(req, resp.status_code, resp.headers, body)
                return self.core.rebuild(resp, body)
            return self.core.wrap_ok(resp, _TeeAsync, _SSEParser(self.core.rec))
        return resp

    async def aclose(self) -> None:
        await self.inner.aclose()


# --------------------------------------------------------------------------- agent


def build_model(model_slug: str, token_provider, opts: Options, rec: Recorder,
                inner: httpx.BaseTransport | None = None, inner_async: httpx.AsyncBaseTransport | None = None):
    from langchain_openai import ChatOpenAI

    sync_client = httpx.Client(
        transport=RewriteTransport(inner or httpx.HTTPTransport(), token_provider, opts, rec),
        timeout=httpx.Timeout(300.0, connect=30.0),
    )
    async_client = httpx.AsyncClient(
        transport=AsyncRewriteTransport(inner_async or httpx.AsyncHTTPTransport(), token_provider, opts, rec),
        timeout=httpx.Timeout(300.0, connect=30.0),
    )
    return ChatOpenAI(
        model=model_slug,
        base_url=spike_login.API_BASE,
        api_key="placeholder-replaced-by-transport",
        use_responses_api=True,
        streaming=True,
        store=False,
        max_retries=0,
        http_client=sync_client,
        http_async_client=async_client,
    )


def make_read_tool():
    from langchain_core.tools import tool

    @tool
    def read_sample_file(path: str) -> str:
        """Read a UTF-8 text file from the spike directory and return its full text. Example path: sample.txt"""
        p = (SPIKE_DIR / path).resolve()
        if SPIKE_DIR not in p.parents or not p.is_file():
            return f"error: {path!r} is not a readable file in the spike directory"
        return p.read_text(encoding="utf-8")[:60000]

    return read_sample_file


TURN1 = (
    "Use the read_sample_file tool to read sample.txt. Then summarise it in three sentences and "
    "quote the audit code that appears in its final sentence, exactly."
)
TURN2 = (
    "Without calling any tool, repeat the audit code from the final sentence of the file, "
    "and name the first topic heading in the file."
)


def show_new_messages(msgs: list, start: int) -> dict:
    stats = {"tool_calls": [], "tool_results": [], "final": ""}
    for m in msgs[start:]:
        kind = type(m).__name__
        if kind == "AIMessage":
            for tc in m.tool_calls:
                log(f"  [tool call] {tc['name']} args={json.dumps(tc['args'])[:200]}")
                stats["tool_calls"].append(tc["name"])
            text = m.text if hasattr(m, "text") and isinstance(m.text, str) else str(m.content)
            if text.strip():
                stats["final"] = text
        elif kind == "ToolMessage":
            body = m.content if isinstance(m.content, str) else json.dumps(m.content)
            log(f"  [tool result] {m.name}: {len(body)} chars, sentinel_present={SENTINEL_CODE in body}")
            stats["tool_results"].append(body)
    log(f"  [assistant] {stats['final'][:600]!r}")
    return stats


def run_spike(model_slug: str, token_provider, opts: Options, rec: Recorder, inner=None, inner_async=None) -> dict:
    from deepagents import create_deep_agent

    model = build_model(model_slug, token_provider, opts, rec, inner, inner_async)
    agent = create_deep_agent(
        model=model,
        tools=[make_read_tool()],
        system_prompt="You are a careful research assistant. Use tools when asked.",
    )
    cfg = {"recursion_limit": 40}
    results = {}

    log("=== turn 1 ===")
    out1 = agent.invoke({"messages": [{"role": "user", "content": TURN1}]}, config=cfg)
    msgs = out1["messages"]
    s1 = show_new_messages(msgs, 0)
    log("=== turn 2 (history replayed) ===")
    n_before = len(msgs)
    out2 = agent.invoke({"messages": msgs + [{"role": "user", "content": TURN2}]}, config=cfg)
    s2 = show_new_messages(out2["messages"], n_before + 1)

    results["tool_read_proven"] = any(SENTINEL_CODE in r for r in s1["tool_results"])
    results["turn1_has_sentinel"] = SENTINEL_CODE in s1["final"]
    results["turn2_has_sentinel"] = SENTINEL_CODE in s2["final"]
    results["completed_events"] = rec.completed
    results["requests"] = rec.requests
    results["no_failures"] = not rec.failed and not rec.http_errors
    results["passed"] = (
        results["tool_read_proven"] and results["turn1_has_sentinel"] and results["turn2_has_sentinel"]
        and results["no_failures"] and rec.completed >= 3 and rec.completed == rec.requests
    )
    return results


def pick_model(token: str) -> str | None:
    status, models, rid, raw = spike_login.list_models(token)
    if status != 200:
        log(f"GET /v1/models -> HTTP {status} request_id={rid}\nerror body: {raw}")
        return None
    slugs = spike_login.visible_slugs(models)
    log(f"visible models: {slugs}")
    return slugs[0] if slugs else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool-format", choices=["namespace", "additional"], required=True)
    ap.add_argument("--model", help="model slug; default: first visible slug from GET /v1/models")
    ap.add_argument("--namespace", default="scribe", help="namespace name for --tool-format namespace")
    ap.add_argument("--keep-item-ids", action="store_true", help="do not strip item ids from replayed input")
    ap.add_argument("--include-reasoning", action="store_true", help="send include=[reasoning.encrypted_content]")
    ap.add_argument("--dump-dir", type=Path, help="write each rewritten request body here (contains prompt text)")
    args = ap.parse_args(argv)

    opts = Options(args.tool_format, args.namespace, args.keep_item_ids, args.include_reasoning)
    rec = Recorder(args.dump_dir)
    try:
        token = spike_login.get_access_token()
        slug = args.model or pick_model(token)
        if not slug:
            log("SPIKE RESULT: FAIL (no model slug)")
            return 1
        log(f"model slug: {slug}  tool format: {args.tool_format}")
        t0 = time.time()
        res = run_spike(slug, spike_login.get_access_token, opts, rec)
    except spike_login.AuthError as exc:
        log(f"auth problem: {exc}")
        log("SPIKE RESULT: FAIL (auth)")
        return 1
    except Exception as exc:  # noqa: BLE001 - spike: show everything
        log(f"EXCEPTION {type(exc).__name__}: {exc}")
        body = getattr(exc, "body", None)
        resp = getattr(exc, "response", None)
        if resp is not None:
            log(f"  status={getattr(resp, 'status_code', None)} request_id={resp.headers.get('x-request-id')}")
        if body is not None:
            log(f"  error body: {body!r}")
        log(f"  wire summary: requests={rec.requests} completed={rec.completed} failed={rec.failed} http_errors={rec.http_errors}")
        log("SPIKE RESULT: FAIL (exception)")
        return 1

    log("=== summary ===")
    log(f"elapsed {time.time() - t0:.1f}s  model={slug}  tool_format={args.tool_format}")
    log(f"checks: {json.dumps(res)}")
    log(f"event type sequence (collapsed): {rec.event_types[:40]}")
    log(f"response output items seen: {rec.output_items}")
    log(f"SPIKE RESULT: {'PASS' if res['passed'] else 'FAIL'}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
