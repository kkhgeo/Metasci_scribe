"""Offline checks for the M0 spike. No network, no real credentials, no browser.

    C:\\dev\\venvs\\metasci-spike\\Scripts\\python.exe -m pytest -q test_spike.py
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import urllib.parse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

import spike_login as L
import spike_turn as T


# --------------------------------------------------------------------------- login pieces


def test_pkce_s256_unpadded():
    verifier, challenge = L.make_pkce()
    assert 43 <= len(verifier) <= 128
    assert "=" not in verifier and "=" not in challenge
    expect = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert challenge == expect
    assert L.make_pkce()[0] != verifier


def _qs(url):
    p = urllib.parse.urlparse(url)
    return p, {k: v[0] for k, v in urllib.parse.parse_qs(p.query).items()}


def test_authorize_url_first_registration():
    url = L.build_authorize_url(
        client_id=L.REGISTRATION_CLIENT_ID, redirect_uri="http://127.0.0.1:1455/auth/callback",
        state="S", nonce="N", code_challenge="C", ext_agent_host_id="urn:uuid:abc")
    p, q = _qs(url)
    assert f"{p.scheme}://{p.netloc}{p.path}" == "https://auth.openai.com/api/accounts/authorize"
    assert q["client_id"] == "dynamic_agent_client"
    assert q["agent_name_hint"] == "Metasci_scribe"
    assert q["ext_agent_host_id"] == "urn:uuid:abc"
    assert q["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"
    assert q["response_type"] == "code" and q["resource"] == "https://api.openai.com/v1"
    assert q["scope"] == "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
    assert q["code_challenge_method"] == "S256" and q["code_challenge"] == "C"
    assert q["state"] == "S" and q["nonce"] == "N"
    assert "localhost" not in url
    assert "127.0.0.1" in urllib.parse.unquote(url)


def test_authorize_url_relogin_omits_name_and_adds_hints():
    url = L.build_authorize_url(
        client_id="oaiapp_x", redirect_uri="http://127.0.0.1:54321/auth/callback", state="S", nonce="N",
        code_challenge="C", ext_agent_host_id="urn:uuid:abc", id_token_hint="IDT", login_hint="a@b.c")
    _, q = _qs(url)
    assert q["client_id"] == "oaiapp_x"
    assert "agent_name_hint" not in q
    assert q["id_token_hint"] == "IDT" and q["login_hint"] == "a@b.c"
    assert "login_hint=a%40b.c" in url  # percent-encoded


def test_callback_state_and_client_id_rules():
    ok = {"code": "c", "state": "S", "client_id": "oaiapp_new"}
    assert L.check_callback(ok, "S", L.REGISTRATION_CLIENT_ID) == ("c", "oaiapp_new")
    with pytest.raises(L.AuthError, match="state"):
        L.check_callback({**ok, "state": "other"}, "S", L.REGISTRATION_CLIENT_ID)
    with pytest.raises(L.AuthError, match="state"):  # state is checked before the error param
        L.check_callback({"error": "access_denied", "state": "bad"}, "S", "oaiapp_new")
    with pytest.raises(L.AuthError) as ei:
        L.check_callback({"error": "access_denied", "state": "S"}, "S", "oaiapp_new")
    assert ei.value.code == "access_denied"
    with pytest.raises(L.AuthError, match="no issued client_id"):
        L.check_callback({"code": "c", "state": "S"}, "S", L.REGISTRATION_CLIENT_ID)
    with pytest.raises(L.AuthError, match="no issued client_id"):
        L.check_callback({"code": "c", "state": "S", "client_id": "dynamic_agent_client"}, "S", L.REGISTRATION_CLIENT_ID)
    # reauth: callback may omit client_id, but must not change it
    assert L.check_callback({"code": "c", "state": "S"}, "S", "oaiapp_old") == ("c", "oaiapp_old")
    with pytest.raises(L.AuthError, match="differs"):
        L.check_callback({"code": "c", "state": "S", "client_id": "oaiapp_other"}, "S", "oaiapp_old")


def test_loopback_listener_receives_callback():
    srv, uri = L.start_callback_server()
    assert uri.startswith("http://127.0.0.1:") and uri.endswith("/auth/callback")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        assert httpx.get(uri.replace("/auth/callback", "/other")).status_code == 404
        r = httpx.get(uri + "?code=abc&state=xyz&client_id=oaiapp_1")
        assert r.status_code == 200
        assert srv.done.wait(5)
        assert srv.result == {"code": "abc", "state": "xyz", "client_id": "oaiapp_1"}
    finally:
        srv.shutdown()
        srv.server_close()


# --------------------------------------------------------------------------- ID token


@pytest.fixture(scope="module")
def keys():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(priv.public_key()))
    jwk.update(kid="k1", alg="RS256", use="sig")
    return priv, other, {"keys": [jwk]}


def _idt(priv, **over):
    now = int(time.time())
    claims = {"iss": "https://auth.openai.com", "aud": ["oaiapp_1"], "sub": "user-1", "iat": now,
              "exp": now + 600, "nonce": "N1", "email": "x@example.com"}
    claims.update(over)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, priv, algorithm="RS256", headers={"kid": "k1"})


def _v(tok, jwks, **kw):
    args = dict(jwks=jwks, issuer="https://auth.openai.com", client_id="oaiapp_1", nonce="N1")
    args.update(kw)
    return L.validate_id_token(tok, **args)


def test_id_token_valid(keys):
    priv, _, jwks = keys
    assert _v(_idt(priv), jwks)["sub"] == "user-1"


@pytest.mark.parametrize("over,kw,msg", [
    ({"nonce": "wrong"}, {}, "nonce"),
    ({"aud": ["someone-else"]}, {}, "invalid"),
    ({"iss": "https://evil.example"}, {}, "invalid"),
    ({"exp": 1000, "iat": 900}, {}, "invalid"),
    ({"sub": None}, {}, "invalid"),
])
def test_id_token_rejections(keys, over, kw, msg):
    priv, _, jwks = keys
    with pytest.raises(L.AuthError, match=msg):
        _v(_idt(priv, **over), jwks, **kw)


def test_id_token_bad_signature(keys):
    _, other, jwks = keys
    with pytest.raises(L.AuthError, match="invalid"):
        _v(_idt(other), jwks)


def test_id_token_alg_none_rejected(keys):
    _, _, jwks = keys
    tok = jwt.encode({"sub": "u", "aud": "oaiapp_1", "iss": "https://auth.openai.com", "exp": int(time.time()) + 99,
                      "iat": int(time.time()), "nonce": "N1"}, None, algorithm="none")
    with pytest.raises(L.AuthError, match="alg"):
        _v(tok, jwks)


# --------------------------------------------------------------------------- store / refresh


def test_save_token_and_refresh_roundtrip(tmp_path):
    path = tmp_path / "t" / "spike_token.json"
    rec = {"client_id": "oaiapp_1", "refresh_token": "R0", "access_token": "A0", "scopes": [L.DIRECT_SCOPE],
           "expires_at": time.time() - 10}
    L.save_token(rec, path)
    assert not list(path.parent.glob("*.tmp"))
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["form"] = dict(urllib.parse.parse_qsl(request.content.decode()))
        return httpx.Response(200, json={"access_token": "A1", "refresh_token": "R1", "expires_in": 3600,
                                         "scope": "openid chatgpt.tokens.use.direct", "token_type": "Bearer"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        assert L.get_access_token(path, client=c) == "A1"
    assert seen["url"] == L.TOKEN_URL
    assert seen["form"] == {"grant_type": "refresh_token", "client_id": "oaiapp_1", "refresh_token": "R0",
                            "resource": "https://api.openai.com/v1"}
    saved = L.load_token(path)
    assert saved["refresh_token"] == "R1" and saved["access_token"] == "A1" and saved["expires_at"] > time.time() + 3000
    # still fresh: no second call
    assert L.get_access_token(path, client=httpx.Client(transport=httpx.MockTransport(lambda r: 1 / 0))) == "A1"


def test_refresh_errors(tmp_path):
    rec = {"client_id": "oaiapp_1", "refresh_token": "R0", "access_token": "A0"}

    def reuse(request):
        return httpx.Response(400, json={"error": "refresh_token_reused"})

    with httpx.Client(transport=httpx.MockTransport(reuse)) as c, pytest.raises(L.AuthError) as ei:
        L.refresh_token(rec, c)
    assert ei.value.terminal and ei.value.code == "refresh_token_reused"

    def boom(request):
        raise httpx.ConnectError("down")

    with httpx.Client(transport=httpx.MockTransport(boom)) as c, pytest.raises(L.AuthError) as ei:
        L.refresh_token(rec, c)
    assert not ei.value.terminal  # network trouble must not mark credentials dead

    with pytest.raises(L.AuthError):
        L.refresh_token({"client_id": "dynamic_agent_client", "refresh_token": "x"})


def test_get_access_token_requires_direct_scope(tmp_path):
    path = tmp_path / "t.json"
    L.save_token({"client_id": "oaiapp_1", "scopes": ["openid"], "expires_at": time.time() + 9999, "access_token": "A"}, path)
    with pytest.raises(L.AuthError, match="plan usage is off"):
        L.get_access_token(path)


def test_visible_slugs():
    models = [{"slug": "a", "visibility": "list"}, {"slug": "b", "visibility": "hide"}, {"slug": "c"}]
    assert L.visible_slugs(models) == ["a", "c"]


# --------------------------------------------------------------------------- body normalization


FN = {"type": "function", "name": "read_sample_file", "description": "d",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}, "strict": False}


def _lc_body():
    return {
        "model": "m", "stream": False, "store": True, "temperature": 0.5, "top_p": 1, "max_output_tokens": 9,
        "user": "u", "metadata": {"a": 1}, "previous_response_id": "resp_1", "truncation": "auto",
        "tool_choice": "auto", "parallel_tool_calls": True, "tools": [FN, {"type": "function", "name": "ls", "parameters": {}}],
        "input": [
            {"role": "system", "content": "sys A"},
            {"role": "developer", "content": [{"type": "input_text", "text": "dev B"}]},
            {"role": "user", "content": "hi"},
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read_sample_file", "arguments": "{}", "status": "completed"},
            {"type": "function_call_output", "call_id": "call_1", "output": "text"},
            {"type": "message", "role": "assistant", "id": "msg_1", "content": [{"type": "output_text", "text": "ok", "annotations": []}]},
            {"role": "system", "content": "late system"},
        ],
    }


def test_normalize_common_rules():
    out = T.normalize_body(_lc_body(), T.Options("namespace"))
    for f in T.UNSUPPORTED_FIELDS:
        assert f not in out
    assert out["store"] is False and out["stream"] is True
    assert out["instructions"] == "sys A\n\ndev B"
    roles = [i.get("role") for i in out["input"]]
    assert "system" not in roles
    assert out["input"][-1]["role"] == "developer"  # late system message demoted
    assert not any(i.get("type") == "reasoning" for i in out["input"])  # no encrypted content -> dropped
    assert all("id" not in i and "status" not in i for i in out["input"])
    assert all(i.get("call_id") for i in out["input"] if i.get("type", "").startswith("function_call"))


def test_normalize_does_not_mutate_input():
    b = _lc_body()
    snap = json.dumps(b, sort_keys=True)
    T.normalize_body(b, T.Options("additional"))
    assert json.dumps(b, sort_keys=True) == snap


def test_normalize_namespace_format():
    out = T.normalize_body(_lc_body(), T.Options("namespace", "scribe"))
    assert [t["type"] for t in out["tools"]] == ["namespace"]
    ns = out["tools"][0]
    assert ns["name"] == "scribe" and ns["description"] and [t["name"] for t in ns["tools"]] == ["read_sample_file", "ls"]
    fc = next(i for i in out["input"] if i.get("type") == "function_call")
    assert fc["namespace"] == "scribe"  # replayed call must carry its namespace
    assert not any(i.get("type") == "additional_tools" for i in out["input"])


def test_normalize_additional_format():
    out = T.normalize_body(_lc_body(), T.Options("additional"))
    assert "tools" not in out and "tool_choice" not in out and "parallel_tool_calls" not in out
    first = out["input"][0]
    assert first["type"] == "additional_tools" and first["role"] == "developer"
    assert [t["name"] for t in first["tools"]] == ["read_sample_file", "ls"]
    assert all("namespace" not in i for i in out["input"])


def test_normalize_keeps_encrypted_reasoning_and_ids_on_request():
    b = {"model": "m", "input": [
        {"role": "user", "content": "x"},
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC", "summary": []}]}
    out = T.normalize_body(b, T.Options("namespace", include_reasoning=True))
    r = out["input"][1]
    assert r["encrypted_content"] == "ENC" and "id" not in r
    assert out["include"] == ["reasoning.encrypted_content"]
    keep = T.normalize_body(b, T.Options("namespace", keep_item_ids=True))
    assert keep["input"][1]["id"] == "rs_1"
    assert "tools" not in out


def test_normalize_string_input():
    out = T.normalize_body({"model": "m", "input": "hello"}, T.Options("namespace"))
    assert out["input"] == [{"type": "message", "role": "user", "content": "hello"}]


# --------------------------------------------------------------------------- transport + fake agent


def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


_N = [0]


def _next_id(prefix):
    _N[0] += 1
    return f"{prefix}_{_N[0]}"


def _resp_obj(status, output, rid="resp_f"):
    return {"id": rid, "object": "response", "created_at": 1, "status": status, "model": "fake", "output": output,
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [], "store": False,
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


def _call_events(call_id, name, args, namespace):
    rid = _next_id("resp")
    item = {"type": "function_call", "id": f"fc_{call_id}", "call_id": call_id, "name": name,
            "arguments": json.dumps(args), "status": "completed"}
    if namespace:
        item["namespace"] = namespace
    added = {**item, "arguments": "", "status": "in_progress"}
    return [
        {"type": "response.created", "sequence_number": 0, "response": _resp_obj("in_progress", [], rid)},
        {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": added},
        {"type": "response.function_call_arguments.delta", "sequence_number": 2, "output_index": 0,
         "item_id": item["id"], "delta": item["arguments"]},
        {"type": "response.function_call_arguments.done", "sequence_number": 3, "output_index": 0,
         "item_id": item["id"], "arguments": item["arguments"]},
        {"type": "response.output_item.done", "sequence_number": 4, "output_index": 0, "item": item},
        {"type": "response.completed", "sequence_number": 5, "response": _resp_obj("completed", [item], rid)},
    ]


def _text_events(text):
    rid, mid = _next_id("resp"), _next_id("msg")
    item = {"type": "message", "id": mid, "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}
    part0 = {"type": "output_text", "text": "", "annotations": []}
    return [
        {"type": "response.created", "sequence_number": 0, "response": _resp_obj("in_progress", [], rid)},
        {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
         "item": {**item, "content": [], "status": "in_progress"}},
        {"type": "response.content_part.added", "sequence_number": 2, "output_index": 0, "content_index": 0,
         "item_id": mid, "part": part0},
        {"type": "response.output_text.delta", "sequence_number": 3, "output_index": 0, "content_index": 0,
         "item_id": mid, "delta": text},
        {"type": "response.output_text.done", "sequence_number": 4, "output_index": 0, "content_index": 0,
         "item_id": mid, "text": text},
        {"type": "response.content_part.done", "sequence_number": 5, "output_index": 0, "content_index": 0,
         "item_id": mid, "part": item["content"][0]},
        {"type": "response.output_item.done", "sequence_number": 6, "output_index": 0, "item": item},
        {"type": "response.completed", "sequence_number": 7, "response": _resp_obj("completed", [item], rid)},
    ]


class FakeServer:
    """Plays the model: turn 1 -> read tool call; after the tool result -> answer; turn 2 -> answer from history."""

    def __init__(self, tool_format):
        self.tool_format = tool_format
        self.bodies: list[dict] = []
        self.auth: set[str] = set()
        self.urls: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.add(str(request.url))
        self.auth.add(request.headers["authorization"])
        body = json.loads(request.content)
        self.bodies.append(body)
        assert body["store"] is False and body["stream"] is True
        assert request.headers["content-length"] == str(len(request.content))
        items = body["input"]
        outputs = [i["output"] for i in items if i.get("type") == "function_call_output"]
        last = items[-1]
        if last.get("type") == "function_call_output":
            text = f"Summary: synthetic geoscience text. Audit code: {T.SENTINEL_CODE}"
            events = _text_events(text)
        elif outputs and "Without calling any tool" in json.dumps(last):
            events = _text_events(f"The code is {T.SENTINEL_CODE}; first heading: Basin Stratigraphy")
        else:
            events = _call_events("call_1", "read_sample_file", {"path": "sample.txt"},
                                  "scribe" if self.tool_format == "namespace" else None)
        return httpx.Response(200, headers={"content-type": "text/event-stream", "x-request-id": "req_fake"},
                              content=_sse(events))


@pytest.mark.parametrize("fmt", ["namespace", "additional"])
def test_fake_agent_two_turns(fmt, capsys):
    pytest.importorskip("deepagents")
    fake = FakeServer(fmt)
    rec = T.Recorder()
    opts = T.Options(fmt, "scribe")
    res = T.run_spike("fake-model", lambda: "fake-token", opts, rec,
                      inner=httpx.MockTransport(fake), inner_async=httpx.MockTransport(fake))
    out = capsys.readouterr().out
    assert res["passed"], (res, out)
    assert res["tool_read_proven"] and res["turn1_has_sentinel"] and res["turn2_has_sentinel"]
    assert rec.requests == 3 and rec.completed == 3 and not rec.failed
    assert fake.auth == {"Bearer fake-token"}
    assert fake.urls == {"https://api.openai.com/v1/responses"}
    assert "fake-token" not in out  # tokens never printed
    assert "response.completed" in out and "[tool call] read_sample_file" in out

    first, replay = fake.bodies[0], fake.bodies[2]
    for b in fake.bodies:
        for f in T.UNSUPPORTED_FIELDS:
            assert f not in b
        assert not any(i.get("role") == "system" for i in b["input"])
    assert first["instructions"]
    if fmt == "namespace":
        assert first["tools"][0]["type"] == "namespace"
        names = [t["name"] for t in first["tools"][0]["tools"]]
        assert "read_sample_file" in names
        calls = [i for i in replay["input"] if i.get("type") == "function_call"]
        assert calls and all(c["namespace"] == "scribe" for c in calls)
    else:
        assert "tools" not in first
        add = [i for i in first["input"] if i.get("type") == "additional_tools"]
        assert len(add) == 1 and "read_sample_file" in [t["name"] for t in add[0]["tools"]]
    assert any(i.get("type") == "function_call_output" for i in replay["input"])


def test_http_error_body_and_request_id_are_printed(capsys):
    def deny(request):
        return httpx.Response(429, headers={"x-request-id": "req_429"}, json={
            "error": {"code": "subscription_sharing_usage_limit_exceeded", "message": "limit", "param": None}})

    rec = T.Recorder()
    client = httpx.Client(transport=T.RewriteTransport(httpx.MockTransport(deny), lambda: "tok", T.Options("namespace"), rec))
    r = client.post("https://api.openai.com/v1/responses", json={"model": "m", "input": "hi"})
    assert r.status_code == 429 and r.json()["error"]["code"] == "subscription_sharing_usage_limit_exceeded"
    out = capsys.readouterr().out
    assert "subscription_sharing_usage_limit_exceeded" in out and "req_429" in out and '"message":"limit"' in out
    assert rec.http_errors[0]["status"] == 429 and "tok" not in out


def test_transport_refuses_other_hosts():
    rec = T.Recorder()
    client = httpx.Client(transport=T.RewriteTransport(httpx.MockTransport(lambda r: httpx.Response(200)), lambda: "t", T.Options("namespace"), rec))
    for url in ("https://chatgpt.com/backend-api/codex/responses", "https://api.openai.com.evil.example/v1/responses",
                "https://api.openai.com/backend-api/x"):
        with pytest.raises(RuntimeError, match="refusing"):
            client.get(url)


def test_scripts_import_and_cli_help():
    import subprocess
    import sys
    for script in ("spike_login.py", "spike_turn.py"):
        r = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True,
                           cwd=str(T.SPIKE_DIR), encoding="utf-8")
        assert r.returncode == 0, r.stderr


def test_sample_file_has_unique_sentinel():
    txt = (T.SPIKE_DIR / "sample.txt").read_text(encoding="utf-8")
    assert 35000 <= len(txt) <= 45000
    assert txt.count(T.SENTINEL_CODE) == 1
    assert txt.rstrip().endswith(f"{T.SENTINEL_CODE}.")


def test_failed_browser_open_never_prints_id_token_or_email(tmp_path, monkeypatch, capsys):
    """Re-login with a stored ID token: the fallback URL must not leak id_token_hint / login_hint."""
    id_token = "AAAA.SECRET-ID-TOKEN-PAYLOAD.BBBB"
    email = "someone@example.com"
    monkeypatch.setenv("METASCI_SPIKE_DIR", str(tmp_path))
    L.save_token({"client_id": "oaiapp_saved", "ext_agent_host_id": "urn:uuid:host", "id_token": id_token,
                  "email": email, "subject": "sub-1", "scopes": [L.DIRECT_SCOPE]}, L.token_path())
    opened = []
    monkeypatch.setattr(L.webbrowser, "open", lambda u: opened.append(u) or False)
    monkeypatch.setattr(L, "LOGIN_TIMEOUT_S", 0.2)
    assert L.cmd_login(fresh=False) == 2  # times out: no callback arrives, no network used
    out = capsys.readouterr()
    text = out.out + out.err
    # the browser itself still gets the hinted URL ...
    assert len(opened) == 1 and "id_token_hint=" in opened[0] and "login_hint=" in opened[0]
    # ... but nothing printed contains the token or the email
    assert id_token not in text and "SECRET-ID-TOKEN" not in text
    assert email not in text and "someone%40example.com" not in text
    assert "id_token_hint" not in text and "login_hint" not in text
    # the printed URL is still a usable twin of the same attempt
    printed = next(w for w in text.split() if w.startswith("https://auth.openai.com/api/accounts/authorize"))
    _, pq = _qs(printed)
    _, bq = _qs(opened[0])
    assert pq["client_id"] == "oaiapp_saved"
    for k in ("state", "nonce", "code_challenge", "redirect_uri", "scope", "resource", "ext_agent_host_id"):
        assert pq[k] == bq[k]
