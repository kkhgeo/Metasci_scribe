"""ChatGPT plan sign-in and token storage. Every OpenAI endpoint here is faked;
the only real sockets are the login listener on loopback."""

import json
import os
import re
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from metasci_scribe import paths
from metasci_scribe.auth import chatgpt_plan as cp
from metasci_scribe.auth.chatgpt_plan import (
    Credentials,
    LoginCancelled,
    LoginFailed,
    LoginTimeout,
    NotSignedIn,
    PlanUsageDisabled,
    RefreshRejected,
    RefreshUnavailable,
    build_authorize_url,
    clear,
    get_access_token,
    load,
    login,
    save,
)

R = "http://127.0.0.1:1455/auth/callback"
JWKS_URL = "https://auth.openai.com/.well-known/jwks.json"
REVOKE_URL = "https://auth.openai.com/api/accounts/oauth/revoke"
ISSUED = "oaiapp_issued"
ALL_SCOPES = (
    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
)
ACCESS = "access-secret-1"
REFRESH = "refresh-secret-1"
AUTH_CODE = "auth-code-secret"
SECRET_MARKERS = (ACCESS, REFRESH, AUTH_CODE, "id-token-secret", "verifier")

_SIGNING_KEY = ec.generate_private_key(ec.SECP256R1())
_OTHER_KEY = ec.generate_private_key(ec.SECP256R1())


def _token_path() -> Path:
    return paths.DATA_DIR / "auth" / "chatgpt.json"


def _record() -> dict:
    return json.loads(_token_path().read_text(encoding="utf-8"))


def _creds(**changes) -> Credentials:
    values = {
        "client_id": ISSUED,
        "access_token": ACCESS,
        "refresh_token": REFRESH,
        "id_token": "id-token-secret",
        "expires_at": time.time() + 3600,
        "scopes": frozenset(ALL_SCOPES.split()),
        "email": "user@example.com",
        "subject": "user-1",
    }
    return Credentials(**{**values, **changes})


class FakeOpenAI:
    """The token, discovery and JWKS endpoints, and what they were sent."""

    def __init__(self):
        self.nonce = None
        self.token_forms: list[dict[str, str]] = []
        self.authorize_urls: list[str] = []
        self.token_scope = ALL_SCOPES
        self.id_claims: dict = {}
        self.id_headers: dict = {"kid": "key-1"}
        self.id_key = _SIGNING_KEY
        self.id_algorithm = "ES256"
        self.token_reply = None  # form -> httpx.Response, or raises
        self.revoke_forms: list[dict[str, str]] = []
        self.revoke_reply = None  # form -> httpx.Response, or raises
        self.revocation_endpoint: str | None = REVOKE_URL
        self.lock = threading.Lock()

    def id_token(self) -> str:
        now = int(time.time())
        claims = {
            "iss": cp.ISSUER,
            "aud": self.token_forms[-1]["client_id"],
            "sub": "user-1",
            "iat": now,
            "exp": now + 3600,
            "nonce": self.nonce,
            "email": "user@example.com",
        }
        claims.update(self.id_claims)
        return jwt.encode(
            {k: v for k, v in claims.items() if v is not None},
            self.id_key,
            algorithm=self.id_algorithm,
            headers=self.id_headers,
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url == cp.DISCOVERY_URL:
            document = {"issuer": cp.ISSUER, "jwks_uri": JWKS_URL}
            if self.revocation_endpoint is not None:
                document["revocation_endpoint"] = self.revocation_endpoint
            return httpx.Response(200, json=document)
        if url == self.revocation_endpoint:
            assert request.method == "POST"
            form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
            with self.lock:
                self.revoke_forms.append(form)
            if self.revoke_reply is not None:
                return self.revoke_reply(form)
            return httpx.Response(200)
        if url == JWKS_URL:
            jwk = json.loads(
                jwt.algorithms.ECAlgorithm.to_jwk(_SIGNING_KEY.public_key())
            )
            return httpx.Response(200, json={"keys": [{**jwk, "kid": "key-1"}]})
        assert url == cp.TOKEN_URL, f"unexpected request to {url}"
        assert request.method == "POST"
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        with self.lock:
            self.token_forms.append(form)
        if self.token_reply is not None:
            return self.token_reply(form)
        return httpx.Response(
            200,
            json={
                "access_token": ACCESS,
                "refresh_token": REFRESH,
                "id_token": self.id_token(),
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": self.token_scope,
            },
        )


@pytest.fixture
def openai(monkeypatch):
    fake = FakeOpenAI()
    monkeypatch.setattr(
        cp,
        "_http_client",
        lambda: httpx.Client(transport=httpx.MockTransport(fake)),
    )
    return fake


def _hit(redirect_uri: str, params: dict) -> httpx.Response:
    return httpx.get(redirect_uri, params=params, trust_env=False, timeout=5)


def _browser(fake, *, client_id: str | None = ISSUED, extra=None, stray=()):
    """Stands in for the user's browser: approves and returns to the listener."""

    def open_browser(url: str) -> bool:
        query = parse_qs(urlparse(url).query)
        fake.nonce = query["nonce"][0]
        fake.authorize_urls.append(url)
        redirect_uri = query["redirect_uri"][0]
        for params in stray:
            _hit(redirect_uri, params)
        params = {"code": AUTH_CODE, "state": query["state"][0], **(extra or {})}
        if client_id is not None:
            params["client_id"] = client_id
        _hit(redirect_uri, params)
        return True

    return open_browser


def _query(url: str) -> dict:
    return parse_qs(urlparse(url).query)


def _assert_clean(exc: BaseException) -> None:
    """Nothing secret in the text, and no original exception holding the request."""
    text = f"{exc} {exc!r}"
    for marker in SECRET_MARKERS:
        assert marker not in text
    assert exc.__cause__ is None
    assert exc.__context__ is None


# ---------------------------------------------------------------- authorize URL


def test_first_login_uses_dynamic_client():
    q = parse_qs(
        urlparse(
            build_authorize_url(
                client_id=None,
                redirect_uri=R,
                host_id="h",
                state="s",
                nonce="n",
                code_challenge="c",
            )
        ).query
    )
    assert q["client_id"] == ["dynamic_agent_client"]
    assert q["agent_name_hint"] == ["Metasci_scribe"]
    assert q["code_challenge_method"] == ["S256"]
    assert q["resource"] == ["https://api.openai.com/v1"]
    assert q["ext_agent_host_id"] == ["h"]
    assert q["response_type"] == ["code"]
    assert set(q["scope"][0].split()) == {
        "openid",
        "profile",
        "email",
        "offline_access",
        "resource.invoke",
        "chatgpt.tokens.use.direct",
    }


def test_authorize_url_endpoint():
    url = build_authorize_url(
        client_id=None,
        redirect_uri=R,
        host_id="h",
        state="s",
        nonce="n",
        code_challenge="c",
    )
    assert url.startswith("https://auth.openai.com/api/accounts/authorize?")


def test_relogin_omits_agent_name_hint():
    q = _query(
        build_authorize_url(
            client_id=ISSUED,
            redirect_uri=R,
            host_id="h",
            state="s",
            nonce="n",
            code_challenge="c",
            id_token_hint="hint",
        )
    )
    assert q["client_id"] == [ISSUED]
    assert "agent_name_hint" not in q
    assert q["id_token_hint"] == ["hint"]


@pytest.mark.parametrize(
    "redirect",
    [
        "http://localhost:1455/auth/callback",
        "https://127.0.0.1:1455/auth/callback",
        "http://127.0.0.1:1455/callback",
        "http://127.0.0.1/auth/callback",
        "http://127.0.0.1:1455/auth/callback?x=1",
    ],
)
def test_redirect_only_the_port_may_vary(redirect):
    with pytest.raises(ValueError, match="redirect"):
        build_authorize_url(
            client_id=None,
            redirect_uri=redirect,
            host_id="h",
            state="s",
            nonce="n",
            code_challenge="c",
        )


def test_redirect_uses_127_0_0_1(openai):
    login(open_browser=_browser(openai), timeout=10)
    redirect = urlparse(_query(openai.authorize_urls[0])["redirect_uri"][0])
    assert redirect.scheme == "http"
    assert redirect.hostname == "127.0.0.1"
    assert redirect.path == "/auth/callback"


def test_each_attempt_has_fresh_values_and_same_host_id(openai):
    login(open_browser=_browser(openai), timeout=10)
    login(open_browser=_browser(openai), timeout=10)
    first, second = (_query(url) for url in openai.authorize_urls)
    for name in ("state", "nonce", "code_challenge"):
        assert first[name] != second[name]
    assert first["ext_agent_host_id"] == second["ext_agent_host_id"]


# ---------------------------------------------------------------- login


def test_login_saves_issued_client_and_validated_identity(openai):
    creds = login(open_browser=_browser(openai), timeout=10)
    assert creds.client_id == ISSUED
    assert creds.access_token == ACCESS
    assert creds.refresh_token == REFRESH
    assert creds.email == "user@example.com"
    assert creds.subject == "user-1"
    assert creds.plan_enabled
    assert creds.expires_at == pytest.approx(time.time() + 3600, abs=30)
    assert load() == creds


def test_code_exchange_sends_the_same_redirect_and_verifier(openai):
    login(open_browser=_browser(openai), timeout=10)
    form = openai.token_forms[0]
    query = _query(openai.authorize_urls[0])
    assert form["grant_type"] == "authorization_code"
    assert form["client_id"] == ISSUED
    assert form["code"] == AUTH_CODE
    assert form["redirect_uri"] == query["redirect_uri"][0]
    assert form["resource"] == "https://api.openai.com/v1"
    assert "client_secret" not in form
    expected = cp._b64url(cp.hashlib.sha256(form["code_verifier"].encode()).digest())
    assert query["code_challenge"] == [expected]


def test_rejects_state_mismatch(openai):
    def open_browser(url):
        openai.nonce = _query(url)["nonce"][0]
        redirect = _query(url)["redirect_uri"][0]
        reply = _hit(redirect, {"code": AUTH_CODE, "state": "wrong"})
        assert reply.status_code == 400
        return True

    with pytest.raises(LoginTimeout):
        login(open_browser=open_browser, timeout=0.3)
    assert openai.token_forms == []
    assert load() is None


def test_stray_request_does_not_abort_login(openai):
    stray = (
        {"code": "attacker", "state": "wrong"},
        {"error": "access_denied", "state": "wrong"},
        {"code": "no-state"},
    )
    creds = login(open_browser=_browser(openai, stray=stray), timeout=10)
    assert creds.access_token == ACCESS
    assert [form["code"] for form in openai.token_forms] == [AUTH_CODE]


def test_idle_connection_does_not_block_callback(openai, monkeypatch):
    # Longer than the login, so only serving connections side by side passes.
    monkeypatch.setattr(cp._CallbackHandler, "timeout", 30)
    browser = _browser(openai)
    sockets = []

    def open_browser(url):
        port = urlparse(_query(url)["redirect_uri"][0]).port
        sockets.append(socket.create_connection(("127.0.0.1", port), timeout=5))
        return browser(url)

    try:
        assert login(open_browser=open_browser, timeout=10).client_id == ISSUED
    finally:
        for sock in sockets:
            sock.close()


def test_never_saves_dynamic_client_id(openai):
    browser = _browser(openai, client_id="dynamic_agent_client")
    with pytest.raises(LoginFailed):
        login(open_browser=browser, timeout=10)
    assert openai.token_forms == []
    assert not _token_path().exists()
    with pytest.raises(ValueError, match="dynamic_agent_client"):
        save(_creds(client_id="dynamic_agent_client"))
    assert not _token_path().exists()


def test_saved_dynamic_client_id_is_ignored(openai):
    save(_creds())
    record = _record()
    record["client_id"] = "dynamic_agent_client"
    cp._write_private(_token_path(), json.dumps(record).encode())
    assert load() is None


def test_missing_issued_client_id_is_incomplete(openai):
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai, client_id=None), timeout=10)
    assert caught.value.code == "registration_incomplete"
    assert openai.token_forms == []


def test_plan_disabled_without_direct_scope(openai):
    # The callback's scope is not the grant; only the token response counts.
    openai.token_scope = "openid profile email"
    browser = _browser(openai, extra={"scope": ALL_SCOPES})
    creds = login(open_browser=browser, timeout=10)
    assert not creds.plan_enabled
    assert load().plan_enabled is False
    with pytest.raises(PlanUsageDisabled):
        get_access_token()


def test_token_response_without_scope_leaves_plan_disabled(openai):
    openai.token_scope = None
    creds = login(
        open_browser=_browser(openai, extra={"scope": ALL_SCOPES}), timeout=10
    )
    assert creds.scopes == frozenset()
    assert not creds.plan_enabled


def test_plan_declined_grant_without_refresh_token_keeps_sign_in(openai):
    def reply(form):
        return httpx.Response(
            200,
            json={
                "access_token": ACCESS,
                "id_token": openai.id_token(),
                "expires_in": 3600,
                "scope": "openid profile email",
            },
        )

    openai.token_reply = reply
    creds = login(open_browser=_browser(openai), timeout=10)
    assert creds.refresh_token == ""
    assert load() == creds


def test_access_denied_raises_login_cancelled(openai):
    def open_browser(url):
        _hit(
            _query(url)["redirect_uri"][0],
            {"error": "access_denied", "state": _query(url)["state"][0]},
        )
        return True

    with pytest.raises(LoginCancelled):
        login(open_browser=open_browser, timeout=10)
    assert openai.token_forms == []


def test_other_oauth_error_reports_only_a_clean_code(openai):
    def open_browser(url):
        _hit(
            _query(url)["redirect_uri"][0],
            {
                "error": "server_error",
                "error_description": "<script>" + AUTH_CODE,
                "state": _query(url)["state"][0],
            },
        )
        return True

    with pytest.raises(LoginFailed) as caught:
        login(open_browser=open_browser, timeout=10)
    assert caught.value.code == "server_error"
    _assert_clean(caught.value)


def test_login_times_out_when_browser_closed():
    with pytest.raises(LoginTimeout):
        login(open_browser=lambda url: True, timeout=0.2)


def test_listener_is_closed_after_timeout():
    ports = []

    def open_browser(url):
        ports.append(urlparse(_query(url)["redirect_uri"][0]).port)
        return True

    with pytest.raises(LoginTimeout):
        login(open_browser=open_browser, timeout=0.1)
    probe = socket.socket()
    probe.bind(("127.0.0.1", ports[0]))  # would fail if it were still held
    probe.close()


def test_cancel_event_ends_the_wait_and_closes_the_listener():
    ports = []
    cancel = threading.Event()

    def open_browser(url):
        ports.append(urlparse(_query(url)["redirect_uri"][0]).port)
        threading.Timer(0.3, cancel.set).start()
        return True

    started = time.monotonic()
    with pytest.raises(LoginCancelled) as caught:
        login(open_browser=open_browser, timeout=30, cancel=cancel)
    assert time.monotonic() - started < 5
    _assert_clean(caught.value)
    probe = socket.socket()
    probe.bind(("127.0.0.1", ports[0]))  # would fail if it were still held
    probe.close()


def test_a_cancel_during_the_token_exchange_saves_nothing(openai):
    cancel = threading.Event()

    def cancelled_mid_exchange(form):
        cancel.set()
        return httpx.Response(
            200,
            json={
                "access_token": ACCESS,
                "refresh_token": REFRESH,
                "id_token": openai.id_token(),
                "expires_in": 3600,
                "scope": ALL_SCOPES,
            },
        )

    openai.token_reply = cancelled_mid_exchange
    with pytest.raises(LoginCancelled):
        login(open_browser=_browser(openai), timeout=10, cancel=cancel)
    assert load() is None
    assert ACCESS not in _token_path().read_text(encoding="utf-8")


def test_the_manual_address_goes_to_the_callback_not_to_print(openai, capsys):
    shown = []

    def no_browser(url):
        _browser(openai)(url)
        return False

    login(open_browser=no_browser, timeout=10, show_url=shown.append)
    assert capsys.readouterr().out == ""
    assert len(shown) == 1
    assert "Open this address" in shown[0]
    assert "id_token_hint" not in shown[0]


def test_cancel_already_set_saves_nothing(openai):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(LoginCancelled):
        login(open_browser=lambda url: True, timeout=30, cancel=cancel)
    assert load() is None
    assert openai.token_forms == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows port sharing")
def test_listener_port_cannot_be_shared_on_windows():
    server = cp._start_callback_server("state")
    intruder = socket.socket()
    intruder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        with pytest.raises(OSError, match=r"10013|10048"):
            intruder.bind(server.server_address)
    finally:
        intruder.close()
        server.server_close()


def test_listener_binds_loopback_only():
    server = cp._start_callback_server("state")
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_port_in_use_falls_back_to_another_port(openai, monkeypatch):
    # A plain holder with no exclusive flag: Windows would let a second
    # SO_REUSEADDR socket share this port and steal its traffic.
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen()
    busy = holder.getsockname()[1]
    monkeypatch.setattr(cp, "PREFERRED_PORT", busy)
    try:
        login(open_browser=_browser(openai), timeout=10)
    finally:
        holder.close()
    redirect = urlparse(_query(openai.authorize_urls[0])["redirect_uri"][0])
    assert redirect.port not in (None, busy)
    assert (redirect.scheme, redirect.hostname, redirect.path) == (
        "http",
        "127.0.0.1",
        "/auth/callback",
    )
    assert openai.token_forms[0]["redirect_uri"] == redirect.geturl()


def test_relogin_uses_saved_client_and_id_token_hint(openai):
    login(open_browser=_browser(openai), timeout=10)
    saved_id_token = load().id_token
    login(open_browser=_browser(openai, client_id=None), timeout=10)
    query = _query(openai.authorize_urls[1])
    assert query["client_id"] == [ISSUED]
    assert "agent_name_hint" not in query
    assert query["id_token_hint"] == [saved_id_token]
    assert openai.token_forms[1]["client_id"] == ISSUED


def test_relogin_rejects_different_client_id(openai):
    login(open_browser=_browser(openai), timeout=10)
    before = load()
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai, client_id="oaiapp_other"), timeout=10)
    assert caught.value.code == "client_mismatch"
    assert len(openai.token_forms) == 1
    assert load() == before


def test_relogin_rejects_different_account(openai):
    login(open_browser=_browser(openai), timeout=10)
    before = load()
    openai.id_claims = {"sub": "user-2"}
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai), timeout=10)
    assert caught.value.code == "account_mismatch"
    assert load() == before


def test_invalid_grant_on_exchange_keeps_issued_client_for_next_attempt(openai):
    openai.token_reply = lambda form: httpx.Response(
        400, json={"error": "invalid_grant"}
    )
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai), timeout=10)
    assert caught.value.code == "invalid_grant"
    assert load() is None

    openai.token_reply = None
    login(open_browser=_browser(openai, client_id=None), timeout=10)
    query = _query(openai.authorize_urls[1])
    assert query["client_id"] == [ISSUED]
    assert "agent_name_hint" not in query


def test_exchange_error_text_is_whitelisted(openai):
    openai.token_reply = lambda form: httpx.Response(
        400,
        json={
            "error": "invalid_request",
            "error_description": f"bad code {form['code']} {form['code_verifier']}",
            "code_verifier": form["code_verifier"],
        },
    )
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai), timeout=10)
    assert caught.value.code == "invalid_request"
    _assert_clean(caught.value)
    assert openai.token_forms[0]["code_verifier"] not in str(caught.value)


def test_exchange_network_failure_is_clean(openai):
    def reply(form):
        raise httpx.ConnectError(f"failed {form['code']}")

    openai.token_reply = reply
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai), timeout=10)
    _assert_clean(caught.value)


def test_exchange_garbage_body_is_clean(openai):
    openai.token_reply = lambda form: httpx.Response(
        200, content=f"not json {form['code']}"
    )
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai), timeout=10)
    _assert_clean(caught.value)


@pytest.mark.parametrize(
    "tamper",
    [
        {"id_claims": {"nonce": "other"}},
        {"id_claims": {"aud": "someone-else"}},
        {"id_claims": {"iss": "https://evil.example"}},
        {"id_claims": {"exp": 1000, "iat": 900}},
        {"id_claims": {"sub": None}},
        {"id_claims": {"sub": ""}},
        {"id_claims": {"sid": "   "}},
        {"id_key": _OTHER_KEY},
        {"id_key": "x" * 40, "id_algorithm": "HS256"},
        {"id_headers": {"kid": "unknown"}},
    ],
    ids=[
        "nonce",
        "audience",
        "issuer",
        "expired",
        "no-subject",
        "empty-subject",
        "blank-sid",
        "wrong-signing-key",
        "symmetric-algorithm",
        "unknown-kid",
    ],
)
def test_invalid_id_token_saves_nothing(openai, tamper):
    for name, value in tamper.items():
        setattr(openai, name, value)
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai), timeout=10)
    assert caught.value.code == "id_token_invalid"
    assert load() is None


def test_jwks_failure_saves_nothing(openai, monkeypatch):
    real = cp._http_client

    def failing_discovery(request):
        return httpx.Response(503)

    def client():
        built = real()
        built._transport = httpx.MockTransport(
            lambda request: (
                failing_discovery(request)
                if str(request.url) == cp.DISCOVERY_URL
                else openai(request)
            )
        )
        return built

    monkeypatch.setattr(cp, "_http_client", client)
    with pytest.raises(LoginFailed):
        login(open_browser=_browser(openai), timeout=10)
    assert load() is None


def test_unopened_browser_prints_url_without_hint(openai, capsys):
    login(open_browser=_browser(openai), timeout=10)
    id_token = load().id_token
    browser = _browser(openai, client_id=None)
    seen = []

    def no_browser(url):
        seen.append(url)
        threading.Thread(target=browser, args=(url,), daemon=True).start()
        return False

    login(open_browser=no_browser, timeout=10)
    out = capsys.readouterr()
    assert "id_token_hint" in seen[0]
    assert "https://auth.openai.com/api/accounts/authorize?" in out.out
    text = out.out + out.err
    assert "id_token_hint" not in text
    assert "login_hint" not in text
    assert id_token not in text
    for marker in SECRET_MARKERS:
        assert marker not in text


def test_browser_exception_falls_back_to_printing(openai, capsys):
    def broken(url):
        raise RuntimeError("no display")

    with pytest.raises(LoginTimeout):
        login(open_browser=broken, timeout=0.1)
    assert "https://auth.openai.com/api/accounts/authorize?" in capsys.readouterr().out


def test_login_prints_no_secrets(openai, capsys):
    login(open_browser=_browser(openai), timeout=10)
    out = capsys.readouterr()
    for marker in SECRET_MARKERS:
        assert marker not in out.out + out.err


def test_callback_page_does_not_echo_parameters(openai):
    pages = []
    browser = _browser(openai)

    def open_browser(url):
        reply = _hit(
            _query(url)["redirect_uri"][0],
            {"code": "<b>echo</b>", "state": "wrong"},
        )
        pages.append(reply)
        return browser(url)

    login(open_browser=open_browser, timeout=10)
    assert "echo" not in pages[0].text
    assert pages[0].headers["cache-control"] == "no-store"


# ---------------------------------------------------------------- Credentials


def test_token_file_excludes_from_repr():
    creds = _creds()
    for text in (repr(creds), str(creds), f"{creds}", repr([creds])):
        for secret in (ACCESS, REFRESH, "id-token-secret", "user@example.com"):
            assert secret not in text
    assert ISSUED in repr(creds)


def test_credentials_positional_order_is_the_published_one():
    creds = Credentials("c", "a", "r", "i", 1.0, frozenset({"openid"}), None)
    assert (creds.client_id, creds.access_token, creds.refresh_token) == ("c", "a", "r")
    assert (creds.id_token, creds.expires_at, creds.email) == ("i", 1.0, None)
    assert creds.scopes == frozenset({"openid"})


def test_exception_with_credentials_is_clean():
    assert ACCESS not in str(RuntimeError(_creds()))


# ---------------------------------------------------------------- storage


def test_save_load_round_trip():
    creds = _creds()
    save(creds)
    assert load() == creds
    assert _token_path() == paths.DATA_DIR / "auth" / "chatgpt.json"


def test_load_without_a_file():
    assert load() is None


def test_load_with_a_damaged_file():
    save(_creds())
    cp._write_private(_token_path(), b"{not json")
    assert load() is None
    with pytest.raises(NotSignedIn):
        get_access_token()


def test_save_is_atomic_and_leaves_no_temp_on_failure(monkeypatch):
    save(_creds())

    def refuse(source, target):
        raise OSError("disk trouble")

    monkeypatch.setattr(cp.os, "replace", refuse)
    with pytest.raises(cp.CredentialStoreError):
        save(_creds(access_token="newer"))
    assert load().access_token == ACCESS
    assert [p.name for p in _token_path().parent.glob("*.tmp")] == []


def test_save_retries_while_a_reader_holds_the_file(monkeypatch):
    real = os.replace
    calls = []

    def busy_once(source, target):
        calls.append(source)
        if len(calls) == 1:
            raise PermissionError(13, "sharing violation")
        return real(source, target)

    monkeypatch.setattr(cp.os, "replace", busy_once)
    save(_creds())
    assert len(calls) == 2
    assert load() is not None


def test_stale_temp_from_a_crash_is_removed_on_next_write():
    save(_creds())
    stale = _token_path().with_name("chatgpt.json.crash.tmp")
    stale.write_text(REFRESH, encoding="utf-8")
    save(_creds())
    assert not stale.exists()


def test_save_fails_closed_when_the_file_cannot_be_restricted(monkeypatch):
    def refuse(path):
        raise cp.CredentialStoreError("no ACL here")

    monkeypatch.setattr(cp, "_restrict_file", refuse)
    with pytest.raises(cp.CredentialStoreError):
        save(_creds())
    leftovers = [p.name for p in _token_path().parent.iterdir()]
    assert "chatgpt.json" not in leftovers
    assert not [name for name in leftovers if name.endswith(".tmp")]


def test_save_fails_closed_when_the_folder_cannot_be_restricted(monkeypatch):
    def refuse(path):
        raise cp.CredentialStoreError("no ACL here")

    monkeypatch.setattr(cp, "_restrict_dir", refuse)
    with pytest.raises(cp.CredentialStoreError):
        save(_creds())
    assert not _token_path().exists()


def test_auth_folder_refuses_a_symlink(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    paths.DATA_DIR.mkdir(parents=True)
    try:
        (paths.DATA_DIR / "auth").symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("this account may not create symlinks")
    with pytest.raises(cp.CredentialStoreError):
        save(_creds())
    assert list(target.iterdir()) == []


def _win_dacl(path: Path, tmp_path: Path) -> str:
    """The object's DACL as SDDL, read by the OS tool, not by the code under test."""
    system = Path(os.environ["SystemRoot"], "System32")
    out = tmp_path / f"acl-{time.monotonic_ns()}.txt"
    subprocess.run(
        [system / "icacls.exe", path, "/save", out, "/q"],
        check=True,
        capture_output=True,
    )
    return out.read_bytes().decode("utf-16")


def _win_user_sid() -> str:
    system = Path(os.environ["SystemRoot"], "System32")
    out = subprocess.run(
        [system / "whoami.exe", "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
    ).stdout.decode("ascii", "replace")
    return re.search(r"S-1-[0-9-]+", out).group()


def _assert_owner_only(sddl: str, *, protected: bool = True) -> None:
    if protected:
        assert "D:P" in sddl, sddl  # inheritance from the parent is cut
    entries = re.findall(r"\(([^()]*)\)", sddl)
    assert entries, sddl
    sid = _win_user_sid()
    for entry in entries:
        assert entry.split(";")[0] == "A"
        assert entry.split(";")[-1] == sid, sddl


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL")
def test_windows_acl_limits_folder_and_files_to_the_user(tmp_path):
    save(_creds())
    auth = _token_path().parent
    _assert_owner_only(_win_dacl(auth, tmp_path))
    _assert_owner_only(_win_dacl(_token_path(), tmp_path))
    cp._host_id()
    _assert_owner_only(_win_dacl(auth / "host_id", tmp_path))
    # A file nobody restricted shows what the folder passes on.
    child = auth / "child"
    child.touch()
    _assert_owner_only(_win_dacl(child, tmp_path), protected=False)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL")
def test_windows_acl_replaces_an_existing_wider_acl(tmp_path):
    auth = _token_path().parent
    auth.mkdir(parents=True)
    system = Path(os.environ["SystemRoot"], "System32")
    subprocess.run(
        [system / "icacls.exe", auth, "/grant", "*S-1-1-0:(OI)(CI)R"],
        check=True,
        capture_output=True,
    )
    assert "S-1-1-0" in _win_dacl(auth, tmp_path) or "WD" in _win_dacl(auth, tmp_path)
    save(_creds())
    sddl = _win_dacl(auth, tmp_path)
    assert "WD" not in sddl
    _assert_owner_only(sddl)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows ACL")
def test_windows_temp_file_is_restricted_before_it_is_renamed(tmp_path, monkeypatch):
    real = os.replace
    seen = []

    def check(source, target):
        seen.append(_win_dacl(Path(source), tmp_path))
        return real(source, target)

    monkeypatch.setattr(cp.os, "replace", check)
    save(_creds())
    _assert_owner_only(seen[0])


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_posix_modes_are_owner_only():
    save(_creds())
    auth = _token_path().parent
    assert stat.S_IMODE(auth.stat().st_mode) == 0o700
    assert stat.S_IMODE(_token_path().stat().st_mode) == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_posix_tightens_an_existing_open_folder():
    auth = _token_path().parent
    auth.mkdir(parents=True)
    auth.chmod(0o755)
    save(_creds())
    assert stat.S_IMODE(auth.stat().st_mode) == 0o700


def test_host_id_is_persistent():
    first = cp._host_id()
    assert re.fullmatch(r"urn:uuid:[0-9a-f-]{36}", first)
    assert (paths.DATA_DIR / "auth" / "host_id").read_text("utf-8").strip() == first
    assert cp._host_id() == first


def test_damaged_host_id_is_replaced():
    host_id = paths.DATA_DIR / "auth" / "host_id"
    cp._host_id()
    cp._write_private(host_id, b"\xff\xfe")
    assert re.fullmatch(r"urn:uuid:[0-9a-f-]{36}", cp._host_id())


def test_clear_removes_tokens_and_keeps_registration(openai):
    login(open_browser=_browser(openai), timeout=10)
    host_id = cp._host_id()
    clear()
    assert load() is None
    text = _token_path().read_text(encoding="utf-8")
    assert ACCESS not in text
    assert REFRESH not in text
    assert "id_token" not in text
    assert _record()["client_id"] == ISSUED
    assert cp._host_id() == host_id
    with pytest.raises(NotSignedIn):
        get_access_token()

    login(open_browser=_browser(openai, client_id=None), timeout=10)
    query = _query(openai.authorize_urls[1])
    assert query["client_id"] == [ISSUED]
    assert "id_token_hint" not in query
    assert "agent_name_hint" not in query


def test_clear_removes_stale_temp_files():
    save(_creds())
    stale = _token_path().with_name("chatgpt.json.crash.tmp")
    stale.write_text(REFRESH, encoding="utf-8")
    clear()
    assert not stale.exists()


def test_clear_without_a_file_is_a_no_op():
    clear()
    assert not (paths.DATA_DIR / "auth").exists()


# ---------------------------------------------------------------- sign out


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(cp, "_REVOKE_BACKOFF_S", (0.0, 0.0))


def test_sign_out_revokes_the_refresh_token_and_forgets_everything(openai):
    login(open_browser=_browser(openai), timeout=10)
    host_id = cp._host_id()
    result = cp.sign_out()
    assert result.had_sign_in
    assert result.revoked
    assert result.problem is None
    assert openai.revoke_forms == [
        {"token": REFRESH, "token_type_hint": "refresh_token", "client_id": ISSUED}
    ]
    assert load() is None
    assert not _token_path().exists()
    assert cp._host_id() == host_id
    with pytest.raises(NotSignedIn):
        get_access_token()


def test_account_mismatch_without_sign_out_and_success_after_it(openai):
    login(open_browser=_browser(openai), timeout=10)
    openai.id_claims = {"sub": "user-2"}
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai), timeout=10)
    assert caught.value.code == "account_mismatch"
    cp.sign_out()
    creds = login(open_browser=_browser(openai), timeout=10)
    assert creds.subject == "user-2"
    query = _query(openai.authorize_urls[-1])
    assert query["client_id"] == [cp.REGISTRATION_CLIENT_ID]
    assert "id_token_hint" not in query
    assert query["agent_name_hint"] == [cp.AGENT_NAME_HINT]


def test_sign_out_keeps_the_host_id_for_the_next_account(openai):
    login(open_browser=_browser(openai), timeout=10)
    cp.sign_out()
    login(open_browser=_browser(openai), timeout=10)
    first = _query(openai.authorize_urls[0])["ext_agent_host_id"]
    second = _query(openai.authorize_urls[1])["ext_agent_host_id"]
    assert first == second


@pytest.mark.parametrize("status", [500, 502, 503])
def test_sign_out_retries_a_server_error_then_still_clears(openai, status):
    login(open_browser=_browser(openai), timeout=10)
    openai.revoke_reply = lambda form: httpx.Response(status)
    result = cp.sign_out()
    assert not result.revoked
    assert result.had_sign_in
    assert str(status) in result.problem
    assert len(openai.revoke_forms) == 3
    assert load() is None
    assert not _token_path().exists()


def test_sign_out_stops_trying_when_the_time_budget_is_spent(openai, monkeypatch):
    login(open_browser=_browser(openai), timeout=10)
    monkeypatch.setattr(cp, "_REVOKE_BUDGET_S", 0.0)
    openai.revoke_reply = lambda form: httpx.Response(503)
    result = cp.sign_out()
    assert not result.revoked
    assert "503" in result.problem
    assert len(openai.revoke_forms) == 1
    assert not _token_path().exists()


def test_sign_out_survives_a_server_error_that_clears_up(openai):
    login(open_browser=_browser(openai), timeout=10)
    replies = iter([httpx.Response(503), httpx.Response(200)])
    openai.revoke_reply = lambda form: next(replies)
    result = cp.sign_out()
    assert result.revoked
    assert result.problem is None
    assert len(openai.revoke_forms) == 2


def test_sign_out_network_failure_is_reported_and_clears(openai):
    login(open_browser=_browser(openai), timeout=10)

    def unreachable(form):
        raise httpx.ConnectError("boom " + REFRESH)

    openai.revoke_reply = unreachable
    result = cp.sign_out()
    assert not result.revoked
    assert result.problem
    assert REFRESH not in f"{result} {result!r}"
    assert load() is None
    assert not _token_path().exists()


def test_sign_out_does_not_retry_a_client_error(openai):
    login(open_browser=_browser(openai), timeout=10)
    openai.revoke_reply = lambda form: httpx.Response(400, json={"error": "x"})
    result = cp.sign_out()
    assert not result.revoked
    assert "400" in result.problem
    assert len(openai.revoke_forms) == 1
    assert not _token_path().exists()


def test_sign_out_without_a_revocation_endpoint(openai):
    login(open_browser=_browser(openai), timeout=10)
    openai.revocation_endpoint = None
    result = cp.sign_out()
    assert not result.revoked
    assert result.problem
    assert openai.revoke_forms == []
    assert not _token_path().exists()


def test_sign_out_never_sends_the_token_to_another_host(openai):
    login(open_browser=_browser(openai), timeout=10)
    openai.revocation_endpoint = "https://evil.example/revoke"
    result = cp.sign_out()
    assert not result.revoked
    assert result.problem
    assert openai.revoke_forms == []
    assert not _token_path().exists()


def test_sign_out_without_a_refresh_token_has_nothing_to_revoke(openai):
    save(_creds(refresh_token=""))
    result = cp.sign_out()
    assert result.had_sign_in
    assert not result.revoked
    assert result.problem
    assert openai.revoke_forms == []
    assert not _token_path().exists()


def test_sign_out_when_signed_out_makes_no_request(openai):
    result = cp.sign_out()
    assert not result.had_sign_in
    assert not result.revoked
    assert openai.revoke_forms == []
    assert not (paths.DATA_DIR / "auth").exists()


def test_a_record_that_cannot_be_deleted_is_reported_beside_the_revocation(
    openai, monkeypatch
):
    login(open_browser=_browser(openai), timeout=10)

    def stuck():
        raise cp.CredentialStoreError("Could not remove the saved sign-in (OSError).")

    monkeypatch.setattr(cp, "_remove_record", stuck)
    result = cp.sign_out()
    assert result.revoked
    assert "Could not remove" in result.removal_problem
    assert len(openai.revoke_forms) == 1


def test_a_failed_revocation_and_a_failed_removal_are_both_reported(
    openai, monkeypatch
):
    login(open_browser=_browser(openai), timeout=10)
    openai.revoke_reply = lambda form: httpx.Response(400, json={"error": "x"})

    def stuck():
        raise cp.CredentialStoreError("Could not remove the saved sign-in (OSError).")

    monkeypatch.setattr(cp, "_remove_record", stuck)
    result = cp.sign_out()
    assert not result.revoked
    assert "400" in result.problem
    assert "Could not remove" in result.removal_problem


def test_sign_out_after_clear_drops_the_leftover_registration(openai):
    login(open_browser=_browser(openai), timeout=10)
    clear()
    assert _record()["client_id"] == ISSUED
    result = cp.sign_out()
    assert not result.had_sign_in
    assert openai.revoke_forms == []
    assert not _token_path().exists()


def test_sign_out_result_holds_no_secrets(openai):
    login(open_browser=_browser(openai), timeout=10)
    openai.revoke_reply = lambda form: httpx.Response(
        500, text=f"echo {REFRESH} {ACCESS}"
    )
    result = cp.sign_out()
    for marker in SECRET_MARKERS:
        assert marker not in f"{result} {result!r}"


def test_clear_still_keeps_the_registration(openai):
    login(open_browser=_browser(openai), timeout=10)
    clear()
    assert _record()["client_id"] == ISSUED
    assert openai.revoke_forms == []


# ---------------------------------------------------------------- plan welcome


def test_plan_welcome_is_claimed_once():
    assert cp.claim_plan_welcome() is True
    assert cp.claim_plan_welcome() is False


def test_plan_welcome_survives_sign_out(openai):
    login(open_browser=_browser(openai), timeout=10)
    assert cp.claim_plan_welcome() is True
    cp.sign_out()
    assert cp.claim_plan_welcome() is False


def test_plan_welcome_failing_store_reads_as_already_shown(monkeypatch):
    def broken(*args, **kwargs):
        raise cp.CredentialStoreError("nope")

    monkeypatch.setattr(cp, "_write_private", broken)
    assert cp.claim_plan_welcome() is False


# ---------------------------------------------------------------- refresh


def _rotating(fake, *, access="access-secret-2", refresh="refresh-secret-2", **extra):
    def reply(form):
        return httpx.Response(
            200,
            json={
                "access_token": access,
                "refresh_token": refresh,
                "expires_in": 3600,
                "scope": ALL_SCOPES,
                **extra,
            },
        )

    return reply


def test_not_signed_in():
    with pytest.raises(NotSignedIn):
        get_access_token()


def test_fresh_token_needs_no_network(openai):
    save(_creds(expires_at=time.time() + 120))
    assert get_access_token() == ACCESS
    assert openai.token_forms == []


def test_token_near_expiry_is_refreshed_and_rotation_persisted(openai):
    save(_creds(expires_at=time.time() + 30))
    openai.token_reply = _rotating(openai)
    assert get_access_token() == "access-secret-2"
    assert openai.token_forms == [
        {
            "grant_type": "refresh_token",
            "client_id": ISSUED,
            "refresh_token": REFRESH,
            "resource": "https://api.openai.com/v1",
        }
    ]
    saved = load()
    assert saved.access_token == "access-secret-2"
    assert saved.refresh_token == "refresh-secret-2"
    assert saved.expires_at == pytest.approx(time.time() + 3600, abs=30)
    assert saved.subject == "user-1"
    assert get_access_token() == "access-secret-2"
    assert len(openai.token_forms) == 1


def test_refresh_replaces_granted_scopes(openai):
    save(_creds(expires_at=0))
    openai.token_reply = lambda form: httpx.Response(
        200,
        json={
            "access_token": "new",
            "refresh_token": "new-r",
            "expires_in": 3600,
            "scope": "openid",
        },
    )
    with pytest.raises(PlanUsageDisabled):
        get_access_token()
    assert load().refresh_token == "new-r"
    assert not load().plan_enabled


def test_refresh_keeps_the_sign_in_id_token(openai):
    save(_creds(expires_at=0))
    openai.token_reply = _rotating(openai, id_token="unvalidated")
    get_access_token()
    assert load().id_token == "id-token-secret"


def test_plan_disabled_never_refreshes(openai):
    save(_creds(expires_at=0, scopes=frozenset({"openid"})))
    with pytest.raises(PlanUsageDisabled):
        get_access_token()
    assert openai.token_forms == []


def test_expired_without_refresh_token_is_not_signed_in(openai):
    save(_creds(expires_at=0, refresh_token=""))
    with pytest.raises(NotSignedIn):
        get_access_token()
    assert openai.token_forms == []


def test_concurrent_refresh_uses_token_once(openai):
    save(_creds(expires_at=0))
    replies = _rotating(openai)

    def slow(form):
        time.sleep(0.2)
        return replies(form)

    openai.token_reply = slow
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(get_access_token()))
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert results == ["access-secret-2", "access-secret-2"]
    assert len(openai.token_forms) == 1


def test_refresh_done_by_another_process_is_reused(openai):
    save(_creds(expires_at=0))
    results = []
    worker = threading.Thread(target=lambda: results.append(get_access_token()))
    with cp._locked():
        worker.start()
        time.sleep(0.2)  # the worker now waits on the lock with a stale view
        cp._write_record(
            cp._record_from(
                _creds(access_token="from-other-process", refresh_token="rotated")
            )
        )
    worker.join(20)
    assert results == ["from-other-process"]
    assert openai.token_forms == []


_CHILD = """
import sys, time
from pathlib import Path
from urllib.parse import parse_qs

import httpx

from metasci_scribe.auth import chatgpt_plan as cp

log = Path(sys.argv[1])

def reply(request):
    form = parse_qs(request.content.decode())
    with log.open("a", encoding="utf-8") as fh:
        fh.write(form["refresh_token"][0] + "\\n")
    time.sleep(0.3)
    return httpx.Response(200, json={"access_token": "a2", "refresh_token": "r2",
                                     "expires_in": 3600})

cp._http_client = lambda: httpx.Client(transport=httpx.MockTransport(reply))
Path(sys.argv[2]).touch()
while not Path(sys.argv[3]).exists():
    time.sleep(0.01)
print(cp.get_access_token())
"""


@pytest.mark.timeout(55)
def test_two_processes_spend_the_refresh_token_once(tmp_path):
    data_dir = tmp_path / "data"
    env = {**os.environ, "METASCI_SCRIBE_DATA_DIR": str(data_dir)}
    save(_creds(expires_at=0))
    seeded = data_dir / "auth" / "chatgpt.json"
    seeded.parent.mkdir(parents=True)
    seeded.write_bytes(_token_path().read_bytes())

    log = tmp_path / "spent.log"
    go = tmp_path / "go"
    children = []
    for index in range(2):
        ready = tmp_path / f"ready{index}"
        children.append(
            (
                subprocess.Popen(
                    [sys.executable, "-c", _CHILD, str(log), str(ready), str(go)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
                ready,
            )
        )
    deadline = time.monotonic() + 40
    while not all(ready.exists() for _, ready in children):
        assert time.monotonic() < deadline, "children did not start"
        assert all(child.poll() is None for child, _ in children)
        time.sleep(0.02)
    go.touch()
    outputs = [child.communicate(timeout=30) for child, _ in children]
    assert [child.returncode for child, _ in children] == [0, 0], outputs
    assert [out.strip() for out, _ in outputs] == ["a2", "a2"]
    assert log.read_text(encoding="utf-8").split() == [REFRESH]


@pytest.mark.parametrize(
    "reply",
    [
        lambda form: (_ for _ in ()).throw(httpx.ConnectError("offline")),
        lambda form: (_ for _ in ()).throw(httpx.ReadTimeout("slow")),
        lambda form: httpx.Response(500, text="boom"),
        lambda form: httpx.Response(503, json={"error": "invalid_grant"}),
        lambda form: httpx.Response(429, json={"error": "rate_limited"}),
        lambda form: httpx.Response(400, json={"error": "invalid_request"}),
        lambda form: httpx.Response(401, text="<html>"),
        lambda form: httpx.Response(200, text="not json"),
        lambda form: httpx.Response(200, json={"token_type": "Bearer"}),
        lambda form: httpx.Response(200, json=["a list"]),
    ],
    ids=[
        "connect",
        "timeout",
        "500",
        "503-with-terminal-body",
        "429",
        "unknown-400",
        "401-without-code",
        "200-not-json",
        "200-no-token",
        "200-not-object",
    ],
)
def test_temporary_network_failure_keeps_tokens(openai, reply):
    before = _creds(expires_at=0)
    save(before)
    openai.token_reply = reply
    with pytest.raises(RefreshUnavailable) as caught:
        get_access_token()
    assert load() == before
    _assert_clean(caught.value)


@pytest.mark.parametrize(
    "code",
    [
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
def test_terminal_refresh_error_clears_tokens(openai, code, nested):
    save(_creds(expires_at=0))
    body = {"error": {"code": code}} if nested else {"error": code}
    openai.token_reply = lambda form: httpx.Response(400, json=body)
    with pytest.raises(RefreshRejected) as caught:
        get_access_token()
    assert caught.value.code == code
    assert load() is None
    text = _token_path().read_text(encoding="utf-8")
    assert ACCESS not in text
    assert REFRESH not in text
    assert "id-token-secret" not in text
    assert _record()["client_id"] == ISSUED
    with pytest.raises(NotSignedIn):
        get_access_token()
    assert len(openai.token_forms) == 1


def test_relogin_after_terminal_error_uses_saved_client(openai):
    login(open_browser=_browser(openai), timeout=10)
    cp._write_record(cp._record_from(cp.replace(load(), expires_at=0)))
    openai.token_reply = lambda form: httpx.Response(
        400, json={"error": "refresh_token_reused"}
    )
    with pytest.raises(RefreshRejected):
        get_access_token()
    openai.token_reply = None
    login(open_browser=_browser(openai, client_id=None), timeout=10)
    assert _query(openai.authorize_urls[1])["client_id"] == [ISSUED]
    assert "agent_name_hint" not in _query(openai.authorize_urls[1])
    assert get_access_token() == ACCESS


def test_invalid_client_keeps_tokens(openai):
    before = _creds(expires_at=0)
    save(before)
    openai.token_reply = lambda form: httpx.Response(
        401, json={"error": "invalid_client"}
    )
    with pytest.raises(RefreshRejected) as caught:
        get_access_token()
    assert caught.value.code == "invalid_client"
    assert load() == before


def test_refresh_error_text_is_whitelisted(openai):
    save(_creds(expires_at=0))
    openai.token_reply = lambda form: httpx.Response(
        400,
        json={
            "error": "invalid_grant",
            "error_description": f"token {form['refresh_token']} was reused",
            "refresh_token": form["refresh_token"],
        },
    )
    with pytest.raises(RefreshRejected) as caught:
        get_access_token()
    _assert_clean(caught.value)


def test_error_code_with_injected_text_is_dropped(openai):
    before = _creds(expires_at=0)
    save(before)
    openai.token_reply = lambda form: httpx.Response(
        400, json={"error": f"bad {form['refresh_token']}"}
    )
    with pytest.raises(RefreshUnavailable) as caught:
        get_access_token()
    assert caught.value.code is None
    _assert_clean(caught.value)
    assert load() == before


def test_get_access_token_is_safe_in_a_worker_thread(openai):
    import asyncio

    save(_creds(expires_at=0))
    openai.token_reply = _rotating(openai)

    async def run():
        return await asyncio.gather(
            asyncio.to_thread(get_access_token), asyncio.to_thread(get_access_token)
        )

    assert asyncio.run(run()) == ["access-secret-2", "access-secret-2"]
    assert len(openai.token_forms) == 1


@pytest.mark.parametrize("skew", [-30, 30], ids=["clock-slow", "clock-fast"])
def test_clock_within_a_minute_still_signs_in(openai, skew):
    now = int(time.time())
    openai.id_claims = {"iat": now + skew, "exp": now + 3600 + skew}
    assert login(open_browser=_browser(openai), timeout=10).client_id == ISSUED


def test_id_token_from_far_in_the_future_is_refused(openai):
    now = int(time.time())
    openai.id_claims = {"iat": now + 300, "nbf": now + 300, "exp": now + 3900}
    with pytest.raises(LoginFailed) as caught:
        login(open_browser=_browser(openai), timeout=10)
    assert caught.value.code == "id_token_invalid"


def test_unsecurable_store_fails_before_the_token_is_spent(openai, monkeypatch):
    before = _creds(expires_at=0)
    save(before)
    openai.token_reply = _rotating(openai)

    def refuse(path):
        raise cp.CredentialStoreError("no ACL right now")

    monkeypatch.setattr(cp, "_restrict_dir", refuse)
    with pytest.raises(cp.CredentialStoreError):
        get_access_token()
    assert openai.token_forms == []
    assert load() == before


def test_passing_store_failure_after_a_rotation_keeps_the_new_token(
    openai, monkeypatch
):
    save(_creds(expires_at=0))
    openai.token_reply = _rotating(openai)
    real = cp._restrict_file
    failures = [cp.CredentialStoreError("busy")]

    def flaky(path):
        if failures:
            raise failures.pop()
        real(path)

    monkeypatch.setattr(cp, "_restrict_file", flaky)
    assert get_access_token() == "access-secret-2"
    assert load().refresh_token == "refresh-secret-2"
    assert len(openai.token_forms) == 1


def test_lock_outlasts_the_slowest_refresh():
    assert cp._LOCK_TIMEOUT_S > 3 * cp._HTTP_TIMEOUT_S


def test_lock_timeout_reads_as_temporary(openai, monkeypatch):
    save(_creds(expires_at=0))
    monkeypatch.setattr(cp, "_LOCK_TIMEOUT_S", 0.1)
    caught = []

    def worker():
        try:
            get_access_token()
        except Exception as exc:
            caught.append(exc)

    with cp._locked():
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(10)
    assert isinstance(caught[0], RefreshUnavailable)
    _assert_clean(caught[0])
    assert openai.token_forms == []
