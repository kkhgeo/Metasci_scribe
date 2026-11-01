"""M0 spike: Sign in with ChatGPT (direct registration, open-source flow).

Throwaway code for Metasci_scribe Task 1. Follows
https://developers.openai.com/siwc/token-sharing-open-source/sign-in

    python spike_login.py            # open the system browser and sign in
    python spike_login.py --status   # granted scopes + visible model slugs
    python spike_login.py --fresh    # ignore any saved client_id / host id

The token file lives outside the repository: ~/.metasci-scribe/spike_token.json
(override the directory with METASCI_SPIKE_DIR). Tokens are never printed.

Only https://auth.openai.com and https://api.openai.com/v1 are called.
The one non-stdlib dependency besides httpx is PyJWT (+cryptography) for the
ID-token signature check; the docs require verifying it against OpenAI's JWKS.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
API_BASE = "https://api.openai.com/v1"
RESOURCE = API_BASE
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
REGISTRATION_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME_HINT = "Metasci_scribe"
CALLBACK_PATH = "/auth/callback"
PREFERRED_PORT = 1455
LOGIN_TIMEOUT_S = 300
ID_TOKEN_ALGS = ["RS256", "RS384", "RS512", "PS256", "ES256", "ES384"]
UNUSABLE_REFRESH_CODES = {
    "invalid_grant", "invalid_refresh_token", "token_expired",
    "refresh_token_expired", "refresh_token_invalidated", "refresh_token_reused",
}


class AuthError(RuntimeError):
    """Sign-in or refresh failed. `code` is the OAuth error code if known."""

    def __init__(self, message: str, code: str | None = None, terminal: bool = False):
        super().__init__(message)
        self.code = code
        self.terminal = terminal  # True: token unusable, the user must sign in again


def token_path() -> Path:
    base = os.environ.get("METASCI_SPIKE_DIR")
    root = Path(base) if base else Path.home() / ".metasci-scribe"
    return root / "spike_token.json"


# --------------------------------------------------------------------------- PKCE / URL


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def make_pkce() -> tuple[str, str]:
    """Return (verifier, S256 challenge), both unpadded base64url."""
    verifier = b64url(secrets.token_bytes(48))  # 64 chars, within RFC 7636's 43-128
    challenge = b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    nonce: str,
    code_challenge: str,
    ext_agent_host_id: str,
    agent_name_hint: str | None = None,
    id_token_hint: str | None = None,
    login_hint: str | None = None,
) -> str:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "ext_agent_host_id": ext_agent_host_id,
    }
    if client_id == REGISTRATION_CLIENT_ID:
        params["agent_name_hint"] = agent_name_hint or AGENT_NAME_HINT  # first registration only
    if id_token_hint:
        params["id_token_hint"] = id_token_hint
    if login_hint:
        params["login_hint"] = login_hint
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)


# --------------------------------------------------------------------------- callback


class _CallbackServer(HTTPServer):
    result: dict | None = None
    done: threading.Event


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != CALLBACK_PATH:
                self.send_response(404)
                self.end_headers()
                return
            q = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            self.server.result = q  # type: ignore[attr-defined]
            body = b"<html><body><p>Sign-in finished. You can close this tab.</p></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.server.done.set()  # type: ignore[attr-defined]

        def log_message(self, *args):  # never log the request line (it carries the code)
            pass

    return Handler


def start_callback_server() -> tuple[_CallbackServer, str]:
    """Bind 127.0.0.1 (never localhost). Prefer 1455, fall back to an ephemeral port."""
    last: Exception | None = None
    for port in (PREFERRED_PORT, 0):
        try:
            srv = _CallbackServer(("127.0.0.1", port), _make_handler())
            srv.done = threading.Event()
            srv.result = None
            return srv, f"http://127.0.0.1:{srv.server_address[1]}{CALLBACK_PATH}"
        except OSError as exc:
            last = exc
    raise AuthError(f"cannot bind a loopback port: {last}")


def check_callback(result: dict, expected_state: str, pending_client_id: str) -> tuple[str, str]:
    """Validate state, OAuth errors, code, and issued client_id. Return (code, client_id)."""
    got_state = result.get("state", "")
    if not hmac.compare_digest(got_state.encode(), expected_state.encode()):
        raise AuthError("state mismatch on callback; discarding the result")
    if "error" in result:
        raise AuthError(
            f"authorization error: {result['error']} {result.get('error_description', '')}".strip(),
            code=result["error"],
        )
    code = result.get("code")
    if not code:
        raise AuthError("callback has no authorization code")
    returned = result.get("client_id")
    if pending_client_id == REGISTRATION_CLIENT_ID:
        if not returned or returned == REGISTRATION_CLIENT_ID:
            raise AuthError("registration incomplete: callback carried no issued client_id")
        return code, returned
    if returned and returned != pending_client_id:
        raise AuthError("callback client_id differs from the pending client; rejecting")
    return code, pending_client_id


# --------------------------------------------------------------------------- ID token


def _jwks_key(jwks: dict, kid: str | None) -> dict:
    keys = jwks.get("keys", [])
    for k in keys:
        if kid is None or k.get("kid") == kid:
            return k
    raise AuthError(f"no JWKS key for kid={kid!r}")


def validate_id_token(
    id_token: str, *, jwks: dict, issuer: str, client_id: str, nonce: str
) -> dict:
    """Verify signature, iss, aud, exp, nonce, sub (docs: sign-in step 4)."""
    import jwt  # PyJWT

    header = jwt.get_unverified_header(id_token)
    alg = header.get("alg")
    if alg not in ID_TOKEN_ALGS:
        raise AuthError(f"ID token alg {alg!r} not allowed")
    key = jwt.PyJWK(_jwks_key(jwks, header.get("kid"))).key
    try:
        payload = jwt.decode(
            id_token,
            key,
            algorithms=[alg],
            audience=client_id,
            issuer=issuer,
            leeway=5,
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"ID token invalid: {exc}") from exc
    if not hmac.compare_digest(str(payload.get("nonce", "")).encode(), nonce.encode()):
        raise AuthError("ID token nonce mismatch")
    if not isinstance(payload.get("sub"), str) or not payload["sub"]:
        raise AuthError("ID token has no subject")
    sid = payload.get("sid")
    if sid is not None and (not isinstance(sid, str) or not sid.strip()):
        raise AuthError("ID token has an invalid sid")
    return payload


def fetch_jwks(client: httpx.Client) -> tuple[str, dict]:
    disc = client.get(DISCOVERY_URL)
    disc.raise_for_status()
    doc = disc.json()
    issuer = doc.get("issuer", ISSUER)
    if issuer != ISSUER:
        raise AuthError(f"unexpected issuer in discovery document: {issuer!r}")
    jwks = client.get(doc["jwks_uri"])
    jwks.raise_for_status()
    return issuer, jwks.json()


# --------------------------------------------------------------------------- token store


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_token(path: Path | None = None) -> dict | None:
    p = path or token_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def save_token(record: dict, path: Path | None = None) -> Path:
    p = path or token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, p)
    return p


def apply_token_response(record: dict, resp: dict, *, now: float | None = None) -> dict:
    """Replace access token, refresh token, scopes and expiry together."""
    now = time.time() if now is None else now
    out = dict(record)
    out["access_token"] = resp["access_token"]
    if resp.get("refresh_token"):
        out["refresh_token"] = resp["refresh_token"]
    if resp.get("id_token"):
        out["id_token"] = resp["id_token"]
    out["token_type"] = resp.get("token_type", "Bearer")
    out["expires_in"] = resp.get("expires_in", 3600)
    out["expires_at"] = now + float(out["expires_in"])
    if resp.get("earliest_refresh_at") is not None:
        out["earliest_refresh_at"] = resp["earliest_refresh_at"]
    if resp.get("scope"):
        out["scopes"] = sorted(resp["scope"].split())
    out["saved_at"] = utc_now()
    return out


def _oauth_error(resp: httpx.Response) -> tuple[str | None, str]:
    try:
        body = resp.json()
    except ValueError:
        return None, resp.text[:500]
    err = body.get("error")
    if isinstance(err, dict):
        err = err.get("code")
    safe = {k: v for k, v in body.items() if k in ("error", "error_description", "code", "message", "detail")}
    return err or body.get("code"), json.dumps(safe)


_refresh_lock = threading.Lock()


def refresh_token(record: dict, client: httpx.Client | None = None) -> dict:
    """grant_type=refresh_token with the saved issued client_id (never dynamic_agent_client)."""
    if record.get("client_id") in (None, REGISTRATION_CLIENT_ID):
        raise AuthError("saved record has no issued client_id; sign in again", terminal=True)
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        try:
            resp = client.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "client_id": record["client_id"],
                    "refresh_token": record["refresh_token"],
                    "resource": RESOURCE,
                },
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:  # transient network error: keep credentials
            raise AuthError(f"network error during refresh: {exc}") from exc
        if resp.status_code != 200:
            code, detail = _oauth_error(resp)
            terminal = code in UNUSABLE_REFRESH_CODES
            raise AuthError(
                f"refresh failed HTTP {resp.status_code} code={code}: {detail}"
                + (" -- sign in again" if terminal else ""),
                code=code,
                terminal=terminal,
            )
        return apply_token_response(record, resp.json())
    finally:
        if own:
            client.close()


def get_access_token(path: Path | None = None, skew_s: float = 300.0, client: httpx.Client | None = None) -> str:
    """Return a valid access token, refreshing (and saving) it when near expiry."""
    with _refresh_lock:  # single-process serialization is enough for the spike
        rec = load_token(path)
        if rec is None:
            raise AuthError("no saved token; run: python spike_login.py", terminal=True)
        if DIRECT_SCOPE not in rec.get("scopes", []):
            raise AuthError(f"granted scopes lack {DIRECT_SCOPE}; plan usage is off", terminal=True)
        if rec.get("expires_at", 0) - time.time() <= skew_s:
            rec = refresh_token(rec, client)
            save_token(rec, path)
        return rec["access_token"]


# --------------------------------------------------------------------------- models


def list_models(access_token: str, client: httpx.Client | None = None) -> tuple[int, list[dict], str | None, str]:
    """GET /v1/models. Returns (status, models, request_id, raw_text_if_error)."""
    own = client is None
    client = client or httpx.Client(timeout=30)
    try:
        resp = client.get(f"{API_BASE}/models", headers={"Authorization": f"Bearer {access_token}"})
    finally:
        if own:
            client.close()
    rid = resp.headers.get("x-request-id")
    if resp.status_code != 200:
        return resp.status_code, [], rid, resp.text
    body = resp.json()
    models = body.get("models") if isinstance(body, dict) else None
    if models is None and isinstance(body, dict):
        models = body.get("data", [])
    return 200, models or [], rid, ""


def visible_slugs(models: list[dict]) -> list[str]:
    out = []
    for m in models:
        if m.get("visibility", "list") == "list":
            out.append(m.get("slug") or m.get("id") or "?")
    return out


# --------------------------------------------------------------------------- commands


def cmd_status() -> int:
    rec = load_token()
    if rec is None:
        print("no saved token; run: python spike_login.py")
        return 1
    scopes = rec.get("scopes", [])
    print("granted scopes:", " ".join(scopes) if scopes else "(none)")
    print(f"{DIRECT_SCOPE} present:", DIRECT_SCOPE in scopes)
    try:
        access = get_access_token()
    except AuthError as exc:
        print("cannot get access token:", exc)
        return 1
    status, models, rid, raw = list_models(access)
    if status != 200:
        print(f"GET /v1/models -> HTTP {status} request_id={rid}")
        print("error body:", raw)
        return 1
    slugs = visible_slugs(models)
    print(f"visible model slugs ({len(slugs)}):")
    for s in slugs:
        print("  ", s)
    return 0


def cmd_login(fresh: bool) -> int:
    existing = None if fresh else load_token()
    host_id = (existing or {}).get("ext_agent_host_id") or f"urn:uuid:{uuid.uuid4()}"
    issued = (existing or {}).get("client_id")
    client_id = issued or REGISTRATION_CLIENT_ID

    verifier, challenge = make_pkce()
    state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    server, redirect_uri = start_callback_server()
    url = build_authorize_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        nonce=nonce,
        code_challenge=challenge,
        ext_agent_host_id=host_id,
        id_token_hint=(existing or {}).get("id_token") if issued else None,
        login_hint=(existing or {}).get("email") if issued else None,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print(f"callback listener: {redirect_uri}")
        print(f"mode: {'re-login with saved client_id' if issued else 'first-time registration'}")
        if not webbrowser.open(url):
            print("could not open a browser; open this URL yourself:\n" + url)
        else:
            print(f"browser opened; waiting up to {LOGIN_TIMEOUT_S}s for the callback ...")
        if not server.done.wait(LOGIN_TIMEOUT_S):
            print(f"timed out after {LOGIN_TIMEOUT_S}s without a callback")
            return 2
        result = server.result or {}
    finally:
        server.shutdown()
        server.server_close()

    try:
        code, client_id = check_callback(result, state, client_id)
        with httpx.Client(timeout=30) as http:
            resp = http.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": redirect_uri,
                    "resource": RESOURCE,
                },
                headers={"Accept": "application/json"},
            )
            if resp.status_code != 200:
                ecode, detail = _oauth_error(resp)
                raise AuthError(f"code exchange failed HTTP {resp.status_code} code={ecode}: {detail}", code=ecode)
            tok = resp.json()
            issuer, jwks = fetch_jwks(http)
        if not tok.get("id_token"):
            raise AuthError("token response has no id_token")
        claims = validate_id_token(tok["id_token"], jwks=jwks, issuer=issuer, client_id=client_id, nonce=nonce)
    except (AuthError, httpx.HTTPError) as exc:
        print("sign-in failed:", exc)
        return 1

    if existing and existing.get("subject") and existing["subject"] != claims["sub"]:
        print("signed-in identity differs from the saved account; not replacing credentials")
        return 1

    record = {
        "email": claims.get("email"),
        "issuer": claims.get("iss"),
        "subject": claims["sub"],
        "client_id": client_id,
        "ext_agent_host_id": host_id,
    }
    record = apply_token_response(record, tok)
    if "scopes" not in record:  # fall back to the callback's scope list
        record["scopes"] = sorted((result.get("scope") or "").split())
    path = save_token(record)
    has_direct = DIRECT_SCOPE in record["scopes"]
    print(f"saved credentials to {path}")
    print("granted scopes:", " ".join(record["scopes"]))
    print(f"{DIRECT_SCOPE} present:", has_direct)
    if not has_direct:
        print("WARNING: plan usage is OFF for this grant; do not run inference.")
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="print granted scopes and visible model slugs")
    ap.add_argument("--fresh", action="store_true", help="ignore any saved client_id / host id")
    args = ap.parse_args(argv)
    return cmd_status() if args.status else cmd_login(args.fresh)


if __name__ == "__main__":
    sys.exit(main())
