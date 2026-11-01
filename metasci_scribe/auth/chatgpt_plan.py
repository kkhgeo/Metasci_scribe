"""Sign in with ChatGPT (direct registration) and the tokens it yields.

Follows OpenAI's open-source flow:
https://developers.openai.com/siwc/token-sharing-open-source/sign-in

``DATA_DIR/auth`` holds ``chatgpt.json`` (one account) and ``host_id``. The
folder and everything in it are limited to the current user, and a file that
cannot be limited is not written.

Secrets stay out of exception text and out of the exception chain: an httpx
exception holds its request, form body included, and a JSON error holds the
document it failed on. Failures are therefore raised after the ``except``
block ends, with a fixed message and a code that passed ``_CODE_PATTERN``.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import socket
import socketserver
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import httpx
import jwt
from filelock import FileLock, Timeout

from .. import paths

ISSUER = "https://auth.openai.com"
AUTHORIZE_URL = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
RESOURCE = "https://api.openai.com/v1"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPES = f"openid profile email offline_access resource.invoke {DIRECT_SCOPE}"
REGISTRATION_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME_HINT = "Metasci_scribe"
LOOPBACK = "127.0.0.1"
CALLBACK_PATH = "/auth/callback"
PREFERRED_PORT = 1455
REFRESH_SKEW_S = 60.0
DEFAULT_EXPIRES_IN_S = 3600.0

_ID_TOKEN_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "PS256", "ES256", "ES384"})
_UNUSABLE_REFRESH_CODES = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)
_CODE_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_HOST_ID_PATTERN = re.compile(r"urn:uuid:[0-9a-f-]{36}")
_HTTP_TIMEOUT_S = 30.0
# Held across the refresh request, so a waiter must outlast one.
_LOCK_TIMEOUT_S = 90.0
_SHARING_RETRY_S = 2.0
# Windows does not deliver Ctrl+C to a long Event.wait().
_WAIT_SLICE_S = 0.2
_TEMP_SUFFIX = ".tmp"

_T = TypeVar("_T")


class ChatGPTAuthError(RuntimeError):
    """Sign-in or token failure. ``code`` is a checked OAuth code, when known."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class NotSignedIn(ChatGPTAuthError):
    """No usable sign-in is saved; the user must sign in."""


class LoginCancelled(ChatGPTAuthError):
    """The user declined in the browser."""


class LoginTimeout(ChatGPTAuthError):
    """No valid callback arrived in time."""


class LoginFailed(ChatGPTAuthError):
    """The attempt ended without credentials; saved ones are unchanged."""


class RefreshRejected(ChatGPTAuthError):
    """OpenAI refused the refresh; the user must sign in again.

    Unusable tokens have been cleared. ``invalid_client`` keeps them, since
    that points at the client's configuration and not at the tokens.
    """


class RefreshUnavailable(ChatGPTAuthError):
    """The refresh did not complete. Tokens are kept; retry later."""


class PlanUsageDisabled(ChatGPTAuthError):
    """Signed in, but the grant lacks ``chatgpt.tokens.use.direct``."""


class CredentialStoreError(ChatGPTAuthError):
    """The auth folder cannot be secured, read, written or locked."""


@dataclass(frozen=True)
class Credentials:
    """A signed-in account. Tokens and identity are left out of ``repr``."""

    client_id: str
    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    id_token: str = field(repr=False)
    expires_at: float
    scopes: frozenset[str]
    email: str | None = field(repr=False)
    subject: str | None = field(default=None, repr=False)

    @property
    def plan_enabled(self) -> bool:
        return DIRECT_SCOPE in self.scopes


# ---------------------------------------------------------------- file access


def _retry_sharing(operation: Callable[[], _T]) -> _T:
    """Run ``operation``, waiting out Windows' sharing violations.

    Replacing a file another process has open, or opening one mid-replace,
    fails on Windows for as long as that lasts. Giving up on a rotated
    refresh token because a reader was active would lose the sign-in.
    """
    deadline = time.monotonic() + _SHARING_RETRY_S
    while True:
        try:
            return operation()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _win_set_owner_only_dacl(path: Path, *, inherit: bool) -> None:
    """Replace the object's DACL with one entry: full control for this user.

    The whole list is replaced, so entries somebody added stay out, which
    ``icacls /grant`` would leave in place. A protected DACL takes nothing
    from the parent. ``inherit`` passes the entry on to the folder's children.
    """
    import ctypes
    from ctypes import byref, c_int, c_void_p, wintypes

    TOKEN_QUERY = 0x0008
    TOKEN_USER = 1
    SDDL_REVISION_1 = 1
    SE_FILE_OBJECT = 1
    DACL_SECURITY_INFORMATION = 0x00000004
    PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.LocalFree.argtypes = [c_void_p]
    kernel32.LocalFree.restype = c_void_p
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        c_int,
        c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.ConvertSidToStringSidW.argtypes = [
        c_void_p,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(c_void_p),
        c_void_p,
    ]
    advapi32.GetSecurityDescriptorDacl.argtypes = [
        c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        c_int,
        wintypes.DWORD,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD

    def failed(step: str, error: int) -> CredentialStoreError:
        return CredentialStoreError(
            f"Could not restrict {path} to the current user "
            f"({step}, Windows error {error})."
        )

    token = wintypes.HANDLE()
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), TOKEN_QUERY, byref(token)
    ):
        raise failed("OpenProcessToken", ctypes.get_last_error())
    try:
        size = wintypes.DWORD()
        advapi32.GetTokenInformation(token, TOKEN_USER, None, 0, byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi32.GetTokenInformation(
            token, TOKEN_USER, buffer, size, byref(size)
        ):
            raise failed("GetTokenInformation", ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(token)

    # TOKEN_USER starts with the pointer to the user's SID.
    user_sid = ctypes.cast(buffer, ctypes.POINTER(c_void_p)).contents
    sid_text = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(user_sid, byref(sid_text)):
        raise failed("ConvertSidToStringSid", ctypes.get_last_error())
    sid = sid_text.value
    kernel32.LocalFree(ctypes.cast(sid_text, c_void_p))

    flags = "OICI" if inherit else ""
    descriptor = c_void_p()
    if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        f"D:P(A;{flags};FA;;;{sid})", SDDL_REVISION_1, byref(descriptor), None
    ):
        raise failed("building the descriptor", ctypes.get_last_error())
    try:
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        dacl = c_void_p()
        if not advapi32.GetSecurityDescriptorDacl(
            descriptor, byref(present), byref(dacl), byref(defaulted)
        ):
            raise failed("GetSecurityDescriptorDacl", ctypes.get_last_error())
        error = advapi32.SetNamedSecurityInfoW(
            str(path),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            dacl,
            None,
        )
        if error:
            raise failed("SetNamedSecurityInfo", error)
    finally:
        kernel32.LocalFree(descriptor)


def _restrict_dir(path: Path) -> None:
    if sys.platform == "win32":
        _win_set_owner_only_dacl(path, inherit=True)
    else:
        os.chmod(path, 0o700)


def _restrict_file(path: Path) -> None:
    if sys.platform == "win32":
        _win_set_owner_only_dacl(path, inherit=False)
    else:
        os.chmod(path, 0o600)


def _auth_dir() -> Path:
    # Read at call time: ``paths.DATA_DIR`` is not fixed for the tests.
    return paths.DATA_DIR / "auth"


def _token_path() -> Path:
    return _auth_dir() / "chatgpt.json"


def _make_dir() -> Path:
    directory = _auth_dir()
    try:
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.is_symlink():
            raise CredentialStoreError(f"{directory} must not be a symbolic link.")
    except OSError as exc:
        raise CredentialStoreError(
            f"Could not create {directory} ({type(exc).__name__})."
        ) from None
    return directory


def _secure_dir() -> Path:
    """Limit the auth folder to this user, or refuse. Call with the lock held:
    on Windows this walks the children, which must not be changing."""
    directory = _make_dir()
    try:
        _restrict_dir(directory)
    except OSError as exc:
        raise CredentialStoreError(
            f"Could not secure {directory} ({type(exc).__name__})."
        ) from None
    return directory


@contextmanager
def _locked() -> Iterator[None]:
    """Hold the lock every process takes to write in the auth folder.

    One lock object per path, so a thread that holds it can take it again
    (``save`` from inside a locked section).
    """
    lock = FileLock(str(_make_dir() / "chatgpt.lock"), is_singleton=True)
    try:
        lock.acquire(timeout=_LOCK_TIMEOUT_S)
    except Timeout:
        raise CredentialStoreError(
            "Another Metasci Scribe process is holding the sign-in lock."
        ) from None
    try:
        yield
    finally:
        lock.release()


def _remove_stale_temps(directory: Path) -> None:
    """Remove temp files a crash left. Call with the lock held."""
    for leftover in directory.glob(f"*{_TEMP_SUFFIX}"):
        with contextlib.suppress(OSError):
            leftover.unlink()


def _write_private(path: Path, data: bytes) -> None:
    """Atomically write ``data``. Call with the lock held.

    The temp file is limited to this user before it holds anything, so a
    crash can leave it behind but not readable by anyone else.
    """
    directory = _secure_dir()
    _remove_stale_temps(directory)
    temp: Path | None = None
    try:
        fd, name = tempfile.mkstemp(
            dir=directory, prefix=f"{path.name}.", suffix=_TEMP_SUFFIX
        )
        temp = Path(name)
        with os.fdopen(fd, "wb") as handle:
            _restrict_file(temp)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        written = temp
        _retry_sharing(lambda: os.replace(written, path))
        temp = None
        if sys.platform != "win32":
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except OSError as exc:
        raise CredentialStoreError(
            f"Could not write {path.name} ({type(exc).__name__})."
        ) from None
    finally:
        if temp is not None:
            with contextlib.suppress(OSError):
                temp.unlink()


def _read_record() -> dict[str, Any] | None:
    """The saved record, or ``None`` when there is none or it is damaged.

    Writes are atomic, so no lock is needed for a whole, consistent file.
    """
    path = _token_path()
    try:
        raw = _retry_sharing(path.read_bytes)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CredentialStoreError(
            f"Could not read {path.name} ({type(exc).__name__})."
        ) from None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def _write_record(record: dict[str, Any]) -> None:
    _write_private(_token_path(), json.dumps(record, indent=2).encode("utf-8"))


def _record_from(creds: Credentials) -> dict[str, Any]:
    return {
        "email": creds.email,
        "issuer": ISSUER,
        "subject": creds.subject,
        "client_id": creds.client_id,
        "id_token": creds.id_token,
        "access_token": creds.access_token,
        "refresh_token": creds.refresh_token,
        "token_type": "Bearer",
        "expires_at": creds.expires_at,
        "scopes": sorted(creds.scopes),
        "saved_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _issued_client_id(record: dict[str, Any]) -> str | None:
    client_id = record.get("client_id")
    if isinstance(client_id, str) and client_id != REGISTRATION_CLIENT_ID:
        return client_id or None
    return None


def _credentials_from(record: dict[str, Any] | None) -> Credentials | None:
    """The record as ``Credentials``, or ``None`` if it holds no usable sign-in."""
    if record is None:
        return None
    client_id = _issued_client_id(record)
    access_token = record.get("access_token")
    refresh_token = record.get("refresh_token")
    id_token = record.get("id_token")
    expires_at = record.get("expires_at")
    scopes = record.get("scopes")
    email = record.get("email")
    subject = record.get("subject")
    usable = (
        client_id
        and isinstance(access_token, str)
        and access_token
        and isinstance(refresh_token, str)
        and isinstance(id_token, str)
        and isinstance(expires_at, int | float)
        and not isinstance(expires_at, bool)
        and isinstance(scopes, list)
        and all(isinstance(scope, str) for scope in scopes)
        and isinstance(email, str | None)
        and isinstance(subject, str | None)
    )
    if not usable:
        return None
    return Credentials(
        client_id=client_id,
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
        expires_at=float(expires_at),
        scopes=frozenset(scopes),
        email=email,
        subject=subject,
    )


def load() -> Credentials | None:
    """The saved sign-in, or ``None`` when signed out."""
    return _credentials_from(_read_record())


def save(c: Credentials) -> None:
    """Replace the saved sign-in."""
    if not c.client_id or c.client_id == REGISTRATION_CLIENT_ID:
        raise ValueError(
            f"Only an issued client ID is saved, never {REGISTRATION_CLIENT_ID}."
        )
    with _locked():
        _write_record(_record_from(c))


def _forget_tokens() -> None:
    """Drop the tokens, keeping the account-to-client mapping for the next
    sign-in. Signing out does not end the registration, and a new one would
    create a second client for this account. Call with the lock held."""
    record = _read_record()
    client_id = _issued_client_id(record) if record else None
    if client_id:
        _write_record(
            {
                "issuer": ISSUER,
                "subject": record.get("subject"),
                "client_id": client_id,
            }
        )
    else:
        try:
            _retry_sharing(lambda: _token_path().unlink(missing_ok=True))
        except OSError as exc:
            raise CredentialStoreError(
                f"Could not remove the saved sign-in ({type(exc).__name__})."
            ) from None
    _remove_stale_temps(_auth_dir())


def clear() -> None:
    """Sign out on this machine: erase the access, refresh and ID tokens.

    Does not revoke the tokens at OpenAI.
    """
    if not _auth_dir().is_dir():
        return
    with _locked():
        _forget_tokens()


def _host_id() -> str:
    """This host's stable ``ext_agent_host_id``, created on first use."""
    path = _auth_dir() / "host_id"
    with _locked():
        try:
            existing = _retry_sharing(path.read_bytes).decode("ascii").strip()
        except (OSError, ValueError):
            existing = ""
        if _HOST_ID_PATTERN.fullmatch(existing):
            return existing
        host_id = f"urn:uuid:{uuid.uuid4()}"
        _write_private(path, host_id.encode("ascii"))
        return host_id


# ---------------------------------------------------------------- authorize URL


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_pkce() -> tuple[str, str]:
    """Return (verifier, S256 challenge)."""
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _check_redirect_uri(redirect_uri: str) -> None:
    """Only the port may vary; OpenAI matches the rest exactly."""
    parts = urlsplit(redirect_uri)
    try:
        port = parts.port
    except ValueError:
        port = None
    if (
        parts.scheme != "http"
        or port is None
        or parts.netloc != f"{LOOPBACK}:{port}"
        or parts.path != CALLBACK_PATH
        or parts.query
        or parts.fragment
    ):
        raise ValueError(
            f"redirect_uri must be http://{LOOPBACK}:<port>{CALLBACK_PATH}"
        )


def build_authorize_url(
    *,
    client_id: str | None,
    redirect_uri: str,
    host_id: str,
    state: str,
    nonce: str,
    code_challenge: str,
    id_token_hint: str | None = None,
) -> str:
    """The authorization URL; ``client_id=None`` registers a new client."""
    _check_redirect_uri(redirect_uri)
    registering = client_id is None or client_id == REGISTRATION_CLIENT_ID
    params = {
        "client_id": REGISTRATION_CLIENT_ID if registering else client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "ext_agent_host_id": host_id,
    }
    if registering:
        params["agent_name_hint"] = AGENT_NAME_HINT
    if id_token_hint:
        params["id_token_hint"] = id_token_hint
    return f"{AUTHORIZE_URL}?{urlencode(params, quote_via=quote)}"


# ---------------------------------------------------------------- callback

_DONE_PAGE = (
    b"<!doctype html><title>Metasci Scribe</title>"
    b"<p>You can close this tab and return to the terminal.</p>"
)
_STRAY_PAGE = (
    b"<!doctype html><title>Metasci Scribe</title>"
    b"<p>This request is not part of the sign-in in progress.</p>"
)


class _CallbackHandler(BaseHTTPRequestHandler):
    # An idle connection must not hold its thread forever.
    timeout = 5

    def do_GET(self) -> None:
        server: _CallbackServer = self.server  # type: ignore[assignment]
        url = urlsplit(self.path)
        if url.path != CALLBACK_PATH:
            self._reply(404, _STRAY_PAGE)
            return
        params = {
            name: values[0]
            for name, values in parse_qs(url.query, keep_blank_values=True).items()
        }
        state = params.get("state", "").encode()
        if not hmac.compare_digest(state, server.expected_state.encode()):
            # Anything local can reach this port. It is ignored, not fatal:
            # otherwise a stray request could cancel the user's sign-in.
            self._reply(400, _STRAY_PAGE)
            return
        with server.result_lock:
            first = server.params is None
            if first:
                server.params = params
        try:
            self._reply(200, _DONE_PAGE)
        finally:
            if first:
                server.done.set()

    def _reply(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        """Silent: the request line carries the authorization code."""


class _CallbackServer(ThreadingHTTPServer):
    """One thread per connection: a browser's idle preconnect would otherwise
    stop the real callback from being read."""

    block_on_close = False
    # On Windows SO_REUSEADDR lets a socket share a port another one is
    # listening on, which would hide a busy port and let traffic go astray.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, port: int, expected_state: str) -> None:
        self.expected_state = expected_state
        self.params: dict[str, str] | None = None
        self.result_lock = threading.Lock()
        self.done = threading.Event()
        super().__init__((LOOPBACK, port), _CallbackHandler)

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        # Not HTTPServer.server_bind: it resolves the host name for no use here.
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Silent: the default prints a traceback for a dropped connection."""


def _start_callback_server(state: str) -> _CallbackServer:
    """Bind 127.0.0.1 only: the preferred port, else any free one."""
    try:
        return _CallbackServer(PREFERRED_PORT, state)
    except OSError:
        pass
    try:
        return _CallbackServer(0, state)
    except OSError as exc:
        raise LoginFailed(
            f"Could not open a local port for the sign-in ({type(exc).__name__}).",
            code="no_port",
        ) from None


def _await_callback(server: _CallbackServer, timeout: float) -> dict[str, str]:
    deadline = time.monotonic() + timeout
    while not server.done.wait(_WAIT_SLICE_S):
        if time.monotonic() >= deadline:
            raise LoginTimeout("No answer from the browser in time.")
    return server.params or {}


def _safe_code(value: Any) -> str | None:
    if isinstance(value, str) and _CODE_PATTERN.fullmatch(value):
        return value
    return None


def _resolve_client_id(saved: str | None, returned: str | None) -> str:
    """The client this attempt belongs to, from a state-validated callback."""
    if saved is None:
        if not returned or returned == REGISTRATION_CLIENT_ID:
            raise LoginFailed(
                "The registration did not finish: no client ID was issued.",
                code="registration_incomplete",
            )
        return returned
    if returned and returned != saved:
        raise LoginFailed(
            "The sign-in returned a different client than the saved one.",
            code="client_mismatch",
        )
    return saved


# ---------------------------------------------------------------- token endpoint


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=_HTTP_TIMEOUT_S)


def _post_token(form: dict[str, str]) -> httpx.Response | None:
    """POST to the token endpoint; ``None`` when no answer arrived."""
    try:
        with _http_client() as client:
            return client.post(
                TOKEN_URL, data=form, headers={"Accept": "application/json"}
            )
    except (httpx.HTTPError, OSError):
        return None


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


def _error_code(body: dict[str, Any] | None) -> str | None:
    """The one field read from an error body. The rest can echo the request."""
    if body is None:
        return None
    error = body.get("error")
    if isinstance(error, dict):
        error = error.get("code")
    return _safe_code(error if error is not None else body.get("code"))


def _expires_at(body: dict[str, Any]) -> float:
    expires_in = body.get("expires_in")
    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, int | float)
        or not math.isfinite(expires_in)
        or expires_in <= 0
    ):
        expires_in = DEFAULT_EXPIRES_IN_S
    return time.time() + float(expires_in)


def _nonempty_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _exchange_code(form: dict[str, str]) -> dict[str, Any]:
    response = _post_token(form)
    if response is None:
        raise LoginFailed("Could not reach the sign-in server.", code="network_error")
    body = _json_object(response)
    if response.status_code != 200:
        code = _error_code(body)
        raise LoginFailed(
            f"The sign-in server refused the exchange (HTTP "
            f"{response.status_code}, {code or 'no code'}). Sign in again.",
            code=code,
        )
    if (
        body is None
        or not _nonempty_str(body.get("access_token"))
        or not _nonempty_str(body.get("id_token"))
    ):
        raise LoginFailed(
            "The sign-in server sent an incomplete answer.",
            code="invalid_token_response",
        )
    return body


def _fetch_jwks() -> dict[str, Any] | None:
    try:
        with _http_client() as client:
            discovery = client.get(DISCOVERY_URL)
            document = _json_object(discovery) if discovery.status_code == 200 else None
            if document is None or document.get("issuer") != ISSUER:
                return None
            jwks_uri = document.get("jwks_uri")
            if not isinstance(jwks_uri, str) or not jwks_uri.startswith("https://"):
                return None
            keys = client.get(jwks_uri)
            return _json_object(keys) if keys.status_code == 200 else None
    except (httpx.HTTPError, OSError):
        return None


def _verified_claims(
    id_token: str, jwks: dict[str, Any], client_id: str, nonce: str
) -> dict[str, Any] | None:
    """The claims if signature, issuer, audience, expiry and nonce all hold."""
    try:
        header = jwt.get_unverified_header(id_token)
        algorithm = header.get("alg")
        if algorithm not in _ID_TOKEN_ALGORITHMS:
            return None
        kid = header.get("kid")
        candidates = [
            key
            for key in jwks.get("keys", [])
            if isinstance(key, dict) and (kid is None or key.get("kid") == kid)
        ]
        if not candidates:
            return None
        claims = jwt.decode(
            id_token,
            jwt.PyJWK(candidates[0]).key,
            algorithms=[algorithm],
            audience=client_id,
            issuer=ISSUER,
            leeway=5,
            options={"require": ["iss", "aud", "sub", "exp", "iat"]},
        )
    except (jwt.PyJWTError, ValueError, TypeError, AttributeError):
        return None
    if not hmac.compare_digest(str(claims.get("nonce", "")).encode(), nonce.encode()):
        return None
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        return None
    sid = claims.get("sid")
    if sid is not None and (not isinstance(sid, str) or not sid.strip()):
        return None
    return claims


def _validate_id_token(id_token: str, client_id: str, nonce: str) -> dict[str, Any]:
    jwks = _fetch_jwks()
    if jwks is None:
        raise LoginFailed(
            "Could not fetch OpenAI's signing keys; nothing was saved.",
            code="jwks_unavailable",
        )
    claims = _verified_claims(id_token, jwks, client_id, nonce)
    if claims is None:
        raise LoginFailed(
            "The sign-in's ID token did not validate; nothing was saved.",
            code="id_token_invalid",
        )
    return claims


# ---------------------------------------------------------------- sign-in


def _try_open(open_browser: Callable[[str], bool], url: str) -> bool:
    try:
        return bool(open_browser(url))
    except Exception:
        return False


def login(
    *,
    open_browser: Callable[[str], bool] = webbrowser.open,
    timeout: float = 300.0,
) -> Credentials:
    """Sign in through the system browser and save the result.

    Raises :class:`LoginCancelled`, :class:`LoginTimeout` or
    :class:`LoginFailed`; on any of them saved tokens stay as they were.
    """
    saved = _read_record() or {}
    saved_client_id = _issued_client_id(saved)
    id_token_hint = _nonempty_str(saved.get("id_token")) if saved_client_id else None
    host_id = _host_id()

    verifier, challenge = _make_pkce()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    server = _start_callback_server(state)
    redirect_uri = f"http://{LOOPBACK}:{server.server_address[1]}{CALLBACK_PATH}"
    url_args = {
        "client_id": saved_client_id,
        "redirect_uri": redirect_uri,
        "host_id": host_id,
        "state": state,
        "nonce": nonce,
        "code_challenge": challenge,
    }
    listener = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    listener.start()
    try:
        url = build_authorize_url(**url_args, id_token_hint=id_token_hint)
        if not _try_open(open_browser, url):
            # Same attempt without the hint: it only skips the account
            # selector, and it must not reach the terminal or its scrollback.
            print(
                "Could not open a browser. Open this address to sign in:\n"
                + build_authorize_url(**url_args)
            )
        params = _await_callback(server, timeout)
    finally:
        server.shutdown()
        server.server_close()
        listener.join(5)

    error = params.get("error")
    if error == "access_denied":
        raise LoginCancelled("The sign-in was declined in the browser.")
    if error is not None:
        code = _safe_code(error)
        raise LoginFailed(f"The sign-in failed ({code or 'unknown error'}).", code=code)
    authorization_code = _nonempty_str(params.get("code"))
    if authorization_code is None:
        raise LoginFailed("The callback held no authorization code.", code="no_code")
    client_id = _resolve_client_id(saved_client_id, params.get("client_id"))

    if saved_client_id is None:
        # Kept now: OpenAI wants a failed exchange restarted with this client,
        # not registered anew. No token and no active sign-in depend on it.
        with _locked():
            if _issued_client_id(_read_record() or {}) is None:
                _write_record({"issuer": ISSUER, "client_id": client_id})

    tokens = _exchange_code(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": authorization_code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "resource": RESOURCE,
        }
    )
    claims = _validate_id_token(tokens["id_token"], client_id, nonce)
    saved_subject = saved.get("subject")
    if saved_subject and saved_subject != claims["sub"]:
        raise LoginFailed(
            "This is a different ChatGPT account from the saved one.",
            code="account_mismatch",
        )

    # The token response decides the grant, not the callback's own list.
    scope = tokens.get("scope")
    creds = Credentials(
        client_id=client_id,
        access_token=tokens["access_token"],
        # Absent when the user declined plan usage: the sign-in stands.
        refresh_token=_nonempty_str(tokens.get("refresh_token")) or "",
        id_token=tokens["id_token"],
        expires_at=_expires_at(tokens),
        scopes=frozenset(scope.split()) if isinstance(scope, str) else frozenset(),
        email=_nonempty_str(claims.get("email")),
        subject=claims["sub"],
    )
    save(creds)
    return creds


# ---------------------------------------------------------------- refresh


def _refresh(creds: Credentials) -> Credentials:
    """Spend the refresh token and persist what replaces it. Lock held."""
    response = _post_token(
        {
            "grant_type": "refresh_token",
            "client_id": creds.client_id,
            "refresh_token": creds.refresh_token,
            "resource": RESOURCE,
        }
    )
    if response is None:
        raise RefreshUnavailable("Could not reach the sign-in server.")
    status = response.status_code
    body = _json_object(response)
    if status == 200:
        access_token = _nonempty_str(body.get("access_token")) if body else None
        if body is None or access_token is None:
            raise RefreshUnavailable("The sign-in server sent an incomplete answer.")
        scope = body.get("scope")
        # The ID token stays the one validated at sign-in; it serves as a hint.
        renewed = replace(
            creds,
            access_token=access_token,
            refresh_token=_nonempty_str(body.get("refresh_token"))
            or creds.refresh_token,
            expires_at=_expires_at(body),
            scopes=frozenset(scope.split())
            if isinstance(scope, str) and scope
            else creds.scopes,
        )
        _write_record(_record_from(renewed))
        return renewed

    code = _error_code(body)
    if 400 <= status < 500:  # a 5xx says nothing about the token, whatever its body
        if code in _UNUSABLE_REFRESH_CODES:
            _forget_tokens()
            raise RefreshRejected(
                f"ChatGPT sign-in is no longer valid ({code}). Sign in again.",
                code=code,
            )
        if code == "invalid_client":
            raise RefreshRejected(
                "OpenAI does not accept this client (invalid_client).", code=code
            )
    raise RefreshUnavailable(
        f"The refresh failed (HTTP {status}, {code or 'no code'}).", code=code
    )


def get_access_token() -> str:
    """A valid access token, refreshed when it is within a minute of expiry.

    Blocks on the file lock and on HTTP, so async code must call it through
    ``asyncio.to_thread``.

    Raises :class:`NotSignedIn`, :class:`PlanUsageDisabled`,
    :class:`RefreshRejected`, :class:`RefreshUnavailable` or
    :class:`CredentialStoreError`.
    """
    # The lock covers the read: a token a refresh is about to replace, or
    # one another process just replaced, must not be spent again.
    with _locked():
        creds = _credentials_from(_read_record())
        if creds is None:
            raise NotSignedIn("Not signed in to ChatGPT.")
        if not creds.plan_enabled:
            raise PlanUsageDisabled("ChatGPT plan usage is not enabled.")
        if creds.expires_at - time.time() > REFRESH_SKEW_S:
            return creds.access_token
        if not creds.refresh_token:
            raise NotSignedIn("The ChatGPT sign-in has expired.")
        creds = _refresh(creds)
        if not creds.plan_enabled:
            raise PlanUsageDisabled("ChatGPT plan usage is not enabled.")
        return creds.access_token
