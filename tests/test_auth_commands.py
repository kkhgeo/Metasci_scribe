"""/login, /logout, the not-signed-in message and the model catalog.

Every sign-in and OpenAI call is a fake: no browser opens, nothing reaches
OpenAI, and the credentials are the ones a test writes into the temp data dir.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import typer

from metasci_scribe import paths
from metasci_scribe.auth import chatgpt_plan as cp
from metasci_scribe.auth.chatgpt_plan import Credentials
from metasci_scribe.llm.credentials import ModelNotConfigured
from tests.test_cli_workspace_resolution import _config, _run_prompt

ALL_SCOPES = (
    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
)
EMAIL = "user@example.com"


def _creds(**changes) -> Credentials:
    values = {
        "client_id": "oaiapp_issued",
        "access_token": "access-secret-1",
        "refresh_token": "refresh-secret-1",
        "id_token": "id-token-secret",
        "expires_at": time.time() + 3600,
        "scopes": frozenset(ALL_SCOPES.split()),
        "email": EMAIL,
        "subject": "user-1",
    }
    return Credentials(**{**values, **changes})


# ------------------------------------------------- the typed not-set-up error


class TestModelNotConfigured:
    def test_message_names_login_and_the_api_key_alternative(self):
        text = str(ModelNotConfigured("openai"))
        assert "/login" in text
        assert "API key" in text
        assert "OPENAI_API_KEY" in text

    def test_anthropic_message_names_its_key_and_login(self):
        text = str(ModelNotConfigured("anthropic"))
        assert "ANTHROPIC_API_KEY" in text
        assert "/login" in text

    def test_openai_without_a_key_raises_it(self):
        from metasci_scribe.agent import _build_chat_model

        cfg = SimpleNamespace(model="gpt-6.1-sol", provider="openai")
        with pytest.raises(ModelNotConfigured):
            _build_chat_model(cfg)

    def test_anthropic_without_a_key_raises_it(self):
        from metasci_scribe.agent import _build_chat_model

        cfg = SimpleNamespace(model="claude-opus-5-5", provider="anthropic")
        with pytest.raises(ModelNotConfigured) as caught:
            _build_chat_model(cfg)
        assert "ANTHROPIC_API_KEY" in str(caught.value)

    def test_anthropic_behind_a_gateway_needs_no_key_here(self, monkeypatch):
        from metasci_scribe.llm.credentials import ensure_model_configured

        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
        ensure_model_configured("anthropic")

    def test_plan_mode_without_a_sign_in_raises_it(self, monkeypatch):
        from metasci_scribe.agent import _build_chat_model

        monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
        cfg = SimpleNamespace(model="gpt-6.1-sol", provider="openai")
        with pytest.raises(ModelNotConfigured) as caught:
            _build_chat_model(cfg)
        assert "/login" in str(caught.value)

    def test_plan_mode_with_a_sign_in_builds_without_the_network(self, monkeypatch):
        from metasci_scribe.agent import _build_chat_model

        monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
        cp.save(_creds())
        cfg = SimpleNamespace(model="gpt-6.1-sol", provider="openai")
        assert _build_chat_model(cfg) is not None

    def test_plan_mode_with_an_unreadable_store_lets_the_request_explain(
        self, monkeypatch
    ):
        from metasci_scribe.llm.credentials import ensure_model_configured

        def broken():
            raise cp.CredentialStoreError("cannot read")

        monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
        monkeypatch.setattr(cp, "load", broken)
        ensure_model_configured("openai")

    def test_an_api_key_builds(self, monkeypatch):
        from metasci_scribe.agent import _build_chat_model

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        cfg = SimpleNamespace(model="gpt-6.1-sol", provider="openai")
        assert _build_chat_model(cfg) is not None

    def test_a_leftover_auxiliary_model_without_a_key_is_refused_by_name(
        self, monkeypatch
    ):
        from metasci_scribe import agent

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        cfg = SimpleNamespace(
            model="gpt-6.1-sol",
            provider="openai",
            auxiliary_model="claude-haiku-5-5",
            auxiliary_provider="anthropic",
        )
        monkeypatch.setattr(agent, "_auxiliary_chat_model", None)
        monkeypatch.setattr(agent, "_auxiliary_chat_model_key", None)
        with (
            patch("metasci_scribe.agent._ensure_config", return_value=cfg),
            pytest.raises(ModelNotConfigured) as caught,
        ):
            agent._ensure_auxiliary_chat_model()
        assert "ANTHROPIC_API_KEY" in str(caught.value)

    def test_it_is_a_runtime_error_without_a_traceback_chain(self):
        error = ModelNotConfigured("openai")
        assert isinstance(error, RuntimeError)
        assert error.provider == "openai"


class TestSingleShotNotSetUp:
    """A first-time user running ``-p`` sees the message, not a traceback."""

    def test_load_failure_prints_the_message_and_exits_1(
        self, monkeypatch, tmp_path, capsys
    ):
        monkeypatch.chdir(tmp_path)

        def not_set_up(**kwargs):
            raise ModelNotConfigured("openai")

        with pytest.raises(typer.Exit) as caught:
            _run_prompt(
                monkeypatch,
                _config(),
                thread_id=None,
                stored=None,
                load_agent=not_set_up,
            )
        assert caught.value.exit_code == 1
        out = capsys.readouterr().out
        assert "/login" in out
        assert "API key" in out
        assert "Traceback" not in out

    def test_other_load_failures_still_propagate(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)

        def broken(**kwargs):
            raise ValueError("something else")

        with pytest.raises(ValueError, match="something else"):
            _run_prompt(
                monkeypatch,
                _config(),
                thread_id=None,
                stored=None,
                load_agent=broken,
            )


class TestInteractiveNotSetUp:
    def test_notice_names_login_and_the_api_key(self, capsys):
        from metasci_scribe.cli import interactive

        interactive._report_model_not_configured(ModelNotConfigured("openai"))
        out = capsys.readouterr().out
        assert "/login" in out
        assert "API key" in out

    def test_a_text_that_merely_mentions_api_key_is_not_the_notice(self):
        from metasci_scribe.cli import interactive

        assert not interactive._is_model_not_configured(
            RuntimeError("bad api_key in the request")
        )
        assert interactive._is_model_not_configured(ModelNotConfigured("openai"))

    def test_single_run_reports_it_and_exits_1(self, monkeypatch, capsys):
        from metasci_scribe.cli import interactive

        def refuse(**kwargs):
            raise ModelNotConfigured("openai")

        monkeypatch.setattr(interactive, "run_streaming", refuse)
        with pytest.raises(typer.Exit):
            interactive.cmd_run(
                MagicMock(),
                "hello",
                "thread-1",
                dirs=MagicMock(work_dir="x"),
                runtime_gateways=MagicMock(),
            )
        out = capsys.readouterr().out
        assert "/login" in out

    def test_single_run_no_longer_guesses_from_error_text(self, monkeypatch, capsys):
        from metasci_scribe.cli import interactive

        def refuse(**kwargs):
            raise RuntimeError("authentication failed: api_key was rejected")

        monkeypatch.setattr(interactive, "run_streaming", refuse)
        with pytest.raises(typer.Exit):
            interactive.cmd_run(
                MagicMock(),
                "hello",
                "thread-1",
                dirs=MagicMock(work_dir="x"),
                runtime_gateways=MagicMock(),
            )
        out = capsys.readouterr().out
        assert "API key not configured" not in out
        assert "authentication failed" in out


# ------------------------------------------------------------ command fakes


class FakeUI:
    """Collects what a command prints; answers the model picker with a cancel."""

    supports_interactive = True

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []
        self.pick_entries = None

    def append_system(self, text: str, style: str = "dim") -> None:
        self.lines.append((text, style))

    def mount_renderable(self, renderable) -> None:
        self.lines.append((str(renderable), "renderable"))

    async def wait_for_model_pick(self, entries, current_model, current_provider):
        self.pick_entries = entries
        return None

    async def flush(self) -> None:
        return None

    @property
    def text(self) -> str:
        return "\n".join(line for line, _ in self.lines)

    def styled(self, style: str) -> str:
        return "\n".join(line for line, s in self.lines if s == style)


def _ctx(ui: FakeUI | None = None, agent=None):
    ctx = MagicMock()
    ctx.ui = ui or FakeUI()
    ctx.agent = agent
    return ctx


def _live_config(**changes):
    values = {
        "model": "gpt-6.1-sol",
        "provider": "openai",
        "openai_auth_mode": "api_key",
        "auxiliary_model": "",
        "auxiliary_provider": "",
    }
    return SimpleNamespace(**{**values, **changes})


@pytest.fixture
def switch(monkeypatch):
    """The session switch, faked: what /login asks of ``/model``'s rebuild."""
    from metasci_scribe.commands.implementation.model import ModelCommand

    fake = AsyncMock(return_value=True)
    monkeypatch.setattr(ModelCommand, "_apply_model", fake)
    return fake


@pytest.fixture
def saved_config():
    with patch(
        "metasci_scribe.config.settings.set_config_values",
        return_value=("rev", {}),
    ) as saved:
        yield saved


@pytest.fixture
def live_config():
    cfg = _live_config()
    with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
        yield cfg


def _sign_in_as(monkeypatch, creds: Credentials, seen: dict | None = None):
    def fake_login(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        cp.save(creds)
        return creds

    monkeypatch.setattr(cp, "login", fake_login)


def _login_raises(monkeypatch, error: Exception):
    def fake_login(**kwargs):
        raise error

    monkeypatch.setattr(cp, "login", fake_login)


# ------------------------------------------------------------------- /login


CATALOG = [("gpt-6.1-sol", "GPT 6.1 Sol"), ("gpt-5.6-terra", "GPT 5.6 Terra")]


class TestLoginCommand:
    @pytest.fixture(autouse=True)
    def catalog(self, monkeypatch):
        """The account's model list, faked; nothing here reaches OpenAI."""
        from metasci_scribe.llm import chatgpt_plan as plan_llm

        listing = MagicMock(return_value=list(CATALOG))
        monkeypatch.setattr(plan_llm, "list_models", listing)
        return listing

    def test_description_uses_the_exact_label(self):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        assert LoginCommand.description.startswith("Continue with ChatGPT")

    def test_is_registered(self):
        from metasci_scribe.commands.implementation import auth  # noqa: F401
        from metasci_scribe.commands.manager import manager

        assert manager.get_command("/login").name == "/login"
        assert manager.get_command("/logout").name == "/logout"

    def test_runs_without_waiting_for_the_agent(self):
        from metasci_scribe.commands.implementation.auth import (
            LoginCommand,
            LogoutCommand,
        )

        assert LoginCommand().needs_agent([]) is False
        assert LogoutCommand().needs_agent([]) is False

    async def test_success_shows_account_and_plan_status(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        await LoginCommand().execute(ctx, [])
        text = ctx.ui.text
        assert EMAIL in text
        assert "plan usage is on" in text.lower()
        assert "chatgpt.com/settings/usage" in text

    async def test_success_switches_to_the_plan_and_saves_it(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        await LoginCommand().execute(ctx, [])
        saved_config.assert_called_once_with(
            {"openai_auth_mode": "chatgpt_plan", "provider": "openai"}
        )
        switch.assert_awaited_once_with(
            ctx, "gpt-6.1-sol", "openai", auth_mode="chatgpt_plan"
        )

    async def test_from_another_provider_it_moves_to_an_openai_model(
        self, monkeypatch, switch, saved_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        cfg = _live_config(model="claude-opus-5-5", provider="anthropic")
        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            await LoginCommand().execute(ctx, [])
        saved_config.assert_called_once_with(
            {
                "openai_auth_mode": "chatgpt_plan",
                "provider": "openai",
                "model": "gpt-6.1-sol",
            }
        )
        switch.assert_awaited_once_with(
            ctx, "gpt-6.1-sol", "openai", auth_mode="chatgpt_plan"
        )
        assert "/model" in ctx.ui.text

    async def test_a_catalog_model_the_user_chose_is_kept(
        self, monkeypatch, switch, saved_config, catalog
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        catalog.return_value = [("gpt-from-the-catalog", "X"), *CATALOG]
        cfg = _live_config(model="gpt-from-the-catalog")
        _sign_in_as(monkeypatch, _creds())
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            await LoginCommand().execute(_ctx(), [])
        assert switch.await_args.args[1] == "gpt-from-the-catalog"
        assert "model" not in saved_config.call_args.args[0]

    async def test_a_model_the_account_does_not_list_gives_way_to_the_first_listed(
        self, monkeypatch, switch, saved_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        cfg = _live_config(model="gpt-not-on-this-plan")
        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            await LoginCommand().execute(ctx, [])
        assert switch.await_args.args[1] == "gpt-6.1-sol"
        assert saved_config.call_args.args[0]["model"] == "gpt-6.1-sol"
        assert "gpt-not-on-this-plan" in ctx.ui.text
        assert "gpt-6.1-sol" in ctx.ui.text

    async def test_the_default_model_is_checked_against_the_account_too(
        self, monkeypatch, switch, saved_config, catalog
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        catalog.return_value = [("gpt-only-this", "Only")]
        cfg = _live_config(model="claude-opus-5-5", provider="anthropic")
        _sign_in_as(monkeypatch, _creds())
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            await LoginCommand().execute(_ctx(), [])
        assert switch.await_args.args[1] == "gpt-only-this"
        assert saved_config.call_args.args[0]["model"] == "gpt-only-this"

    async def test_an_unreadable_catalog_keeps_the_model_and_goes_on(
        self, monkeypatch, switch, saved_config, catalog
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand
        from metasci_scribe.llm.errors import PlanAction, PlanStop

        catalog.side_effect = PlanStop(PlanAction.RETRY, "Could not reach OpenAI.")
        cfg = _live_config(model="gpt-not-checked")
        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            await LoginCommand().execute(ctx, [])
        assert switch.await_args.args[1] == "gpt-not-checked"
        assert "model" not in saved_config.call_args.args[0]
        assert "Could not check" in ctx.ui.text

    async def test_a_leftover_auxiliary_model_is_cleared_and_said_so(
        self, monkeypatch, switch, saved_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        cfg = _live_config(
            auxiliary_model="claude-haiku-5-5", auxiliary_provider="anthropic"
        )
        seen = {}

        async def during_switch(*args, **kwargs):
            seen["aux"] = (cfg.auxiliary_model, cfg.auxiliary_provider)
            return True

        switch.side_effect = during_switch
        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            await LoginCommand().execute(ctx, [])
        assert seen["aux"] == ("", "")
        changes = saved_config.call_args.args[0]
        assert changes["auxiliary_model"] is None
        assert changes["auxiliary_provider"] is None
        assert "co-pilot" in ctx.ui.text

    async def test_the_auxiliary_model_comes_back_when_the_switch_fails(
        self, monkeypatch, switch, saved_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        switch.return_value = False
        cfg = _live_config(
            auxiliary_model="claude-haiku-5-5", auxiliary_provider="anthropic"
        )
        _sign_in_as(monkeypatch, _creds())
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            await LoginCommand().execute(_ctx(), [])
        assert (cfg.auxiliary_model, cfg.auxiliary_provider) == (
            "claude-haiku-5-5",
            "anthropic",
        )

    async def test_no_auxiliary_model_means_nothing_to_clear(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        await LoginCommand().execute(ctx, [])
        assert "auxiliary_model" not in saved_config.call_args.args[0]
        assert "co-pilot" not in ctx.ui.text

    async def test_the_manual_address_is_shown_through_the_command_ui(
        self, monkeypatch, switch, saved_config, live_config, capsys
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        def fake_login(**kwargs):
            kwargs["show_url"]("Could not open a browser. Open this address: https://x")
            cp.save(_creds())
            return _creds()

        monkeypatch.setattr(cp, "login", fake_login)
        ctx = _ctx()
        await LoginCommand().execute(ctx, [])
        assert "Open this address: https://x" in ctx.ui.text
        assert capsys.readouterr().out == ""

    async def test_ctrl_c_during_the_exchange_returns_within_a_second(
        self, monkeypatch, switch, saved_config, live_config
    ):
        """The sign-in is past the browser wait and cannot see the event; the
        command stops waiting, and nothing is switched when the thread finishes."""
        import signal

        from metasci_scribe.commands.implementation.auth import LoginCommand

        started = threading.Event()
        finished = threading.Event()

        def stuck_in_the_exchange(**kwargs):
            started.set()
            time.sleep(2.5)  # a token request that is slow to answer
            finished.set()
            return _creds()

        monkeypatch.setattr(cp, "login", stuck_in_the_exchange)
        ctx = _ctx()
        task = asyncio.ensure_future(LoginCommand().execute(ctx, []))
        while not started.is_set():
            await asyncio.sleep(0.01)
        signal.raise_signal(signal.SIGINT)
        began = time.monotonic()
        await asyncio.wait_for(task, 10)
        assert time.monotonic() - began < 2.0
        assert "cancel" in ctx.ui.text.lower()
        assert finished.wait(5)
        await asyncio.sleep(0.1)
        switch.assert_not_awaited()
        saved_config.assert_not_called()

    async def test_warns_in_two_lines_before_the_browser_opens(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        ctx = _ctx()
        order: list[str] = []

        def fake_login(**kwargs):
            order.append("login")
            cp.save(_creds())
            return _creds()

        monkeypatch.setattr(cp, "login", fake_login)
        original = ctx.ui.append_system

        def spy(text, style="dim"):
            order.append("print")
            original(text, style)

        ctx.ui.append_system = spy
        await LoginCommand().execute(ctx, [])
        assert order.index("login") > 1  # the warning came first
        warning = ctx.ui.styled("yellow").splitlines()
        assert len(warning) == 2
        assert "paper folder" in warning[0]
        assert "approval" in warning[1]
        assert "no shell" in warning[1]

    async def test_passes_a_cancel_event_to_the_sign_in(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        seen: dict = {}
        _sign_in_as(monkeypatch, _creds(), seen)
        await LoginCommand().execute(_ctx(), [])
        assert isinstance(seen["cancel"], threading.Event)
        assert not seen["cancel"].is_set()

    async def test_plan_usage_off_points_to_the_api_key_and_switches_nothing(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        _sign_in_as(
            monkeypatch,
            _creds(scopes=frozenset(ALL_SCOPES.split()) - {cp.DIRECT_SCOPE}),
        )
        ctx = _ctx()
        await LoginCommand().execute(ctx, [])
        text = ctx.ui.text
        assert EMAIL in text
        assert "off" in text
        assert "API key" in text
        assert "/login" in text
        saved_config.assert_not_called()
        switch.assert_not_awaited()

    async def test_the_welcome_is_shown_once(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        _sign_in_as(monkeypatch, _creds())
        first = _ctx()
        await LoginCommand().execute(first, [])
        second = _ctx()
        await LoginCommand().execute(second, [])
        assert "You're using your ChatGPT plan." in first.ui.text
        assert "You're using your ChatGPT plan." not in second.ui.text

    async def test_a_save_failure_is_reported_and_the_session_still_switches(
        self, monkeypatch, switch, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand
        from metasci_scribe.config.store import ConfigError

        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        with patch(
            "metasci_scribe.config.settings.set_config_values",
            side_effect=ConfigError("locked"),
        ):
            await LoginCommand().execute(ctx, [])
        assert "Could not save" in ctx.ui.styled("red")
        switch.assert_awaited_once()

    async def test_a_failed_rebuild_is_not_called_a_success(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        switch.return_value = False
        _sign_in_as(monkeypatch, _creds())
        ctx = _ctx()
        await LoginCommand().execute(ctx, [])
        assert "restart" in ctx.ui.text.lower() or "/model" in ctx.ui.text

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (cp.LoginCancelled("x"), "cancel"),
            (cp.LoginTimeout("x"), "/login"),
            (
                cp.LoginFailed("The sign-in failed (server_error).", "server_error"),
                "server_error",
            ),
            (
                cp.CredentialStoreError("Could not secure the folder."),
                "Could not secure",
            ),
            (cp.NotSignedIn("x"), "/login"),
            (cp.RefreshRejected("x"), "/login"),
            (cp.RefreshUnavailable("x"), "kept"),
            (cp.PlanUsageDisabled("x"), "API key"),
        ],
    )
    async def test_every_auth_error_ends_as_a_message_and_changes_nothing(
        self, monkeypatch, switch, saved_config, live_config, error, expected
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        _login_raises(monkeypatch, error)
        ctx = _ctx()
        await LoginCommand().execute(ctx, [])
        assert expected.lower() in ctx.ui.text.lower()
        assert "Traceback" not in ctx.ui.text
        saved_config.assert_not_called()
        switch.assert_not_awaited()

    async def test_a_different_account_is_told_to_log_out_first(
        self, monkeypatch, switch, saved_config, live_config
    ):
        from metasci_scribe.commands.implementation.auth import LoginCommand

        _login_raises(
            monkeypatch,
            cp.LoginFailed(
                "This is a different ChatGPT account from the saved one.",
                code="account_mismatch",
            ),
        )
        ctx = _ctx()
        await LoginCommand().execute(ctx, [])
        assert "/logout" in ctx.ui.text

    async def test_ctrl_c_scoped_to_the_wait_cancels_the_sign_in(
        self, monkeypatch, switch, saved_config, live_config
    ):
        """The command's task is cancelled (as an interrupt does): the wait
        ends through the event, the thread is gone, and the cancel propagates."""
        from metasci_scribe.commands.implementation.auth import LoginCommand

        started = threading.Event()
        finished = threading.Event()
        seen: dict = {}

        def waiting_login(**kwargs):
            seen["cancel"] = kwargs["cancel"]
            started.set()
            try:
                if kwargs["cancel"].wait(10):
                    raise cp.LoginCancelled("The sign-in was cancelled.")
                raise cp.LoginTimeout("late")
            finally:
                finished.set()

        monkeypatch.setattr(cp, "login", waiting_login)
        task = asyncio.ensure_future(LoginCommand().execute(_ctx(), []))
        while not started.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert seen["cancel"].is_set()
        assert finished.wait(5)
        saved_config.assert_not_called()

    async def test_a_keyboard_interrupt_cancels_and_reports(
        self, monkeypatch, switch, saved_config, live_config
    ):
        import signal

        from metasci_scribe.commands.implementation.auth import LoginCommand

        started = threading.Event()

        def waiting_login(**kwargs):
            started.set()
            if kwargs["cancel"].wait(10):
                raise cp.LoginCancelled("The sign-in was cancelled.")
            raise cp.LoginTimeout("late")

        monkeypatch.setattr(cp, "login", waiting_login)
        previous = signal.getsignal(signal.SIGINT)
        ctx = _ctx()
        task = asyncio.ensure_future(LoginCommand().execute(ctx, []))
        while not started.is_set():
            await asyncio.sleep(0.01)
        signal.raise_signal(signal.SIGINT)  # the handler the command installed
        await asyncio.wait_for(task, 10)
        assert "cancel" in ctx.ui.text.lower()
        assert signal.getsignal(signal.SIGINT) is previous
        switch.assert_not_awaited()


# ------------------------------------------------------------------ /logout


class TestLogoutCommand:
    @pytest.fixture(autouse=True)
    def live_config(self):
        with patch("metasci_scribe.agent._ensure_config", return_value=_live_config()):
            yield

    @pytest.fixture
    def revoked(self, monkeypatch):
        """OpenAI's endpoints, faked: discovery and revocation."""
        import httpx

        calls: list[dict] = []
        endpoint = "https://auth.openai.com/api/accounts/oauth/revoke"

        def handler(request: httpx.Request) -> httpx.Response:
            if str(request.url) == cp.DISCOVERY_URL:
                return httpx.Response(
                    200,
                    json={"issuer": cp.ISSUER, "revocation_endpoint": endpoint},
                )
            assert str(request.url) == endpoint
            calls.append({"body": request.content.decode()})
            return handler.reply(request)

        handler.reply = lambda request: httpx.Response(200)
        monkeypatch.setattr(
            cp,
            "_http_client",
            lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        )
        monkeypatch.setattr(cp, "_REVOKE_BACKOFF_S", (0.0, 0.0))
        handler.calls = calls
        return handler

    async def test_logout_revokes_and_signs_out(self, revoked):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        cp.save(_creds())
        ctx = _ctx()
        await LogoutCommand().execute(ctx, [])
        assert cp.load() is None
        assert len(revoked.calls) == 1
        assert "revoked" in ctx.ui.text.lower()
        assert not (paths.DATA_DIR / "auth" / "chatgpt.json").exists()

    async def test_a_revocation_failure_is_reported_and_tokens_still_go(self, revoked):
        import httpx

        from metasci_scribe.commands.implementation.auth import LogoutCommand

        revoked.reply = lambda request: httpx.Response(503)
        cp.save(_creds())
        ctx = _ctx()
        await LogoutCommand().execute(ctx, [])
        assert cp.load() is None
        text = ctx.ui.text
        assert "not confirmed" in text.lower() or "could not confirm" in text.lower()
        assert "503" in text
        assert "ChatGPT Settings" in text
        assert "secret" not in text

    async def test_the_account_registration_goes_so_another_account_can_sign_in(
        self, revoked
    ):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        cp.save(_creds())
        await LogoutCommand().execute(_ctx(), [])
        assert cp._read_record() is None

    async def test_host_id_is_kept(self, revoked):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        cp.save(_creds())
        host_id = cp._host_id()
        await LogoutCommand().execute(_ctx(), [])
        assert cp._host_id() == host_id

    async def test_when_not_signed_in_it_says_so(self, revoked):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        ctx = _ctx()
        await LogoutCommand().execute(ctx, [])
        assert "not signed in" in ctx.ui.text.lower()
        assert revoked.calls == []

    async def test_a_store_problem_is_a_message(self, monkeypatch):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        def broken():
            raise cp.CredentialStoreError("Could not remove the saved sign-in.")

        monkeypatch.setattr(cp, "sign_out", broken)
        ctx = _ctx()
        await LogoutCommand().execute(ctx, [])
        assert "Could not remove" in ctx.ui.styled("red")

    async def test_it_points_to_login_and_the_api_key(self, revoked):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        cp.save(_creds())
        ctx = _ctx()
        await LogoutCommand().execute(ctx, [])
        assert "/login" in ctx.ui.text

    async def test_it_leaves_the_configuration_alone(self, revoked):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        cp.save(_creds())
        with patch("metasci_scribe.config.settings.set_config_values") as saved:
            await LogoutCommand().execute(_ctx(), [])
        saved.assert_not_called()

    async def test_it_says_what_happens_to_the_running_session(self, revoked):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        cp.save(_creds())
        ctx = _ctx()
        await LogoutCommand().execute(ctx, [])
        assert "this session" in ctx.ui.text.lower()

    async def test_a_record_that_stays_is_reported_beside_the_revocation(
        self, monkeypatch
    ):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        monkeypatch.setattr(
            cp,
            "sign_out",
            lambda: cp.SignOutResult(
                True, True, None, "Could not remove the saved sign-in (OSError)."
            ),
        )
        ctx = _ctx()
        with patch("metasci_scribe.config.settings.set_config_values") as saved:
            await LogoutCommand().execute(ctx, [])
        text = ctx.ui.text
        assert "revoked" in text.lower()
        assert "Could not remove" in text
        assert "chatgpt.json" in text
        saved.assert_not_called()  # the sign-in is still there: keep the route

    async def test_a_failed_revocation_and_a_failed_removal_are_both_said(
        self, monkeypatch
    ):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        monkeypatch.setattr(
            cp,
            "sign_out",
            lambda: cp.SignOutResult(
                True,
                False,
                "OpenAI answered HTTP 503.",
                "Could not remove the saved sign-in (OSError).",
            ),
        )
        ctx = _ctx()
        await LogoutCommand().execute(ctx, [])
        assert "503" in ctx.ui.text
        assert "Could not remove" in ctx.ui.text


class TestLogoutOnThePlanRoute:
    """``/logout`` under ``openai_auth_mode=chatgpt_plan`` returns the route to
    the key, so the session follows the configured state; plan traffic is never
    sent on a key without saying so."""

    @pytest.fixture(autouse=True)
    def signed_in(self, monkeypatch):
        import httpx

        def handler(request):
            if str(request.url) == cp.DISCOVERY_URL:
                return httpx.Response(200, json={"issuer": cp.ISSUER})
            raise AssertionError(request.url)

        monkeypatch.setattr(
            cp,
            "_http_client",
            lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        )
        monkeypatch.setattr(cp, "_REVOKE_BACKOFF_S", (0.0, 0.0))
        cp.save(_creds())

    async def _logout(self, cfg):
        from metasci_scribe.commands.implementation.auth import LogoutCommand

        ctx = _ctx()
        with (
            patch("metasci_scribe.agent._ensure_config", return_value=cfg),
            patch("metasci_scribe.agent.set_active_config") as activate,
        ):
            await LogoutCommand().execute(ctx, [])
        return ctx, activate

    async def test_with_a_key_the_session_is_rebuilt_on_it_and_told_so(
        self, monkeypatch, switch, saved_config
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        cfg = _live_config(openai_auth_mode="chatgpt_plan")
        ctx, _ = await self._logout(cfg)
        saved_config.assert_called_once_with({"openai_auth_mode": "api_key"})
        switch.assert_awaited_once_with(
            ctx, "gpt-6.1-sol", "openai", auth_mode="api_key"
        )
        assert "API key" in ctx.ui.text
        assert "billed" in ctx.ui.text.lower()

    async def test_without_a_key_the_route_returns_and_the_session_says_it_is_stuck(
        self, switch, saved_config
    ):
        cfg = _live_config(openai_auth_mode="chatgpt_plan")
        ctx, activate = await self._logout(cfg)
        saved_config.assert_called_once_with({"openai_auth_mode": "api_key"})
        switch.assert_not_awaited()
        assert cfg.openai_auth_mode == "api_key"
        activate.assert_called_once_with(cfg)
        text = ctx.ui.text
        assert "/login" in text
        assert "OPENAI_API_KEY" in text
        assert "cannot send requests" in text

    async def test_another_provider_keeps_its_session(
        self, monkeypatch, switch, saved_config
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        cfg = _live_config(
            openai_auth_mode="chatgpt_plan",
            provider="anthropic",
            model="claude-opus-5-5",
        )
        await self._logout(cfg)
        saved_config.assert_called_once_with({"openai_auth_mode": "api_key"})
        switch.assert_not_awaited()

    async def test_a_failed_rebuild_is_said(self, monkeypatch, switch, saved_config):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        switch.return_value = False
        cfg = _live_config(openai_auth_mode="chatgpt_plan")
        ctx, _ = await self._logout(cfg)
        assert "could not" in ctx.ui.text.lower()

    async def test_a_save_failure_is_reported(self, monkeypatch, switch):
        from metasci_scribe.commands.implementation.auth import LogoutCommand
        from metasci_scribe.config.store import ConfigError

        cfg = _live_config(openai_auth_mode="chatgpt_plan")
        ctx = _ctx()
        with (
            patch("metasci_scribe.agent._ensure_config", return_value=cfg),
            patch("metasci_scribe.agent.set_active_config"),
            patch(
                "metasci_scribe.config.settings.set_config_values",
                side_effect=ConfigError("locked"),
            ),
        ):
            await LogoutCommand().execute(ctx, [])
        assert "Could not save" in ctx.ui.styled("red")

    async def test_the_key_route_is_left_alone(self, switch, saved_config):
        cfg = _live_config()
        ctx, activate = await self._logout(cfg)
        saved_config.assert_not_called()
        switch.assert_not_awaited()
        activate.assert_not_called()
        assert "unaffected" in ctx.ui.text.lower()


class TestPlanRouteMessage:
    def test_the_message_gives_the_real_next_step(self):
        from metasci_scribe.llm.credentials import ModelNotConfigured

        text = str(ModelNotConfigured("openai", plan=True))
        assert "/login" in text
        assert "config set openai_auth_mode api_key" in text
        assert "OPENAI_API_KEY" in text

    def test_a_key_never_stands_in_for_the_plan(self, monkeypatch):
        from metasci_scribe.llm.credentials import (
            ModelNotConfigured,
            ensure_model_configured,
        )

        monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
        with pytest.raises(ModelNotConfigured) as caught:
            ensure_model_configured("openai")
        assert "openai_auth_mode api_key" in str(caught.value)


# ------------------------------------------------------------------- /model


class TestModelCommandOnThePlan:
    CATALOG: ClassVar[list[tuple[str, str]]] = [
        ("gpt-plan-a", "GPT Plan A"),
        ("gpt-plan-b", "GPT Plan B"),
    ]

    @pytest.fixture
    def plan_catalog(self, monkeypatch):
        from metasci_scribe.llm import chatgpt_plan as plan_llm

        listing = MagicMock(return_value=list(self.CATALOG))
        monkeypatch.setattr(plan_llm, "list_models", listing)
        return listing

    @pytest.fixture
    def apply(self, monkeypatch):
        from metasci_scribe.commands.implementation.model import ModelCommand

        fake = AsyncMock(return_value=True)
        monkeypatch.setattr(ModelCommand, "_apply_model", fake)
        return fake

    def _plan_cfg(self, **changes):
        return _live_config(openai_auth_mode="chatgpt_plan", **changes)

    async def test_the_picker_lists_the_accounts_models(self, plan_catalog, apply):
        from metasci_scribe.commands.implementation.model import ModelCommand

        ctx = _ctx()
        with patch(
            "metasci_scribe.agent._ensure_config", return_value=self._plan_cfg()
        ):
            await ModelCommand().execute(ctx, [])
        names = [name for name, _id, _provider in ctx.ui.pick_entries]
        assert "gpt-plan-a" in names
        assert "gpt-plan-b" in names
        assert "gpt-6.1-sol" not in names  # the static list is not the account's
        providers = {p for n, _i, p in ctx.ui.pick_entries if n.startswith("gpt-plan")}
        assert providers == {"openai"}

    async def test_a_catalog_slug_is_accepted(self, plan_catalog, apply):
        from metasci_scribe.commands.implementation.model import ModelCommand

        ctx = _ctx()
        with patch(
            "metasci_scribe.agent._ensure_config", return_value=self._plan_cfg()
        ):
            await ModelCommand().execute(ctx, ["gpt-plan-b"])
        apply.assert_awaited_once_with(ctx, "gpt-plan-b", "openai", save=False)

    async def test_a_name_outside_the_catalog_is_refused(self, plan_catalog, apply):
        from metasci_scribe.commands.implementation.model import ModelCommand

        ctx = _ctx()
        with patch(
            "metasci_scribe.agent._ensure_config", return_value=self._plan_cfg()
        ):
            await ModelCommand().execute(ctx, ["gpt-6-sol"])
        apply.assert_not_awaited()
        assert "ChatGPT plan" in ctx.ui.styled("red")

    async def test_a_claude_model_still_goes_by_its_own_key(self, plan_catalog, apply):
        from metasci_scribe.commands.implementation.model import ModelCommand

        ctx = _ctx()
        with patch(
            "metasci_scribe.agent._ensure_config", return_value=self._plan_cfg()
        ):
            await ModelCommand().execute(ctx, ["claude-opus-5-5"])
        apply.assert_awaited_once_with(ctx, "claude-opus-5-5", "anthropic", save=False)
        plan_catalog.assert_not_called()

    async def test_a_plan_stop_is_shown_with_its_message_and_nothing_switches(
        self, monkeypatch, apply
    ):
        from metasci_scribe.commands.implementation.model import ModelCommand
        from metasci_scribe.llm import chatgpt_plan as plan_llm
        from metasci_scribe.llm.errors import PlanAction, PlanStop

        stop = PlanStop(
            PlanAction.RELOGIN,
            "ChatGPT did not accept this sign-in. Run /login. (HTTP 401)",
            code="subscription_sharing_invalid_user",
            request_id="req_123",
        )

        def refuse():
            raise stop

        monkeypatch.setattr(plan_llm, "list_models", refuse)
        ctx = _ctx()
        with patch(
            "metasci_scribe.agent._ensure_config", return_value=self._plan_cfg()
        ):
            await ModelCommand().execute(ctx, ["gpt-plan-a"])
            await ModelCommand().execute(ctx, [])
        apply.assert_not_awaited()
        assert ctx.ui.pick_entries is None
        assert "Run /login" in ctx.ui.styled("red")

    async def test_a_retryable_stop_is_a_warning_not_an_error(self, monkeypatch, apply):
        from metasci_scribe.commands.implementation.model import ModelCommand
        from metasci_scribe.llm import chatgpt_plan as plan_llm
        from metasci_scribe.llm.errors import PlanAction, PlanStop

        def refuse():
            raise PlanStop(
                PlanAction.RETRY, "Could not reach OpenAI. Your sign-in is kept."
            )

        monkeypatch.setattr(plan_llm, "list_models", refuse)
        ctx = _ctx()
        with patch(
            "metasci_scribe.agent._ensure_config", return_value=self._plan_cfg()
        ):
            await ModelCommand().execute(ctx, [])
        assert "sign-in is kept" in ctx.ui.styled("yellow")

    async def test_the_key_route_never_asks_the_account(self, plan_catalog, apply):
        from metasci_scribe.commands.implementation.model import ModelCommand

        ctx = _ctx()
        with patch("metasci_scribe.agent._ensure_config", return_value=_live_config()):
            await ModelCommand().execute(ctx, ["gpt-6-sol"])
            await ModelCommand().execute(ctx, [])
        plan_catalog.assert_not_called()
        apply.assert_awaited_once_with(ctx, "gpt-6-sol", "openai", save=False)
        assert any(n == "gpt-6.1-sol" for n, _i, _p in ctx.ui.pick_entries)


class TestApplyModelAuthMode:
    """``/login`` rebuilds the session through ``/model``'s transactional switch."""

    ENV = "METASCI_SCRIBE_OPENAI_AUTH_MODE"

    def _patches(self, cfg, build, load_agent):
        return (
            patch("metasci_scribe.agent._ensure_config", return_value=cfg),
            patch("metasci_scribe.agent._build_chat_model", side_effect=build),
            patch("metasci_scribe.agent.set_active_config"),
            patch("metasci_scribe.agent.set_chat_model_instance"),
            patch("metasci_scribe.cli.agent._load_agent", side_effect=load_agent),
        )

    async def test_the_trial_build_runs_in_the_new_mode_and_commits_it(self):
        import os
        from contextlib import ExitStack

        from metasci_scribe.commands.implementation.model import ModelCommand

        cfg = _live_config()
        seen = {}

        def build(temp_cfg):
            seen["env"] = os.environ.get(self.ENV)
            seen["mode"] = temp_cfg.openai_auth_mode
            return MagicMock()

        new_agent = MagicMock()
        ctx = _ctx()
        ctx.work_dir = "work"
        with ExitStack() as stack:
            for p in self._patches(cfg, build, lambda **kw: new_agent):
                stack.enter_context(p)
            ok = await ModelCommand()._apply_model(
                ctx, "gpt-6.1-sol", "openai", auth_mode="chatgpt_plan"
            )
        assert ok is True
        assert seen == {"env": "chatgpt_plan", "mode": "chatgpt_plan"}
        assert cfg.openai_auth_mode == "chatgpt_plan"
        assert ctx.agent is new_agent

    async def test_a_failed_build_puts_the_old_mode_back(self, monkeypatch):
        import os
        from contextlib import ExitStack

        from metasci_scribe.commands.implementation.model import ModelCommand

        monkeypatch.setenv(self.ENV, "api_key")
        cfg = _live_config()

        def build(temp_cfg):
            raise RuntimeError("cannot build")

        ctx = _ctx()
        ctx.work_dir = "work"
        with ExitStack() as stack:
            for p in self._patches(cfg, build, lambda **kw: MagicMock()):
                stack.enter_context(p)
            ok = await ModelCommand()._apply_model(
                ctx, "gpt-6.1-sol", "openai", auth_mode="chatgpt_plan"
            )
        assert ok is False
        assert os.environ[self.ENV] == "api_key"
        assert cfg.openai_auth_mode == "api_key"
        assert "Failed to switch model" in ctx.ui.styled("red")

    async def test_a_failed_agent_load_puts_the_old_mode_back(self, monkeypatch):
        import os
        from contextlib import ExitStack

        from metasci_scribe.commands.implementation.model import ModelCommand

        monkeypatch.delenv(self.ENV, raising=False)
        cfg = _live_config()

        def load(**kwargs):
            raise RuntimeError("cannot load")

        ctx = _ctx()
        ctx.work_dir = "work"
        with ExitStack() as stack:
            for p in self._patches(cfg, lambda c: MagicMock(), load):
                stack.enter_context(p)
            ok = await ModelCommand()._apply_model(
                ctx, "gpt-6.1-sol", "openai", auth_mode="chatgpt_plan"
            )
        assert ok is False
        assert self.ENV not in os.environ

    async def test_without_an_auth_mode_nothing_about_it_changes(self):
        import os
        from contextlib import ExitStack

        from metasci_scribe.commands.implementation.model import ModelCommand

        cfg = _live_config()
        ctx = _ctx()
        ctx.work_dir = "work"
        with ExitStack() as stack:
            for p in self._patches(
                cfg, lambda c: MagicMock(), lambda **kw: MagicMock()
            ):
                stack.enter_context(p)
            ok = await ModelCommand()._apply_model(ctx, "gpt-6-sol", "openai")
        assert ok is True
        assert cfg.openai_auth_mode == "api_key"
        assert self.ENV not in os.environ
