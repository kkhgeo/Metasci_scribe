"""``/login`` and ``/logout``: Continue with ChatGPT, and sign out again.

https://developers.openai.com/siwc/token-sharing-open-source/sign-in

The sign-in itself lives in :mod:`metasci_scribe.auth.chatgpt_plan`; these
commands only wrap it for the prompt: a warning before the browser opens, a
wait that Ctrl+C ends at once, a message for every way it can fail, and the
switch of the running session to the plan, or back to the key.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import threading
from collections.abc import Callable
from typing import Any, ClassVar

from ... import paths
from ...auth import chatgpt_plan as plan_auth
from ...auth.notices import (
    CONTINUE_LABEL,
    LOGIN_WARNING,
    USAGE_URL,
    WELCOME,
    describe_auth_error,
    plan_off_lines,
)
from ..base import Command, CommandContext
from ..manager import manager

# After a cancel the sign-in thread may be in a token request that no event can
# interrupt. The command waits this long for it, then stops waiting; the thread
# ends on its own and, seeing the cancel, saves nothing.
_CANCEL_GRACE_S = 1.0
_POLL_S = 0.2


def _cancel_on_interrupt(cancel: threading.Event):
    """Route Ctrl+C to ``cancel`` while the sign-in waits; returns the undo.

    Without it the interrupt would unwind the whole session and leave the
    listener thread holding its port until it timed out. A second Ctrl+C
    before the wait ends is the ordinary interrupt.
    """
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum: int, frame: Any) -> None:
        if cancel.is_set():
            signal.default_int_handler(signum, frame)
            return
        cancel.set()

    signal.signal(signal.SIGINT, handler)

    def restore() -> None:
        signal.signal(signal.SIGINT, previous)

    return restore


async def _sign_in(show_url: Callable[[str], None]) -> plan_auth.Credentials:
    """Run the browser sign-in off the event loop; Ctrl+C returns within about
    a second wherever the sign-in is, in the browser wait or in a token request."""
    cancel = threading.Event()
    restore = _cancel_on_interrupt(cancel)
    sign_in = asyncio.ensure_future(
        asyncio.to_thread(plan_auth.login, cancel=cancel, show_url=show_url)
    )
    # A thread we stop waiting for may still fail later; nobody reads that.
    sign_in.add_done_callback(lambda f: f.cancelled() or f.exception())
    try:
        while True:
            done, _ = await asyncio.wait({sign_in}, timeout=_POLL_S)
            if done:
                return sign_in.result()
            if cancel.is_set():
                done, _ = await asyncio.wait({sign_in}, timeout=_CANCEL_GRACE_S)
                if done:
                    return sign_in.result()
                raise plan_auth.LoginCancelled("The sign-in was cancelled.")
    except plan_auth.ChatGPTAuthError:
        raise  # the sign-in is over, one way or another: nothing left to stop
    except BaseException:
        # The task was cancelled or interrupted: end the wait, and give the
        # thread a moment to close its listener before anything else happens.
        cancel.set()
        with contextlib.suppress(BaseException):
            await asyncio.wait({sign_in}, timeout=_CANCEL_GRACE_S)
        raise
    finally:
        restore()


class LoginCommand(Command):
    """Continue with ChatGPT and use the plan for this session."""

    name = "/login"
    description = f"{CONTINUE_LABEL} to use your plan"
    category = "Model"
    requires_agent: ClassVar[bool] = False

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        ui = ctx.ui
        for line in LOGIN_WARNING:
            ui.append_system(line, style="yellow")
        ui.append_system(
            f"Opening your browser to {CONTINUE_LABEL.lower()}. "
            "Press Ctrl+C to cancel.",
            style="dim",
        )
        loop = asyncio.get_running_loop()

        def show_url(text: str) -> None:
            # Called from the sign-in thread: print through the command UI.
            loop.call_soon_threadsafe(ui.append_system, text, "yellow")

        try:
            creds = await _sign_in(show_url)
        except plan_auth.ChatGPTAuthError as exc:
            message, style = describe_auth_error(exc)
            ui.append_system(message, style=style)
            return

        who = f"Signed in as {creds.email}" if creds.email else "Signed in to ChatGPT"
        if not creds.plan_enabled:
            headline, advice = plan_off_lines(who)
            ui.append_system(headline, style="yellow")
            ui.append_system(advice, style="dim")
            return

        ui.append_system(f"{who}. ChatGPT plan usage is on.", style="green")
        await self._switch_to_plan(ctx)
        if await asyncio.to_thread(plan_auth.claim_plan_welcome):
            ui.append_system(WELCOME[0], style="green")
            ui.append_system(WELCOME[1], style="dim")
        ui.append_system(f"Manage usage: {USAGE_URL}", style="dim")

    async def _account_models(self, ctx: CommandContext) -> list[str] | None:
        """Slugs the account lists, or ``None`` after saying the check failed."""
        from ...llm import chatgpt_plan as plan_llm
        from ...llm.errors import PlanStop

        try:
            catalog = await asyncio.to_thread(plan_llm.list_models)
        except Exception as exc:  # best effort: the sign-in itself worked
            why = str(exc) if isinstance(exc, PlanStop) else type(exc).__name__
            ctx.ui.append_system(
                f"Could not check your ChatGPT plan's model list ({why}).",
                style="yellow",
            )
            return None
        return [slug for slug, _name in catalog] or None

    async def _switch_to_plan(self, ctx: CommandContext) -> None:
        """Save the plan as the way in, and move the running session onto it."""
        from ...agent import _ensure_config
        from ...config.settings import SettingsError, set_config_values
        from ...config.store import ConfigError
        from ...gateway.server import LangGraphServerGateway
        from ...llm.registry import DEFAULT_MODEL
        from .model import ModelCommand

        ui = ctx.ui
        cfg = _ensure_config()
        changes: dict[str, str | None] = {
            "openai_auth_mode": "chatgpt_plan",
            "provider": "openai",
        }

        # The plan serves the account's OpenAI models only: the model the
        # session keeps must be one of them.
        model = cfg.model if cfg.provider == "openai" else DEFAULT_MODEL
        listed = await self._account_models(ctx)
        if listed and model not in listed:
            ui.append_system(
                f"{model} is not on your ChatGPT plan; using {listed[0]}. "
                "Use /model to choose another.",
                style="yellow",
            )
            model = listed[0]
        elif cfg.provider != "openai":
            ui.append_system(
                f"Switching to {model}. Use /model to choose another.", style="dim"
            )
        if model != cfg.model:
            changes["model"] = model

        # A co-pilot model from before would be built on the plan route too,
        # and may be a Claude model with no key or a slug outside the account.
        aux = (
            getattr(cfg, "auxiliary_model", "") or "",
            getattr(cfg, "auxiliary_provider", "") or "",
        )
        if any(aux):
            changes["auxiliary_model"] = None
            changes["auxiliary_provider"] = None
            cfg.auxiliary_model = ""
            cfg.auxiliary_provider = ""
            ui.append_system(
                "Cleared the co-pilot (auxiliary) model; the main model is "
                "used for background work. Set one again with "
                "`metasci-scribe config set auxiliary_model <name>`.",
                style="yellow",
            )

        try:
            await asyncio.to_thread(set_config_values, changes)
        except (SettingsError, ConfigError) as exc:
            ui.append_system(
                f"Could not save to config: {exc}. The plan applies to this "
                "session only.",
                style="red",
            )

        switched = await ModelCommand()._apply_model(
            ctx, model, "openai", auth_mode="chatgpt_plan"
        )
        if not switched:
            if any(aux):
                cfg.auxiliary_model, cfg.auxiliary_provider = aux
                # The saved config was cleared before the switch: put it back.
                try:
                    await asyncio.to_thread(
                        set_config_values,
                        {
                            "auxiliary_model": aux[0] or None,
                            "auxiliary_provider": aux[1] or None,
                        },
                    )
                except (SettingsError, ConfigError) as exc:
                    ui.append_system(
                        f"Could not restore the co-pilot model in the config "
                        f"({exc}). Set it again: metasci-scribe config set "
                        f"auxiliary_model {aux[0]}",
                        style="red",
                    )
            ui.append_system(
                "You are signed in, but this session could not switch to the "
                "plan. Try /model, or restart metasci-scribe.",
                style="yellow",
            )
            return
        execute_gateway = getattr(
            ctx.graph_gateway, "execute_gateway", ctx.graph_gateway
        )
        if isinstance(execute_gateway, LangGraphServerGateway):
            ui.append_system(
                "Restart metasci-scribe so the background server uses the plan.",
                style="yellow",
            )


class LogoutCommand(Command):
    """Sign out of ChatGPT: revoke the sign-in and forget the account."""

    name = "/logout"
    description = "Sign out of ChatGPT"
    category = "Model"
    requires_agent: ClassVar[bool] = False

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        ui = ctx.ui
        try:
            result = await asyncio.to_thread(plan_auth.sign_out)
        except plan_auth.ChatGPTAuthError as exc:
            message, style = describe_auth_error(exc)
            ui.append_system(message, style=style)
            ui.append_system(
                "Nothing was signed out: this session keeps its current route "
                "and sign-in.",
                style="dim",
            )
            return

        if not result.had_sign_in:
            ui.append_system("You are not signed in to ChatGPT.", style="dim")
        elif result.revoked:
            ui.append_system(
                "The sign-in was revoked at OpenAI.",
                style="green",
            )
        else:
            ui.append_system(
                "OpenAI could not confirm the revocation. "
                f"{result.problem or ''}".rstrip(),
                style="yellow",
            )
            ui.append_system(
                "To be sure, disconnect Metasci Scribe in ChatGPT Settings.",
                style="dim",
            )
        if result.removal_problem:
            # Still on disk: the route stays as it is, and the person is told.
            ui.append_system(
                "The saved sign-in could not be removed from this machine: "
                f"{result.removal_problem}",
                style="red",
            )
            ui.append_system(
                f"Delete it yourself: {paths.DATA_DIR / 'auth' / 'chatgpt.json'}",
                style="dim",
            )
            if result.revoked:
                ui.append_system(
                    "This session still points at the ChatGPT plan, but the "
                    "token was revoked, so its requests will fail until you "
                    "run /login again.",
                    style="yellow",
                )
            else:
                ui.append_system(
                    "This session still uses your ChatGPT sign-in until the "
                    "file is gone.",
                    style="yellow",
                )
            return
        if result.had_sign_in:
            ui.append_system("Signed out on this machine.", style="green")
        await self._follow_configured_route(ctx)
        ui.append_system(
            "Run /login to sign in again, with this or another ChatGPT account.",
            style="dim",
        )

    async def _follow_configured_route(self, ctx: CommandContext) -> None:
        """Bring the session in step with the signed-out state.

        On the plan route the saved mode returns to ``api_key``, so the next
        start is the configured state too. The session moves onto the OpenAI
        key only when one is set, and says so: plan traffic never reaches a
        key unannounced.
        """
        from ...agent import _ensure_config, set_active_config
        from ...config.settings import SettingsError, set_config_values
        from ...config.store import ConfigError
        from .model import ModelCommand

        ui = ctx.ui
        cfg = _ensure_config()
        if getattr(cfg, "openai_auth_mode", "api_key") != "chatgpt_plan":
            ui.append_system(
                "This session is unaffected: it does not use your ChatGPT plan.",
                style="dim",
            )
            return
        try:
            await asyncio.to_thread(set_config_values, {"openai_auth_mode": "api_key"})
        except (SettingsError, ConfigError) as exc:
            ui.append_system(
                f"Could not save to config: {exc}. Run `metasci-scribe config "
                "set openai_auth_mode api_key` to leave the plan route.",
                style="red",
            )
        if cfg.provider != "openai":
            cfg.openai_auth_mode = "api_key"
            set_active_config(cfg)
            ui.append_system(
                f"This session uses {cfg.provider}: unaffected.", style="dim"
            )
            return
        if os.environ.get("OPENAI_API_KEY"):
            switched = await ModelCommand()._apply_model(
                ctx, cfg.model, "openai", auth_mode="api_key"
            )
            if switched:
                ui.append_system(
                    "This session now uses your OpenAI API key: requests are "
                    "billed to your API account, not your ChatGPT plan.",
                    style="yellow",
                )
            else:
                ui.append_system(
                    "Could not move this session onto your OpenAI API key. "
                    "Restart metasci-scribe, or run /login.",
                    style="yellow",
                )
            return
        cfg.openai_auth_mode = "api_key"
        set_active_config(cfg)
        ui.append_system(
            "This session cannot send requests until you run /login, or add an "
            "OpenAI API key (OPENAI_API_KEY, or `metasci-scribe onboard`) and "
            "restart.",
            style="yellow",
        )


manager.register(LoginCommand())
manager.register(LogoutCommand())
