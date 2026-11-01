"""``/login`` and ``/logout``: Sign in with ChatGPT, and sign out again.

https://developers.openai.com/siwc/token-sharing-open-source/sign-in

The sign-in itself lives in :mod:`metasci_scribe.auth.chatgpt_plan`; these
commands only wrap it for the prompt: a warning before the browser opens, a
wait that Ctrl+C ends, a message for every way it can fail, and the switch of
the running session to the plan.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import threading
from typing import Any, ClassVar

from ...auth import chatgpt_plan as plan_auth
from ...auth.notices import (
    LOGIN_WARNING,
    USAGE_URL,
    WELCOME,
    describe_auth_error,
    plan_off_lines,
)
from ..base import Command, CommandContext
from ..manager import manager


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


async def _sign_in() -> plan_auth.Credentials:
    """Run the browser sign-in off the event loop; cancellable at any moment."""
    cancel = threading.Event()
    restore = _cancel_on_interrupt(cancel)
    sign_in = asyncio.ensure_future(asyncio.to_thread(plan_auth.login, cancel=cancel))
    try:
        return await asyncio.shield(sign_in)
    except BaseException:
        # The task was cancelled or interrupted: end the wait, and let the
        # thread close its listener before anything else happens.
        cancel.set()
        with contextlib.suppress(BaseException):
            await sign_in
        raise
    finally:
        restore()


class LoginCommand(Command):
    """Sign in with ChatGPT and use the plan for this session."""

    name = "/login"
    description = "Sign in with ChatGPT to use your plan"
    category = "Model"
    requires_agent: ClassVar[bool] = False

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        ui = ctx.ui
        for line in LOGIN_WARNING:
            ui.append_system(line, style="yellow")
        ui.append_system(
            "Opening your browser to continue with ChatGPT. Press Ctrl+C to cancel.",
            style="dim",
        )
        try:
            creds = await _sign_in()
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
        model = cfg.model
        changes: dict[str, str] = {
            "openai_auth_mode": "chatgpt_plan",
            "provider": "openai",
        }
        if cfg.provider != "openai":
            # The plan serves OpenAI models only; /model lists the account's.
            model = DEFAULT_MODEL
            changes["model"] = model
            ui.append_system(
                f"Switching to {model}. Use /model to choose another.", style="dim"
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
            return

        if not result.had_sign_in:
            ui.append_system("You are not signed in to ChatGPT.", style="dim")
            return
        if result.revoked:
            ui.append_system(
                "Signed out. The sign-in was revoked at OpenAI and removed from "
                "this machine.",
                style="green",
            )
        else:
            ui.append_system(
                "Signed out on this machine, but OpenAI could not confirm the "
                f"revocation. {result.problem or ''}".rstrip(),
                style="yellow",
            )
            ui.append_system(
                "To be sure, disconnect Metasci Scribe in ChatGPT Settings.",
                style="dim",
            )
        ui.append_system(
            "Run /login to sign in again, with this or another ChatGPT account.",
            style="dim",
        )


manager.register(LoginCommand())
manager.register(LogoutCommand())
