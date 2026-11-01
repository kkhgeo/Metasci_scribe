"""``/run``: write the draft unattended, in a separate process.

The chat keeps its thread and its agent. ``/run`` starts ``metasci-scribe run``
in the session's paper folder as a child process and shows what it prints in
the chat. The child owns the model calls, with the model and provider the chat
uses now; its exit code is the ``RunExit`` value of :mod:`metasci_scribe.run`.

Reading the child's output happens on worker threads, so the prompt's event
loop keeps running. Ctrl+C is passed on to the child (Ctrl+Break on Windows,
which ``metasci-scribe run`` treats as Ctrl+C), and the command then reports
how the child ended; the run resumes with the next ``/run``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from rich.text import Text

from ..base import Command, CommandContext
from ..manager import manager

__all__ = ["RunCommand"]

_GRACE_SECONDS = 20.0  # how long a child may take to stop after Ctrl+C
_INTERRUPTED = 130  # the shell's code for Ctrl+C, as ``metasci-scribe run`` exits

# The RunExit values of metasci_scribe.run (not imported: it is heavy, and this
# module loads with every command).
_MEANING = {
    0: "finished",
    2: "did not finish; see the message above",
    3: "stopped at the plan usage limit; /run again later to continue",
    4: "stopped for your decision; settle it, then /run again",
    _INTERRUPTED: "stopped by Ctrl+C; /run again to continue",
}

# One run at a time in this process. ``metasci-scribe run`` also locks the
# folder, so a run started from another terminal is refused by the child.
_active = False


def _child_argv(
    root: Path, model: str | None = None, provider: str | None = None
) -> list[str]:
    """The command that runs the paper in *root*, with no need for PATH.

    ``-I`` keeps the paper folder off the child's import path (it is the
    agent's working folder), ``-X utf8`` and ``-u`` give UTF-8 and unbuffered
    output whatever the Windows code page is. (They are flags, not ``PYTHON*``
    variables: ``-I`` makes the child ignore those.) *model* and *provider* are
    the chat's live choice, so the run uses the model the session shows.
    """
    argv = [sys.executable, "-I", "-X", "utf8", "-u", "-m", "metasci_scribe", "run"]
    argv += ["--workdir", str(root)]
    if model:
        argv += ["--model", model]
    if provider:
        argv += ["--provider", provider]
    return argv


def _child_env(auth_mode: str | None = None) -> dict[str, str]:
    """The environment of the child. *auth_mode* is the chat's live OpenAI
    route (``api_key`` or ``chatgpt_plan``): the environment outranks the
    config file in the child, so it cannot pick a different billing path."""
    env = dict(os.environ)
    if auth_mode:
        env["METASCI_SCRIBE_OPENAI_AUTH_MODE"] = auth_mode
    return env


def _live_settings(ctx: CommandContext) -> tuple[str | None, str | None, str | None]:
    """The model, provider and OpenAI route the chat uses now (``/model`` may
    have changed them since the session started)."""
    try:
        from ...agent import _ensure_config

        cfg = _ensure_config()
    except Exception:
        cfg = ctx.config
    return (
        getattr(cfg, "model", None),
        getattr(cfg, "provider", None),
        getattr(cfg, "openai_auth_mode", None),
    )


def _describe(model: str | None, provider: str | None, auth_mode: str | None) -> str:
    plan = provider in (None, "openai") and auth_mode == "chatgpt_plan"
    route = ", ChatGPT plan" if plan else ""
    return f"model {model or 'default'} (provider {provider or 'default'}{route})"


def _popen_group_args() -> dict:
    """Put the child in its own process group, so only ``/run`` signals it."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _interrupt_child(proc: subprocess.Popen) -> None:
    """Pass Ctrl+C on to the child."""
    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
    except (OSError, ValueError):
        pass  # already gone


@contextlib.contextmanager
def _sigint_to(on_first):
    """While active, the first Ctrl+C calls *on_first*; a second one is the
    usual ``KeyboardInterrupt`` (force quit).

    ``asyncio.run`` installs a SIGINT handler that counts interrupts for the
    life of the app. If ``/run`` left that count at 1 after swallowing a
    cancel, the next Ctrl+C, in any later turn, would end the chat (see
    ``interactive._run_rich_cli_streaming_turn``). So the runner's handler is
    replaced for the duration of the command and put back after it.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    hit = False

    def handler(signum, frame):
        nonlocal hit
        if hit:
            signal.default_int_handler(signum, frame)
            return
        hit = True
        on_first()

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)


async def _pump(proc: subprocess.Popen, ctx: CommandContext, tail: list[str]) -> None:
    """Show the child's output line by line, decoding what is not UTF-8 with
    replacement characters. The last non-blank line is kept in *tail*."""
    assert proc.stdout is not None
    while True:
        raw = await asyncio.to_thread(proc.stdout.readline)
        if not raw:
            return
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line.strip():
            tail[:] = [line.strip()]
        # ``Text`` is literal: the child's brackets are not Rich markup.
        ctx.ui.mount_renderable(Text(line, style="dim"))


async def _settle(proc: subprocess.Popen, reader: asyncio.Future) -> int:
    """Wait for an interrupted child, killing it after the grace period."""
    try:
        await asyncio.to_thread(proc.wait, _GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        await asyncio.to_thread(proc.wait)
    await reader
    return proc.returncode


async def _finish(proc: subprocess.Popen, reader: asyncio.Future) -> int:
    # Shielded: an interrupt must not cut the reader off before the child's
    # last lines are shown.
    await asyncio.shield(reader)
    await asyncio.to_thread(proc.wait)
    return proc.returncode


class RunCommand(Command):
    """Write the draft unattended: ``metasci-scribe run`` in this folder."""

    name = "/run"
    description = (
        "Write the draft unattended in a separate process (resumes where it stopped)"
    )

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        global _active
        if _active:
            ctx.ui.append_system(
                "A run is already in progress. Wait for it to finish, or press "
                "Ctrl+C to stop it.",
                style="yellow",
            )
            return
        root = ctx.workspace.root
        _active = True
        try:
            await self._run(ctx, root)
        finally:
            _active = False

    async def _run(self, ctx: CommandContext, root: Path) -> None:
        model, provider, auth_mode = _live_settings(ctx)
        try:
            proc = subprocess.Popen(
                _child_argv(root, model, provider),
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=_child_env(auth_mode),
                **_popen_group_args(),
            )
        except OSError as exc:
            ctx.ui.append_system(f"✗ Could not start the run: {exc}", style="red")
            return
        ctx.ui.append_system(
            f"Run started in {root} with {_describe(model, provider, auth_mode)}. "
            "Your chat is not part of it. Ctrl+C stops it; /run again continues.",
            style="cyan",
        )
        loop = asyncio.get_running_loop()
        interrupt = asyncio.Event()
        tail: list[str] = []
        reader = asyncio.ensure_future(_pump(proc, ctx, tail))
        finisher = asyncio.ensure_future(_finish(proc, reader))
        signalled = asyncio.ensure_future(interrupt.wait())
        cancelled: asyncio.CancelledError | None = None
        try:
            with _sigint_to(lambda: loop.call_soon_threadsafe(interrupt.set)):
                try:
                    await asyncio.wait(
                        {finisher, signalled}, return_when=asyncio.FIRST_COMPLETED
                    )
                except asyncio.CancelledError as exc:
                    cancelled = exc
                if cancelled is None and finisher.done():
                    code = finisher.result()
                else:
                    _interrupt_child(proc)
                    code = await _settle(proc, reader)
        finally:
            signalled.cancel()
            if not finisher.done():
                finisher.cancel()
            if proc.poll() is None:
                proc.kill()
        self._report(ctx, code, tail)
        if cancelled is not None:
            task = asyncio.current_task()
            # Ctrl+C meant "stop the run", not "close the chat": go back to
            # the prompt unless something else is cancelling this task too.
            if task is not None and task.uncancel() > 0:
                raise cancelled

    @staticmethod
    def _report(ctx: CommandContext, code: int, tail: list[str]) -> None:
        meaning = _MEANING.get(code)
        if meaning is None:
            # Not a result of the run: the child failed before it could say
            # anything (an import error, a crash, a kill). Say what it printed last.
            last = f" Its last output was: {tail[0][:200]}" if tail else ""
            meaning = f"it ended without a result.{last} See the output above"
        ok = code == 0
        ctx.ui.append_system(
            f"{'✓' if ok else '✗'} Run ended with exit code {code}: {meaning}.",
            style="green" if ok else "yellow",
        )


manager.register(RunCommand())
