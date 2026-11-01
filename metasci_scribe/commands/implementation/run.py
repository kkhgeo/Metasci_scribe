"""``/run``: write the draft unattended, in a separate process.

The chat keeps its thread and its agent. ``/run`` starts ``metasci-scribe run``
in the session's paper folder as a child process and shows what it prints in
the chat. The child owns the model calls; its exit code is the ``RunExit``
value of :mod:`metasci_scribe.run`.

Reading the child's output happens on worker threads, so the prompt's event
loop keeps running. Ctrl+C is passed on to the child (Ctrl+Break on Windows,
which ``metasci-scribe run`` treats as Ctrl+C), and the command then reports
how the child ended; the run resumes with the next ``/run``.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
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


def _child_argv(root: Path) -> list[str]:
    """The command that runs the paper in *root*, with no need for PATH.

    ``-I`` keeps the paper folder off the child's import path (it is the
    agent's working folder), ``-X utf8`` and ``-u`` give UTF-8 and unbuffered
    output whatever the Windows code page is.
    """
    return [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        "-u",
        "-m",
        "metasci_scribe",
        "run",
        "--workdir",
        str(root),
    ]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    return env


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


async def _pump(proc: subprocess.Popen, ctx: CommandContext) -> None:
    """Show the child's output line by line, decoding what is not UTF-8 with
    replacement characters."""
    assert proc.stdout is not None
    while True:
        raw = await asyncio.to_thread(proc.stdout.readline)
        if not raw:
            return
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
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
        try:
            proc = subprocess.Popen(
                _child_argv(root),
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=_child_env(),
                **_popen_group_args(),
            )
        except OSError as exc:
            ctx.ui.append_system(f"✗ Could not start the run: {exc}", style="red")
            return
        ctx.ui.append_system(
            f"Run started in {root}. Your chat is not part of it. Ctrl+C stops "
            "it; /run again continues.",
            style="cyan",
        )
        reader = asyncio.ensure_future(_pump(proc, ctx))
        interrupted: BaseException | None = None
        try:
            # The reader is shielded: an interrupt must not cut it off before
            # the child's last lines are shown.
            await asyncio.shield(reader)
            await asyncio.to_thread(proc.wait)
            code = proc.returncode
        except (asyncio.CancelledError, KeyboardInterrupt) as exc:
            interrupted = exc
            _interrupt_child(proc)
            code = await _settle(proc, reader)
        finally:
            if proc.poll() is None:
                proc.kill()
        self._report(ctx, code)
        if isinstance(interrupted, asyncio.CancelledError):
            task = asyncio.current_task()
            # Ctrl+C meant "stop the run", not "close the chat": go back to
            # the prompt unless something else is cancelling this task too.
            if task is not None and task.uncancel() > 0:
                raise interrupted

    @staticmethod
    def _report(ctx: CommandContext, code: int) -> None:
        meaning = _MEANING.get(code, "ended unexpectedly")
        ok = code == 0
        ctx.ui.append_system(
            f"{'✓' if ok else '✗'} Run ended with exit code {code}: {meaning}.",
            style="green" if ok else "yellow",
        )


manager.register(RunCommand())
