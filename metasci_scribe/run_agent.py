"""The real agent of an autonomous run (``metasci-scribe run``).

``real_invoke(root)`` is the ``invoke`` that :func:`metasci_scribe.run.run_project`
calls once per pass. It builds the CLI agent with ``create_cli_agent`` and runs
one turn through the local gateway, the same event stream the single-shot
modes use.

What the autonomous agent is, and is not:

* ``auto_mode`` and ``auto_approve`` on, ``ask_user`` off: nobody answers.
* No shell. A run with no human gate gets no ``execute`` and no
  ``run_in_background`` (``ShellGateMiddleware``), and no skill installs.
* No recursive ``delete``: with the interrupt off nothing else would stop an
  injected agent from removing ``Knowledge/`` or ``Manuscript/``.
* No MCP tools at all, and no async sub-agents or ``langgraph dev`` server.
* ``mark_paragraph`` is added to the main agent only (not to a sub-agent, and
  never to a chat session).

The single-shot text path (``cmd_run``) turns every error into a ``typer.Exit``,
which would hide a :class:`~metasci_scribe.llm.errors.PlanStop` from
``run_project``. This module reads the gateway's event stream itself, so a
failure reaches ``run_project`` with its type. It prints one line per
``mark_paragraph`` result and nothing of the model's text or reasoning.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

from .paths import SessionDirs, Workspace
from .run import _short, make_mark_paragraph_tool

__all__ = ["RunInvoker", "autonomous_config", "build_run_agent", "real_invoke"]

Progress = Callable[[str], None]


def autonomous_config(config: Any) -> Any:
    """A copy of *config* for an unattended run. The caller's config is not touched."""
    cfg = copy.copy(config)
    cfg.auto_mode = True
    cfg.auto_approve = True
    cfg.enable_ask_user = False
    cfg.enable_async_subagents = False
    return cfg


def build_run_agent(root: Path, cfg: Any, checkpointer: Any, chat_model: Any = None):
    """The autonomous agent for the paper in *root*, with ``mark_paragraph``.

    *cfg* is an :func:`autonomous_config`. It is also made the active config,
    which the shell gate and the retry tier read at run time, so a model passed
    in (tests) cannot leave the run looking attended.
    """
    from .agent import create_cli_agent, set_active_config

    root = Path(root)
    set_active_config(cfg)
    return create_cli_agent(
        work_dir=str(root),
        workspace=Workspace(root),
        checkpointer=checkpointer,
        config=cfg,
        chat_model=chat_model,
        extra_tools=[make_mark_paragraph_tool(root)],
        with_mcp=False,
        refuse_delete=True,
    )


def _echo(text: str) -> None:
    click.echo(text)
    # A parent (``/run``) reads this through a pipe, line by line.
    import sys

    sys.stdout.flush()


def _report(event: dict[str, Any], say: Progress) -> None:
    """One human-readable line per paragraph mark. Everything else is dropped."""
    if event.get("type") == "tool_result" and event.get("name") == "mark_paragraph":
        say(f"  {_short(event.get('content', ''))}")


async def _one_pass(
    root: Path,
    cfg: Any,
    chat_model: Any,
    thread_id: str,
    prompt: str,
    say: Progress,
) -> None:
    from .cli._constants import build_metadata
    from .gateway import GraphTarget, RunRequest, create_runtime_gateways_for_config
    from .middleware.code_interpreter import aclose_code_interpreters
    from .sessions import get_checkpointer

    dirs = SessionDirs(Workspace(root))
    graph_gateway = create_runtime_gateways_for_config(
        cfg, backend="local"
    ).graph_gateway
    # The thread lives in the sessions database, so a later invocation of
    # ``run`` continues it by the thread ID in run.json.
    async with get_checkpointer() as checkpointer:
        agent = build_run_agent(root, cfg, checkpointer, chat_model)
        request = RunRequest(
            message=prompt,
            thread_id=thread_id,
            metadata=build_metadata(dirs, cfg.model),
            target=GraphTarget(local_graph=agent, **dirs.metadata()),
        )
        try:
            async for event in graph_gateway.stream_events(request):
                _report(event, say)
        finally:
            await aclose_code_interpreters()


class RunInvoker:
    """The ``invoke`` of the real agent, and the event loop its passes share.

    The chat model caches one HTTP client (one connection pool) for the whole
    process, and a pool is tied to the loop it first ran on. A fresh loop per
    pass would hand pass 2 a client whose loop is closed ("Event loop is
    closed"), so every pass runs on the one ``asyncio.Runner`` this object
    owns. A Runner keeps Ctrl+C working on Windows, where a loop on another
    thread cannot be interrupted. Use it as a context manager, or call
    :meth:`close`.
    """

    def __init__(
        self, root: Path, cfg: Any, chat_model: Any, progress: Progress
    ) -> None:
        self._root = root
        self._cfg = cfg
        self._chat_model = chat_model
        self._say = progress
        self._runner: asyncio.Runner | None = None

    def __call__(self, thread_id: str, prompt: str) -> None:
        if self._runner is None:
            self._runner = asyncio.Runner()
        self._runner.run(
            _one_pass(
                self._root, self._cfg, self._chat_model, thread_id, prompt, self._say
            )
        )

    def close(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.close()

    def __enter__(self) -> RunInvoker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def real_invoke(
    root: Path,
    *,
    config: Any = None,
    chat_model: Any = None,
    progress: Progress | None = None,
) -> RunInvoker:
    """The ``invoke(thread_id, prompt)`` of the real agent in *root*.

    Each call runs one turn of the thread on the invoker's own loop (see
    :class:`RunInvoker`) and lets any failure out as it is, so ``run_project``
    can tell a plan usage limit from other errors. *chat_model* is for tests;
    without it the model comes from the configuration.
    """
    from .config import get_effective_config

    root = Path(root)
    cfg = autonomous_config(config if config is not None else get_effective_config())
    return RunInvoker(root, cfg, chat_model, progress or _echo)
