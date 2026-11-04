"""Typer command registrations — onboard, config, mcp, main callback."""

import copy
import json
import logging
import os
import sys
from contextlib import contextmanager, nullcontext
from importlib.metadata import version as _pkg_version
from typing import Annotated, Any, cast

import click
import typer
from rich.markup import escape
from rich.table import Table

from .. import project
from ..gateway import (
    GraphGateway,
    GraphTarget,
    RunRequest,
)
from ..llm.context_window import DEFAULT_CONTEXT_WINDOW_FALLBACK, resolve_context_window
from ..llm.credentials import ModelNotConfigured
from ..paths import (
    SessionDirs,
    Workspace,
    WorkspaceRootError,
    ensure_dirs,
    reload_env_dirs,
    require_workspace_root,
    start_workspace_path,
)
from ..runtime import AsyncRuntime
from ..stream.console import console
from . import (
    server_cmd,  # noqa: F401 — registers `metasci-scribe server` commands
)
from ._app import app, config_app, configure_app, mcp_app, sessions_app
from ._constants import build_metadata
from .agent import (
    _load_agent,
)
from .mcp_ui import (
    _mcp_add_server_from_kwargs,
    _mcp_edit_server_fields,
    _mcp_list_servers,
    _mcp_remove_server,
    _show_mcp_config,
)

_ASYNC_RUNTIME_META_KEY = "metasci-scribe.async_runtime"


def _close_cli_async_runtime(runtime: AsyncRuntime) -> None:
    """Close the owned runtime or surface a controlled CLI shutdown failure."""
    try:
        runtime.close()
    except TimeoutError as exc:
        click.echo(
            f"Error: Async runtime shutdown did not complete: {exc}",
            err=True,
        )
        raise click.exceptions.Exit(1) from None


def _get_cli_async_runtime(ctx: typer.Context) -> AsyncRuntime:
    """Return the application-scoped runtime owned by this CLI invocation."""
    root = ctx.find_root()
    runtime = root.meta.get(_ASYNC_RUNTIME_META_KEY)
    if runtime is None:
        runtime = AsyncRuntime()
        root.meta[_ASYNC_RUNTIME_META_KEY] = runtime
        root.call_on_close(lambda: _close_cli_async_runtime(runtime))
    if not isinstance(runtime, AsyncRuntime):  # pragma: no cover - defensive
        raise RuntimeError("CLI async runtime context is invalid")
    return runtime


# =============================================================================
# Onboard command
# =============================================================================


@app.command()
def onboard(
    ctx: typer.Context,
    skip_validation: bool = typer.Option(
        False, "--skip-validation", help="Skip API key validation during setup"
    ),
    # ---- Pre-fill answers (any subset; remaining prompts stay interactive)
    provider: str | None = typer.Option(
        None, "--provider", help="Pre-set LLM provider (e.g. anthropic, openai)"
    ),
    model: str | None = typer.Option(None, "--model", help="Pre-set model name"),
    api_key: str | None = typer.Option(
        None, "--api-key", help="Pre-set API key for the chosen --provider"
    ),
    show_thinking: bool | None = typer.Option(
        None,
        "--show-thinking/--no-show-thinking",
        help="Pre-set thinking-panel visibility",
    ),
    port: int | None = typer.Option(
        None, "--port", help="Pre-set langgraph dev server port"
    ),
    # ---- Skip flags
    skip_skills: bool = typer.Option(
        False, "--skip-skills", help="Skip skills install"
    ),
    skip_mcp: bool = typer.Option(False, "--skip-mcp", help="Skip MCP server setup"),
    skip_latex: bool = typer.Option(False, "--skip-latex", help="Skip LaTeX setup"),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Run without prompts — every required answer must come from a flag",
    ),
):
    """Interactive setup wizard for Metasci Scribe.

    Guides you through configuring API keys, model selection,
    workspace settings, and agent parameters.

    Any answer can be pre-set via a flag (``--provider anthropic
    --model claude-sonnet-5-5 ...``); prompts for unset answers stay
    interactive unless ``--non-interactive`` is passed, in which case any
    missing required answer aborts the wizard.
    """
    from ..config.onboard.constants import (
        VALID_PROVIDERS,
    )
    from ..config.onboard.prompter import NonInteractivePrompter

    # Validate constrained string flags up-front so a typo doesn't silently
    # poison the saved config. Allowed-value sets live in
    # ``metasci_scribe/config/onboard/constants.py``; a drift test in
    # ``tests/test_onboard.py`` keeps them aligned with the interactive
    # ``Choice(value=...)`` lists in ``steps.py``.
    if provider is not None and provider not in VALID_PROVIDERS:
        raise typer.BadParameter(
            f"--provider must be one of {sorted(VALID_PROVIDERS)}",
            param_hint="--provider",
        )
    # Match the interactive prompt's range (1024 < port < 65536). Without
    # this check, --port 80 or --port 99999 would land in config and break
    # the langgraph dev server on startup.
    if port is not None and not (1024 < port < 65536):
        raise typer.BadParameter(
            "--port must be in the user-port range (1025 — 65535)",
            param_hint="--port",
        )

    # Collect flag-supplied answers keyed by the prompt_id wizard steps use.
    answers: dict = {}
    if port is not None:
        answers["port"] = str(port)
    if provider is not None:
        answers["provider"] = provider
    if model is not None:
        answers["model"] = model
    if api_key is not None:
        answers["api_key"] = api_key
    if show_thinking is not None:
        answers["show_thinking"] = show_thinking

    skip_set = {
        section
        for section, flag in (
            ("skills", skip_skills),
            ("mcp", skip_mcp),
            ("latex", skip_latex),
        )
        if flag
    }

    prompter = None
    if answers or skip_set or non_interactive:
        prompter = NonInteractivePrompter(
            answers=answers,
            skip_set=skip_set,
            strict=non_interactive,
        )

    _run_onboard_cli(
        skip_validation=skip_validation,
        prompter=prompter,
        runtime=_get_cli_async_runtime(ctx),
    )


# =============================================================================
# Setup command
# =============================================================================


@app.command()
def setup(
    manifest: bool = typer.Option(
        False, "--manifest", help="Print the setup stages as JSON and exit"
    ),
    stage: str | None = typer.Option(
        None, "--stage", help="Run only this stage (ids from --manifest)"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print one JSON event per line instead of progress text"
    ),
    cn: bool = typer.Option(
        False, "--cn", help="Download from mainland China mirrors and remember it"
    ),
    skip: Annotated[
        list[str] | None,
        typer.Option(
            "--skip", help="Skip this stage (repeatable; ids from --manifest)"
        ),
    ] = None,
):
    """Install what Metasci Scribe needs beyond the Python package (Node.js, Git
    for Windows on Windows, and a Python for the agent's shell when none is on
    PATH).

    Runs every stage that applies to this platform, in order. With ``--json``
    stdout carries only the JSON event lines; everything else goes to stderr.
    """

    from ..config import load_config, set_config_value
    from ..setup import STAGES, get_stage, run_stages
    from ..setup import manifest as setup_manifest
    from ..setup.protocol import ConsoleEmitter, JsonEmitter

    if manifest:
        sys.stdout.write(json.dumps(setup_manifest()) + "\n")
        return

    if json_output:
        # The shared console also carries log warnings; stdout belongs to the
        # event lines.
        from ..stream.json_sink import redirect_console_to_stderr

        redirect_console_to_stderr()

    known = ", ".join(s.id for s in STAGES)
    skipped = frozenset(skip or ())
    unknown = sorted(skipped - {s.id for s in STAGES})
    if unknown:
        typer.echo(
            f"Unknown stage {', '.join(map(repr, unknown))}. Known stages: {known}",
            err=True,
        )
        raise typer.Exit(2)
    if skipped and stage is not None:
        typer.echo("--skip cannot be combined with --stage.", err=True)
        raise typer.Exit(2)

    selected = tuple(s for s in STAGES if s.applies())
    if stage is not None:
        found = get_stage(stage)
        if found is None:
            typer.echo(f"Unknown stage {stage!r}. Known stages: {known}", err=True)
            raise typer.Exit(2)
        selected = (found,)

    if cn:
        from ..config.store import ConfigError

        try:
            set_config_value("mirror", "cn")
        except (OSError, ConfigError) as exc:
            # Saving the choice is secondary; this run still uses the mirror.
            logging.getLogger(__name__).warning(
                f"Could not save mirror: cn to the config ({exc}); "
                "using the mirror for this run only."
            )
    mirror = "cn" if cn else load_config().mirror

    emit = JsonEmitter() if json_output else ConsoleEmitter(console)
    code = run_stages(selected, emit, mirror, skip=skipped)
    if code:
        raise typer.Exit(code)


# =============================================================================
# `metasci-scribe configure <section>` — re-run one onboarding section
# =============================================================================


_CONFIGURE_SECTIONS = {
    "port": "LangGraph server port",
    "provider": "LLM provider + API key",
    "model": "Model + reasoning effort",
    "thinking": "Thinking panel",
    "skills": "Skills",
    "mcp": "MCP servers",
    "latex": "LaTeX (TinyTeX)",
}


def _run_onboard_cli(**kwargs: Any) -> None:
    """Invoke the wizard, presenting non-interactive errors as a clean
    message + exit code 1 instead of a raw Python traceback.

    The wizard raises ``RuntimeError`` for *expected* non-interactive
    failures: a rejected ``--api-key`` preset or missing
    required flags under ``--non-interactive``. A
    refused config file (one that does not parse, or the lock held
    elsewhere) or a write that keeps failing raises ``ConfigError``, a
    ``RuntimeError`` too. Those are user-input problems, not bugs — surface them like any other
    CLI validation error rather than dumping a stack trace.
    """
    from ..config import run_onboard

    try:
        run_onboard(**kwargs)
    except RuntimeError as exc:
        console.print(f"[red]✗ {escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc


def _configure_section(
    section: str,
    skip_validation: bool = False,
    *,
    runtime: AsyncRuntime | None = None,
) -> None:
    """Run a single onboarding section, reusing the wizard's step logic."""
    kwargs: dict[str, Any] = {
        "skip_validation": skip_validation,
        "only_sections": {section},
    }
    if runtime is not None:
        kwargs["runtime"] = runtime
    _run_onboard_cli(
        **kwargs,
    )


@configure_app.command("port")
def configure_port():
    """Re-run langgraph dev server port selection."""
    _configure_section("port")


@configure_app.command("provider")
def configure_provider(
    skip_validation: bool = typer.Option(False, "--skip-validation"),
):
    """Re-run LLM provider and API key prompts.

    Model selection is automatically re-run after provider — the model list
    depends on the provider, and silently leaving e.g. ``model="claude-...""``
    when the provider was switched to ``openai`` would break the first
    request. Press Enter on the model picker to keep the current default.
    """
    _run_onboard_cli(
        skip_validation=skip_validation,
        only_sections={"provider", "model"},
    )


@configure_app.command("model")
def configure_model():
    """Re-run model selection (and reasoning effort for OpenAI)."""
    _configure_section("model")


@configure_app.command("thinking")
def configure_thinking():
    """Re-run thinking-panel visibility selection."""
    _configure_section("thinking")


@configure_app.command("skills")
def configure_skills():
    """Re-run skills install/sync."""
    _configure_section("skills")


@configure_app.command("mcp")
def configure_mcp():
    """Re-run MCP server selection."""
    _configure_section("mcp")


@configure_app.command("latex")
def configure_latex():
    """Re-run LaTeX (TinyTeX) setup."""
    _configure_section("latex")


# =============================================================================
# Compact helper
# =============================================================================

_COMPACT_CONTEXT_WINDOW_FALLBACK = DEFAULT_CONTEXT_WINDOW_FALLBACK
_MANUAL_COMPACT_MIN_FRACTION = 0.40
_MANUAL_COMPACT_MIN_PERCENT = int(_MANUAL_COMPACT_MIN_FRACTION * 100)


class CompactResult:
    """Structured result from compact_conversation.

    Attributes:
        status: "noop" (nothing to compact), "ok" (compacted), or "error".
        message: Short human-readable message (used as fallback text).
        messages_compacted: Number of messages summarized (0 for noop/error).
        messages_kept: Number of messages unchanged.
        tokens_before: Total tokens before compaction.
        tokens_after: Total tokens after compaction.
        tokens_summarized: Tokens in the summarized portion (before).
        tokens_summary: Tokens in the summary message (after).
        pct_decrease: Percentage decrease.
        context_window: Model context window used for thresholding.
        context_percent: Effective context utilization percent.
        summary_text: Human-readable compact summary content for UI display.
    """

    __slots__ = (
        "context_percent",
        "context_window",
        "message",
        "messages_compacted",
        "messages_kept",
        "pct_decrease",
        "status",
        "summary_text",
        "tokens_after",
        "tokens_before",
        "tokens_summarized",
        "tokens_summary",
    )

    def __init__(
        self,
        status: str,
        message: str,
        *,
        messages_compacted: int = 0,
        messages_kept: int = 0,
        tokens_before: int = 0,
        tokens_after: int = 0,
        tokens_summarized: int = 0,
        tokens_summary: int = 0,
        pct_decrease: int = 0,
        context_window: int = 0,
        context_percent: int = 0,
        summary_text: str = "",
    ):
        self.status = status
        self.message = message
        self.messages_compacted = messages_compacted
        self.messages_kept = messages_kept
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.tokens_summarized = tokens_summarized
        self.tokens_summary = tokens_summary
        self.pct_decrease = pct_decrease
        self.context_window = context_window
        self.context_percent = context_percent
        self.summary_text = summary_text

    def __str__(self) -> str:
        return self.message


class CompactSummaryRenderable:
    """Rich renderable payload for the manual compact summary content."""

    __slots__ = ("summary_text",)

    def __init__(self, summary_text: str):
        self.summary_text = (summary_text or "").strip()

    def __rich_console__(self, console, options):
        yield render_compact_summary_panel(self.summary_text)


def _ensure_async_subagent_server(
    config: Any, *, dirs: SessionDirs, backend: str | None = None
) -> None:
    """Start the langgraph dev subprocess for background agent work.

    Shared by the interactive and single-shot entries so the user-visible
    status message and workspace-mismatch handling stay in one place.

    ``backend`` is the calling surface's resolved gateway backend, forwarded to
    ``ensure_langgraph_dev`` so whether a server is needed follows the
    surface's choice rather than re-reading the global flag. ``None`` keeps the
    global-read behavior.

    Raises ``typer.Exit(1)`` (after surfacing a red error) when an
    externally-managed langgraph dev is already running for a different
    workspace — e.g., a ``metasci-scribe --workdir /A`` session left its server
    up and the user is starting ``metasci-scribe`` in /B. Continuing in that
    state would route async sub-agent calls to a process pinned to /A
    while the main agent runs in /B.
    """
    from ..langgraph_dev.manager import (
        _DEFAULT_HOST,
        WorkspaceMismatchError,
        _is_loopback_host,
        ensure_langgraph_dev,
        is_async_subagents_available,
        needs_langgraph_dev,
    )

    # With the defaults nothing asks for the server: no spinner, no message,
    # and ``ensure_langgraph_dev`` returns without starting anything.
    status = (
        console.status(
            "[dim]Starting background agent server (langgraph dev)...[/dim]",
            spinner="dots",
        )
        if needs_langgraph_dev(config, backend=backend)
        else nullcontext()
    )
    try:
        with status:
            ensure_langgraph_dev(
                config,
                workspace_dir=dirs.workspace.root,
                run_dir=dirs.run_dir,
                backend=backend,
            )
            _reconcile_autoskill_schedule(config, workspace=dirs.workspace)
    except WorkspaceMismatchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    from ..langgraph_dev import manager as _lg_manager

    if _lg_manager.CONFIG_DRIFT_SINCE_LAUNCH:
        console.print(
            "[yellow]⚠ Configuration or version changed since the background "
            "agent server was launched — it still runs the old settings or "
            "version. Apply them with [bold]metasci-scribe server stop[/bold], then "
            "restart metasci-scribe.[/yellow]"
        )
    if _lg_manager.AGENT_SHELL_DRIFT is not None:
        console.print(f"[yellow]⚠ {escape(_lg_manager.AGENT_SHELL_DRIFT)}[/yellow]")

    # The exposure warning lives here, where every session starts the
    # backend. Gated on the server being up: warning
    # about a bind that never happened would be worse than saying nothing.
    bind_host = str(getattr(config, "langgraph_dev_host", _DEFAULT_HOST) or "").strip()
    if (
        bind_host
        and not _is_loopback_host(bind_host)
        and is_async_subagents_available()
    ):
        console.print(
            "[bold white on red] ⚠ PUBLIC BIND [/bold white on red] "
            f"[bold red]Agent server listening on {bind_host} — no auth, and "
            f"the agent can run shell. Set langgraph_dev_host to 127.0.0.1 on "
            f"untrusted networks.[/bold red]"
        )


def warn_server_backend_hitl_caveats(
    backend: str | None, *, surface_label: str
) -> None:
    """Warn that the server gateway backend is lossy for a HITL surface.

    The interactive CLI create a per-session workspace and switch models
    mid-session, but on the server backend neither reaches the run: execution
    stays in the server's import-time workspace, and per-run config (model,
    active teams, HITL suppression) is dropped when a turn resumes after a tool
    approval. Both are gaps the server path has not closed yet, so a surface
    that opts into the server backend surfaces this once at startup rather than
    silently misrouting.
    """
    if backend != "langgraph_server":
        return
    console.print(
        f"[yellow]⚠ Server gateway backend active for the {surface_label}. "
        "Until the remaining gaps close, the per-session workspace is not "
        "applied server-side (execute / write_file run in the server's "
        "workspace), and a model or team switch is dropped for the rest of a "
        "turn that resumes after a tool approval.[/yellow]"
    )


def _reconcile_autoskill_schedule(config: Any, *, workspace: Workspace) -> None:
    """Best-effort reconciliation for EvoMemory's hidden AutoSkills cron.

    The cron belongs to the workspace, whichever run folder the session
    works in. Only for a server known to serve *workspace*, as for
    the reconcile replaces crons it counts as the workspace's, and in another
    workspace's store those are not.
    """
    try:
        from ..langgraph_dev import manager
        from ..memory.autoskills.schedule import reconcile_autoskill_schedule

        if manager.SERVED_WORKSPACE != workspace.root:
            return
        reconcile_autoskill_schedule(config, workspace_dir=workspace.root)
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to reconcile EvoMemory AutoSkills schedule", exc_info=True
        )


# How long a CLI waits for EvoMemory work before it exits or moves the server.
_MEMORY_WAIT_SECONDS = 120.0
_MEMORY_WAIT_POLL_SECONDS = 0.5
_MEMORY_WAIT_OUTPUT_GRACE_SECONDS = 3.0


def _wait_for_memory_workers(*, timeout_action: str = "shutting down") -> None:
    """Wait, bounded, for queued EvoMemory work to finish.

    One-shot runs wait so post-run memory persists before atexit cleanup;
    moving the server waits because a move stops runs in flight.
    ``timeout_action`` says what happens when the wait runs out. Returns at
    once when nothing runs.
    """
    try:
        from ..memory.worker_activity import (
            MemoryActivityPhase,
            MemoryWorkerStatusSnapshot,
            wait_for_memory_pipeline_idle,
        )
    except Exception:
        return

    announced = False

    def print_saved(observed: MemoryWorkerStatusSnapshot) -> None:
        saved = []
        if observed.observations_recorded:
            saved.append(f"{observed.observations_recorded} observation(s)")
        if observed.profile_updates:
            saved.append(f"{observed.profile_updates} profile update(s)")
        if saved:
            console.print(f"[dim]EvoMemory saved {', '.join(saved)}.[/dim]")

    def print_waiting(phase: MemoryActivityPhase) -> None:
        nonlocal announced
        if not announced:
            console.print(f"[dim]Waiting for EvoMemory {phase}...[/dim]")
            announced = True

    def print_timeout(phase: MemoryActivityPhase) -> None:
        console.print(
            f"[dim]EvoMemory {phase} is still running; {timeout_action}.[/dim]"
        )

    wait_for_memory_pipeline_idle(
        timeout_seconds=_MEMORY_WAIT_SECONDS,
        poll_seconds=_MEMORY_WAIT_POLL_SECONDS,
        output_grace_seconds=_MEMORY_WAIT_OUTPUT_GRACE_SECONDS,
        on_saved=print_saved,
        on_waiting=print_waiting,
        on_timeout=print_timeout,
    )


_RESUME_SYNC_STATUS = (
    "[dim]Syncing background agent server to resumed workspace...[/dim]"
)


async def _sync_background_agent_server_workspace(
    config: Any,
    *,
    dirs: SessionDirs,
    backend: str | None = None,
    status_message: str = _RESUME_SYNC_STATUS,
) -> None:
    """Sync langgraph dev to a resumed workspace for background agent work.

    ``ensure_langgraph_dev`` is intentionally always called: EvoMemory
    background workers require the server even when async subagents are disabled.
    WorkspaceMismatchError is left for callers to handle according to their UI
    flow.

    ``backend`` is the calling surface's resolved gateway backend, forwarded so
    whether a server is needed follows the surface's choice; ``None`` keeps
    the global-read behavior.
    """
    import asyncio

    from ..langgraph_dev.manager import (
        ensure_langgraph_dev,
        owned_server_pinned_elsewhere,
    )

    with console.status(status_message, spinner="dots"):
        # Only a move stops runs in flight; a sync to the served folders is a
        # health check and must not wait on memory work.
        if await asyncio.to_thread(
            owned_server_pinned_elsewhere, dirs.workspace.root, dirs.run_dir
        ):
            await asyncio.to_thread(
                _wait_for_memory_workers, timeout_action="moving the server anyway"
            )
        await asyncio.to_thread(
            ensure_langgraph_dev,
            config,
            workspace_dir=dirs.workspace.root,
            run_dir=dirs.run_dir,
            backend=backend,
        )
        await asyncio.to_thread(
            _reconcile_autoskill_schedule,
            config,
            workspace=dirs.workspace,
        )


async def _move_server_or_keep_session(
    config: Any,
    *,
    target: SessionDirs,
    current: SessionDirs | None,
    backend: str | None = None,
    status_message: str = _RESUME_SYNC_STATUS,
) -> bool:
    """Move the background agent server to *target* before a session goes there.

    Until the server serves several workspaces it is pinned to one workspace
    and run folder, so background workers and deployed sub-agents would
    otherwise keep working in *current*, the folders the server was last
    synced to (``None`` when it was not started yet). Nothing happens when
    the two are the same.

    Raises ``WorkspaceMismatchError`` when another metasci-scribe session holds the
    server for other folders: the caller keeps its session as it is. Returns
    False when the move failed otherwise: the session can go to *target*, but
    async sub-agents and EvoMemory workers may be unavailable.
    """
    if target == current:
        return True

    from ..langgraph_dev.manager import WorkspaceMismatchError

    try:
        await _sync_background_agent_server_workspace(
            config, dirs=target, backend=backend, status_message=status_message
        )
    except WorkspaceMismatchError:
        raise
    except Exception:
        logging.getLogger(__name__).warning(
            "Failed to sync background agent server to %s; continuing in degraded mode",
            target.work_dir,
            exc_info=True,
        )
        return False
    return True


@contextmanager
def _local_folders_only():
    """Turn a network-folder refusal into the message and exit code 1."""
    try:
        yield
    except project.NetworkPathError as exc:
        console.print(f"[red]✗ {escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc


async def _restore_thread_dirs(
    thread_id: str,
    *,
    dirs: SessionDirs,
    served: SessionDirs | None,
    graph_gateway: GraphGateway,
    config: Any,
    backend: str | None = None,
) -> SessionDirs:
    """Return the folders to resume *thread_id* in, with the server moved there.

    A thread resumes in the workspace and run folder it was stored with,
    wherever the CLI was started; a thread stored without folders resumes in
    *dirs*, the folders the session started in. *served* are the folders the
    server was started for, ``None`` when it was not started yet. Raises
    ``typer.Exit(1)`` when another session holds the server.
    """
    from ..langgraph_dev.manager import WorkspaceMismatchError

    metadata = await graph_gateway.get_thread_metadata(thread_id) or {}
    # Refuse a network folder by its stored spelling, before anything resolves
    # it (which would touch the share), then again once resolved: a local
    # looking link can lead to one.
    with _local_folders_only():
        project.require_local_stored(
            metadata.get("workspace_dir"), metadata.get("run_dir")
        )
        restored = (
            SessionDirs.from_stored(
                metadata.get("workspace_dir"), metadata.get("run_dir")
            )
            or dirs
        )
        project.require_local_path(restored.workspace.root)
    try:
        synced = await _move_server_or_keep_session(
            config, target=restored, current=served, backend=backend
        )
    except WorkspaceMismatchError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not synced:
        console.print(
            "[yellow]Background agent server sync failed; async subagents and "
            "EvoMemory workers may be unavailable.[/yellow]"
        )
    return restored


def _resolve_context_window(
    model: Any, fallback: int = _COMPACT_CONTEXT_WINDOW_FALLBACK
) -> int:
    """Resolve a model context window with a stable fallback."""
    return resolve_context_window(model, fallback=fallback)


def _percent_used(tokens: int, context_window: int) -> int:
    """Return a clamped utilization percent."""
    if context_window <= 0:
        return 0
    return max(0, min(100, round((tokens / context_window) * 100)))


def render_compact_result(result: CompactResult):  # -> rich.text.Text
    """Render a CompactResult as styled Rich Text.

    Uses the same visual language as the token usage display:
    cyan for numbers, green for savings, dim for labels.
    """
    from rich.text import Text

    output = Text()

    if result.status == "noop":
        output.append("○ ", style="dim")
        output.append("Manual compact not needed", style="dim")
        if result.tokens_before > 0:
            output.append("  [", style="dim")
            output.append(f"{result.tokens_before:,}", style="cyan")
            if result.context_window > 0:
                output.append(" / ", style="dim")
                output.append(f"{result.context_window:,}", style="cyan")
                output.append(" tokens", style="dim")
                output.append("  │  ", style="dim")
                output.append(f"{result.context_percent}%", style="cyan")
                output.append(" of window", style="dim")
            else:
                output.append(" tokens", style="dim")
            output.append("]", style="dim")
        if result.message:
            output.append("\n  ", style="")
            output.append(result.message, style="dim")
        return output

    if result.status == "error":
        output.append("✗ ", style="red")
        output.append(result.message, style="red")
        return output

    # status == "ok"
    output.append("✓ ", style="green")
    output.append("Compacted ", style="dim")
    output.append(f"{result.messages_compacted}", style="bold")
    output.append(" messages", style="dim")
    output.append("  [", style="dim")
    output.append(f"{result.tokens_before:,}", style="cyan")
    output.append(" → ", style="dim")
    output.append(f"{result.tokens_after:,}", style="green")
    output.append(" tokens", style="dim")
    output.append(f"  ↓{result.pct_decrease}%", style="green bold")
    output.append("]", style="dim")

    # Second line: detail breakdown
    output.append("\n  ", style="")
    output.append("Summarized: ", style="dim")
    output.append(f"{result.tokens_summarized:,}", style="cyan")
    output.append(" → ", style="dim")
    output.append(f"{result.tokens_summary:,}", style="green")
    output.append("  │  ", style="dim")
    output.append("Kept: ", style="dim")
    output.append(f"{result.messages_kept}", style="cyan")
    output.append(" messages unchanged", style="dim")
    if result.context_window > 0:
        output.append("  │  ", style="dim")
        output.append("Window: ", style="dim")
        output.append(f"{result.context_percent}%", style="cyan")
        output.append(" used", style="dim")

    return output


def render_compact_summary_panel(summary_text: str):
    """Render the compacted summary content as a Rich panel."""
    from rich.panel import Panel
    from rich.text import Text

    content = (summary_text or "").strip()
    body = Text(content or "(empty summary)", style="dim italic")
    return Panel(
        body,
        title="Context Compacted",
        border_style="#f59e0b",
        padding=(0, 1),
    )


def build_compact_summary_renderable(
    result: CompactResult,
) -> CompactSummaryRenderable | None:
    """Build the UI summary payload for a successful compact operation."""
    if result.status != "ok" or not result.summary_text.strip():
        return None
    return CompactSummaryRenderable(result.summary_text)


async def compact_conversation(
    graph_gateway: GraphGateway,
    thread_id: str,
    target: GraphTarget,
    *,
    workspace: Workspace,
    input_tokens_hint: int | None = None,
) -> CompactResult:
    """Compact the conversation by summarizing old messages.

    Reads the graph's checkpointed state, creates a temporary
    ``SummarizationMiddleware``, generates a summary, and writes
    the compacted state back through ``GraphGateway``.

    ``input_tokens_hint`` is the real LLM input token count from the last
    ``usage_metadata`` (includes system prompt + tool schemas).  When
    provided it is used for the display values in ``CompactResult`` so the
    panel stays in sync with the status bar; the internal compact logic
    (cutoff determination) still uses message-level token counts.

    Returns a structured ``CompactResult``.
    """
    from langchain_core.messages.utils import count_tokens_approximately

    try:
        state_values = await graph_gateway.get_state_values(target, thread_id)
    except Exception as exc:
        return CompactResult("error", f"Failed to read state: {exc}")

    messages = state_values.get("messages", [])
    if not messages:
        return CompactResult(
            "noop", "Nothing to compact — no messages in conversation."
        )

    from deepagents.middleware.summarization import (
        SummarizationEvent,
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    from ..agent import _ensure_chat_model, _get_default_backend

    try:
        model = _ensure_chat_model()
    except Exception as exc:
        return CompactResult(
            "error", f"Compaction requires a working model configuration: {exc}"
        )

    backend = _get_default_backend(
        workspace, work_dir=target.run_dir or target.workspace_dir
    )
    context_window = _resolve_context_window(model)

    defaults = compute_summarization_defaults(model)
    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        keep=defaults["keep"],
        trim_tokens_to_summarize=None,
    )

    # Rebuild effective message list accounting for prior compaction
    event = state_values.get("_summarization_event")
    effective = middleware._apply_event_to_messages(messages, event)
    effective_tokens = count_tokens_approximately(effective)

    # For display and threshold we prefer the real LLM input token count
    # (includes system prompt + tool schemas) so the panel stays in sync with
    # the status bar.  The internal compact logic (cutoff, partition, savings)
    # still uses effective_tokens (message-level) because compact only reduces
    # messages, not the constant system/tool overhead.
    display_tokens = (
        input_tokens_hint
        if input_tokens_hint is not None and input_tokens_hint > 0
        else effective_tokens
    )
    display_percent = _percent_used(display_tokens, context_window)

    if display_percent < _MANUAL_COMPACT_MIN_PERCENT:
        return CompactResult(
            "noop",
            "Conversation is below the manual compact threshold "
            f"({display_percent}% < {_MANUAL_COMPACT_MIN_PERCENT}%).",
            tokens_before=display_tokens,
            context_window=context_window,
            context_percent=display_percent,
        )

    cutoff = middleware._determine_cutoff_index(effective)
    if cutoff == 0:
        return CompactResult(
            "noop",
            f"Conversation (~{display_tokens:,} tokens) is within the retention budget.",
            tokens_before=display_tokens,
            context_window=context_window,
            context_percent=display_percent,
        )

    to_summarize, to_keep = middleware._partition_messages(effective, cutoff)

    tokens_summarized = count_tokens_approximately(to_summarize)
    tokens_kept = count_tokens_approximately(to_keep)
    tokens_before = tokens_summarized + tokens_kept

    # Skip if savings would be negligible — compacting ≤2 messages with
    # <2% of total tokens prevents the infinite 1-message-at-a-time loop
    # that occurs when the conversation sits just above the keep budget.
    _MIN_COMPACT_MESSAGES = 3
    _MIN_COMPACT_TOKEN_FRACTION = 0.02
    if (
        len(to_summarize) < _MIN_COMPACT_MESSAGES
        and tokens_summarized < tokens_before * _MIN_COMPACT_TOKEN_FRACTION
    ):
        return CompactResult(
            "noop",
            f"Nothing to compact — only {len(to_summarize)} message(s) "
            f"({tokens_summarized:,} tokens) would be summarized, "
            f"not worth the overhead.",
            tokens_before=display_tokens,
            context_window=context_window,
            context_percent=display_percent,
        )

    # Generate summary (LLM call)
    summary = await middleware._acreate_summary(to_summarize)

    # Reuse the persisted _summarization_session_id (or generate one) so
    # history keeps appending to a single file; re-persisted below.
    session_id = middleware._get_session_id(state_values)

    # Offload old messages to backend
    file_path: str | None = None
    try:
        file_path = await middleware._aoffload_to_backend(
            backend, to_summarize, session_id
        )
    except Exception:
        pass  # non-fatal — proceed without offloaded history

    from langchain_core.messages import HumanMessage

    summary_msg = cast(
        HumanMessage,
        middleware._build_new_messages_with_path(summary, file_path)[0],
    )

    # Compute token savings (message-level, used for pct calculation)
    tokens_summary = count_tokens_approximately([summary_msg])
    tokens_after = tokens_summary + tokens_kept
    pct = (
        round((tokens_before - tokens_after) / tokens_before * 100)
        if tokens_before > 0
        else 0
    )

    # Adjust display totals: preserve real overhead (system + tools) by
    # offsetting from input_tokens_hint rather than using bare message counts.
    msg_reduction = tokens_before - tokens_after  # how many message tokens saved
    display_before = display_tokens
    display_after = max(0, display_tokens - msg_reduction)
    display_after_percent = _percent_used(display_after, context_window)

    # Append savings note to summary message for model awareness
    savings_note = (
        f"\n\n{len(to_summarize)} messages were compacted "
        f"({tokens_summarized:,} → {tokens_summary:,} tokens). "
        f"Total context: {display_before:,} → {display_after:,} tokens "
        f"({pct}% decrease), "
        f"{len(to_keep)} messages unchanged."
    )
    summary_msg.content += savings_note

    state_cutoff = middleware._compute_state_cutoff(event, cutoff)

    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": summary_msg,
        "file_path": file_path,
        # Kept messages before this index lose reasoning bound to the old history.
        "created_at_index": len(messages),
    }

    await graph_gateway.update_state_values(
        target,
        thread_id,
        {"_summarization_event": new_event, "_summarization_session_id": session_id},
    )

    return CompactResult(
        "ok",
        f"Compacted {len(to_summarize)} messages "
        f"({display_before:,} → {display_after:,} tokens, {pct}% decrease)",
        messages_compacted=len(to_summarize),
        messages_kept=len(to_keep),
        tokens_before=display_before,
        tokens_after=display_after,
        tokens_summarized=tokens_summarized,
        tokens_summary=tokens_summary,
        pct_decrease=pct,
        context_window=context_window,
        context_percent=display_after_percent,
        summary_text=summary,
    )


# =============================================================================
# Config commands
# =============================================================================


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context):
    """Configuration management commands"""
    if ctx.invoked_subcommand is None:
        config_list()


@config_app.command("list")
def config_list():
    """List all configuration values"""
    from ..config import get_config_path, list_config
    from ..config.settings import display_config_value

    config_data = list_config()

    table = Table(title="Metasci Scribe Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    # Mask API keys, tokens, secrets and passwords
    def format_value(key: str, value: Any) -> str:
        if value == "":
            return "[dim](not set)[/dim]"
        return escape(display_config_value(key, value))

    for key, value in config_data.items():
        table.add_row(key, format_value(key, value))

    console.print(table)
    console.print(f"\n[dim]Config file: {get_config_path()}[/dim]")


@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Configuration key to get")):
    """Get a single configuration value"""
    from ..config import get_config_value
    from ..config.settings import display_config_value

    value = get_config_value(key)
    if value is None:
        console.print(f"[red]Unknown key: {key}[/red]")
        raise typer.Exit(1)

    # Mask API keys, tokens, secrets and passwords
    display_value = display_config_value(key, value)

    console.print(f"[cyan]{key}[/cyan]: {escape(display_value)}")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="New value"),
):
    """Set a single configuration value"""
    from ..config.settings import SettingsError, set_config_values
    from ..config.store import ConfigError

    try:
        _, warnings = set_config_values({key: value})
    except SettingsError as e:
        reason = e.errors.get(key, "invalid key or value")
        console.print(f"[red]Could not set {escape(key)}: {escape(reason)}[/red]")
        raise typer.Exit(1) from None
    except (ConfigError, OSError) as e:
        # OSError: for example a Windows replace that kept failing while
        # another program had the file open; nothing was written.
        console.print(f"[red]Could not set {escape(key)}: {escape(str(e))}[/red]")
        raise typer.Exit(1) from None
    console.print(f"[green]Set {escape(key)}[/green]")
    if key in warnings:
        console.print(f"[yellow]Note: {escape(warnings[key])}[/yellow]")


@config_app.command("reset")
def config_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """Reset configuration to defaults"""
    from ..config import get_config_path, reset_config

    config_path = get_config_path()

    if not config_path.exists():
        console.print("[yellow]No config file to reset.[/yellow]")
        return

    if not yes:
        confirm = typer.confirm("Reset configuration to defaults?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    from ..config.store import ConfigError

    try:
        reset_config()
    except ConfigError as e:
        console.print(f"[red]{escape(str(e))}[/red]")
        raise typer.Exit(1) from None
    console.print("[green]Configuration reset to defaults.[/green]")


@config_app.command("path")
def config_path():
    """Show the configuration file path"""
    from ..config import get_config_path

    path = get_config_path()
    exists = path.exists()
    status = "[green]exists[/green]" if exists else "[dim]not created yet[/dim]"
    console.print(f"{path} ({status})")


# =============================================================================
# MCP commands
# =============================================================================


@mcp_app.callback(invoke_without_command=True)
def mcp_callback(ctx: typer.Context):
    """MCP server management commands"""
    if ctx.invoked_subcommand is None:
        mcp_list()


@mcp_app.command("list")
def mcp_list():
    """List configured MCP servers"""
    _mcp_list_servers()


@mcp_app.command("config")
def mcp_config(
    name: str | None = typer.Argument(None, help="Server name (omit to show all)"),
):
    """Show detailed configuration for MCP servers

    \b
    Examples:
      metasci-scribe mcp config             # Show all servers in detail
      metasci-scribe mcp config filesystem  # Show one server
    """
    status = _show_mcp_config(name or "", show_blank_line=False)
    if status == "empty":
        console.print(
            "[dim]Add one with:[/dim] metasci-scribe mcp add <name> <transport> <command-or-url> [args...]"
        )
        return
    if status == "missing":
        raise typer.Exit(1)


@mcp_app.command("add")
def mcp_add(
    name: Annotated[str, typer.Argument(help="Server name")],
    target: Annotated[str, typer.Argument(help="Command (stdio) or URL (http/sse)")],
    args: Annotated[
        list[str] | None, typer.Argument(help="Extra args for stdio command")
    ] = None,
    transport: Annotated[
        str | None,
        typer.Option("--transport", "-T", help="Transport type (default: auto-detect)"),
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            "-t",
            help="Comma-separated tool allowlist (supports wildcards: *_exa, read_*)",
        ),
    ] = None,
    expose_to: Annotated[
        str | None,
        typer.Option("--expose-to", "-e", help="Comma-separated target agents"),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header as Key:Value (repeatable)"),
    ] = None,
    env: Annotated[
        list[str] | None,
        typer.Option("--env", help="Env var as KEY=VALUE for stdio (repeatable)"),
    ] = None,
    env_ref: Annotated[
        list[str] | None,
        typer.Option(
            "--env-ref", help="Env var name as ${NAME} runtime ref (repeatable)"
        ),
    ] = None,
):
    """Add an MCP server to user config

    \b
    Transport is auto-detected: URLs default to http, commands default to stdio.

    \b
    Examples:
      metasci-scribe mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking
      metasci-scribe mcp add docs-langchain https://docs.langchain.com/mcp
      metasci-scribe mcp add my-sse https://example.com/sse --transport sse -e research-agent
      metasci-scribe mcp add brave-search npx --env-ref BRAVE_API_KEY -- -y @modelcontextprotocol/server-brave-search
    """
    from ..mcp import build_mcp_add_kwargs

    # Merge env and env_ref into a single dict
    env_dict: dict[str, str] = {}
    for e in env or []:
        if "=" in e:
            k, v = e.split("=", 1)
            env_dict[k.strip()] = v.strip()
    for ref in env_ref or []:
        env_dict[ref] = "${" + ref + "}"

    kwargs = build_mcp_add_kwargs(
        name=name,
        target=target,
        extra_args=list(args) if args else None,
        transport=transport,
        tools=[t.strip() for t in tools.split(",") if t.strip()] if tools else None,
        expose_to=[a.strip() for a in expose_to.split(",") if a.strip()]
        if expose_to
        else None,
        headers={
            k.strip(): v.strip()
            for h in (header or [])
            for k, v in [h.split(":", 1)]
            if ":" in h
        }
        or None,
        env=env_dict or None,
    )

    if not _mcp_add_server_from_kwargs(kwargs, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("edit")
def mcp_edit(
    name: Annotated[str, typer.Argument(help="Server name to edit")],
    transport: Annotated[
        str | None, typer.Option("--transport", help="New transport type")
    ] = None,
    command: Annotated[
        str | None, typer.Option("--command", help="New command (stdio)")
    ] = None,
    url: Annotated[
        str | None, typer.Option("--url", help="New URL (http/sse/websocket)")
    ] = None,
    tools: Annotated[
        str | None,
        typer.Option(
            "--tools",
            "-t",
            help="Comma-separated tool allowlist, supports wildcards ('none' to clear)",
        ),
    ] = None,
    expose_to: Annotated[
        str | None,
        typer.Option(
            "--expose-to",
            "-e",
            help="Comma-separated target agents ('none' to clear)",
        ),
    ] = None,
    header: Annotated[
        list[str] | None,
        typer.Option("--header", "-H", help="HTTP header as Key:Value (repeatable)"),
    ] = None,
    env: Annotated[
        list[str] | None,
        typer.Option("--env", help="Env var as KEY=VALUE for stdio (repeatable)"),
    ] = None,
):
    """Edit an existing MCP server in user config

    \b
    Examples:
      metasci-scribe mcp edit filesystem --expose-to main,research-agent
      metasci-scribe mcp edit filesystem -t read_file,write_file
      metasci-scribe mcp edit my-api --url http://new-host:9090/mcp
      metasci-scribe mcp edit my-api --tools none
    """
    from ..mcp import build_mcp_edit_fields

    fields = build_mcp_edit_fields(
        transport=transport,
        command=command,
        url=url,
        tools=tools,
        expose_to=expose_to,
        headers=header,
        env=env,
    )

    if not _mcp_edit_server_fields(name, fields, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Server name to remove"),
):
    """Remove an MCP server from user config"""
    if not _mcp_remove_server(name, show_reload_hint=False):
        raise typer.Exit(1)


@mcp_app.command("install")
def mcp_install(
    source: Annotated[
        str | None, typer.Argument(help="Server name or tag filter")
    ] = None,
):
    """Browse and install MCP servers from the registry and marketplace

    \b
    Examples:
      metasci-scribe mcp install                       # Interactive browser
      metasci-scribe mcp install search                # Filter by 'search' tag
      metasci-scribe mcp install sequential-thinking   # Install by name
    """
    from .mcp_install_cmd import _cmd_install_mcp

    _cmd_install_mcp(source or "")


# =============================================================================
# Sessions commands — read-only diagnostics for ~/.metasci-scribe/sessions.db
# =============================================================================


def _format_bytes(n: int) -> str:
    """Render a byte count as a human-readable string (KB / MB / GB)."""
    if n < 1024:
        return f"{n} B"
    units = ["KB", "MB", "GB", "TB"]
    size = float(n) / 1024.0
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


@sessions_app.callback(invoke_without_command=True)
def sessions_callback(ctx: typer.Context):
    """Inspect and manage the sessions DB.

    Running ``metasci-scribe sessions`` with no subcommand defaults to ``stats``
    so the bare command is informative rather than silent.
    """
    if ctx.invoked_subcommand is None:
        sessions_stats(ctx)


@sessions_app.command("stats")
def sessions_stats(ctx: typer.Context):
    """Show DB size, thread count, total checkpoints, top heaviest threads."""
    from ..sessions import db_stats

    runtime = _get_cli_async_runtime(ctx)
    stats = runtime.run_sync(db_stats)

    table = Table(title="Metasci Scribe sessions DB", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value")
    table.add_row("Path", stats["db_path"])
    table.add_row("Size", _format_bytes(int(stats["size_bytes"])))
    table.add_row("Threads", str(stats["thread_count"]))
    table.add_row("Checkpoints", str(stats["checkpoint_count"]))
    table.add_row("Writes", str(stats["write_count"]))
    console.print(table)

    if stats["top_threads"]:
        top = Table(title="Heaviest threads (checkpoints per thread)")
        top.add_column("thread_id", style="yellow")
        top.add_column("checkpoints", justify="right")
        for row in stats["top_threads"]:
            top.add_row(str(row["thread_id"]), str(row["count"]))
        console.print(top)


# =============================================================================
# Main callback (default behavior)
# =============================================================================


def _version_callback(value: bool):
    if value:
        typer.echo(f"Metasci Scribe {_pkg_version('metasci-scribe')}")
        raise typer.Exit()


# 2 is Click's usage-error code; 1 stays "the run failed".
RUBRIC_UNMET_EXIT_CODE = 3


def _resolve_rubric_option(value: str | None, *, prompt: str | None) -> str | None:
    """Return the text given by ``--rubric``, reading ``@path`` from disk."""
    if value is None:
        return None
    if not prompt:
        raise typer.BadParameter("--rubric requires -p/--prompt (single-shot mode)")
    from ..rubric_text import RubricTextError, read_rubric_text

    try:
        return read_rubric_text(value)
    except RubricTextError as exc:
        raise typer.BadParameter(f"--rubric {exc}") from exc


def _rubric_exit_code(rubric: str | None, result: str | None) -> int | None:
    """Exit code for a single-shot run that finished; ``None`` keeps the default."""
    if rubric is None or result == "satisfied":
        return None
    return RUBRIC_UNMET_EXIT_CODE


def _rubric_exit_notice(result: str | None, exit_code: int) -> str:
    """The line a graded single-shot run prints before it exits with *exit_code*."""
    if result is None:
        # No verdict at all: a cancelled approval or question, the round limit.
        return (
            "The run stopped before the acceptance criteria were checked "
            f"(exit code {exit_code})."
        )
    if result == "failed":
        return (
            f"The acceptance criteria could not be evaluated (exit code {exit_code})."
        )
    if result == "grader_error":
        return (
            "The acceptance criteria could not be checked: the grader failed "
            f"(exit code {exit_code})."
        )
    return f"Acceptance criteria not met (exit code {exit_code})."


def _single_shot_agent_config(config: Any, rubric: str | None) -> Any:
    """The config a single-shot run builds its agent from.

    The grader runs when the agent stops, so a graded run keeps sub-agents in-turn. The
    agent is built from a copy; the background server is compared with the user's config.
    """
    if rubric is None:
        return config
    agent_config = copy.copy(config)
    agent_config.enable_async_subagents = False
    return agent_config


def _resolve_stream_json_auto_mode(auto_mode: bool | None, output_format: str) -> bool:
    """Resolve the effective ``--auto-mode`` for a single-shot run.

    stream-json is headless, so auto-mode defaults on there when the caller did
    not pass the flag (``None``); an explicit ``--auto-mode`` / ``--no-auto-mode``
    always wins, and non-stream-json runs keep the historical off-by-default.
    """
    if auto_mode is not None:
        return auto_mode
    return output_format == "stream-json"


@app.callback(invoke_without_command=True)
def _main_callback(
    ctx: typer.Context,
    version: bool | None = typer.Option(
        None,
        "-V",
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    prompt: str | None = typer.Option(
        None, "-p", "--prompt", help="Query to execute (single-shot mode)"
    ),
    thread_id: str | None = typer.Option(
        None,
        "--resume",
        "--thread-id",
        help="Thread ID (or prefix) to resume a previous session.",
    ),
    workdir: str | None = typer.Option(
        None, "--workdir", help="Override workspace directory for this session"
    ),
    use_cwd: bool = typer.Option(
        False, "--use-cwd", help="Use current working directory as workspace"
    ),
    no_thinking: bool = typer.Option(
        False, "--no-thinking", help="Disable thinking display"
    ),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Skip tool approval prompts for HITL actions",
    ),
    auto_mode: bool | None = typer.Option(
        None,
        "--auto-mode/--no-auto-mode",
        help="Run unattended: skip ask_user and tool approval prompts "
        "(default: on when --output-format stream-json)",
    ),
    ask_user: bool = typer.Option(
        False,
        "--ask-user",
        help="Enable agent to ask clarifying questions about your research preferences",
    ),
    output_format: str | None = typer.Option(
        None,
        "--output-format",
        help=(
            "Output format for single-shot (-p) mode: 'text' (default) or "
            "'stream-json' (line-delimited JSON events to stdout)."
        ),
    ),
    # Annotated, so a direct call without these arguments gets a plain default
    # rather than Typer's OptionInfo default, which is truthy.
    rubric: Annotated[
        str | None,
        typer.Option(
            "--rubric",
            metavar="TEXT|@PATH",
            help=(
                "Acceptance criteria for a single-shot (-p) run. The work is "
                "graded each time the agent stops and sent back with the gaps, "
                "up to 3 passes. Literal text, or '@path' to read a file. "
                "Sub-agents run in-turn for this run. Exit code 3 when the run "
                "ends without a satisfied verdict."
            ),
        ),
    ] = None,
):
    """Metasci Scribe: a terminal agent that writes a scientific paper with you."""
    # If a subcommand was invoked, don't run the default behavior
    if ctx.invoked_subcommand is not None:
        return

    async_runtime = _get_cli_async_runtime(ctx)

    # Load and apply configuration
    from ..config import apply_config_to_env, get_effective_config

    # Resolve the output format first. In stream-json mode stdout must carry
    # only JSONL, so establish the mode and redirect the console to stderr
    # BEFORE anything below (validation, server start) can print a
    # human-readable line to stdout.
    effective_output_format = (output_format or "text").lower()
    if effective_output_format not in ("text", "stream-json"):
        raise typer.BadParameter("--output-format must be 'text' or 'stream-json'")
    if effective_output_format == "stream-json":
        if not prompt:
            raise typer.BadParameter(
                "--output-format stream-json requires -p/--prompt (single-shot mode)"
            )
        from ..stream.json_sink import redirect_console_to_stderr

        redirect_console_to_stderr()

    rubric_text = _resolve_rubric_option(rubric, prompt=prompt)

    # stream-json is headless, so auto-mode defaults on (auto-handle approval and
    # ask_user gates) unless the caller explicitly passed --no-auto-mode. Without
    # it the run would stall at the first gate and end without doing the work;
    # --no-auto-mode is the (experimental) opt-in to receiving interrupt/ask_user
    # events and driving resume yourself.
    effective_auto_mode = _resolve_stream_json_auto_mode(
        auto_mode, effective_output_format
    )
    if effective_output_format == "stream-json" and auto_mode is False:
        console.print(
            "[yellow]--no-auto-mode with stream-json is experimental: an "
            "interrupt/ask_user event ends the run early and is not yet "
            "resumable.[/yellow]"
        )

    # Build CLI overrides dict
    cli_overrides = {}
    if workdir:
        cli_overrides["default_workdir"] = workdir
    if no_thinking:
        cli_overrides["show_thinking"] = False
    if auto_approve:
        cli_overrides["auto_approve"] = True
    if effective_auto_mode:
        cli_overrides["auto_mode"] = True
        cli_overrides["auto_approve"] = True
        cli_overrides["enable_ask_user"] = False
    else:
        # An explicit --no-auto-mode (auto_mode is False, not None) must win over
        # a config file that enables auto-mode; without this the resolved-off
        # value writes nothing and silently falls back to the config default.
        if auto_mode is False:
            cli_overrides["auto_mode"] = False
        if ask_user:
            cli_overrides["enable_ask_user"] = True

    config = get_effective_config(cli_overrides)
    apply_config_to_env(config)

    if rubric_text is not None:
        from ..config import GatewaySurface, resolve_gateway_backend

        single_shot_backend = resolve_gateway_backend(
            config, GatewaySurface.SINGLE_SHOT
        )
        if single_shot_backend == "langgraph_server":
            raise typer.BadParameter(
                "--rubric needs the local gateway for single-shot runs: on "
                "langgraph_server the agent runs in the server process, which "
                "still dispatches background sub-agents the grader cannot wait for"
            )

    show_thinking = config.show_thinking if not no_thinking else False

    # Validate mutually exclusive options
    if workdir and use_cwd:
        raise typer.BadParameter("Use either --workdir or --use-cwd, not both.")

    # Resolve the session's workspace, which is also the paper folder ``run``
    # writes in. Priority: --use-cwd / --workdir > default_workdir > cwd.
    if use_cwd:
        workspace_root = start_workspace_path()
    elif workdir:
        workspace_root = start_workspace_path(workdir)
    else:
        workspace_root = start_workspace_path(default_workdir=config.default_workdir)
    # Before the path is resolved (a mapped drive resolves to a UNC path) or
    # created: network folders are not supported.
    with _local_folders_only():
        project.require_local_path(workspace_root)
    if workdir and not use_cwd:
        os.makedirs(workspace_root, exist_ok=True)
    workspace = Workspace(workspace_root)
    with _local_folders_only():
        # Resolved now: a junction or link on a local drive may lead to a share.
        project.require_local_path(workspace.root)
    try:
        require_workspace_root(workspace.root)
    except WorkspaceRootError as exc:
        console.print(f"[red]\u2717 {escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc

    # The project .env was merged into os.environ by get_effective_config().
    reload_env_dirs()
    # Ensure memory and skills subdirs exist in workspace
    ensure_dirs()

    dirs = SessionDirs(workspace)

    # Resolve the gateway backend for whichever surface this callback launches:
    # single-shot when a prompt is given, else the interactive CLI (the
    # same value each inner entry re-resolves for its own factory call). Drives
    # both the server start and the single-shot factory below.
    from ..config import GatewaySurface, resolve_gateway_backend

    if prompt:
        _gateway_surface = GatewaySurface.SINGLE_SHOT
    else:
        _gateway_surface = GatewaySurface.INTERACTIVE
    gateway_backend = resolve_gateway_backend(config, _gateway_surface)

    # Auto-start langgraph dev (after workspace resolution, so deployed
    # async sub-agents inherit the CLI's workspace via METASCI_SCRIBE_WORKSPACE_DIR).
    # A one-shot resume starts it for the thread's own folders instead, unless
    # the server gateway backend needs a running server to be built.
    prespawn = not (prompt and thread_id) or gateway_backend == "langgraph_server"
    if prespawn:
        _ensure_async_subagent_server(config, dirs=dirs, backend=gateway_backend)

    if prompt:
        # Single-shot mode: wrap in persistent checkpointer
        import asyncio

        from ..gateway import create_runtime_gateways_for_config
        from ..sessions import get_checkpointer
        from ..stream.json_sink import stream_json
        from .interactive import cmd_run
        from .resume_hint import print_resume_hint

        runtime_gateways = create_runtime_gateways_for_config(
            config, backend=gateway_backend
        )
        graph_gateway = runtime_gateways.graph_gateway

        async def _single_shot():
            async with get_checkpointer() as checkpointer:
                # Resolve resume target first so a bad --resume/--thread-id
                # exits before the slow _load_agent() provider setup.
                if thread_id:
                    resolution = await graph_gateway.resolve_thread(thread_id)
                    if resolution.thread_id:
                        tid = resolution.thread_id
                        session_dirs = await _restore_thread_dirs(
                            tid,
                            dirs=dirs,
                            served=dirs if prespawn else None,
                            graph_gateway=graph_gateway,
                            config=config,
                            backend=gateway_backend,
                        )
                    elif resolution.matches:
                        console.print(
                            f"[yellow]Ambiguous thread ID '{escape(thread_id)}'. Matches:[/yellow]"
                        )
                        for s in resolution.matches:
                            console.print(f"  [cyan]{escape(s)}[/cyan]")
                        raise typer.Exit(1)
                    else:
                        console.print(
                            f"[red]Thread '{escape(thread_id)}' not found.[/red]"
                        )
                        raise typer.Exit(1)
                else:
                    tid = await graph_gateway.create_thread()
                    session_dirs = dirs
                console.print("[dim]Loading agent...[/dim]")
                try:
                    agent = await asyncio.to_thread(
                        _load_agent,
                        work_dir=str(session_dirs.work_dir),
                        workspace=session_dirs.workspace,
                        checkpointer=checkpointer,
                        config=_single_shot_agent_config(config, rubric_text),
                        runtime=async_runtime,
                    )
                except ModelNotConfigured as exc:
                    # A first run before any sign-in or key: the one message,
                    # not a traceback.
                    console.print(f"[red]✗ {escape(str(exc))}[/red]")
                    raise typer.Exit(1) from exc
                rubric_result: str | None = None
                try:
                    if effective_output_format == "stream-json":
                        # Headless JSONL path: drive the sink through the gateway
                        # directly. We are already inside the async single-shot
                        # loop, so this is a plain await — no nested-loop juggling,
                        # and the gateway seam keeps it execution-backend agnostic.
                        request = RunRequest(
                            message=prompt,
                            thread_id=tid,
                            metadata=build_metadata(session_dirs, config.model),
                            target=GraphTarget(
                                local_graph=agent, **session_dirs.metadata()
                            ),
                            rubric=rubric_text,
                        )

                        def _track_verdict(event: dict) -> None:
                            nonlocal rubric_result
                            if event.get("type") == "rubric_evaluation":
                                result = event.get("result")
                                rubric_result = (
                                    result if isinstance(result, str) else None
                                )

                        try:
                            await stream_json(
                                graph_gateway, request, on_event=_track_verdict
                            )
                        except Exception as exc:
                            # stream_events already emitted a terminal `error`
                            # event onto the JSON stream before re-raising; exit
                            # cleanly so the stream ends with that event instead
                            # of a raw traceback.
                            raise typer.Exit(1) from exc
                        finally:
                            # Let post-run memory workers persist before exit,
                            # matching the text path (cmd_run does this itself).
                            _wait_for_memory_workers()
                    else:
                        stream_worker = asyncio.create_task(
                            asyncio.to_thread(
                                cmd_run,
                                agent,
                                prompt,
                                thread_id=tid,
                                show_thinking=show_thinking,
                                dirs=session_dirs,
                                model=config.model,
                                runtime_gateways=runtime_gateways,
                                async_runtime=async_runtime,
                                rubric=rubric_text,
                            )
                        )
                        try:
                            rubric_result = await asyncio.shield(stream_worker)
                        except asyncio.CancelledError:
                            from ..stream.display import request_stream_cancel
                            from .stream_runtime import settle_cancelled_worker

                            await settle_cancelled_worker(
                                stream_worker,
                                on_cancel=request_stream_cancel,
                            )
                            raise
                    exit_code = _rubric_exit_code(rubric_text, rubric_result)
                    if exit_code is not None:
                        notice = _rubric_exit_notice(rubric_result, exit_code)
                        console.print(f"[yellow]{notice}[/yellow]")
                        raise typer.Exit(exit_code)
                finally:
                    # Model failures can bypass middleware ``after_agent``
                    # hooks. Close any remaining QuickJS workers while this
                    # event loop is still available; their synchronous GC
                    # fallback can deadlock during interpreter shutdown.
                    from ..middleware.code_interpreter import (
                        aclose_code_interpreters,
                    )

                    await aclose_code_interpreters()
                    try:
                        print_resume_hint(tid, console=console)
                    except Exception:
                        pass

        async_runtime.run_sync(_single_shot)
    else:
        from .interactive import cmd_interactive

        # Interactive mode (default) — checkpointer managed inside cmd_interactive
        cmd_interactive(
            show_thinking=show_thinking,
            dirs=dirs,
            model=config.model,
            provider=config.provider,
            thread_id=thread_id,
            config=config,
            async_runtime=async_runtime,
        )


# 130 is the shell's code for a run ended by Ctrl+C; 0, 2, 3 and 4 are RunExit.
RUN_INTERRUPTED_EXIT_CODE = 130


@app.command("run")
def run_command(
    workdir: Annotated[
        str | None,
        typer.Option(
            "--workdir",
            help="The paper folder (default: the current folder)",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", help="Model for this run (default: the configured one)"
        ),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", help="Provider for this run (default: the configured one)"
        ),
    ] = None,
    section: Annotated[
        str | None,
        typer.Option(
            "--section",
            help="Write only this section of outline.md, for example Results",
        ),
    ] = None,
):
    """Write the draft unattended, following outline.md, then stop.

    Each section goes to its own file in Manuscript/, and manuscript.md is
    joined from them. With --section, only that section is written.

    Exit codes: 0 every paragraph (of the section) is marked done, 2 the run failed or could not
    start, 3 the ChatGPT plan usage limit was reached, 4 a paragraph needs your
    decision. Run it again to continue where it stopped.
    """
    import signal

    from ..config import apply_config_to_env, get_effective_config
    from ..run import run_project
    from ..run_agent import real_invoke

    if hasattr(signal, "SIGBREAK"):
        # ``/run`` stops its child with Ctrl+Break on Windows; treat it as Ctrl+C.
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    root = start_workspace_path(workdir)
    config = get_effective_config(
        {
            "auto_mode": True,
            "auto_approve": True,
            "enable_ask_user": False,
            "model": model,
            "provider": provider,
        }
    )
    apply_config_to_env(config)
    reload_env_dirs()
    ensure_dirs()
    typer.echo(
        f"Metasci Scribe run: {root}" + (f" (section {section})" if section else "")
    )
    typer.echo(
        f"Model: {config.model} (provider {config.provider}"
        + (
            ", ChatGPT plan"
            if config.provider in (None, "openai")
            and config.openai_auth_mode == "chatgpt_plan"
            else ""
        )
        + ")"
    )
    try:
        with real_invoke(root, config=config) as invoke:
            result = run_project(root, invoke, section=section)
    except KeyboardInterrupt:
        typer.echo(
            "\nInterrupted. Progress is saved; run `metasci-scribe run` again "
            "to resume."
        )
        raise typer.Exit(RUN_INTERRUPTED_EXIT_CODE) from None
    typer.echo(result.message)
    if result.total:
        typer.echo(f"Paragraphs marked done: {result.done}/{result.total}")
    if result.exit:
        raise typer.Exit(int(result.exit))


# Upstream's logger and its two records for a grader attempt that raised.
_GRADER_LOGGER = "deepagents.middleware.rubric"
_GRADER_FAILURE_RECORDS = ("grader failed", "coverage retry raised")


def _grader_failure_on_one_line(record: logging.LogRecord) -> logging.LogRecord:
    """A copy of upstream's grader-failure record with the reason instead of the traceback.

    The verdict already says why; the record itself keeps its traceback for other handlers.
    """
    if record.name != _GRADER_LOGGER or not record.exc_info:
        return record
    message = record.getMessage()
    if not any(marker in message for marker in _GRADER_FAILURE_RECORDS):
        return record
    exc = record.exc_info[1]
    short = copy.copy(record)
    short.msg = f"{message}: {type(exc).__name__}: {exc}"
    short.args = None
    short.exc_info = None
    short.exc_text = None
    return short


def _configure_logging():
    """Configure logging with warning symbols for better visibility."""
    from rich.logging import RichHandler

    from ..config import get_effective_config

    def _resolve_log_level() -> int:
        """Resolve the root log level from config/env with a safe fallback."""
        try:
            raw = (get_effective_config().log_level or "").strip().upper()
        except Exception:
            raw = ""
        if raw == "WARN":
            raw = "WARNING"
        return getattr(logging, raw, logging.WARNING)

    resolved_level = _resolve_log_level()
    verbose_logging = resolved_level <= logging.DEBUG

    class DimWarningHandler(RichHandler):
        """Custom handler that renders warnings in dim style."""

        def emit(self, record: logging.LogRecord) -> None:
            if not verbose_logging:
                record = _grader_failure_on_one_line(record)
            if record.levelno == logging.WARNING:
                # Use Rich console to print dim warning
                msg = record.getMessage()
                console.print(
                    f"[dim yellow]\u26a0\ufe0f  Warning:[/dim yellow] [dim]{escape(msg)}[/dim]"
                )
            else:
                super().emit(record)

    # Configure root logger to use our handler for WARNING and above
    handler = DimWarningHandler(
        console=console,
        show_time=verbose_logging,
        show_path=verbose_logging,
        show_level=verbose_logging,
    )
    handler.setLevel(resolved_level)

    # Apply to root logger (catches all loggers including deepagents)
    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicate output
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    root_logger.addHandler(handler)
    root_logger.setLevel(resolved_level)
