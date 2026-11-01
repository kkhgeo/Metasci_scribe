"""Agent loading and workspace helpers."""

import os
from typing import TYPE_CHECKING

from ..paths import Workspace

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ..runtime import AsyncRuntime


def _shorten_path(path: str) -> str:
    """Shorten absolute path to relative path from current directory."""
    if not path:
        return path
    try:
        cwd = os.getcwd()
        if path.startswith(cwd):
            rel = path[len(cwd) :].lstrip(os.sep)
            return (
                os.path.join(os.path.basename(cwd), rel)
                if rel
                else os.path.basename(cwd)
            )
        return path
    except Exception:
        return path


def _load_agent(
    work_dir: str | None = None,
    checkpointer=None,
    config=None,
    chat_model=None,
    *,
    workspace: Workspace,
    on_mcp_progress=None,
    events=None,
    runtime: "AsyncRuntime | None" = None,
) -> "CompiledStateGraph":
    """Load the CLI agent with optional persistent checkpointer.

    Args:
        work_dir: The folder the agent works in (defaults to the
            workspace root).
        workspace: The session's workspace.
        checkpointer: Optional LangGraph checkpointer (e.g. ``AsyncSqliteSaver``).
            Falls back to ``InMemorySaver`` when ``None``.
        config: Optional pre-loaded ``ScribeConfig``.  Forwarded to
            ``create_cli_agent`` to avoid double config loading.
        chat_model: Optional pre-built chat model.  Forwarded to
            ``create_cli_agent``; combined with an explicit ``config`` it
            selects the pure (no module-global write) build path.
        on_mcp_progress: Optional per-server MCP progress callback.
            Signature ``(event, server_name, detail) -> None``.
        runtime: Optional application-scoped runtime used for MCP discovery.
    """
    from ..agent import create_cli_agent

    return create_cli_agent(
        work_dir=work_dir,
        workspace=workspace,
        checkpointer=checkpointer,
        config=config,
        chat_model=chat_model,
        on_mcp_progress=on_mcp_progress,
        events=events,
        runtime=runtime,
    )
