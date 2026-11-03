"""Shared constants and utilities for CLI modules."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..paths import SessionDirs


def _agent_name() -> str:
    # Deferred import: ``sessions`` pulls in langgraph/aiosqlite (~300 ms)
    # and is only needed when ``build_metadata`` is actually called.
    from ..sessions import AGENT_NAME

    return AGENT_NAME


# One is shown under the name at start-up. ``{newline_key}`` is the key that
# inserts a line break on this platform. Each fits an 80-column row after the
# mark and "Tip: ".
STARTUP_TIPS = [
    "Type /init to set up this folder for a new paper.",
    "Type /run to write the whole draft from outline.md, unattended.",
    "Type /rubric to set criteria the agent's work is checked against.",
    "Type @ and a file name, like @outline.md, to add the file.",
    "Press {newline_key} for a new line; Enter sends the message.",
    "Type /help to see every command.",
]

# The start-screen mark: a small "M", two lines high.
LOGO_LINES = (
    "█▀▄▀█",
    "█ ▀ █",
)

# One colour per mark line: royal blue over sky blue.
LOGO_GRADIENT = ["#1e88e5", "#64b5f6"]


def build_metadata(dirs: SessionDirs, model: str | None) -> dict:
    """Build metadata dict for LangGraph checkpoint persistence.

    ``workspace_dir`` is the workspace root; ``run_dir`` is added only for a
    thread stored with a run folder.
    """
    return {
        "agent_name": _agent_name(),
        "updated_at": datetime.now(UTC).isoformat(),
        **dirs.metadata(),
        "model": model or "",
    }
