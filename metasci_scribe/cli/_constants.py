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


WELCOME_SLOGANS = [
    "Your outline, your sources, your paper.",
    "One paragraph at a time.",
    "Write the outline. The draft follows.",
    "Put your PDFs in Knowledge/ and ask for a paragraph.",
    "Stuck on a paragraph? Start there.",
    "Start with /init. Finish with /run.",
]

# ASCII art logo — "METASCI" over "SCRIBE", 62 columns wide so it fits an
# 80-column terminal.
LOGO_LINES = (
    r" ███╗   ███╗ ███████╗ ████████╗  █████╗  ███████╗  ██████╗ ██╗",
    r" ████╗ ████║ ██╔════╝ ╚══██╔══╝ ██╔══██╗ ██╔════╝ ██╔════╝ ██║",
    r" ██╔████╔██║ █████╗      ██║    ███████║ ███████╗ ██║      ██║",
    r" ██║╚██╔╝██║ ██╔══╝      ██║    ██╔══██║ ╚════██║ ██║      ██║",
    r" ██║ ╚═╝ ██║ ███████╗    ██║    ██║  ██║ ███████║ ╚██████╗ ██║",
    r" ╚═╝     ╚═╝ ╚══════╝    ╚═╝    ╚═╝  ╚═╝ ╚══════╝  ╚═════╝ ╚═╝",
    r" ███████╗  ██████╗ ██████╗  ██╗ ██████╗  ███████╗",
    r" ██╔════╝ ██╔════╝ ██╔══██╗ ██║ ██╔══██╗ ██╔════╝",
    r" ███████╗ ██║      ██████╔╝ ██║ ██████╔╝ █████╗",
    r" ╚════██║ ██║      ██╔══██╗ ██║ ██╔══██╗ ██╔══╝",
    r" ███████║ ╚██████╗ ██║  ██║ ██║ ██████╔╝ ███████╗",
    r" ╚══════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═╝ ╚═════╝  ╚══════╝",
)

# Blue gradient, one colour per logo line: deep navy -> royal blue -> sky blue
LOGO_GRADIENT = [
    "#1a237e",
    "#1a3a9e",
    "#1565c0",
    "#1976d2",
    "#1e88e5",
    "#2196f3",
    "#42a5f5",
    "#55aef7",
    "#64b5f6",
    "#7abff7",
    "#90caf9",
    "#a6d4fa",
]


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
