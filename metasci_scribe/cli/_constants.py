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

# ASCII art logo — "METASCI", five lines high, 55 columns wide. A narrower
# terminal gets the name as plain text instead.
LOGO_LINES = (
    r" ███    ███ ███████ ████████  █████  ███████  ██████ ██",
    r" ████  ████ ██         ██    ██   ██ ██      ██      ██",
    r" ██ ████ ██ █████      ██    ███████ ███████ ██      ██",
    r" ██  ██  ██ ██         ██    ██   ██      ██ ██      ██",
    r" ██      ██ ███████    ██    ██   ██ ███████  ██████ ██",
)

# Teal gradient: deep teal -> sea green -> light aqua
LOGO_GRADIENT = ["#00796b", "#00897b", "#26a69a", "#4db6ac", "#80cbc4"]


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
