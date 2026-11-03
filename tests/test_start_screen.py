"""The start screen: the METASCI logo, the model, the folder and the key hint."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from metasci_scribe.cli import interactive
from metasci_scribe.cli._constants import LOGO_GRADIENT, LOGO_LINES


def _print_banner(monkeypatch, width: int) -> list[str]:
    out = StringIO()
    monkeypatch.setattr(
        interactive,
        "console",
        Console(file=out, width=width, color_system=None, legacy_windows=False),
    )
    interactive.print_banner("/papers/my-paper", "gpt-5.5", "openai")
    return out.getvalue().splitlines()


def test_the_logo_is_five_lines_that_fit_80_columns():
    assert len(LOGO_LINES) == len(LOGO_GRADIENT) == 5
    assert max(len(line) for line in LOGO_LINES) <= 80


def test_an_80_column_terminal_shows_the_logo_and_the_session(monkeypatch):
    lines = _print_banner(monkeypatch, 80)

    assert lines[:5] == list(LOGO_LINES)
    assert lines[5] == "  Model: gpt-5.5  Provider: openai"
    assert lines[6] == "  Directory: /papers/my-paper"
    assert "Type / for commands" in lines[7]


def test_a_narrow_terminal_shows_the_name_instead_of_the_logo(monkeypatch):
    lines = _print_banner(monkeypatch, 50)

    assert lines[0] == "  Metasci"
    assert lines[1] == "  Model: gpt-5.5  Provider: openai"


@pytest.mark.parametrize(
    ("provider", "auth_mode", "expected"),
    [
        ("openai", "chatgpt_plan", True),
        (None, "chatgpt_plan", True),
        ("openai", "api_key", False),
        ("anthropic", "chatgpt_plan", False),
    ],
)
def test_the_plan_route_is_read_from_the_config(provider, auth_mode, expected):
    cfg = SimpleNamespace(provider=provider, openai_auth_mode=auth_mode)

    assert interactive._uses_chatgpt_plan(cfg) is expected
