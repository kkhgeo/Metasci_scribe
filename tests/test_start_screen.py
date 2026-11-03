"""The start screen: the one-row logo, the model, the folder and the key hint."""

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


def test_the_logo_is_one_row_of_six_lines():
    assert len(LOGO_LINES) == len(LOGO_GRADIENT) == 6
    assert max(len(line) for line in LOGO_LINES) <= 120


def test_a_wide_terminal_shows_the_logo_and_the_session(monkeypatch):
    lines = _print_banner(monkeypatch, 120)

    assert lines[:6] == list(LOGO_LINES)
    assert lines[6] == "  Model: gpt-5.5  Provider: openai"
    assert lines[7] == "  Directory: /papers/my-paper"
    assert "Type / for commands" in lines[8]


def test_a_narrow_terminal_shows_the_name_instead_of_the_logo(monkeypatch):
    lines = _print_banner(monkeypatch, 80)

    assert lines[0] == "  Metasci Scribe"
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
