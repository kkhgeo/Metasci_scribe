"""The start screen: a two-line mark with the name, the version and a tip."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from metasci_scribe.cli import interactive
from metasci_scribe.cli._constants import LOGO_LINES, STARTUP_TIPS


def _print_banner(monkeypatch, width: int, tip: str | None = None) -> list[str]:
    out = StringIO()
    monkeypatch.setattr(
        interactive,
        "console",
        Console(file=out, width=width, color_system=None, legacy_windows=False),
    )
    monkeypatch.setattr(interactive, "_app_version", lambda: "0.1.0.dev0")
    interactive.print_banner(tip)
    return out.getvalue().splitlines()


def test_the_banner_is_two_lines(monkeypatch):
    lines = _print_banner(monkeypatch, 80, tip="Type /help to see every command.")

    assert lines == [
        f" {LOGO_LINES[0]}  metasci scribe 0.1.0.dev0   "
        "/ commands · @ files · "
        + ("Option+Enter" if interactive.sys.platform == "darwin" else "Ctrl+J")
        + " newline",
        f" {LOGO_LINES[1]}  Tip: Type /help to see every command.",
    ]


def test_a_narrow_terminal_leaves_out_the_key_hint(monkeypatch):
    lines = _print_banner(monkeypatch, 40)

    assert len(lines) == 2
    assert lines[0] == f" {LOGO_LINES[0]}  metasci scribe 0.1.0.dev0"
    assert all(len(line) <= 40 for line in lines)


def test_a_random_tip_is_shown_by_default(monkeypatch):
    shown = _print_banner(monkeypatch, 120)[1]

    assert any(
        shown.endswith(tip.format(newline_key=key))
        for tip in STARTUP_TIPS
        for key in ("Ctrl+J", "Option+Enter")
    )


@pytest.mark.parametrize("tip", STARTUP_TIPS)
def test_every_tip_fits_an_80_column_terminal(tip):
    line = f" {LOGO_LINES[1]}  Tip: {tip.format(newline_key='Option+Enter')}"

    assert len(line) <= 80


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
