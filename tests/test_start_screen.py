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


def test_the_input_sits_between_two_rules():
    from prompt_toolkit import PromptSession
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.layout import Window
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.shortcuts import CompleteStyle

    session = PromptSession(
        input=DummyInput(),
        output=DummyOutput(),
        completer=interactive.SlashCommandCompleter(),
        complete_style=CompleteStyle.COLUMN,
        complete_while_typing=True,
    )
    interactive._add_input_rule(session)

    main_input = session.layout.container.children[0].alternative_content
    rule, room = main_input.content.children[-2:]
    assert isinstance(rule.content, Window)
    assert rule.content.char == "─"
    # The menu room moved from the input window to below the rule.
    assert session.reserve_space_for_menu == 0
    assert room.content.height().min == 8
