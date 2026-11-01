"""Which calls of a TUI turn carry the session "approve all" grant to the graph."""

from __future__ import annotations

import pytest

pytest.importorskip("textual")

from tests.fakes.common import no_config_rule as _no_config_rule
from tests.fakes.tui import (
    APPROVE,
    FINISHED_REPLY,
    GRANTING,
    run_turn,
)
from tests.fakes.tui import approval_interrupt as _interrupt
from tests.fakes.tui import widget_answers as _widget_answers


def _keyed(extras: list[dict]) -> list[bool]:
    """Per call: does it carry the session grant? (Other extras ride along.)"""
    return [extra.get("hitl_approve_all") is True for extra in extras]


async def test_a_turn_under_the_grant_is_keyed_from_its_first_call(
    monkeypatch, workspace
):
    extras = await run_turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], FINISHED_REPLY],
        prepare=lambda app: setattr(app, "_hitl_auto_approve", True),
    )
    assert _keyed(extras) == [True, True]


async def test_the_granting_answer_is_sent_unkeyed(monkeypatch, workspace):
    """The widget's answer must be applied as given; the key rides from the
    call after it, the first auto-answered resume."""
    _no_config_rule(monkeypatch)
    extras = await run_turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [_interrupt(2)], FINISHED_REPLY],
        prepare=lambda app: _widget_answers(monkeypatch, app, GRANTING),
    )
    assert _keyed(extras) == [False, False, True]


async def test_a_config_rule_answer_is_not_keyed(monkeypatch, workspace):
    import metasci_scribe.stream.interaction as interaction_mod

    monkeypatch.setattr(
        interaction_mod, "config_policy_snapshot", lambda reqs: (list(APPROVE), {})
    )
    extras = await run_turn(monkeypatch, workspace, [[_interrupt(1)], FINISHED_REPLY])
    assert _keyed(extras) == [False, False]


async def test_other_extras_ride_on_every_call(monkeypatch, workspace):
    _no_config_rule(monkeypatch)

    def _prepare(app):
        app._session_runtime.active_teams.append("reviewer")
        _widget_answers(monkeypatch, app, GRANTING)

    extras = await run_turn(
        monkeypatch,
        workspace,
        [[_interrupt(1)], [_interrupt(2)], FINISHED_REPLY],
        prepare=_prepare,
    )
    assert _keyed(extras) == [False, False, True]
    assert [extra["active_teams"] for extra in extras] == [["reviewer"]] * 3
    # The marker the TUI sends on every call of its own thread is untouched.
    assert all("rubric_run" in extra for extra in extras)


ANSWER = {"answers": ["yes"], "status": "answered"}
