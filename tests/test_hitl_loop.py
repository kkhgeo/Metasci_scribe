"""Tests for HITL resume-round budgeting (issue #469).

The per-turn HITL resume loops must count only human-prompted rounds
against their budget. A session "approve all" grant or a config
auto-approving allow-list resumes guarded tool calls with no human
involved, so a long unattended turn must run to completion instead of
halting mid-work after 50 rounds. When the budget IS spent, the loop must
stop visibly — a message for the user plus a WARNING — refuse the NEXT
pending BEFORE prompting for it (the 50th decision is always resumed),
and close the parked checkpoint without resuming the agent. A rejecting
resume would run another model step; the close writes tool results and
clears ``next`` instead, so the checkpoint cannot poison the next turn.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from metasci_scribe.stream import display as display_mod
from tests.fakes.common import FakeCheckpointAgent, FakeGraphGateway
from tests.fakes.common import attended_config as _attended_config
from tests.fakes.common import interrupt_event as _interrupt_event


class TestRichCliHitlRoundBudget:
    """Rich CLI ``_run_streaming`` resume-loop budgeting."""

    def test_session_auto_approve_resumes_past_50_rounds(self, monkeypatch, caplog):
        """An unattended "approve all (session)" turn with 60 guarded tool
        calls must run to completion — auto-resolved rounds do not count
        against the human-prompted budget (issue #469)."""
        from langgraph.types import Command  # type: ignore[import-untyped]

        monkeypatch.setattr(display_mod, "_session_auto_approve", True)
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 60:
                yield _interrupt_event(stream_calls)
                return
            yield {"type": "text", "content": "All sixty rounds done"}
            yield {"type": "done", "content": "All sixty rounds done"}

        gateway = FakeGraphGateway(stream=_fake_stream)
        state = display_mod.StreamState()

        with caplog.at_level(logging.WARNING, logger="metasci_scribe.stream.display"):
            result = display_mod._run_streaming(
                agent=MagicMock(),
                message="hello",
                thread_id="t1",
                show_thinking=False,
                interactive=True,
                gateway=gateway,
                _state=state,
            )

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert stream_calls == 61  # initial stream + 60 auto-resumed rounds
        assert len(resumes) == 60
        assert result == "All sixty rounds done"
        assert state.pending_interrupt is None  # nothing left parked
        assert not any("max rounds" in r.getMessage() for r in caplog.records)

    def test_human_round_budget_exhaustion_stops_visibly_and_drains(
        self, monkeypatch, caplog
    ):
        """51 human-prompted interrupts: 50 prompts are served, then the
        loop stops with a visible notice, keeps the partial response, and
        closes the parked interrupt without another model step (issue #469)."""
        from langchain_core.messages import AIMessage
        from langgraph.types import Command  # type: ignore[import-untyped]

        from metasci_scribe.backends import HITL_ROUND_LIMIT_REJECT_MESSAGE

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            "metasci_scribe.agent._ensure_config",
            lambda: SimpleNamespace(
                auto_approve=False, dangerous_mode=False, shell_allow_list=""
            ),
        )
        printed: list[str] = []

        def _record_print(*args, **_kwargs):
            printed.append(str(args[0]) if args else "")

        monkeypatch.setattr(display_mod.console, "print", _record_print)

        prompt_calls = 0

        def _human_approves(_requests):
            nonlocal prompt_calls
            prompt_calls += 1
            return [{"type": "approve"}]

        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield {"type": "text", "content": "Partial work so far"}
            if stream_calls <= 51:
                yield _interrupt_event(stream_calls)
                return
            yield {"type": "text", "content": "should not stream a drain"}
            yield {"type": "done", "content": "should not stream a drain"}

        dangling = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "execute",
                    "args": {"command": "echo step"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
        # A main-agent approval parks before any tool of the batch runs.
        parked_at = "HumanInTheLoopMiddleware.after_model"
        agent = FakeCheckpointAgent(values={"messages": [dangling]}, next=(parked_at,))
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
        state = display_mod.StreamState()

        with caplog.at_level(logging.WARNING, logger="metasci_scribe.stream.display"):
            result = display_mod._run_streaming(
                agent=agent,
                message="hello",
                thread_id="t1",
                show_thinking=False,
                interactive=True,
                hitl_prompt_fn=_human_approves,
                gateway=gateway,
                _state=state,
            )

        assert prompt_calls == 50  # the 51st round hits the budget, not a prompt
        assert result == "Partial work so far"  # partial-response semantics kept
        assert stream_calls == 51  # initial + 50 approve resumes, no drain stream
        assert state.pending_interrupt is None  # parked interrupt was closed
        assert any("Approval round limit reached" in p for p in printed)
        assert any("max rounds" in r.getMessage() for r in caplog.records)

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert len(resumes) == 50
        assert "i51" not in resumes[-1].message.resume
        assert [as_node for _values, as_node in agent.updates] == [
            "__end__",
            parked_at,
            "__end__",
        ]
        tool_update = next(update for update in agent.updates if update[1] == parked_at)
        content = tool_update[0]["messages"][0].content
        assert HITL_ROUND_LIMIT_REJECT_MESSAGE in content
        assert "Do not retry this tool call" in content
        assert agent.updates[-1][0] is None
        assert agent.updates[-1][1] == "__end__"

    def test_ask_user_rounds_share_the_human_budget(self, monkeypatch):
        """ask_user rounds are always human-prompted and count toward the
        same budget; exhaustion closes the parked ask_user checkpoint
        without a cancelled resume (issue #469)."""

        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 51:
                yield {
                    "type": "ask_user",
                    "interrupt_id": f"ask-{stream_calls}",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "text", "content": "never reached"}
            yield {"type": "done", "content": "never reached"}

        agent = FakeCheckpointAgent()
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
        state = display_mod.StreamState()

        def _ask_user_fn(_pending):
            return {"answers": ["yes"], "status": "answered"}

        result = display_mod._run_streaming(
            agent=agent,
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            ask_user_prompt_fn=_ask_user_fn,
            gateway=gateway,
            _state=state,
        )

        assert stream_calls == 51  # 50 ask_user rounds + parked 51st, no drain
        assert state.pending_ask_user is None
        assert [as_node for _values, as_node in agent.updates] == ["__end__", "__end__"]
        assert result == ""

    def test_total_cap_closes_without_streaming_another_step(self, monkeypatch, caplog):
        """The runaway guard stops an auto-approved turn without a rejecting
        resume, which would otherwise run another model step (issue #469)."""
        from langgraph.types import Command  # type: ignore[import-untyped]

        import metasci_scribe.stream.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 2)
        monkeypatch.setattr(display_mod, "_session_auto_approve", True)
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            yield _interrupt_event(stream_calls)

        agent = FakeCheckpointAgent()
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
        state = display_mod.StreamState()
        printed: list[str] = []
        monkeypatch.setattr(
            display_mod.console,
            "print",
            lambda *args, **_kwargs: printed.append(str(args[0]) if args else ""),
        )

        with caplog.at_level(logging.WARNING, logger="metasci_scribe.stream.display"):
            display_mod._run_streaming(
                agent=agent,
                message="hello",
                thread_id="t1",
                show_thinking=False,
                interactive=True,
                gateway=gateway,
                _state=state,
            )

        resumes = [r for r in gateway.requests if isinstance(r.message, Command)]
        assert stream_calls == 2
        assert len(resumes) == 1  # the second interrupt is closed, not resumed
        assert agent.updates[-1][1] == "__end__"
        assert any("Approval round limit reached" in line for line in printed)

    def test_session_grant_after_human_budget_still_auto_resolves(self, monkeypatch):
        """The human budget is checked only when a prompt would be shown."""
        import metasci_scribe.stream.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HUMAN_HITL_ROUNDS", 1)
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            "metasci_scribe.agent._ensure_config",
            lambda: SimpleNamespace(
                auto_approve=False, dangerous_mode=False, shell_allow_list=""
            ),
        )
        prompts = 0

        def _grant_on_first(_requests):
            nonlocal prompts
            prompts += 1
            display_mod._session_auto_approve = True
            return [{"type": "approve"}]

        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls <= 3:
                yield _interrupt_event(stream_calls)
                return
            yield {"type": "text", "content": "kept going"}
            yield {"type": "done", "content": "kept going"}

        gateway = FakeGraphGateway(stream=_fake_stream)
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            hitl_prompt_fn=_grant_on_first,
            gateway=gateway,
            _state=display_mod.StreamState(),
        )

        assert prompts == 1
        assert result == "kept going"
        assert stream_calls == 4

    def test_close_failure_keeps_the_partial_response(self, monkeypatch, caplog):
        import metasci_scribe.stream.hitl_budget as budget_mod

        monkeypatch.setattr(budget_mod, "MAX_HUMAN_HITL_ROUNDS", 0)
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            "metasci_scribe.agent._ensure_config",
            lambda: SimpleNamespace(
                auto_approve=False, dangerous_mode=False, shell_allow_list=""
            ),
        )

        async def _fake_stream(_request):
            yield {"type": "text", "content": "partial"}
            yield _interrupt_event(1)

        agent = FakeCheckpointAgent(update_error=RuntimeError("checkpoint store down"))
        gateway = FakeGraphGateway(stream=_fake_stream, checkpoint=agent)
        monkeypatch.setattr(display_mod.console, "print", lambda *_a, **_k: None)

        with caplog.at_level(logging.WARNING, logger="metasci-scribe"):
            result = display_mod._run_streaming(
                agent=agent,
                message="hello",
                thread_id="t1",
                show_thinking=False,
                interactive=True,
                hitl_prompt_fn=lambda _requests: [{"type": "approve"}],
                gateway=gateway,
                _state=display_mod.StreamState(),
            )

        assert result == "partial"
        assert len(gateway.requests) == 1
        failures = [
            r
            for r in caplog.records
            if "Could not close the parked checkpoint" in r.getMessage()
        ]
        assert len(failures) == 1
        assert failures[0].exc_info is None

    def test_channel_budget_stop_reaches_the_channel_reply(self, monkeypatch):
        from metasci_scribe.stream.hitl_budget import HITL_BUDGET_STOP_NOTICE
        from metasci_scribe.stream.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            "metasci_scribe.agent._ensure_config",
            lambda: SimpleNamespace(
                auto_approve=False, dangerous_mode=False, shell_allow_list=""
            ),
        )
        monkeypatch.setattr(display_mod.console, "print", lambda *a, **k: None)
        stream_calls = 0

        async def _fake_stream(_request):
            nonlocal stream_calls
            stream_calls += 1
            if stream_calls == 1:
                yield {"type": "text", "content": "Partial work so far"}
            yield _interrupt_event(stream_calls)

        prompted = 0

        def _bridge(_requests, exhausted):
            nonlocal prompted
            if exhausted:
                return ApprovalOutcome(budget_exhausted=True)
            prompted += 1
            return ApprovalOutcome(decisions=[{"type": "approve"}], prompted=True)

        gateway = FakeGraphGateway(
            stream=_fake_stream, checkpoint=FakeCheckpointAgent()
        )
        result = display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            gateway=gateway,
            _state=display_mod.StreamState(),
            hitl_outcome_fn=_bridge,
        )

        assert prompted == 50
        assert stream_calls == 51
        assert result == f"Partial work so far\n\n{HITL_BUDGET_STOP_NOTICE}"


GRANT = {"hitl_approve_all": True}
ANSWER = {"answers": ["yes"], "status": "answered"}


def _interrupts_then_done(interrupts: int):
    """A stream that parks ``interrupts`` approvals, one per call, then answers."""
    calls = 0

    async def _fake_stream(_request):
        nonlocal calls
        calls += 1
        if calls <= interrupts:
            yield _interrupt_event(calls)
            return
        yield {"type": "text", "content": "finished"}
        yield {"type": "done", "content": "finished"}

    return _fake_stream


class TestRichCliSessionGrantReachesTheGraph:
    """Which calls of a turn carry ``hitl_approve_all`` (the graph then skips the pause).

    The fake stream keeps parking approvals whatever it is sent, so these pin
    the client's side only: the real graph stops interrupting once a call
    carries the key (``tests/test_hitl_approve_all_graph.py``).
    """

    @staticmethod
    def _run(gateway, **kwargs):
        return display_mod._run_streaming(
            agent=MagicMock(),
            message="hello",
            thread_id="t1",
            show_thinking=False,
            interactive=True,
            gateway=gateway,
            _state=display_mod.StreamState(),
            **kwargs,
        )

    @staticmethod
    def _extras(gateway):
        return [request.configurable_extra for request in gateway.requests]

    def test_a_turn_under_the_grant_is_keyed_from_its_first_call(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", True)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(2))
        assert self._run(gateway) == "finished"
        assert self._extras(gateway) == [GRANT, GRANT, GRANT]

    def test_the_granting_answer_is_sent_unkeyed(self, monkeypatch):
        """The human's answer must be applied as given, so the key rides from
        the call after it: the first auto-answered resume."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)

        def _grant(_requests):
            display_mod._session_auto_approve = True
            return [{"type": "approve"}]

        gateway = FakeGraphGateway(stream=_interrupts_then_done(3))
        assert self._run(gateway, hitl_prompt_fn=_grant) == "finished"
        assert self._extras(gateway) == [None, None, GRANT, GRANT]

    def test_an_allow_list_answer_is_not_keyed(self, monkeypatch):
        import metasci_scribe.stream.interaction as interaction_mod

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            interaction_mod,
            "config_policy_snapshot",
            lambda reqs: ([{"type": "approve"} for _ in reqs], {}),
        )
        gateway = FakeGraphGateway(stream=_interrupts_then_done(2))
        assert self._run(gateway) == "finished"
        assert self._extras(gateway) == [None, None, None]

    @pytest.mark.parametrize("granted", [True, False])
    def test_a_bridge_answer_is_keyed_only_when_the_chat_is_granted(
        self, monkeypatch, granted
    ):
        from metasci_scribe.stream.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(1))
        self._run(
            gateway,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs], session_granted=granted
            ),
        )
        assert self._extras(gateway) == [None, GRANT if granted else None]

    def test_a_turn_that_starts_granted_is_keyed_from_its_first_call(self, monkeypatch):
        """A channel-origin turn: the caller knows the chat's grant and says so
        with a flag; the loop leaves the caller's extras alone."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        extras = {"active_teams": ["reviewer"]}
        gateway = FakeGraphGateway(stream=_interrupts_then_done(0))
        self._run(gateway, configurable_extra=extras, approve_all_at_start=True)
        assert self._extras(gateway) == [
            {"active_teams": ["reviewer"], "hitl_approve_all": True}
        ]
        assert extras == {"active_teams": ["reviewer"]}

    def test_a_turn_that_starts_ungranted_is_not_keyed(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(0))
        self._run(gateway, configurable_extra={"active_teams": ["reviewer"]})
        assert self._extras(gateway) == [{"active_teams": ["reviewer"]}]

    def test_a_starting_grant_is_dropped_from_a_call_that_must_not_carry_it(
        self, monkeypatch
    ):
        from metasci_scribe.stream.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        extras = {"active_teams": ["reviewer"]}
        gateway = FakeGraphGateway(stream=_interrupts_then_done(1))
        self._run(
            gateway,
            configurable_extra=extras,
            approve_all_at_start=True,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs], prompted=True
            ),
        )
        assert self._extras(gateway) == [
            {"active_teams": ["reviewer"], "hitl_approve_all": True},
            {"active_teams": ["reviewer"]},
        ]
        assert extras == {"active_teams": ["reviewer"]}

    @pytest.mark.parametrize("granted", [True, False])
    def test_an_ask_user_answer_is_keyed_only_under_a_grant(self, monkeypatch, granted):
        monkeypatch.setattr(display_mod, "_session_auto_approve", granted)
        calls = 0

        async def _fake_stream(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        gateway = FakeGraphGateway(stream=_fake_stream)
        self._run(
            gateway,
            ask_user_prompt_fn=lambda _p: {"answers": ["yes"], "status": "answered"},
        )
        expected = GRANT if granted else None
        assert self._extras(gateway) == [expected, expected]

    def test_a_grant_answer_with_a_non_approve_decision_is_not_keyed(self, monkeypatch):
        """Belt and braces: the key is sent only when every decision approves."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            display_mod,
            "_resolve_hitl_approval",
            lambda *_a, **_k: display_mod._HitlResolution(
                [{"type": "reject", "message": "no"}],
                human_prompted=False,
                session_granted=True,
            ),
        )
        gateway = FakeGraphGateway(stream=_interrupts_then_done(1))
        self._run(gateway)
        assert self._extras(gateway) == [None, None]

    @staticmethod
    def _approval_then_question():
        """One approval, then one question, then the answer."""
        calls = 0

        async def _fake_stream(_request):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield _interrupt_event(1)
                return
            if calls == 2:
                yield {
                    "type": "ask_user",
                    "interrupt_id": "ask-1",
                    "questions": [{"question": "Continue?"}],
                }
                return
            yield {"type": "text", "content": "finished"}
            yield {"type": "done", "content": "finished"}

        return _fake_stream

    def test_a_question_after_the_granting_prompt_is_answered_keyed(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)

        def _grant(_requests):
            display_mod._session_auto_approve = True
            return [{"type": "approve"}]

        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(gateway, hitl_prompt_fn=_grant, ask_user_prompt_fn=lambda _p: ANSWER)
        assert self._extras(gateway) == [None, None, GRANT]

    def test_a_question_after_a_bridge_grant_answer_is_answered_keyed(
        self, monkeypatch
    ):
        """A chat grant seen during the turn holds for the rest of it."""
        from metasci_scribe.stream.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(
            gateway,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs],
                session_granted=True,
                grant_in_force=True,
            ),
            ask_user_prompt_fn=lambda _p: ANSWER,
        )
        assert self._extras(gateway) == [None, GRANT, GRANT]

    def test_a_grant_given_on_the_channel_keys_the_next_question_answer(
        self, monkeypatch
    ):
        """The reply that grants is a human answer (unkeyed), but the chat's
        grant is in force afterwards: the answer to a question that comes next
        carries it, so no approval pauses after the question."""
        from metasci_scribe.stream.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(
            gateway,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs],
                prompted=True,
                grant_in_force=True,
            ),
            ask_user_prompt_fn=lambda _p: ANSWER,
        )
        assert self._extras(gateway) == [None, None, GRANT]

    def test_a_question_after_a_plain_approval_is_answered_unkeyed(self, monkeypatch):
        """Approving once grants nothing: the answer to a question that follows
        is unkeyed, so the turn's later gated calls still ask."""
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(
            gateway,
            hitl_prompt_fn=lambda reqs: [{"type": "approve"} for _ in reqs],
            ask_user_prompt_fn=lambda _p: ANSWER,
        )
        assert self._extras(gateway) == [None, None, None]

    @pytest.mark.parametrize("prompted", [True, False])
    def test_a_question_after_an_ungranted_bridge_answer_is_answered_unkeyed(
        self, monkeypatch, prompted
    ):
        """The same for a channel answer with no grant behind it, whether the
        user approved once or the bridge answered on its own."""
        from metasci_scribe.stream.interaction import ApprovalOutcome

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(
            gateway,
            hitl_outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
                decisions=[{"type": "approve"} for _ in reqs], prompted=prompted
            ),
            ask_user_prompt_fn=lambda _p: ANSWER,
        )
        assert self._extras(gateway) == [None, None, None]

    def test_a_question_after_a_config_rule_answer_is_answered_unkeyed(
        self, monkeypatch
    ):
        """An allow-list answer is not a grant either."""
        import metasci_scribe.stream.interaction as interaction_mod

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            interaction_mod,
            "config_policy_snapshot",
            lambda reqs: ([{"type": "approve"} for _ in reqs], {}),
        )
        gateway = FakeGraphGateway(stream=self._approval_then_question())
        self._run(gateway, ask_user_prompt_fn=lambda _p: ANSWER)
        assert self._extras(gateway) == [None, None, None]

    def test_a_config_rule_answer_drops_a_key_the_turn_started_with(self, monkeypatch):
        """A channel-origin turn whose chat is granted (the caller says so), yet
        this batch is answered by the config rules before the bridge is asked:
        not the grant's answer, so the resume goes out unkeyed."""
        import metasci_scribe.stream.interaction as interaction_mod

        def _bridge_must_not_be_asked(_reqs, _exhausted):
            raise AssertionError("the config rule answers first")

        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        monkeypatch.setattr(
            interaction_mod,
            "config_policy_snapshot",
            lambda reqs: ([{"type": "approve"} for _ in reqs], {}),
        )
        gateway = FakeGraphGateway(stream=_interrupts_then_done(1))
        self._run(
            gateway,
            approve_all_at_start=True,
            hitl_outcome_fn=_bridge_must_not_be_asked,
        )
        assert self._extras(gateway) == [GRANT, None]

    def test_a_declined_prompt_sends_nothing_more(self, monkeypatch):
        monkeypatch.setattr(display_mod, "_session_auto_approve", False)
        _attended_config(monkeypatch)
        gateway = FakeGraphGateway(stream=_interrupts_then_done(3))
        self._run(gateway, hitl_prompt_fn=lambda _requests: None)
        assert self._extras(gateway) == [None]


def test_resolver_reports_the_grant_only_on_its_fast_path(monkeypatch):
    import metasci_scribe.stream.interaction as interaction_mod
    from metasci_scribe.stream.interaction import ApprovalOutcome

    pending = _interrupt_event(1)
    monkeypatch.setattr(
        "metasci_scribe.agent._ensure_config",
        lambda: SimpleNamespace(
            auto_approve=False, dangerous_mode=False, shell_allow_list=""
        ),
    )

    monkeypatch.setattr(display_mod, "_session_auto_approve", True)
    assert display_mod._resolve_hitl_approval(pending).session_granted is True

    monkeypatch.setattr(display_mod, "_session_auto_approve", False)

    def _grant(_requests):
        display_mod._session_auto_approve = True
        return [{"type": "approve"}]

    granting = display_mod._resolve_hitl_approval(pending, prompt_fn=_grant)
    assert granting.human_prompted is True
    assert granting.session_granted is False

    monkeypatch.setattr(display_mod, "_session_auto_approve", False)
    bridged = display_mod._resolve_hitl_approval(
        pending,
        outcome_fn=lambda reqs, _exhausted: ApprovalOutcome(
            decisions=[{"type": "approve"}], session_granted=True
        ),
    )
    assert bridged.session_granted is True

    monkeypatch.setattr(
        interaction_mod,
        "config_policy_snapshot",
        lambda reqs: ([{"type": "approve"}], {}),
    )
    assert display_mod._resolve_hitl_approval(pending).session_granted is False


def test_hitl_budget_stop_ignores_human_cap_for_auto_rounds():
    import metasci_scribe.stream.hitl_budget as budget_mod

    assert not budget_mod.hitl_budget_stop(
        human_rounds=50, total_rounds=51, needs_human=False
    )
    assert budget_mod.hitl_budget_stop(
        human_rounds=50, total_rounds=51, needs_human=True
    )
    assert budget_mod.hitl_budget_stop(
        human_rounds=0, total_rounds=1000, needs_human=False
    )


def test_tui_loop_does_not_build_a_resume_past_the_total_cap(monkeypatch):
    """TUI HITL loop, without a Pilot harness.

    Each pause branch calls ``hitl_budget_stop`` before it prompts or
    builds a resume. Auto rounds ignore the human cap; at the total cap the
    branch stops with no resume left unsent.
    """
    import metasci_scribe.stream.hitl_budget as budget_mod
    from metasci_scribe.stream.hitl_budget import hitl_budget_stop

    monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 2)
    resumes_built = 0
    unsent_resume = None
    for total_rounds in range(1, 6):
        # The previous iteration's resume, if any, is what this round streams.
        unsent_resume = None
        if hitl_budget_stop(
            human_rounds=0, total_rounds=total_rounds, needs_human=False
        ):
            break
        unsent_resume = {"type": "approve", "round": total_rounds}
        resumes_built += 1

    assert resumes_built == 1
    assert unsent_resume is None


def test_tui_loop_human_cap_does_not_stop_an_auto_branch():
    """A session-grant branch keeps resuming after the human cap; a widget
    branch does not."""
    from metasci_scribe.stream.hitl_budget import hitl_budget_stop

    assert not hitl_budget_stop(human_rounds=50, total_rounds=51, needs_human=False)
    assert hitl_budget_stop(human_rounds=50, total_rounds=51, needs_human=True)


def test_tui_loop_level_total_cap_uses_completed_round_helper(monkeypatch):
    """The TUI loop-top guard is ``hitl_completed_round_cap_reached``,
    not a rewritten copy of the stop condition (issue #469 review)."""
    import metasci_scribe.stream.hitl_budget as budget_mod
    from metasci_scribe.stream.hitl_budget import hitl_completed_round_cap_reached

    monkeypatch.setattr(budget_mod, "MAX_HITL_TOTAL_ROUNDS", 2)
    assert not hitl_completed_round_cap_reached(0)
    assert not hitl_completed_round_cap_reached(1)
    assert hitl_completed_round_cap_reached(2)
    assert hitl_completed_round_cap_reached(3)


def test_channel_response_with_budget_stop_keeps_partial_then_notice():
    from metasci_scribe.stream.hitl_budget import (
        HITL_BUDGET_STOP_NOTICE,
        channel_response_with_budget_stop,
    )

    assert channel_response_with_budget_stop("") == HITL_BUDGET_STOP_NOTICE
    assert channel_response_with_budget_stop("   ") == HITL_BUDGET_STOP_NOTICE
    assert channel_response_with_budget_stop("partial") == (
        f"partial\n\n{HITL_BUDGET_STOP_NOTICE}"
    )
    assert (
        channel_response_with_budget_stop(HITL_BUDGET_STOP_NOTICE)
        == HITL_BUDGET_STOP_NOTICE
    )


def test_abandoned_tool_messages_closes_invalid_tool_calls():
    """HITL close must patch invalid calls too, matching crash recovery."""
    from langchain_core.messages import AIMessage, ToolMessage

    from metasci_scribe.backends import (
        HITL_ROUND_LIMIT_REJECT_MESSAGE,
        abandoned_tool_messages,
    )

    patch = abandoned_tool_messages(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {},
                        "id": "ok-1",
                        "type": "tool_call",
                    }
                ],
                invalid_tool_calls=[
                    {
                        "name": "execute",
                        "args": "not-json",
                        "id": "bad-1",
                        "error": "parse",
                        "type": "invalid_tool_call",
                    }
                ],
            ),
            ToolMessage(content="done", name="execute", tool_call_id="ok-1"),
        ],
        reason="round_limit",
    )
    assert [message.tool_call_id for message in patch] == ["bad-1"]
    assert HITL_ROUND_LIMIT_REJECT_MESSAGE in patch[0].content


def test_hitl_pause_unresolved_does_not_replay_without_a_resume():
    from metasci_scribe.stream.hitl_budget import hitl_pause_unresolved

    assert hitl_pause_unresolved(resuming=False, pending=True)
    assert not hitl_pause_unresolved(resuming=True, pending=True)
    assert not hitl_pause_unresolved(resuming=False, pending=False)


class _StickyCheckpointAgent:
    """``aupdate_state`` is a no-op so recovery's verify still sees ``next``."""

    async def aget_state(self, _config):
        return SimpleNamespace(
            next=("tools",),
            tasks=(),
            interrupts=(object(),),
            values={},
        )

    async def aupdate_state(self, *_args, **_kwargs):
        return None


async def test_close_parked_checkpoint_keeps_finished_sibling_results():
    """Clear first so an answered sibling (ask_user beside execute) stays
    in history; only the unanswered call is patched (issue #469 review)."""
    from langchain_core.messages import AIMessage, ToolMessage

    from metasci_scribe.backends import (
        HITL_ROUND_LIMIT_REJECT_MESSAGE,
        close_parked_checkpoint,
    )
    from metasci_scribe.gateway import GraphTarget

    pending_ask = ToolMessage(
        content="user answered from the widget",
        name="ask_user",
        tool_call_id="ask-1",
    )
    agent = FakeCheckpointAgent(
        values={
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "ask_user",
                            "args": {"questions": [{"question": "Continue?"}]},
                            "id": "ask-1",
                            "type": "tool_call",
                        },
                        {
                            "name": "execute",
                            "args": {"command": "echo sibling"},
                            "id": "exec-1",
                            "type": "tool_call",
                        },
                    ],
                )
            ]
        },
        pending_write=pending_ask,
    )

    gateway = FakeGraphGateway(checkpoint=agent)
    await close_parked_checkpoint(
        gateway, GraphTarget(), "t-sibling", reason="round_limit"
    )

    assert [as_node for _values, as_node in agent.updates] == [
        "__end__",
        "tools",
        "__end__",
    ]
    patch = next(values for values, node in agent.updates if node == "tools")[
        "messages"
    ]
    assert [message.tool_call_id for message in patch] == ["exec-1"]
    assert HITL_ROUND_LIMIT_REJECT_MESSAGE in patch[0].content
    seen = {
        getattr(message, "tool_call_id", None): getattr(message, "content", None)
        for message in agent.values["messages"]
        if getattr(message, "type", None) == "tool"
    }
    assert seen["ask-1"] == "user answered from the widget"
    assert HITL_ROUND_LIMIT_REJECT_MESSAGE in seen["exec-1"]
    verify = await agent.aget_state({})
    assert verify.next == ()
    assert not verify.interrupts


async def test_close_parked_checkpoint_trailing_end_even_without_dangling_calls():
    """With no unanswered calls the first END still gets a trailing END,
    matching recovery: the first clear can leave next == ('model',)."""
    from metasci_scribe.backends import close_parked_checkpoint
    from metasci_scribe.gateway import GraphTarget

    agent = FakeCheckpointAgent()
    gateway = FakeGraphGateway(checkpoint=agent)
    await close_parked_checkpoint(
        gateway, GraphTarget(), "t-empty", reason="round_limit"
    )
    assert [as_node for _values, as_node in agent.updates] == ["__end__", "__end__"]
    assert all(values is None for values, _node in agent.updates)
    verify = await agent.aget_state({})
    assert verify.next == ()
    assert not verify.interrupts


async def test_close_parked_checkpoint_raises_when_verify_still_stuck():
    """Unlike the old parallel close, a failed repair is visible."""
    from metasci_scribe.backends import close_parked_checkpoint
    from metasci_scribe.gateway import GraphTarget

    with pytest.raises(RuntimeError, match="Could not close parked HITL"):
        await close_parked_checkpoint(
            FakeGraphGateway(checkpoint=_StickyCheckpointAgent()),
            GraphTarget(),
            "t-sticky",
            reason="round_limit",
        )


async def test_close_parked_checkpoint_ignores_local_graph():
    """After #470, close must use the gateway even if local_graph is present."""
    from metasci_scribe.backends import close_parked_checkpoint
    from metasci_scribe.gateway import GraphTarget

    agent = FakeCheckpointAgent()
    sentinel = SimpleNamespace(
        aget_state=AsyncMock(side_effect=AssertionError("used local_graph")),
        aupdate_state=AsyncMock(side_effect=AssertionError("used local_graph")),
    )
    gateway = FakeGraphGateway(checkpoint=agent)
    await close_parked_checkpoint(
        gateway, GraphTarget(local_graph=sentinel), "t-gateway", reason="round_limit"
    )
    assert [as_node for _values, as_node in agent.updates] == ["__end__", "__end__"]
    assert sentinel.aget_state.await_count == 0
    assert [node for _target, _tid, _values, node in gateway.updated_states] == [
        "__end__",
        "__end__",
    ]


async def test_close_parked_checkpoint_via_local_gateway():
    """The in-process gateway still closes through its snapshot/update seam."""
    from metasci_scribe.backends import close_parked_checkpoint
    from metasci_scribe.gateway import GraphTarget, LocalGraphGateway

    agent = FakeCheckpointAgent()
    await close_parked_checkpoint(
        LocalGraphGateway(),
        GraphTarget(local_graph=agent),
        "t-local",
        reason="round_limit",
    )
    assert [as_node for _values, as_node in agent.updates] == ["__end__", "__end__"]
