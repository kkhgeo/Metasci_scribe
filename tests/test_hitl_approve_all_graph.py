"""A session "approve all" on the real main graph: the grant key and the order it must travel in."""

from __future__ import annotations

import logging

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from metasci_scribe.agent import create_cli_agent
from metasci_scribe.config import ScribeConfig
from metasci_scribe.paths import Workspace
from metasci_scribe.stream.events import stream_agent_events

pytestmark = pytest.mark.usefixtures("no_mcp")


class _Scripted(GenericFakeChatModel):
    """Returns the scripted messages in order, for the main agent and its sub-agents alike."""

    def bind_tools(self, tools, **kwargs):
        return self


def _call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _execute(call_id: str, command: str) -> AIMessage:
    return AIMessage("", tool_calls=[_call(call_id, "execute", command=command)])


def _agent(tmp_path, messages):
    cfg = ScribeConfig(enable_async_subagents=False, memory_workers_enabled=False)
    return create_cli_agent(
        workspace=Workspace(tmp_path),
        config=cfg,
        chat_model=_Scripted(messages=iter([*messages, *[AIMessage("extra")] * 4])),
        checkpointer=InMemorySaver(),
    )


def _config(*, granted: bool = False, suppressed: bool = False) -> dict:
    configurable = {"thread_id": "t", "hitl_suppressed": suppressed}
    if granted:
        configurable["hitl_approve_all"] = True
    return {"configurable": configurable}


def _resume(interrupt, *decisions) -> Command:
    return Command(resume={interrupt.id: {"decisions": list(decisions)}})


def _written(tmp_path) -> list[str]:
    return sorted(p.name for p in tmp_path.glob("*.txt"))


def _tool_text(result, call_id: str) -> str:
    return next(
        str(m.content)
        for m in result["messages"]
        if m.type == "tool" and m.tool_call_id == call_id
    )


APPROVE = {"type": "approve"}
REJECT = {"type": "reject", "message": "not this one"}


def _edit(command: str) -> dict:
    return {
        "type": "edit",
        "edited_action": {"name": "execute", "args": {"command": command}},
    }


async def test_a_granted_call_runs_gated_tools_without_interrupting(tmp_path):
    def _script():
        return [
            AIMessage("", tool_calls=[_delete("d1", "one.txt")]),
            AIMessage("Done."),
        ]

    for folder in ("armed", "granted"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "one.txt").write_text("x", encoding="utf-8")

    armed = _agent(tmp_path / "armed", _script())
    parked = await armed.ainvoke({"messages": [("user", "go")]}, config=_config())
    assert len(parked["__interrupt__"]) == 1
    assert _written(tmp_path / "armed") == ["one.txt"]

    granted = _agent(tmp_path / "granted", _script())
    result = await granted.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    assert not result.get("__interrupt__")
    assert _written(tmp_path / "granted") == []


async def test_the_granting_answer_holds_and_the_next_call_silences_the_turn(tmp_path):
    """The answer that grants goes out without the key, so its decisions apply;
    the key rides on the call after it and nothing interrupts again."""
    agent = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call("a", "execute", command="echo a > a.txt"),
                    _call("b", "execute", command="echo b > b.txt"),
                    _call("x", "execute", command="echo x > x.txt"),
                ],
            ),
            AIMessage("", tool_calls=[_delete("c", "c.txt")]),
            AIMessage("", tool_calls=[_delete("d", "d.txt")]),
            AIMessage("All done."),
        ],
    )
    (tmp_path / "c.txt").write_text("x", encoding="utf-8")
    (tmp_path / "d.txt").write_text("x", encoding="utf-8")
    first = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    (parked,) = first["__interrupt__"]
    assert len(parked.value["action_requests"]) == 3

    second = await agent.ainvoke(
        _resume(parked, APPROVE, REJECT, _edit("echo edited > e.txt")), config=_config()
    )
    assert _written(tmp_path) == ["a.txt", "c.txt", "d.txt", "e.txt"]
    (next_parked,) = second["__interrupt__"]

    third = await agent.ainvoke(
        _resume(next_parked, APPROVE), config=_config(granted=True)
    )
    assert not third.get("__interrupt__")
    assert _written(tmp_path) == ["a.txt", "e.txt"]
    assert "rejected" in _tool_text(third, "b")
    assert third["messages"][-1].text == "All done."


async def test_a_sync_subagent_sees_the_parents_grant(tmp_path):
    def _script():
        return [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "t1",
                        "task",
                        subagent_type="general-purpose",
                        description="write a file",
                    )
                ],
            ),
            AIMessage("", tool_calls=[_delete("s1", "sub.txt")]),
            AIMessage("Sub done."),
            AIMessage("Main done."),
        ]

    for folder in ("armed", "granted"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "sub.txt").write_text("x", encoding="utf-8")

    armed = await _agent(tmp_path / "armed", _script()).ainvoke(
        {"messages": [("user", "go")]}, config=_config()
    )
    assert len(armed["__interrupt__"]) == 1
    assert _written(tmp_path / "armed") == ["sub.txt"]

    granted = await _agent(tmp_path / "granted", _script()).ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    assert not granted.get("__interrupt__")
    assert _written(tmp_path / "granted") == []


async def test_a_granted_fresh_turn_does_not_run_a_batch_left_parked(tmp_path):
    """A declined approval leaves the thread parked; the next message is new
    input, not an answer, so the grant must not release the declined call."""
    agent = _agent(
        tmp_path, [_execute("c1", "echo 1 > declined.txt"), AIMessage("Fresh.")]
    )
    parked = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    assert len(parked["__interrupt__"]) == 1

    events = [
        event
        async for event in stream_agent_events(
            agent, "something else", "t", configurable_extra={"hitl_approve_all": True}
        )
    ]
    assert _written(tmp_path) == []
    assert not [e for e in events if e["type"] == "interrupt"]
    assert next(e for e in events if e["type"] == "done")["response"] == "Fresh."


async def test_a_granted_fresh_turn_does_not_run_a_subagents_parked_batch(tmp_path):
    """The same for an approval parked inside a sync sub-agent."""
    agent = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "t1",
                        "task",
                        subagent_type="general-purpose",
                        description="write a file",
                    )
                ],
            ),
            _execute("s1", "echo sub > declined.txt"),
            AIMessage("Fresh."),
        ],
    )
    parked = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    assert len(parked["__interrupt__"]) == 1

    events = [
        event
        async for event in stream_agent_events(
            agent, "something else", "t", configurable_extra={"hitl_approve_all": True}
        )
    ]
    assert _written(tmp_path) == []
    assert not [e for e in events if e["type"] == "interrupt"]
    assert next(e for e in events if e["type"] == "done")["response"] == "Fresh."


async def test_the_grant_still_prompts_but_suppression_removes_the_shell(tmp_path):
    (tmp_path / "granted").mkdir()
    granted = await _agent(
        tmp_path / "granted", [_execute("c1", "echo true | bash"), AIMessage("Done.")]
    ).ainvoke({"messages": [("user", "go")]}, config=_config(granted=True))
    assert len(granted["__interrupt__"]) == 1

    (tmp_path / "both").mkdir()
    both = await _agent(
        tmp_path / "both", [_execute("c1", "echo true | bash"), AIMessage("Done.")]
    ).ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True, suppressed=True)
    )
    assert not both.get("__interrupt__")
    assert "not available" in _tool_text(both, "c1")


# --- a declining answer never travels with the grant --------------------------


def _flat(_interrupt, *decisions) -> Command:
    return Command(resume={"decisions": list(decisions)})


@pytest.mark.parametrize("shape", [_resume, _flat], ids=["by interrupt id", "flat"])
async def test_a_declining_answer_sent_with_the_grant_still_declines(
    tmp_path, caplog, shape
):
    """A caller that keys a rejecting answer by mistake: the grant is dropped
    from that run, so the answer is applied instead of the whole batch."""
    agent = _agent(
        tmp_path, [_execute("c1", "echo 1 > declined.txt"), AIMessage("Understood.")]
    )
    parked = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    (interrupt,) = parked["__interrupt__"]

    with caplog.at_level(logging.WARNING, logger="metasci_scribe.stream.events"):
        events = [
            event
            async for event in stream_agent_events(
                agent,
                shape(interrupt, REJECT),
                "t",
                configurable_extra={"hitl_approve_all": True},
            )
        ]
    assert _written(tmp_path) == []
    assert next(e for e in events if e["type"] == "done")["response"] == "Understood."
    assert "hitl_approve_all" in caplog.text


async def test_an_edited_answer_sent_with_the_grant_runs_the_edit(tmp_path):
    agent = _agent(
        tmp_path, [_execute("c1", "echo 1 > original.txt"), AIMessage("Done.")]
    )
    parked = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    (interrupt,) = parked["__interrupt__"]
    async for _ in stream_agent_events(
        agent,
        _resume(interrupt, _edit("echo 1 > edited.txt")),
        "t",
        configurable_extra={"hitl_approve_all": True},
    ):
        pass
    assert _written(tmp_path) == ["edited.txt"]


async def test_an_approving_answer_keeps_the_grant(tmp_path, caplog, monkeypatch):
    """The grant's own answer approves everything, so the key stays on the run."""
    agent = _agent(tmp_path, [_execute("c1", "echo 1 > one.txt"), AIMessage("Done.")])
    parked = await agent.ainvoke({"messages": [("user", "go")]}, config=_config())
    (interrupt,) = parked["__interrupt__"]
    configs = []
    stream = agent.astream_events

    def _spy(*args, **kwargs):
        configs.append(kwargs["config"])
        return stream(*args, **kwargs)

    monkeypatch.setattr(agent, "astream_events", _spy)
    with caplog.at_level(logging.WARNING, logger="metasci_scribe.stream.events"):
        async for _ in stream_agent_events(
            agent,
            _resume(interrupt, APPROVE),
            "t",
            configurable_extra={"hitl_approve_all": True},
        ):
            pass
    assert _written(tmp_path) == ["one.txt"]
    assert configs[-1]["configurable"]["hitl_approve_all"] is True
    assert "hitl_approve_all" not in caplog.text


# --- the graph a server serves does not take the grant -------------------------


async def test_a_served_graph_still_asks_when_a_client_sends_the_grant(
    tmp_path, monkeypatch
):
    """The served main graph is built without the grant: a client that sends
    the key anyway gets the approval pause, and nothing runs."""
    import metasci_scribe.agent as es_mod

    build = es_mod._build_hitl_interrupt_on
    monkeypatch.setattr(
        es_mod,
        "_build_hitl_interrupt_on",
        lambda **_kwargs: build(honor_approve_all=False),
    )
    agent = _agent(tmp_path, [_execute("c1", "echo 1 > one.txt"), AIMessage("Done.")])
    result = await agent.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    assert len(result["__interrupt__"]) == 1
    assert _written(tmp_path) == []

    (tmp_path / "unattended").mkdir()
    unattended = _agent(
        tmp_path / "unattended", [_execute("c1", "echo 1 > one.txt"), AIMessage("Ok.")]
    )
    result = await unattended.ainvoke(
        {"messages": [("user", "go")]}, config=_config(suppressed=True)
    )
    assert not result.get("__interrupt__")
    # Nobody approves there, so there is no shell at all.
    assert _written(tmp_path / "unattended") == []


# --- the grant does not reach the shell ----------------------------------------


def _delete(call_id: str, name: str) -> dict:
    return _call(call_id, "delete", file_path=f"/{name}")


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("execute", {"command": "echo 1 > one.txt"}),
        ("run_in_background", {"command": "echo 1 > one.txt"}),
    ],
    ids=["execute", "run_in_background"],
)
async def test_a_grant_does_not_suppress_the_shell_prompt(tmp_path, tool, args):
    agent = _agent(
        tmp_path,
        [AIMessage("", tool_calls=[_call("c1", tool, **args)]), AIMessage("Done.")],
    )
    result = await agent.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    (parked,) = result["__interrupt__"]
    assert [r["name"] for r in parked.value["action_requests"]] == [tool]
    assert _written(tmp_path) == []


async def test_a_prompted_command_runs_on_the_persons_approval(tmp_path):
    agent = _agent(tmp_path, [_execute("c1", "echo 1 > one.txt"), AIMessage("Done.")])
    parked = await agent.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    (interrupt,) = parked["__interrupt__"]
    done = await agent.ainvoke(
        _resume(interrupt, APPROVE), config=_config(granted=True)
    )
    assert not done.get("__interrupt__")
    assert _written(tmp_path) == ["one.txt"]


async def test_a_rejected_command_does_not_run_under_a_grant(tmp_path):
    agent = _agent(tmp_path, [_execute("c1", "echo 1 > one.txt"), AIMessage("Done.")])
    parked = await agent.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    (interrupt,) = parked["__interrupt__"]
    await agent.ainvoke(_resume(interrupt, REJECT), config=_config(granted=True))
    assert _written(tmp_path) == []


async def test_a_grant_still_covers_other_gated_tools(tmp_path):
    (tmp_path / "gone.txt").write_text("x", encoding="utf-8")
    agent = _agent(
        tmp_path,
        [AIMessage("", tool_calls=[_delete("d1", "gone.txt")]), AIMessage("Done.")],
    )
    result = await agent.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    assert not result.get("__interrupt__")
    assert _written(tmp_path) == []


@pytest.mark.parametrize("carry_grant", [True, False], ids=["carried", "dropped"])
async def test_a_batch_with_a_command_asks_for_all_of_it(tmp_path, carry_grant):
    """Which calls pause must not change between the pause and its resume, which
    may or may not carry the grant, or the answers no longer line up."""
    (tmp_path / "gone.txt").write_text("x", encoding="utf-8")
    agent = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _delete("d1", "gone.txt"),
                    _call("c1", "execute", command="echo 1 > one.txt"),
                ],
            ),
            AIMessage("Done."),
        ],
    )
    parked = await agent.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    (interrupt,) = parked["__interrupt__"]
    assert [r["name"] for r in interrupt.value["action_requests"]] == [
        "delete",
        "execute",
    ]
    done = await agent.ainvoke(
        _resume(interrupt, APPROVE, REJECT), config=_config(granted=carry_grant)
    )
    assert not done.get("__interrupt__")
    assert _written(tmp_path) == []
    assert "rejected" in _tool_text(done, "c1")


async def test_a_subagents_command_is_prompted_under_a_grant(tmp_path):
    agent = _agent(
        tmp_path,
        [
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "t1",
                        "task",
                        subagent_type="general-purpose",
                        description="write a file",
                    )
                ],
            ),
            _execute("s1", "echo sub > sub.txt"),
        ],
    )
    result = await agent.ainvoke(
        {"messages": [("user", "go")]}, config=_config(granted=True)
    )
    assert len(result["__interrupt__"]) == 1
    assert _written(tmp_path) == []


@pytest.fixture
def session_grant(monkeypatch):
    from metasci_scribe.stream import display

    monkeypatch.setattr(display, "_session_auto_approve", True)
    return display


@pytest.mark.parametrize("tool", ["execute", "run_in_background"])
def test_the_client_prompts_for_a_command_under_a_session_grant(session_grant, tool):
    requests = [{"name": tool, "args": {"command": "python run.py"}}]
    seen = []

    def prompt(action_requests):
        seen.append(action_requests)
        return [REJECT]

    resolution = session_grant._resolve_hitl_approval(
        {"action_requests": requests}, prompt_fn=prompt
    )
    assert seen == [requests]
    assert resolution.human_prompted is True
    assert resolution.session_granted is False
    assert resolution.decisions == [REJECT]


def test_the_client_asks_about_a_batch_with_a_command(session_grant):
    requests = [
        {"name": "delete", "args": {"file_path": "/x"}},
        {"name": "execute", "args": {"command": "python run.py"}},
    ]
    resolution = session_grant._resolve_hitl_approval(
        {"action_requests": requests}, prompt_fn=lambda _requests: [REJECT, REJECT]
    )
    assert resolution.human_prompted is True


def test_the_client_still_approves_other_tools_under_a_session_grant(session_grant):
    def prompt(_requests):
        raise AssertionError("no prompt expected")

    resolution = session_grant._resolve_hitl_approval(
        {"action_requests": [{"name": "delete", "args": {"file_path": "/x"}}]},
        prompt_fn=prompt,
    )
    assert resolution.decisions == [APPROVE]
    assert resolution.session_granted is True
