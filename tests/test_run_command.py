"""``metasci-scribe run`` and ``/run``: the real agent graph with a fake model,
and the child process of the chat command. No provider is ever called."""

from __future__ import annotations

import asyncio
import sys
import textwrap
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from typer.testing import CliRunner

from metasci_scribe.commands import CommandContext
from metasci_scribe.llm.errors import PlanAction, PlanStop
from metasci_scribe.paths import SessionDirs, Workspace
from metasci_scribe.run import RunExit, RunState
from tests.fakes.common import FakeCommandUI

OUTLINE = """\
# Outline

## Results

### R1 · ① First result
- 핵심 내용: growth rises

### R2 · ② Second result
- 핵심 내용: growth saturates
"""

REASONING = "PRIVATE-REASONING-TEXT"


def call(name: str, call_id: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def write_para(pid: str, call_id: str) -> AIMessage:
    """What a good model does for one paragraph: write it, then mark it."""
    return AIMessage(
        REASONING,
        tool_calls=[
            call(
                "write_file",
                f"w-{call_id}",
                file_path=f"/{pid}.txt",
                content=f"{pid} text",
            ),
            call("mark_paragraph", f"m-{call_id}", paragraph_id=pid, status="done"),
        ],
    )


class Scripted(BaseChatModel):
    """Replies in order; an exception is raised. Records what it was offered
    and every message list it saw. Text only (no tool call) ends the turn."""

    replies: list = Field(default_factory=list)
    offered: list = Field(default_factory=list)
    seen: list = Field(default_factory=list)
    bound: list = Field(default_factory=list)
    profile: dict = Field(default_factory=lambda: {"max_input_tokens": 200_000})

    @property
    def _llm_type(self) -> str:
        return "scripted-run"

    def bind_tools(self, tools, **kwargs):
        self.bound = [t["name"] if isinstance(t, dict) else t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.offered.append(self.bound)
        self.seen.append(list(messages))
        reply = self.replies.pop(0) if self.replies else AIMessage("finished")
        if isinstance(reply, BaseException):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    root = tmp_path / "paper"
    root.mkdir()
    (root / "program.md").write_text("# Program\n", encoding="utf-8")
    (root / "outline.md").write_text(OUTLINE, encoding="utf-8")
    (root / "manuscript.md").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def model(monkeypatch):
    """The model every real agent of the test is built with. Only the model is
    replaced; the agent, the gate, the tools and the gateway are the real ones."""
    import metasci_scribe.agent as agent_mod

    fake = Scripted()
    monkeypatch.setattr(agent_mod, "_config", None)
    monkeypatch.setattr(agent_mod, "_chat_model", None)
    monkeypatch.setattr(agent_mod, "_chat_model_key", None)
    monkeypatch.setattr(agent_mod, "_build_chat_model", lambda cfg: fake)
    monkeypatch.setattr(agent_mod, "_load_mcp_tools_cached", _mcp_must_not_load)
    return fake


def _mcp_must_not_load(*args, **kwargs):
    raise AssertionError("the autonomous agent must not load MCP tools")


@pytest.fixture
def cli():
    from metasci_scribe.cli import app

    runner = CliRunner()

    def run(root: Path):
        return runner.invoke(app, ["run", "--workdir", str(root)])

    return run


def manuscript(root: Path, text: str = "Draft.\n") -> None:
    (root / "manuscript.md").write_text(text, encoding="utf-8")


# ------------------------------------------------------------ the CLI, exit codes


def test_complete_run_exits_0(paper, model, cli):
    manuscript(paper)
    model.replies = [write_para("R1", "1"), write_para("R2", "2")]
    result = cli(paper)
    assert result.exit_code == 0, result.output
    state = RunState.load(paper)
    assert list(state.paragraphs.values()) == ["done", "done"]
    assert (paper / "R1.txt").read_text(encoding="utf-8") == "R1 text"
    # progress: a line per mark, then the final message and the count
    assert "R1 marked done (1/2 done)" in result.output
    assert "R2 marked done (2/2 done)" in result.output
    assert "The agent marked 2 of 2 paragraphs done." in result.output
    assert "Paragraphs marked done: 2/2" in result.output


def test_progress_never_shows_the_model_text(paper, model, cli):
    manuscript(paper)
    model.replies = [write_para("R1", "1"), write_para("R2", "2")]
    result = cli(paper)
    assert REASONING not in result.output


def test_usage_limit_exits_3_and_keeps_progress(paper, model, cli):
    manuscript(paper)
    model.replies = [
        write_para("R1", "1"),
        PlanStop(PlanAction.STOP_USAGE, "usage limit reached", code="usage_limit"),
    ]
    result = cli(paper)
    assert result.exit_code == int(RunExit.USAGE_LIMIT) == 3, result.output
    assert "usage limit" in result.output
    assert "Run again" in result.output or "run again" in result.output
    assert RunState.load(paper).paragraphs == {"R1": "done", "R2": "pending"}


def test_another_plan_stop_exits_2(paper, model, cli):
    manuscript(paper)
    model.replies = [PlanStop(PlanAction.RELOGIN, "sign in again")]
    result = cli(paper)
    assert result.exit_code == int(RunExit.FAILED) == 2, result.output
    assert "/login" in result.output


def test_blocked_paragraph_exits_4(paper, model, cli):
    manuscript(paper)
    model.replies = [
        write_para("R1", "1"),
        AIMessage(
            "",
            tool_calls=[
                call(
                    "mark_paragraph",
                    "b1",
                    paragraph_id="R2",
                    status="blocked",
                    reason="no data supports saturation",
                )
            ],
        ),
    ]
    result = cli(paper)
    assert result.exit_code == int(RunExit.NEEDS_DECISION) == 4, result.output
    assert "R2: no data supports saturation" in result.output
    assert RunState.load(paper).paragraphs == {"R1": "done", "R2": "blocked"}


def test_a_folder_without_outline_exits_2_without_a_model_call(tmp_path, model, cli):
    result = cli(tmp_path)
    assert result.exit_code == 2
    assert "outline.md" in result.output
    assert model.offered == []


def test_resume_after_usage_limit_in_a_second_invocation(paper, model, cli):
    manuscript(paper)
    model.replies = [
        write_para("R1", "1"),
        PlanStop(PlanAction.STOP_USAGE, "usage limit reached"),
    ]
    first = cli(paper)
    assert first.exit_code == 3
    thread = RunState.load(paper).thread_id
    seen_before = len(model.seen)

    model.replies = [write_para("R2", "2")]
    second = cli(paper)
    assert second.exit_code == 0, second.output
    state = RunState.load(paper)
    assert state.thread_id == thread
    assert list(state.paragraphs.values()) == ["done", "done"]

    # The same thread: the second invocation's first model call sees the
    # first one's conversation, and its prompt asks only for what is left.
    first_call = model.seen[seen_before]
    texts = [str(m.content) for m in first_call]
    assert REASONING in texts, "the earlier turns were not resumed"
    prompt = [t for t in texts if "finished paragraphs: R1" in t]
    assert prompt, "the second prompt does not say R1 is finished"
    assert "- R2" in prompt[-1]


def test_ctrl_c_exits_130_and_says_run_resumes(paper, monkeypatch, cli):
    import metasci_scribe.run as run_mod

    def interrupted(root, invoke):
        raise KeyboardInterrupt

    monkeypatch.setattr(run_mod, "run_project", interrupted)
    result = cli(paper)
    assert result.exit_code == 130
    assert "run" in result.output
    assert "resume" in result.output


# ------------------------------------------------ what the autonomous agent is


def test_autonomous_agent_has_no_shell_no_mcp_no_ask_user(paper, model, cli):
    manuscript(paper)
    model.replies = [write_para("R1", "1"), write_para("R2", "2")]
    assert cli(paper).exit_code == 0
    offered = model.offered[0]
    assert "mark_paragraph" in offered
    assert "write_file" in offered
    assert "execute" not in offered
    assert "run_in_background" not in offered
    assert "ask_user" not in offered
    assert not [n for n in offered if n.startswith("mcp") or "__" in n]


def test_a_shell_call_is_refused(paper, model, cli):
    manuscript(paper)
    model.replies = [
        AIMessage(
            "",
            tool_calls=[call("execute", "x1", command="echo hi > shell-ran.txt")],
        ),
        write_para("R1", "1"),
        write_para("R2", "2"),
    ]
    assert cli(paper).exit_code == 0
    assert not (paper / "shell-ran.txt").exists()
    tool_replies = [
        str(m.content) for m in model.seen[1] if getattr(m, "type", "") == "tool"
    ]
    assert tool_replies, "the refusal never reached the model"


def test_the_config_the_agent_is_built_from_is_unattended(monkeypatch):
    from metasci_scribe.config import ScribeConfig
    from metasci_scribe.run_agent import autonomous_config

    user = ScribeConfig(
        auto_mode=False,
        auto_approve=False,
        enable_ask_user=True,
        enable_async_subagents=True,
        dangerous_mode=True,
    )
    cfg = autonomous_config(user)
    assert cfg.auto_mode
    assert cfg.auto_approve
    assert not cfg.enable_ask_user
    assert not cfg.enable_async_subagents
    assert not cfg.dangerous_mode
    # the user's own config is not changed
    assert user.dangerous_mode
    assert not user.auto_mode


def test_a_chat_session_agent_never_gets_mark_paragraph(tmp_path, model):
    from langgraph.checkpoint.memory import InMemorySaver

    from metasci_scribe.agent import create_cli_agent
    from metasci_scribe.config import ScribeConfig

    agent = create_cli_agent(
        workspace=Workspace(tmp_path),
        config=ScribeConfig(enable_async_subagents=False),
        chat_model=model,
        checkpointer=InMemorySaver(),
        with_mcp=False,
    )
    assert agent is not None
    agent.invoke(
        {"messages": [("user", "hi")]}, {"configurable": {"thread_id": "chat"}}
    )
    assert "mark_paragraph" not in model.offered[0]
    assert "think_tool" in model.offered[0]


def test_extra_tools_go_to_the_main_agent_only(tmp_path, model):
    from langgraph.checkpoint.memory import InMemorySaver

    from metasci_scribe.agent import create_cli_agent
    from metasci_scribe.config import ScribeConfig
    from metasci_scribe.run import make_mark_paragraph_tool

    create_cli_agent(
        workspace=Workspace(tmp_path),
        config=ScribeConfig(enable_async_subagents=False),
        chat_model=model,
        checkpointer=InMemorySaver(),
        extra_tools=[make_mark_paragraph_tool(tmp_path)],
        with_mcp=False,
    ).invoke({"messages": [("user", "hi")]}, {"configurable": {"thread_id": "t"}})
    assert "mark_paragraph" in model.offered[0]


def test_mcp_tools_are_still_loaded_for_a_chat_session(tmp_path, model, monkeypatch):
    """``with_mcp=False`` is the autonomous run's choice, not the default."""
    from langgraph.checkpoint.memory import InMemorySaver

    import metasci_scribe.agent as agent_mod
    from metasci_scribe.config import ScribeConfig

    asked: list[int] = []
    monkeypatch.setattr(
        agent_mod, "_load_mcp_tools_cached", lambda *a, **k: asked.append(1) or {}
    )
    agent_mod.create_cli_agent(
        workspace=Workspace(tmp_path),
        config=ScribeConfig(enable_async_subagents=False),
        chat_model=model,
        checkpointer=InMemorySaver(),
    )
    assert asked == [1]


def test_the_plan_stop_reaches_run_project_typed(paper, model):
    """The stream layer must hand run_project the PlanStop itself, not text."""
    from metasci_scribe.config import ScribeConfig
    from metasci_scribe.run_agent import real_invoke

    manuscript(paper)
    stop = PlanStop(PlanAction.STOP_USAGE, "usage limit reached")
    model.replies = [stop]
    invoke = real_invoke(
        paper,
        config=ScribeConfig(enable_async_subagents=False),
        chat_model=model,
        progress=lambda _text: None,
    )
    with pytest.raises(PlanStop) as caught:
        invoke("run-direct", "go")
    assert caught.value.action is PlanAction.STOP_USAGE


# ------------------------------------------------------------------- /run


STUB = textwrap.dedent(
    """\
    import os, signal, sys, time
    mode = sys.argv[1]
    out = sys.stdout.buffer
    def say(text):
        out.write(text.encode("utf-8") + b"\\n"); out.flush()
    if mode == "lines":
        say("한글 진행 ✓ [evidence needed]")
        say("cwd=" + os.getcwd())
        say("args=" + "|".join(sys.argv[2:]))
        sys.exit(int(sys.argv[2]))
    if mode == "badbytes":
        out.write(b"bad \\xff\\xfe bytes\\n"); out.flush()
        sys.exit(0)
    if mode == "slow":
        say("started")
        time.sleep(float(sys.argv[2]))
        say("finished")
        sys.exit(0)
    if mode == "wait":
        if sys.platform == "win32":
            signal.signal(signal.SIGBREAK, signal.default_int_handler)
        import asyncio
        say("started")
        try:
            # like the real child, whose model turn runs in asyncio.run
            asyncio.run(asyncio.sleep(60))
        except KeyboardInterrupt:
            say("got the interrupt")
            sys.exit(130)
    """
)


@pytest.fixture
def stub(tmp_path, monkeypatch):
    """Replace the child command with a small script, to test ``/run`` alone."""
    import metasci_scribe.commands.implementation.run as run_cmd

    script = tmp_path / "stub_child.py"
    script.write_text(STUB, encoding="utf-8")

    def use(*args: str) -> None:
        monkeypatch.setattr(
            run_cmd,
            "_child_argv",
            lambda root: [sys.executable, "-X", "utf8", str(script), *args],
        )

    return use


def make_ctx(root: Path):
    ui = FakeCommandUI()
    marker = object()
    ctx = CommandContext(
        agent=marker,
        thread_id="chat-thread",
        ui=ui,
        dirs=SessionDirs(Workspace(root)),
    )
    return ctx, ui, marker


def shown(ui: FakeCommandUI) -> list[str]:
    lines = list(ui.system_messages)
    lines += [getattr(r, "plain", str(r)) for r in ui.renderables]
    return lines


@pytest.fixture
def run_command():
    from metasci_scribe.commands.implementation.run import RunCommand

    return RunCommand()


def test_run_is_registered_with_the_command_manager():
    from metasci_scribe.commands import manager

    cmd = manager.get_command("/run")
    assert cmd is not None
    assert cmd.name == "/run"
    assert not cmd.requires_agent


async def test_run_shows_the_child_output_and_leaves_the_chat_alone(
    tmp_path, stub, run_command
):
    stub("lines", "0")
    ctx, ui, marker = make_ctx(tmp_path)
    await run_command.execute(ctx, [])
    lines = shown(ui)
    assert "한글 진행 ✓ [evidence needed]" in lines  # UTF-8, literal brackets
    assert f"cwd={tmp_path}" in lines  # the session's folder
    assert ctx.thread_id == "chat-thread"
    assert ctx.agent is marker
    assert ctx.session_runtime is None
    assert any("exit code 0" in line for line in lines)


@pytest.mark.parametrize(
    ("code", "word"),
    [(0, "finished"), (2, "did not finish"), (3, "usage limit"), (4, "decision")],
)
async def test_run_reports_each_run_exit_code(tmp_path, stub, run_command, code, word):
    stub("lines", str(code))
    ctx, ui, _ = make_ctx(tmp_path)
    await run_command.execute(ctx, [])
    report = [m for m in ui.system_messages if "Run ended" in m]
    assert len(report) == 1
    assert f"exit code {code}" in report[0]
    assert word in report[0]


def test_the_meanings_cover_exactly_the_run_exit_values():
    from metasci_scribe.commands.implementation.run import _INTERRUPTED, _MEANING

    assert set(_MEANING) == {int(e) for e in RunExit} | {_INTERRUPTED}


async def test_output_that_is_not_utf8_does_not_break_it(tmp_path, stub, run_command):
    stub("badbytes")
    ctx, ui, _ = make_ctx(tmp_path)
    await run_command.execute(ctx, [])
    assert "bad �� bytes" in shown(ui)


def test_the_real_child_command_runs_from_the_folder_without_path(tmp_path):
    from metasci_scribe.commands.implementation.run import _child_argv, _child_env

    argv = _child_argv(tmp_path)
    assert argv[0] == sys.executable
    assert argv[argv.index("-m") + 1] == "metasci_scribe"
    assert argv[-3:] == ["run", "--workdir", str(tmp_path)]
    assert "-I" in argv  # the paper folder is not on the child's import path
    env = _child_env()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


async def test_the_prompt_loop_keeps_running_while_the_child_runs(
    tmp_path, stub, run_command
):
    stub("slow", "1.5")
    ctx, _ui, _ = make_ctx(tmp_path)
    ticks = 0
    child_done = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not child_done.is_set():
            await asyncio.sleep(0.05)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    await run_command.execute(ctx, [])
    child_done.set()
    await beat
    assert ticks >= 10, f"the event loop was blocked (ticks={ticks})"


async def test_output_is_shown_while_the_child_is_still_running(
    tmp_path, stub, run_command
):
    stub("slow", "1.5")
    ctx, ui, _ = make_ctx(tmp_path)
    task = asyncio.create_task(run_command.execute(ctx, []))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if "started" in shown(ui):
            break
    assert "started" in shown(ui)
    assert not task.done()
    assert "finished" not in shown(ui)
    await task
    assert "finished" in shown(ui)


async def test_a_second_run_is_refused_while_one_is_running(
    tmp_path, stub, run_command, monkeypatch
):
    import metasci_scribe.commands.implementation.run as run_cmd

    stub("slow", "1.0")
    ctx, _ui, _ = make_ctx(tmp_path)
    first = asyncio.create_task(run_command.execute(ctx, []))
    await asyncio.sleep(0.3)
    started: list[int] = []
    real_popen = run_cmd.subprocess.Popen
    monkeypatch.setattr(
        run_cmd.subprocess,
        "Popen",
        lambda *a, **k: started.append(1) or real_popen(*a, **k),
    )
    ctx2, ui2, _ = make_ctx(tmp_path)
    await run_command.execute(ctx2, [])
    assert started == []
    assert any("already in progress" in m for m in ui2.system_messages)
    await first
    # and once it ended, the next one runs
    await run_command.execute(ctx2, [])
    assert started == [1]


async def test_a_failed_start_is_reported_and_frees_the_command(
    tmp_path, run_command, monkeypatch
):
    import metasci_scribe.commands.implementation.run as run_cmd

    def broken(*args, **kwargs):
        raise OSError("no such interpreter")

    monkeypatch.setattr(run_cmd.subprocess, "Popen", broken)
    ctx, ui, _ = make_ctx(tmp_path)
    await run_command.execute(ctx, [])
    assert any("Could not start" in m for m in ui.system_messages)
    assert run_cmd._active is False


async def test_ctrl_c_is_passed_to_the_child_and_the_result_reported(
    tmp_path, stub, run_command
):
    stub("wait")
    ctx, ui, _ = make_ctx(tmp_path)
    task = asyncio.create_task(run_command.execute(ctx, []))
    for _ in range(100):
        await asyncio.sleep(0.1)
        if "started" in shown(ui):
            break
    assert "started" in shown(ui)
    task.cancel()
    await asyncio.wait_for(task, timeout=30)
    assert not task.cancelled()  # back to the prompt, not a closed chat
    lines = shown(ui)
    assert "got the interrupt" in lines
    assert any("exit code 130" in line and "Ctrl+C" in line for line in lines)
    import metasci_scribe.commands.implementation.run as run_cmd

    assert run_cmd._active is False


async def test_the_real_child_starts_without_path_and_reports_its_exit(
    tmp_path, run_command
):
    """No stub: ``python -I -m metasci_scribe run`` in a folder with no outline
    refuses before any model call, and ``/run`` shows why and the exit code 2."""
    ctx, ui, _ = make_ctx(tmp_path)
    await run_command.execute(ctx, [])
    lines = shown(ui)
    assert any("outline.md" in line for line in lines), lines
    assert any("exit code 2" in line for line in lines), lines
