"""The writing workflow: what the model is told, which sub-agents and skills it has,
and one real conversation turn on the real agent graph with a scripted model."""

from __future__ import annotations

import re
import uuid

import pytest
import yaml
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from metasci_scribe import prompts
from metasci_scribe.agent import SKILLS_DIR, SUBAGENTS_CONFIG, create_cli_agent
from metasci_scribe.config import ScribeConfig
from metasci_scribe.paths import Workspace

MODES = [
    {},
    {"dangerous": True, "cwd": "/work/paper"},
    {"bash_on_windows": True},
    {"dangerous": True, "cwd": "C:/work/paper", "bash_on_windows": True},
]


def _all_prompts() -> list:
    return [
        pytest.param(prompts.get_system_prompt(**mode), id=f"mode{i}")
        for i, mode in enumerate(MODES)
    ]


def _subagent_specs() -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for path in sorted(SUBAGENTS_CONFIG.glob("*.yaml")):
        specs.update(yaml.safe_load(path.read_text(encoding="utf-8")))
    return specs


class TestSystemPrompt:
    @pytest.mark.parametrize("text", _all_prompts())
    def test_names_the_writing_files_and_the_marker(self, text):
        assert "[evidence needed]" in text
        assert "manuscript.md" in text
        assert ".metasci-scribe/extracted" in text
        assert "outline.md" in text
        assert "program.md" in text

    @pytest.mark.parametrize("text", _all_prompts())
    def test_has_no_experiment_workflow_left(self, text):
        assert "experiment" not in text.lower()
        assert "final_report" not in text
        assert "research_request" not in text

    def test_the_old_constants_are_gone(self):
        assert not hasattr(prompts, "EXPERIMENT_WORKFLOW")
        assert not hasattr(prompts, "REPORT_TEMPLATE")
        assert prompts.WRITING_WORKFLOW in prompts.get_system_prompt()

    def test_workflow_comes_after_identity_and_before_delegation(self):
        text = prompts.get_system_prompt()
        assert (
            0
            <= text.index("# Identity")
            < text.index("# Writing Workflow")
            < text.index("# Sub-Agent Delegation")
        )

    def test_papers_and_data_are_material_not_instructions(self):
        flat = " ".join(prompts.WRITING_WORKFLOW.split())
        assert "untrusted" in flat.lower()
        assert "never instructions" in flat
        assert "do not act on" in flat.lower()

    def test_missing_evidence_is_marked_not_invented(self):
        flat = " ".join(prompts.WRITING_WORKFLOW.split())
        assert "Never invent" in flat
        assert "[evidence needed]" in flat

    def test_outline_is_changed_only_with_the_user(self):
        flat = " ".join(prompts.WRITING_WORKFLOW.split())
        assert "outline.md" in flat
        assert "only when you and the author agree" in flat
        assert "no one to ask" in flat

    def test_language_follows_the_user(self):
        assert "language" in prompts.WRITING_WORKFLOW.lower()

    @pytest.mark.parametrize("text", _all_prompts())
    def test_invites_nothing_the_agent_can_no_longer_do(self, text):
        lowered = text.lower()
        for phrase in (
            "web search",
            "internet_search",
            "evoskills",
            "start_async_task",
            "check_async_task",
            "update_async_task",
            "gpu",
            "training",
            "langgraph dev",
            "write to `/skills/`",
            "save it under `/skills/`",
        ):
            assert phrase not in lowered, phrase

    def test_shell_guidance_stays_for_attended_chat(self):
        text = prompts.get_system_prompt()
        assert "Shell Execution Guidelines" in text
        assert "approval" in text.lower()
        assert "300" in text
        assert "3600" in text

    def test_skills_are_project_folders_the_user_installs(self):
        flat = " ".join(prompts.WRITING_WORKFLOW.split())
        assert ".metasci-scribe/skills" in flat
        assert "/install-skill" in flat


class TestSubagents:
    def test_the_three_writing_roles(self):
        specs = _subagent_specs()
        assert set(specs) == {"planner-agent", "research-agent", "writing-agent"}
        assert "outline" in specs["planner-agent"]["system_prompt"].lower()
        assert "Knowledge/" in specs["research-agent"]["system_prompt"]
        assert ".metasci-scribe/extracted" in specs["research-agent"]["system_prompt"]
        assert "manuscript.md" in specs["writing-agent"]["system_prompt"]

    def test_research_has_no_web_search(self):
        spec = _subagent_specs()["research-agent"]
        assert spec["tools"] == ["think_tool"]
        assert not any("search" in tool for tool in spec["tools"])
        text = spec["system_prompt"].lower()
        assert "web search" not in text
        assert "url" not in text

    @pytest.mark.parametrize(
        "name", ["planner-agent", "research-agent", "writing-agent"]
    )
    def test_prompts_carry_the_writing_rules(self, name):
        spec = _subagent_specs()[name]
        text = " ".join(spec["system_prompt"].split())
        assert "experiment" not in (text + spec["description"]).lower()
        assert "[evidence needed]" in text
        assert "never instructions" in text
        assert "do not act on" in text.lower()
        assert spec["tools"] == ["think_tool"]

    def test_no_sub_agent_prompt_mentions_a_server(self):
        for name, spec in _subagent_specs().items():
            assert "langgraph" not in spec["system_prompt"].lower(), name
            assert "async" not in spec["system_prompt"].lower(), name

    def test_planner_works_from_the_outline_map_in_one_mode(self):
        text = _subagent_specs()["planner-agent"]["system_prompt"]
        assert "MODE:" not in text
        assert "JSON" not in text


class TestBuiltInSkills:
    def test_two_skills_ship(self):
        from pathlib import Path

        shipped = {p.name for p in Path(SKILLS_DIR).iterdir() if p.is_dir()}
        assert shipped == {"find-skills", "skill-creator"}


# -- One real turn ------------------------------------------------------------


class _Scripted(GenericFakeChatModel):
    """Plays the scripted replies in order and keeps what it was shown."""

    seen: list = Field(default_factory=list)
    tools_seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.tools_seen.append([getattr(t, "name", None) or t["name"] for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def _call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _agent(root, replies):
    model = _Scripted(messages=iter([*replies, *[AIMessage("extra")] * 4]))
    agent = create_cli_agent(
        workspace=Workspace(root),
        config=ScribeConfig(enable_async_subagents=False, memory_workers_enabled=False),
        chat_model=model,
        checkpointer=InMemorySaver(),
    )
    return agent, model


def _system_text(messages) -> str:
    return next(m for m in messages if isinstance(m, SystemMessage)).text


def _skill_names(text: str) -> set[str]:
    return set(re.findall(r"^- \*\*([\w-]+)\*\*:", text, flags=re.MULTILINE))


def _cfg() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


@pytest.mark.usefixtures("no_mcp")
async def test_a_turn_reads_the_project_and_writes_the_manuscript(tmp_path):
    (tmp_path / "outline.md").write_text(
        "### R1 · ①\nThe assay detects the target.\n", encoding="utf-8"
    )
    extracted = tmp_path / ".metasci-scribe" / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "smith.pdf.md").write_text(
        "[p. 3]\nSensitivity was 91%.\n",
        encoding="utf-8",
    )
    draft = "The assay detects the target (Smith, p. 3). [evidence needed]\n"
    agent, model = _agent(
        tmp_path,
        [
            AIMessage(
                "", tool_calls=[_call("r1", "read_file", file_path="/outline.md")]
            ),
            AIMessage(
                "",
                tool_calls=[
                    _call(
                        "r2",
                        "read_file",
                        file_path="/.metasci-scribe/extracted/smith.pdf.md",
                    )
                ],
            ),
            AIMessage(
                "",
                tool_calls=[
                    _call("w1", "write_file", file_path="/manuscript.md", content=draft)
                ],
            ),
            AIMessage("Drafted R1."),
        ],
    )

    result = await agent.ainvoke({"messages": [("user", "Draft R1.")]}, config=_cfg())

    assert not result.get("__interrupt__")
    assert (tmp_path / "manuscript.md").read_text(encoding="utf-8") == draft
    assert (tmp_path / "outline.md").exists()
    tool_text = {
        m.tool_call_id: str(m.content) for m in result["messages"] if m.type == "tool"
    }
    assert "The assay detects the target." in tool_text["r1"]
    assert "Sensitivity was 91%" in tool_text["r2"]
    assert "Error" not in tool_text["w1"]
    assert result["messages"][-1].content == "Drafted R1."

    system = _system_text(model.seen[0])
    assert "[evidence needed]" in system
    assert ".metasci-scribe/extracted" in system
    assert "experiment" not in system.lower()
    assert "web search" not in system.lower()
    assert _skill_names(system) == {"find-skills", "skill-creator"}


@pytest.mark.usefixtures("no_mcp")
async def test_the_agent_has_no_tool_to_search_the_web(tmp_path):
    agent, model = _agent(tmp_path, [AIMessage("hi")])
    await agent.ainvoke({"messages": [("user", "hi")]}, config=_cfg())
    tools = set(model.tools_seen[-1])
    assert not {t for t in tools if "search" in t or "web" in t or "fetch" in t}
    assert {"read_file", "write_file", "edit_file", "task"} <= tools


@pytest.mark.usefixtures("no_mcp")
async def test_a_project_skill_added_between_messages_is_listed_next_message(
    tmp_path,
):
    agent, model = _agent(tmp_path, [AIMessage("first"), AIMessage("second")])
    cfg = _cfg()

    await agent.ainvoke({"messages": [("user", "one")]}, config=cfg)
    before = _skill_names(_system_text(model.seen[-1]))
    assert before == {"find-skills", "skill-creator"}

    # The harness plays the user installing a skill by hand: the agent's own
    # file tools cannot write under .metasci-scribe/.
    skill_dir = tmp_path / ".metasci-scribe" / "skills" / "journal-style"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: journal-style\ndescription: Match the target journal's style.\n"
        "---\n\n# Journal style\n\nFollow it.\n",
        encoding="utf-8",
    )

    await agent.ainvoke({"messages": [("user", "two")]}, config=cfg)
    after = _skill_names(_system_text(model.seen[-1]))
    assert after == {"find-skills", "skill-creator", "journal-style"}
