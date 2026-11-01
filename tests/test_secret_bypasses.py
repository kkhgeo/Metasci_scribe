"""Regression tests for ways around the sign-in guard, each through the real tool
or backend, with a fake home and a fake token."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from deepagents.backends import CompositeBackend
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from metasci_scribe import backends, paths
from metasci_scribe.agent import _get_default_backend, create_cli_agent
from metasci_scribe.backends import (
    CustomSandboxBackend,
    MergedSkillsBackend,
    build_autoskill_agent_backend,
    build_memory_agent_backend,
    build_memory_worker_backend,
)
from metasci_scribe.config import ScribeConfig
from metasci_scribe.paths import Workspace
from metasci_scribe.tools.skill_manager import make_skill_manager_tool
from metasci_scribe.tools.skills_manager import install_skill

SECRET = "refresh-secret-value"
TOKEN_REL = ".metasci-scribe/auth/chatgpt.json"
SKILL_MD = "---\nname: helper\ndescription: A helper.\n---\n\nBody.\n"


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "home"
    auth = root / ".metasci-scribe" / "auth"
    auth.mkdir(parents=True)
    (auth / "chatgpt.json").write_text(
        json.dumps({"refresh_token": SECRET}), encoding="utf-8"
    )
    (root / "notes.txt").write_text("refresh_token mentioned", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_DIR", root / ".metasci-scribe")
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", root / ".metasci-scribe" / "skills")
    monkeypatch.setattr(paths, "MEMORIES_DIR", root / ".metasci-scribe" / "memories")
    return root


@pytest.fixture
def token(home) -> Path:
    return home / TOKEN_REL


@pytest.fixture
def project(home) -> Path:
    folder = home / "paper"
    folder.mkdir()
    return folder


def _assert_no_copy_of_the_secret(root: Path) -> None:
    for path in root.rglob("*"):
        if (
            path.is_file()
            and not path.is_symlink()
            and "auth" not in path.relative_to(root).parts[:2]
        ):
            assert SECRET not in path.read_text(encoding="utf-8", errors="replace"), (
                path
            )


def _assert_refused(result: Any) -> None:
    assert getattr(result, "error", None), result
    assert SECRET not in str(result)


def _read_is_refused(backend, key: str) -> None:
    try:
        result = backend.read(key)
    except ValueError:
        return
    _assert_refused(result)


def _link(link: Path, target: Path, *, folder: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=folder)
    except OSError:
        pytest.skip("this account may not create symlinks")


# ---------------------------------------------------------------- skill_manager


def test_skill_tool_will_not_install_a_folder_holding_the_auth_folder(home):
    backend = CustomSandboxBackend(root_dir=str(home), virtual_mode=True)
    assert not backend.write("/.metasci-scribe/SKILL.md", SKILL_MD).error
    tool = make_skill_manager_tool(Workspace(home))

    answer = tool.invoke({"action": "install", "source": "/.metasci-scribe"})

    assert "sign-in" in answer
    assert SECRET not in answer
    assert not (home / ".metasci-scribe" / "skills" / "helper").exists()
    _assert_no_copy_of_the_secret(home)


def test_install_refuses_a_parent_of_the_data_folder(home):
    (home / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    result = install_skill(str(home), workspace=Workspace(home))
    assert result["success"] is False
    _assert_no_copy_of_the_secret(home)


def test_install_refuses_the_auth_folder_and_anything_in_it(home, token):
    (token.parent / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    inner = token.parent / "inner"
    inner.mkdir()
    (inner / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    for source in (token.parent, inner):
        result = install_skill(str(source), workspace=Workspace(home))
        assert result["success"] is False, source
    _assert_no_copy_of_the_secret(home)


def test_install_refuses_a_link_to_the_auth_folder(home, token, project):
    (token.parent / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    _link(project / "looks-fine", token.parent, folder=True)
    result = install_skill(str(project / "looks-fine"), workspace=Workspace(project))
    assert result["success"] is False
    _assert_no_copy_of_the_secret(home)


def test_dangerous_mode_install_from_another_workspace_is_refused(home, token, project):
    (home / ".metasci-scribe" / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    tool = make_skill_manager_tool(Workspace(project))
    answer = tool.invoke({"action": "install", "source": str(home / ".metasci-scribe")})
    assert "sign-in" in answer
    _assert_no_copy_of_the_secret(home)


def _skill_folder(root: Path) -> Path:
    skill = root / "helper"
    skill.mkdir()
    (skill / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    return skill


def test_install_does_not_follow_a_link_to_a_file(home, token, project):
    skill = _skill_folder(project)
    _link(skill / "notes.json", token)
    result = install_skill(str(skill), workspace=Workspace(project))
    assert result["success"] is True
    installed = Path(result["path"])
    assert (installed / "SKILL.md").is_file()
    assert not os.path.lexists(installed / "notes.json")
    _assert_no_copy_of_the_secret(home)


def test_install_does_not_follow_a_link_to_a_folder(home, token, project):
    skill = _skill_folder(project)
    _link(skill / "refs", token.parent, folder=True)
    result = install_skill(str(skill), workspace=Workspace(project))
    assert result["success"] is True
    assert not os.path.lexists(Path(result["path"]) / "refs")
    _assert_no_copy_of_the_secret(home)


def test_install_from_github_does_not_follow_a_link(home, token, project, monkeypatch):
    def fake_clone(_repo, _ref, clone_dir):
        repo = Path(clone_dir)
        repo.mkdir(parents=True)
        (repo / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
        _link(repo / "notes.json", token)

    monkeypatch.setattr("metasci_scribe.tools.skills_manager._clone_repo", fake_clone)
    monkeypatch.setattr(
        "metasci_scribe.tools.skills_manager._current_commit",
        lambda *_a, **_k: None,
        raising=False,
    )
    result = install_skill("owner/repo", workspace=Workspace(project))
    assert result["success"] is True, result
    assert not os.path.lexists(Path(result["path"]) / "notes.json")
    _assert_no_copy_of_the_secret(home)


def test_failed_copy_leaves_nothing_and_keeps_the_old_install(
    home, project, monkeypatch
):
    first = install_skill(str(_skill_folder(project)), workspace=Workspace(project))
    assert first["success"] is True
    installed = Path(first["path"])
    (installed / "old.txt").write_text("kept", encoding="utf-8")
    real = shutil.copytree

    def partly(source, target, **kwargs):
        real(source, target, **kwargs)
        raise shutil.Error([("a", "b", "boom")])

    monkeypatch.setattr("metasci_scribe.tools.skills_manager.shutil.copytree", partly)
    result = install_skill(str(project / "helper"), workspace=Workspace(project))

    assert result["success"] is False
    assert (installed / "old.txt").is_file()
    assert sorted(p.name for p in installed.parent.iterdir() if p.is_dir()) == [
        "helper"
    ]


# ---------------------------------------------------------------- // paths


def _device_forms(token: Path) -> list[str]:
    posix = token.as_posix()
    forms = [f"//{posix.lstrip('/')}", f"//?/{posix}", f"\\\\?\\{token}"]
    if re.match(r"[A-Za-z]:", posix):
        rest = posix[3:]
        share = f"{posix[0]}$/{rest}"
        forms += [
            f"//localhost/{share}",
            f"//127.0.0.1/{share}",
            f"//./{posix}",
            f"/\\?\\{token}",
        ]
    return forms


@pytest.mark.parametrize("dangerous", [True, False], ids=["dangerous", "default"])
def test_double_slash_paths_are_refused_for_every_operation(
    home, token, project, dangerous
):
    backend = CustomSandboxBackend(
        root_dir=str(project), virtual_mode=True, dangerous=dangerous
    )
    for form in _device_forms(token):
        _read_is_refused(backend, form)
        _assert_refused(backend.write(form + ".new", "x"))
        _assert_refused(backend.edit(form, SECRET, "x"))
        _assert_refused(backend.ls(form))
        assert not backend.grep("refresh", form).matches
        assert backend.download_files([form])[0].error
    assert SECRET in token.read_text(encoding="utf-8")


def test_auth_check_compares_by_file_identity(home, token, project):
    # A spelling the string check cannot see, reaching the same folder.
    assert paths.is_auth_path(f"//?/{token}") or sys.platform != "win32"
    assert paths.is_auth_path(token)
    assert paths.is_auth_path(token.parent)
    assert paths.is_auth_path(token.parent / "not-yet-there.json")
    assert not paths.is_auth_path(home / "notes.txt")
    assert not paths.is_auth_path(token.parent.with_name("auth-notes"))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows device paths")
def test_auth_check_sees_through_the_device_prefix_without_the_key_check(
    home, token, project, monkeypatch
):
    monkeypatch.setattr(backends, "_DEVICE_PATH", re.compile(r"^$(?!.)x"))
    backend = CustomSandboxBackend(root_dir=str(project), dangerous=True)
    _assert_refused(backend.read(f"//?/{token.as_posix()}"))


class _Scripted(BaseChatModel):
    script: list[Any]
    offered: list[set[str]]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.offered.append(
            {t.get("name") if isinstance(t, dict) else t.name for t in tools}
        )
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self.script.pop(0) if self.script else AIMessage("done")
        return ChatResult(generations=[ChatGeneration(message=message)])


def _call(call_id: str, name: str, **args) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.mark.usefixtures("no_mcp")
@pytest.mark.parametrize("tool", ["readFile", "grep", "ls"])
def test_code_interpreter_reads_nothing_through_double_slash(
    home, token, project, tool
):
    form = f"//?/{token.as_posix()}"
    argument = {
        "readFile": {"file_path": form},
        "grep": {"pattern": "refresh", "path": form},
        "ls": {"path": form},
    }[tool]
    code = (
        "let out;\n"
        f"try {{ out = await tools.{tool}({json.dumps(argument)}); }}\n"
        "catch (e) { out = String(e); }\n"
        "JSON.stringify(out)"
    )
    model = _Scripted(
        script=[AIMessage("", tool_calls=[_call("c1", "code_interpreter", code=code)])],
        offered=[],
    )
    agent = create_cli_agent(
        workspace=Workspace(project),
        config=ScribeConfig(
            enable_async_subagents=False,
            memory_workers_enabled=False,
            dangerous_mode=True,
        ),
        chat_model=model,
        checkpointer=InMemorySaver(),
    )
    result = asyncio.run(
        agent.ainvoke(
            {"messages": [HumanMessage("go")]},
            {"configurable": {"thread_id": "t", "hitl_suppressed": False}},
        )
    )
    answers = [str(m.content) for m in result["messages"] if m.type == "tool"]
    assert answers
    assert SECRET not in " ".join(answers)


# ---------------------------------------------------------------- every backend


@pytest.mark.parametrize(
    "build",
    [
        build_memory_agent_backend,
        build_memory_worker_backend,
    ],
    ids=["memory-agent", "memory-worker"],
)
def test_memory_backends_refuse_it_in_a_home_workspace(home, token, build):
    backend = build(workspace_dir=home, memory_dir=home / "memories")
    _read_is_refused(backend, f"/{TOKEN_REL}")
    _read_is_refused(backend, f"//?/{token.as_posix()}")
    _assert_refused(backend.ls("/.metasci-scribe/auth"))
    assert not [
        item
        for item in backend.ls("/.metasci-scribe").entries
        if item["path"].rstrip("/").endswith("/auth")
    ]
    assert not [
        item
        for item in backend.glob("**/*.json", "/.metasci-scribe").matches
        if "auth" in item["path"]
    ]
    hits = backend.grep("refresh", "/.metasci-scribe").matches
    assert SECRET not in str(hits)


def test_memory_route_refuses_a_link_to_it(home, token):
    memories = home / "memories"
    memories.mkdir()
    _link(memories / "looks-fine.md", token)
    backend = build_memory_agent_backend(workspace_dir=home, memory_dir=memories)
    _read_is_refused(backend, "/memories/looks-fine.md")


def test_autoskill_routes_refuse_it(home, token):
    proposals = home / "proposals"
    skills = home / "skills"
    memories = home / "memories"
    for folder in (proposals, skills, memories):
        folder.mkdir()
    _link(proposals / "a.json", token)
    _link(skills / "b.json", token)
    _link(memories / "c.json", token)
    backend = build_autoskill_agent_backend(
        memory_dir=memories, proposals_dir=proposals, skills_dir=skills
    )
    for key in ("/autoskill-proposals/a.json", "/skills/b.json", "/memories/c.json"):
        _read_is_refused(backend, key)


def test_skills_backend_primary_tier_is_guarded(home, token):
    primary = home / "skills-primary"
    primary.mkdir()
    secondary = home / "skills-builtin"
    secondary.mkdir()
    _link(primary / "looks-fine.json", token)
    backend = MergedSkillsBackend(
        primary_dir=str(primary), secondary_dir=str(secondary)
    )
    _read_is_refused(backend, "/looks-fine.json")


def test_every_filesystem_backend_goes_through_the_guard():
    from deepagents.backends import FilesystemBackend

    source = Path(backends.__file__).read_text(encoding="utf-8")
    assert not re.search(r"(?<![A-Za-z])FilesystemBackend\(root_dir", source)
    for name in (
        "GuardedFilesystemBackend",
        "ReadOnlyFilesystemBackend",
        "MemoryFilesystemBackend",
        "CustomSandboxBackend",
        "AutoskillProposalSandboxBackend",
    ):
        cls = getattr(backends, name)
        assert issubclass(cls, FilesystemBackend)
        assert issubclass(cls, backends._GuardedFiles), name


def test_composite_default_refuses_a_double_slash(home, token, project):
    backend = CompositeBackend(
        default=CustomSandboxBackend(root_dir=str(project), dangerous=True), routes={}
    )
    _read_is_refused(backend, f"//?/{token.as_posix()}")


# ---------------------------------------------------------------- memory agents


def test_memory_agents_have_no_shell():
    from metasci_scribe.memory.agents._factory import (
        MEMORY_MAINTENANCE_EXCLUDED_TOOLS,
    )
    from metasci_scribe.memory.agents.autoskills import _AUTOSKILLS_EXCLUDED_TOOLS
    from metasci_scribe.memory.agents.memory_worker import (
        _MEMORY_WORKER_EXCLUDED_TOOLS,
    )

    for excluded in (
        MEMORY_MAINTENANCE_EXCLUDED_TOOLS,
        _AUTOSKILLS_EXCLUDED_TOOLS,
        _MEMORY_WORKER_EXCLUDED_TOOLS,
    ):
        assert {"execute", "run_in_background"} <= excluded


def test_autoskills_prompt_does_not_invite_the_shell():
    from metasci_scribe.memory.agents.autoskills import _autoskills_system_prompt

    prompt = _autoskills_system_prompt()
    assert "execute" not in prompt
    assert "shell" not in prompt.lower()


def test_built_autoskills_graph_offers_no_shell(tmp_path, monkeypatch):
    from metasci_scribe.memory.agents.autoskills import build_autoskills_graph

    model = _Scripted(script=[], offered=[])
    monkeypatch.setattr(
        "metasci_scribe.agent._ensure_auxiliary_chat_model", lambda: model
    )
    (tmp_path / "ws").mkdir()
    graph = build_autoskills_graph(
        workspace=Workspace(tmp_path / "ws"), memory_dir=tmp_path / "memories"
    )
    asyncio.run(graph.ainvoke({"messages": [HumanMessage("go")]}))
    assert model.offered
    for offered in model.offered:
        assert not {"execute", "run_in_background"} & offered
    assert "read_file" in model.offered[0]


# ---------------------------------------------------------------- workspace root


def test_workspace_root_check(home, project):
    data_dir = home / ".metasci-scribe"
    assert paths.holds_data_dir(home) is True
    assert paths.holds_data_dir(data_dir) is True
    assert paths.holds_data_dir(home.parent) is True
    assert paths.holds_data_dir(project) is False
    assert paths.holds_data_dir(data_dir / "skills") is False
    paths.require_workspace_root(project)
    with pytest.raises(paths.WorkspaceRootError) as caught:
        paths.require_workspace_root(home)
    assert str(data_dir) in str(caught.value)
    assert isinstance(caught.value, ValueError)


def test_workspace_root_check_follows_links(home, tmp_path):
    _link(tmp_path / "shortcut", home, folder=True)
    assert paths.holds_data_dir(tmp_path / "shortcut") is True


def test_agent_build_refuses_a_workspace_holding_the_data_folder(home):
    with pytest.raises(paths.WorkspaceRootError):
        _get_default_backend(Workspace(home))
    with pytest.raises(paths.WorkspaceRootError):
        create_cli_agent(
            workspace=Workspace(home),
            config=ScribeConfig(),
            chat_model=_Scripted(script=[], offered=[]),
            checkpointer=InMemorySaver(),
        )


def test_memory_graphs_refuse_a_workspace_holding_the_data_folder(home):
    from metasci_scribe.memory.agents._factory import resolve_memory_agent_paths

    with pytest.raises(paths.WorkspaceRootError):
        resolve_memory_agent_paths(workspace=Workspace(home))


def test_cli_refuses_to_start_there(home, monkeypatch, capsys):
    import typer

    from tests.test_cli_workspace_resolution import _config, _run

    monkeypatch.chdir(home)
    with pytest.raises(typer.Exit):
        _run(monkeypatch, _config())


def test_resume_does_not_move_into_such_a_workspace(home, project):
    from types import SimpleNamespace

    from metasci_scribe.commands.implementation.session import ResumeCommand
    from metasci_scribe.paths import SessionDirs

    messages = []
    start = SessionDirs(Workspace(project))
    ctx = SimpleNamespace(
        dirs=start,
        thread_id="before",
        ui=SimpleNamespace(
            append_system=lambda text, style=None: messages.append(text)
        ),
    )
    command = ResumeCommand()

    async def resolve(_prefix, _ctx):
        return "other"

    class _Gateway:
        async def get_thread_metadata(self, _thread):
            return {"workspace_dir": str(home)}

    command._resolve_thread_id = resolve
    import metasci_scribe.commands.implementation.session as session

    original = session._graph_gateway
    session._graph_gateway = lambda _ctx: _Gateway()
    try:
        asyncio.run(command.execute(ctx, ["other"]))
    finally:
        session._graph_gateway = original
    assert ctx.dirs is start
    assert ctx.thread_id == "before"
    assert any(str(home) in text for text in messages)
