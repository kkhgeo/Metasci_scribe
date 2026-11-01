"""Child runs carry their folders, and the server refuses runs meant for others."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from metasci_scribe.langgraph_dev.folder_check import (
    RunFolderMismatchError,
    check_run_folders,
    folder_checked,
)
from metasci_scribe.paths import SessionDirs, Workspace
from metasci_scribe.sessions import FOLDER_GRAPH_IDS

RUN = "20260930_120000"


@pytest.fixture
def served(workspace, monkeypatch) -> SessionDirs:
    """A server pinned to a run folder of ``workspace``."""
    dirs = SessionDirs(workspace, workspace.runs_dir / RUN)
    monkeypatch.setattr(
        "metasci_scribe.langgraph_dev.folder_check.process_session_dirs", lambda: dirs
    )
    return dirs


def _config(dirs: SessionDirs | None) -> dict:
    return {"configurable": dirs.metadata() if dirs is not None else {}}


# ---------------------------------------------------------------------------
# The server's check
# ---------------------------------------------------------------------------


def test_runs_that_forward_no_folders_are_not_checked(served):
    check_run_folders(_config(None), works_in_folder=True)


def test_runs_for_the_served_folders_run(served, tmp_path):
    check_run_folders(_config(served), works_in_folder=True)

    # The same folders written another way: a symlink and a trailing slash.
    link = tmp_path / "link"
    link.symlink_to(served.workspace.root, target_is_directory=True)
    check_run_folders(
        {
            "configurable": {
                "workspace_dir": f"{link}/",
                "run_dir": str(link / "runs" / RUN),
            }
        },
        works_in_folder=True,
    )


def test_runs_for_another_workspace_are_refused(served, tmp_path):
    other = SessionDirs(Workspace(tmp_path / "other"))
    for works_in_folder in (True, False):
        with pytest.raises(RunFolderMismatchError) as exc:
            check_run_folders(_config(other), works_in_folder=works_in_folder)
        assert str(served.workspace.root) in str(exc.value)
        assert str(other.workspace.root) in str(exc.value)


def test_check_run_folders_refuses_unreadable_folders(served):
    """Present but unreadable is a mismatch, not an unchecked run."""
    for workspace_dir in ("/tmp/evil\x00x", 123):
        with pytest.raises(RunFolderMismatchError, match="cannot read"):
            check_run_folders(
                {"configurable": {"workspace_dir": workspace_dir}},
                works_in_folder=True,
            )


def test_check_run_folders_refuses_run_dir_outside_workspace(served, tmp_path):
    with pytest.raises(RunFolderMismatchError):
        check_run_folders(
            {
                "configurable": {
                    "workspace_dir": served.workspace.key,
                    "run_dir": str(tmp_path / "elsewhere" / "runs" / RUN),
                }
            },
            works_in_folder=True,
        )


def test_folder_checked_factory_matches_server_signature():
    """langgraph-api classifies the factory by its ``runtime`` annotation."""
    import inspect
    import typing

    from langgraph_sdk.runtime import ServerRuntime

    factory = folder_checked(object(), graph_id="writing-agent")
    assert list(inspect.signature(factory).parameters) == ["config", "runtime"]
    assert typing.get_type_hints(factory)["runtime"] is ServerRuntime


def test_only_graphs_working_in_a_folder_check_the_run_folder(served):
    other_run = SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    with pytest.raises(RunFolderMismatchError):
        check_run_folders(_config(other_run), works_in_folder=True)
    # Scheduled tasks and memory workers work in the root, whatever run
    # folder a session holds the server for.
    check_run_folders(_config(other_run), works_in_folder=False)
    check_run_folders(_config(SessionDirs(served.workspace)), works_in_folder=False)


def test_factory_checks_runs_only_and_returns_the_prebuilt_graph(served, tmp_path):
    graph = object()
    factory = folder_checked(graph, graph_id="writing-agent")
    mismatch = _config(SessionDirs(Workspace(tmp_path / "other")))

    reading = SimpleNamespace(execution_runtime=None)
    assert asyncio.run(factory(mismatch, reading)) is graph

    running = SimpleNamespace(execution_runtime=object())
    assert asyncio.run(factory(_config(served), running)) is graph
    with pytest.raises(RunFolderMismatchError):
        asyncio.run(factory(mismatch, running))


def test_memory_workers_and_the_linker_forward_the_root(workspace, tmp_path):
    from metasci_scribe.memory.launch import (
        _memory_worker_run_payload,
        _observation_linker_run_payload,
    )
    from tests.fakes.memory import linker_context, memory_source_context

    worker = _memory_worker_run_payload(
        context=memory_source_context(
            memory_dir=tmp_path / "memories", workspace_dir=workspace.root
        ),
        thread_id="t1",
    )
    linker = _observation_linker_run_payload(
        context=linker_context(
            memory_dir=tmp_path / "memories",
            workspace_dir=workspace.root,
            observation_ids=("o1",),
        ),
        thread_id="t2",
    )
    for payload in (worker, linker):
        assert payload["config"]["configurable"]["workspace_dir"] == workspace.key


def test_manual_autoskills_runs_forward_the_root(workspace, monkeypatch):
    import metasci_scribe.memory.autoskills.schedule as schedule

    client = MagicMock()
    client.threads.create.return_value = {"thread_id": "t1"}
    client.runs.create.return_value = {"run_id": "r1"}
    monkeypatch.setattr(schedule, "get_langgraph_sync_client", lambda **_k: client)

    config = SimpleNamespace(
        memory_skill_synthesis_mode=SimpleNamespace(value="propose"),
        memory_skill_synthesis_cadence=SimpleNamespace(value="daily"),
        memory_skill_synthesis_time="03:00",
    )
    monkeypatch.setattr(schedule, "langgraph_dev_url", lambda _c: "http://x")
    schedule.run_autoskill_now(config, workspace_dir=workspace.root)

    configurable = client.runs.create.call_args.kwargs["config"]["configurable"]
    assert configurable["workspace_dir"] == workspace.key


def test_registered_graphs_check_the_folders_they_work_in(served, monkeypatch, request):
    """Every registered graph refuses another workspace. Graphs that work in
    the session's folder also refuse another run folder; graphs built for the
    workspace root only check the workspace."""
    import importlib
    import json
    import sys
    from pathlib import Path

    import metasci_scribe.agent as evo
    import metasci_scribe.memory.agents as memory_agents
    import metasci_scribe.subagents._factory as factory_mod

    for module, name in (
        (factory_mod, "build_async_subagent_graph"),
        (memory_agents, "build_memory_worker_graph"),
        (memory_agents, "build_observation_linker_graph"),
        (memory_agents, "build_autoskills_graph"),
    ):
        monkeypatch.setattr(module, name, lambda *a, **k: object())
    monkeypatch.setattr(evo, "_get_default_agent", lambda: MagicMock())
    for name in (
        "metasci_scribe.langgraph_dev.graphs",
        "metasci_scribe.langgraph_dev.main_graph",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
        # Built below from stubbed graphs: later imports must build their own.
        request.addfinalizer(lambda name=name: sys.modules.pop(name, None))

    registry = json.loads(
        (Path(evo.__file__).parent / "langgraph_dev" / "langgraph.json").read_text()
    )["graphs"]
    other_run = _config(
        SessionDirs(served.workspace, served.workspace.runs_dir / "other")
    )
    other_workspace = _config(SessionDirs(Workspace(served.workspace.root / "other")))
    running = SimpleNamespace(execution_runtime=object())
    assert FOLDER_GRAPH_IDS <= registry.keys()

    for graph_id, target in registry.items():
        module_path, attr = target.rsplit(":", 1)
        factory = getattr(importlib.import_module(module_path), attr)
        with pytest.raises(RunFolderMismatchError):
            asyncio.run(factory(other_workspace, running))
        if graph_id in FOLDER_GRAPH_IDS:
            with pytest.raises(RunFolderMismatchError):
                asyncio.run(factory(other_run, running))
        else:
            asyncio.run(factory(other_run, running))


# ---------------------------------------------------------------------------
# Failed child runs report why
# ---------------------------------------------------------------------------


def test_run_errors_keep_exception_type():
    """langgraph-api stores a placeholder text for most exception types."""
    from metasci_scribe.llm.patches import _RunErrors

    failed = {"status": "error", "thread_id": "t1", "run_id": "r1"}
    attached = _RunErrors().attach(
        failed,
        {"error": {"error": "RateLimitError", "message": "An internal error occurred"}},
    )
    assert "RateLimitError" in attached["error"]


def test_run_errors_return_run_without_id_unchanged():
    from metasci_scribe.llm.patches import _RunErrors

    run = {"status": "error", "thread_id": "t1"}
    thread = {"error": {"error": "RuntimeError", "message": "boom"}}
    assert _RunErrors().attach(run, thread) == run


def test_merge_runs_config_replaces_caller_folders():
    """The launching graph's folders win over a caller's stale ones."""
    from metasci_scribe.llm.patches import _merge_runs_config_kwargs

    caller = {"workspace_dir": "/c", "run_dir": "/c/runs/r", "thread_id": "t1"}
    with patch("metasci_scribe.llm.patches._read_cfg_configurable", lambda: {}):
        merged = _merge_runs_config_kwargs(
            {"config": {"configurable": caller}}, {"workspace_dir": "/ws"}
        )
    assert merged["config"]["configurable"] == {
        "workspace_dir": "/ws",
        "thread_id": "t1",
    }


def test_server_gateway_run_config_forwards_target_folders(monkeypatch):
    """The gateway's own main-agent runs carry their folders like child runs,
    over any a caller put in ``configurable_extra``."""
    import metasci_scribe.agent as evo
    from metasci_scribe.gateway.server import LangGraphServerGateway
    from metasci_scribe.gateway.types import GraphTarget

    monkeypatch.setattr(
        evo,
        "_ensure_config",
        lambda: SimpleNamespace(model="m", provider="p", recursion_limit=None),
    )
    monkeypatch.setattr(
        "metasci_scribe.backends.hitl_suppressed_for_run", lambda _cfg: False
    )
    gateway = object.__new__(LangGraphServerGateway)
    stale = {"workspace_dir": "/b", "run_dir": "/b/runs/y", "active_teams": ["t"]}

    run_mode = gateway._resolve_run_config(
        "t1", stale, target=GraphTarget(workspace_dir="/ws", run_dir="/ws/runs/x")
    )["configurable"]
    daemon = gateway._resolve_run_config(
        "t1", stale, target=GraphTarget(workspace_dir="/ws")
    )["configurable"]
    bare = gateway._resolve_run_config("t1", None)["configurable"]

    assert (run_mode["workspace_dir"], run_mode["run_dir"]) == ("/ws", "/ws/runs/x")
    assert run_mode["model"] == "m"
    assert run_mode["active_teams"] == ["t"]
    assert daemon["workspace_dir"] == "/ws"
    assert "run_dir" not in daemon
    assert "workspace_dir" not in bare


# ---------------------------------------------------------------------------
# The server moves with the session
# ---------------------------------------------------------------------------


async def _sync_order(workspace, *, pinned_elsewhere: bool) -> list[str]:
    """Sync the server to ``workspace`` and return what ran, in order."""
    from metasci_scribe.cli import commands

    order: list[str] = []
    with (
        patch.object(
            commands, "_wait_for_memory_workers", lambda **_k: order.append("memory")
        ),
        patch(
            "metasci_scribe.langgraph_dev.manager.owned_server_pinned_elsewhere",
            lambda *a, **k: pinned_elsewhere,
        ),
        patch(
            "metasci_scribe.langgraph_dev.manager.ensure_langgraph_dev",
            lambda *a, **k: order.append("move"),
        ),
        patch.object(commands, "_reconcile_autoskill_schedule", lambda *a, **k: None),
    ):
        await commands._sync_background_agent_server_workspace(
            MagicMock(), dirs=SessionDirs(workspace)
        )
    return order


async def test_server_move_waits_for_memory_work(workspace):
    """Moving the pinned server stops runs in flight, so the previous
    session's memory work finishes before it moves."""
    assert await _sync_order(workspace, pinned_elsewhere=True) == ["memory", "move"]


async def test_server_sync_to_served_folders_skips_memory_wait(workspace):
    """A daemon-mode /resume syncs to the folders the server already serves."""
    assert await _sync_order(workspace, pinned_elsewhere=False) == ["move"]


def test_owned_server_pinned_elsewhere_compares_folder_pair(workspace, monkeypatch):
    from metasci_scribe.langgraph_dev import manager

    alive = SimpleNamespace(poll=lambda: None)
    run = workspace.runs_dir / RUN
    monkeypatch.setattr(manager, "_PROCESS", alive)
    monkeypatch.setattr(manager, "_PROCESS_WORKSPACE", workspace.root)
    monkeypatch.setattr(manager, "_PROCESS_RUN_DIR", run)
    assert manager.owned_server_pinned_elsewhere(workspace.root, run) is False
    assert manager.owned_server_pinned_elsewhere(workspace.root, None) is True
    assert manager.owned_server_pinned_elsewhere(workspace.root / "other", run) is True
    monkeypatch.setattr(manager, "_PROCESS", None)
    assert manager.owned_server_pinned_elsewhere(workspace.root, None) is False


async def _new_session(workspace, sync) -> SessionDirs:
    """Run the Rich CLI ``/new`` folder switch from a thread in a run folder."""
    from metasci_scribe.cli import commands
    from metasci_scribe.cli.interactive import _new_session_dirs

    with patch.object(commands, "_sync_background_agent_server_workspace", sync):
        return await _new_session_dirs(
            SessionDirs(workspace, workspace.runs_dir / "exp"),
            config=MagicMock(),
            backend=None,
        )


async def test_new_session_returns_to_the_workspace_root(workspace):
    from unittest.mock import AsyncMock

    sync = AsyncMock()
    new_dirs = await _new_session(workspace, sync)

    assert new_dirs == SessionDirs(workspace)
    assert sync.await_args.kwargs["dirs"] == new_dirs
    assert not workspace.runs_dir.exists()


async def test_new_session_dirs_refused_keeps_the_session(workspace):
    """Another session holds the server: ``/new`` fails and the session
    keeps its folders."""
    from unittest.mock import AsyncMock

    from metasci_scribe.langgraph_dev.manager import WorkspaceMismatchError

    sync = AsyncMock(side_effect=WorkspaceMismatchError("held by another session"))
    with pytest.raises(WorkspaceMismatchError):
        await _new_session(workspace, sync)


async def test_new_session_dirs_survive_failed_sync(workspace):
    """Any other sync failure starts the new session without background work."""
    from unittest.mock import AsyncMock

    new_dirs = await _new_session(workspace, AsyncMock(side_effect=OSError("down")))

    assert new_dirs == SessionDirs(workspace)
