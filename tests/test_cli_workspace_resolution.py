"""Tests for how the CLI entry point picks the session's workspace and work dir."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from metasci_scribe.cli import commands
from metasci_scribe.paths import SessionDirs, Workspace
from tests.fakes.common import FakeGraphGateway, FakeThreadStore


def _config(*, default_workdir: str = ""):
    return SimpleNamespace(
        default_workdir=default_workdir,
        openai_auth_mode="api_key",
        show_thinking=True,
        ui_backend="cli",
        model="test-model",
        provider="anthropic",
    )


def _run(
    monkeypatch,
    config,
    *,
    workdir: str | None = None,
    use_cwd: bool = False,
    mode: str | None = None,
    name: str | None = None,
) -> dict:
    """Drive the main callback up to ``cmd_interactive`` and capture its inputs."""
    import metasci_scribe.cli.interactive as interactive_mod
    import metasci_scribe.config as config_mod

    captured: dict = {}

    def _fake_get_effective_config(cli_overrides=None):
        merged = vars(config).copy()
        merged.update(cli_overrides or {})
        return SimpleNamespace(**merged)

    def _fake_ensure_server(cfg, *, dirs, backend=None):
        captured["server_dirs"] = dirs

    def _fake_cmd_interactive(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(config_mod, "get_effective_config", _fake_get_effective_config)
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)
    monkeypatch.setattr(config_mod, "resolve_gateway_backend", lambda *_a: "local")
    monkeypatch.setattr(commands, "_get_cli_async_runtime", lambda _ctx: None)
    order: list[str] = []
    captured["order"] = order
    monkeypatch.setattr(commands, "ensure_dirs", lambda: order.append("ensure_dirs"))
    monkeypatch.setattr(
        commands, "reload_env_dirs", lambda: order.append("reload_env_dirs")
    )
    monkeypatch.setattr(commands, "_ensure_async_subagent_server", _fake_ensure_server)
    monkeypatch.setattr(interactive_mod, "cmd_interactive", _fake_cmd_interactive)

    commands._main_callback(
        SimpleNamespace(invoked_subcommand=None),
        version=None,
        prompt=None,
        thread_id=None,
        workdir=workdir,
        use_cwd=use_cwd,
        no_thinking=False,
        auto_approve=False,
        auto_mode=None,
        ask_user=False,
        output_format=None,
    )
    return captured


def test_cwd_is_default_workspace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run(monkeypatch, _config())
    assert got["dirs"].workspace == Workspace(tmp_path)
    assert got["dirs"].work_dir == Workspace(tmp_path).root
    assert got["dirs"].run_dir is None
    assert got["server_dirs"] == got["dirs"]
    assert got["order"] == ["reload_env_dirs", "ensure_dirs"]


def test_symlinked_start_folder_gives_one_spelling(monkeypatch, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    got = _run(monkeypatch, _config(default_workdir=str(link)))
    assert got["dirs"].work_dir == got["dirs"].workspace.root
    assert got["dirs"].work_dir == real.resolve()


def test_use_cwd_ignores_default_workdir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run(
        monkeypatch, _config(default_workdir=str(tmp_path / "cfg")), use_cwd=True
    )
    assert got["dirs"].workspace == Workspace(tmp_path)


def test_workdir_is_created_and_used(monkeypatch, tmp_path):
    target = tmp_path / "proj"
    got = _run(monkeypatch, _config(), workdir=str(target))
    assert target.is_dir()
    assert got["dirs"].workspace == Workspace(target)
    assert got["dirs"].work_dir == Workspace(target).root
    assert got["dirs"].run_dir is None


def test_default_workdir_from_config(monkeypatch, tmp_path):
    target = tmp_path / "cfg"
    target.mkdir()
    monkeypatch.chdir(tmp_path)
    got = _run(monkeypatch, _config(default_workdir=str(target)))
    assert got["dirs"].workspace == Workspace(target)
    assert got["dirs"].work_dir == Workspace(target).root
    assert got["dirs"].run_dir is None


def test_chat_never_makes_a_run_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run(monkeypatch, _config())
    assert got["dirs"].run_dir is None
    assert not (tmp_path / "runs").exists()


def _run_prompt(
    monkeypatch,
    config,
    *,
    thread_id: str | None,
    stored: dict | None,
    backend: str = "local",
    load_agent=None,
) -> dict:
    """Drive the main callback through a ``-p`` run and capture its folders."""
    import metasci_scribe.cli.interactive as interactive_mod
    import metasci_scribe.config as config_mod
    import metasci_scribe.gateway as gateway_mod
    import metasci_scribe.sessions as sessions_mod

    captured: dict = {"synced": [], "prespawned": []}
    gateway = FakeGraphGateway(
        generated_thread_ids=["new-thread"],
        thread_store=FakeThreadStore(resolved_thread_id=thread_id, metadata=stored),
    )

    def _fake_get_effective_config(cli_overrides=None):
        merged = vars(config).copy()
        merged.update(cli_overrides or {})
        return SimpleNamespace(**merged)

    @asynccontextmanager
    async def _fake_checkpointer():
        yield None

    async def _fake_sync(_config, *, dirs, backend=None, **_kwargs):
        captured["synced"].append(dirs)

    def _fake_load_agent(**kwargs):
        captured["agent_workspace"] = kwargs["workspace"]
        captured["agent_work_dir"] = kwargs["work_dir"]
        return object()

    def _fake_cmd_run(_agent, _prompt, *, thread_id, dirs, **_kwargs):
        captured["thread_id"] = thread_id
        captured["run_dirs"] = dirs

    monkeypatch.setattr(config_mod, "get_effective_config", _fake_get_effective_config)
    monkeypatch.setattr(config_mod, "apply_config_to_env", lambda _cfg: None)
    monkeypatch.setattr(config_mod, "resolve_gateway_backend", lambda *_a: backend)
    monkeypatch.setattr(
        commands,
        "_get_cli_async_runtime",
        lambda _ctx: SimpleNamespace(run_sync=lambda fn: asyncio.run(fn())),
    )
    monkeypatch.setattr(commands, "ensure_dirs", lambda: None)
    monkeypatch.setattr(
        commands,
        "_ensure_async_subagent_server",
        lambda _config, *, dirs, backend=None: captured["prespawned"].append(dirs),
    )
    monkeypatch.setattr(commands, "_sync_background_agent_server_workspace", _fake_sync)
    monkeypatch.setattr(commands, "_load_agent", load_agent or _fake_load_agent)
    monkeypatch.setattr(
        gateway_mod,
        "create_runtime_gateways_for_config",
        lambda *_a, **_k: SimpleNamespace(graph_gateway=gateway),
    )
    monkeypatch.setattr(sessions_mod, "get_checkpointer", _fake_checkpointer)
    monkeypatch.setattr(interactive_mod, "cmd_run", _fake_cmd_run)

    commands._main_callback(
        SimpleNamespace(invoked_subcommand=None),
        version=None,
        prompt="hello",
        thread_id=thread_id,
        workdir=None,
        use_cwd=False,
        no_thinking=False,
        auto_approve=False,
        auto_mode=None,
        ask_user=False,
        output_format=None,
    )
    return captured


def test_one_shot_resume_uses_thread_workspace(monkeypatch, tmp_path):
    launch, other = tmp_path / "launch", tmp_path / "other"
    launch.mkdir()
    monkeypatch.chdir(launch)

    got = _run_prompt(
        monkeypatch,
        _config(),
        thread_id="t-other",
        stored={"workspace_dir": other.as_posix()},
    )

    restored = SessionDirs(Workspace(other))
    assert got["thread_id"] == "t-other"
    assert got["agent_workspace"] == Workspace(other)
    assert Path(got["agent_work_dir"]) == other.resolve()
    assert got["run_dirs"] == restored
    assert got["synced"] == [restored]
    # The server is started once, for the thread's folders.
    assert got["prespawned"] == []


def test_one_shot_resume_on_server_backend_starts_server_first(monkeypatch, tmp_path):
    """The server gateway backend is built against a running server."""
    launch, other = tmp_path / "launch", tmp_path / "other"
    launch.mkdir()
    monkeypatch.chdir(launch)

    got = _run_prompt(
        monkeypatch,
        _config(),
        thread_id="t-other",
        stored={"workspace_dir": other.as_posix()},
        backend="langgraph_server",
    )

    assert got["prespawned"] == [SessionDirs(Workspace(launch))]
    assert got["synced"] == [SessionDirs(Workspace(other))]


def test_one_shot_resume_uses_thread_run_dir(monkeypatch, tmp_path):
    root, launch = tmp_path / "proj", tmp_path / "launch"
    run = root / "runs" / "exp"
    launch.mkdir()
    monkeypatch.chdir(launch)

    got = _run_prompt(
        monkeypatch,
        _config(),
        thread_id="t-run",
        stored={"workspace_dir": root.as_posix(), "run_dir": run.as_posix()},
    )

    assert got["agent_workspace"] == Workspace(root)
    assert Path(got["agent_work_dir"]) == run.resolve()
    assert got["run_dirs"] == SessionDirs(Workspace(root), run)


def test_one_shot_resume_without_stored_dirs_keeps_launch_dirs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run_prompt(monkeypatch, _config(), thread_id="t-bare", stored=None)
    assert got["agent_workspace"] == Workspace(tmp_path)
    assert got["run_dirs"] == SessionDirs(Workspace(tmp_path))


def test_one_shot_new_thread_keeps_launch_dirs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    got = _run_prompt(monkeypatch, _config(), thread_id=None, stored=None)
    assert got["thread_id"] == "new-thread"
    assert got["run_dirs"] == SessionDirs(Workspace(tmp_path))
    assert got["prespawned"] == [SessionDirs(Workspace(tmp_path))]
    assert got["synced"] == []


def test_network_start_folder_stops_with_the_local_disk_message(monkeypatch, tmp_path):
    import typer

    from metasci_scribe import project

    printed: list[str] = []
    monkeypatch.setattr(commands.console, "print", lambda *a, **k: printed.append(a[0]))
    monkeypatch.setattr(project, "is_network_path", lambda p: True)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.Exit):
        _run(monkeypatch, _config())
    assert any("local" in line.lower() for line in printed)


def test_network_workdir_is_not_created(monkeypatch, tmp_path):
    import typer

    from metasci_scribe import project

    monkeypatch.setattr(project, "is_network_path", lambda p: True)
    target = tmp_path / "paper"
    with pytest.raises(typer.Exit):
        _run(monkeypatch, _config(), workdir=str(target))
    assert not target.exists()


def _link_dir(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
        return
    except OSError:
        pass
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, OSError):
        pytest.skip("cannot make a link or junction here")


def test_a_local_looking_link_to_a_network_folder_is_refused(monkeypatch, tmp_path):
    """A junction on C: that points at a share passes the raw-path check only."""
    import typer

    from metasci_scribe import project

    share = tmp_path / "share-target"
    share.mkdir()
    link = tmp_path / "paper"
    _link_dir(link, share)
    # Stand-in for the share: whatever resolves into ``share-target``.
    monkeypatch.setattr(
        project, "is_network_path", lambda p: str(p).endswith("share-target")
    )
    printed: list[str] = []
    monkeypatch.setattr(commands.console, "print", lambda *a, **k: printed.append(a[0]))
    with pytest.raises(typer.Exit):
        _run(monkeypatch, _config(), workdir=str(link))
    assert any("local" in line.lower() for line in printed)
