"""Scheduled tasks belong to a workspace: tagged, listed and managed within it."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from metasci_scribe.cron import schedule as crons
from metasci_scribe.paths import Workspace


def _cron(cron_id: str, workspace_dir: str | None = None, **metadata) -> dict:
    meta = {"run_kind": "scheduled_task", "name": cron_id, **metadata}
    if workspace_dir is not None:
        meta["workspace_dir"] = workspace_dir
    return {"cron_id": cron_id, "schedule": "0 7 * * *", "metadata": meta}


@pytest.fixture
def other(tmp_path) -> Workspace:
    """Another live workspace on this machine (its folder exists)."""
    ws = Workspace(tmp_path / "other")
    ws.root.mkdir(parents=True)
    return ws


def test_belongs_to_matches_tagged_workspace(workspace, other, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(workspace.root, target_is_directory=True)

    assert crons.belongs_to(_cron("a", workspace.key), workspace)
    assert crons.belongs_to(_cron("a", f"{link}/"), workspace)
    assert not crons.belongs_to(_cron("a", other.key), workspace)
    # A tag that is no path here names no workspace here.
    assert not crons.belongs_to(_cron("a", "/tmp/evil\x00x"), workspace)
    # Created before tasks were tagged: the server serves one workspace.
    assert crons.belongs_to(_cron("a"), workspace)
    # The project was moved or renamed: its store, and the cron in it, came along.
    assert crons.belongs_to(_cron("a", str(tmp_path / "gone")), workspace)


@pytest.mark.parametrize("served", ["unknown", "other", "workspace"])
def test_startup_autoskills_reconcile_needs_server_serving_the_workspace(
    workspace, other, monkeypatch, served
):
    """The reconcile replaces crons it counts as this workspace's, so it runs
    only on a server known to hold this workspace's store."""
    from metasci_scribe.cli import commands
    from metasci_scribe.langgraph_dev import manager
    from metasci_scribe.memory.autoskills import schedule as autoskills_schedule

    calls = []
    roots = {"unknown": None, "other": other.root, "workspace": workspace.root}
    monkeypatch.setattr(manager, "SERVED_WORKSPACE", roots[served])
    monkeypatch.setattr(
        autoskills_schedule,
        "reconcile_autoskill_schedule",
        lambda config, *, workspace_dir: calls.append(workspace_dir),
    )

    commands._reconcile_autoskill_schedule(object(), workspace=workspace)

    assert calls == ([workspace.root] if served == "workspace" else [])


# ---------------------------------------------------------------------------
# AutoSkills keeps one cron per workspace
# ---------------------------------------------------------------------------


def _autoskill_cron(cron_id: str, workspace_dir: str, *, forwarded: bool) -> dict:
    from metasci_scribe.memory.autoskills.schedule import AUTOSKILL_RUN_KIND

    payload = {"config": {"configurable": {"workspace_dir": workspace_dir}}}
    return {
        "cron_id": cron_id,
        "schedule": "0 3 * * *",
        "enabled": True,
        "metadata": {
            "run_kind": AUTOSKILL_RUN_KIND,
            "workspace_dir": workspace_dir,
            "mode": "propose",
        },
        "payload": payload if forwarded else {},
    }


@pytest.fixture
def autoskills(monkeypatch):
    import metasci_scribe.langgraph_dev.manager as manager
    import metasci_scribe.memory.autoskills.schedule as schedule

    fake = MagicMock()
    fake.crons.create.return_value = {"cron_id": "new"}
    monkeypatch.setattr(schedule, "get_langgraph_sync_client", lambda **_k: fake)
    monkeypatch.setattr(schedule, "langgraph_dev_url", lambda _c: "http://x")
    monkeypatch.setattr(manager, "is_langgraph_dev_running", lambda **_k: True)
    config = SimpleNamespace(
        memory_skill_synthesis_enabled=True,
        memory_skill_synthesis_mode=SimpleNamespace(value="propose"),
        memory_skill_synthesis_cadence=SimpleNamespace(value="nightly"),
        memory_skill_synthesis_time="03:00",
    )
    return schedule, fake, config


def test_reconcile_autoskills_ignores_other_workspaces(
    workspace, other, autoskills, tmp_path
):
    schedule, fake, config = autoskills
    # Ours forwards the workspace written another way: still up to date.
    link = tmp_path / "link"
    link.symlink_to(workspace.root, target_is_directory=True)
    fake.crons.search.return_value = [
        _autoskill_cron("mine", f"{link}/", forwarded=True),
        _autoskill_cron("theirs", other.key, forwarded=True),
    ]

    result = schedule.reconcile_autoskill_schedule(config, workspace_dir=workspace.root)

    assert result["status"] == "unchanged"
    fake.crons.delete.assert_not_called()


def test_reconcile_autoskills_replaces_unforwarded_cron(workspace, other, autoskills):
    """A cron from before runs forwarded the workspace is recreated once."""
    schedule, fake, config = autoskills
    fake.crons.search.return_value = [
        _autoskill_cron("old", workspace.key, forwarded=False),
        _autoskill_cron("theirs", other.key, forwarded=False),
    ]

    result = schedule.reconcile_autoskill_schedule(config, workspace_dir=workspace.root)

    assert result["status"] == "created"
    fake.crons.delete.assert_called_once_with("old")
    assert fake.crons.create.call_args.kwargs["config"] == {
        "configurable": {"workspace_dir": workspace.key}
    }


def test_reconcile_autoskills_replaces_moved_folder_cron(
    workspace, autoskills, tmp_path
):
    """The project was renamed: the store, and its cron tagged with the old
    folder, came along. The cron is ours and is recreated for the new root."""
    schedule, fake, config = autoskills
    gone = str(tmp_path / "gone")
    fake.crons.search.return_value = [_autoskill_cron("stale", gone, forwarded=True)]

    result = schedule.reconcile_autoskill_schedule(config, workspace_dir=workspace.root)

    assert result["status"] == "created"
    fake.crons.delete.assert_called_once_with("stale")
