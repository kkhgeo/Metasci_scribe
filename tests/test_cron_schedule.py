"""Unit tests for the cron wrapper (mock langgraph_sdk client)."""

from unittest.mock import MagicMock


def test_production_loaders_accept_utf8_content(tmp_path, monkeypatch):
    """Production config readers must handle localized UTF-8 content."""
    import os

    from metasci_scribe import paths
    from metasci_scribe.langgraph_dev import manager
    from metasci_scribe.mcp import client, registry

    mcp_config = tmp_path / "mcp.yaml"
    mcp_config.write_text(
        """
écho:
  transport: stdio
  command: python
  args: ["-m", "démo"]
  env:
    MESSAGE: "bonjour café"
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(client, "USER_MCP_CONFIG", mcp_config)
    assert client._load_user_config()["écho"]["env"]["MESSAGE"] == "bonjour café"
    assert client.load_mcp_config()["écho"]["args"] == ["-m", "démo"]

    marketplace_yaml = tmp_path / "marketplace-écho.yaml"
    marketplace_yaml.write_text(
        """
name: écho
label: "Écho café"
description: "localized marketplace entry"
tags: "démo, marché"
transport: stdio
command: python
args: ["-m", "écho"]
""".lstrip(),
        encoding="utf-8",
    )
    entry = registry.parse_marketplace_yaml(marketplace_yaml)
    assert entry.name == "écho"
    assert entry.tags == ["démo", "marché"]

    runtime = manager.LanggraphRuntimePaths.for_directory(tmp_path / "runtime")
    monkeypatch.setattr(manager, "RUNTIME", runtime)
    workspace = tmp_path / "workspace-écho"
    manager._write_workspace_sidecar(workspace, 12345)
    assert manager._read_workspace_sidecar() == {
        "workspace": str(workspace),
        "pid": 12345,
        "data_dir": str(paths.DATA_DIR),
        "config_dir": str(paths.CONFIG_DIR),
    }

    runtime.pid_file.write_text(str(os.getpid()), encoding="utf-8")

    class _NotLanggraphProcess:
        def __init__(self, pid):
            self.pid = pid

        def cmdline(self):
            return ["python", "localized-test"]

        def kill(self):
            raise AssertionError("non-langgraph pid must not be killed")

    monkeypatch.setattr(manager, "_list_pids_on_port", lambda port: [os.getpid()])
    monkeypatch.setattr(manager.psutil, "Process", _NotLanggraphProcess)
    assert manager._kill_owned_stale_process(6174) is False
    assert not runtime.pid_file.exists()
    assert not runtime.workspace_sidecar.exists()


def test_create_cron_without_a_thread_passes_no_new_arguments(workspace):
    from metasci_scribe.cron.schedule import create_cron

    fake = MagicMock()
    create_cron(
        fake,
        workspace=workspace,
        assistant_id="scheduler",
        schedule="0 9 * * *",
        input={},
        metadata={"run_kind": "scheduled_task"},
        timezone="UTC",
    )
    fake.crons.create_for_thread.assert_not_called()
    kw = fake.crons.create.call_args.kwargs
    assert set(kw) == {
        "assistant_id",
        "schedule",
        "input",
        "metadata",
        "config",
        "timezone",
    }


def test_create_cron_on_a_thread_keeps_the_workspace_tag_and_run_config(workspace):
    from metasci_scribe.cron.schedule import create_cron, run_config

    fake = MagicMock()
    create_cron(
        fake,
        workspace=workspace,
        assistant_id="proactive",
        schedule="*/10 * * * *",
        input={"candidates": []},
        metadata={"run_kind": "proactive_check"},
        timezone="UTC",
        thread_id="t-cron",
        multitask_strategy="reject",
        durability="exit",
    )
    fake.crons.create.assert_not_called()
    args = fake.crons.create_for_thread.call_args
    assert args.args == ("t-cron", "proactive")
    assert args.kwargs["metadata"] == {
        "run_kind": "proactive_check",
        "workspace_dir": workspace.key,
    }
    assert args.kwargs["config"] == run_config(workspace)
    assert args.kwargs["multitask_strategy"] == "reject"
    assert args.kwargs["durability"] == "exit"
    assert "context" not in args.kwargs
