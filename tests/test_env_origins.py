"""Tests for the record of where config-mapped environment variables came from
(metasci_scribe.config.settings)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from metasci_scribe.config import settings
from metasci_scribe.config.settings import (
    _ENV_MAPPINGS,
    ScribeConfig,
    apply_config_to_env,
    config_value_origin,
    env_origin,
    get_config_path,
    get_effective_config,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(settings, "_ENV_ORIGINS", {})


def _use_dotenv(tmp_path: Path, monkeypatch) -> None:
    """A workspace ``.env`` (the suite turns ``.env`` discovery off)."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("TAVILY_API_KEY=tvly-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(dotenv))


@pytest.fixture
def config_file() -> Path:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class TestConfigValueOrigin:
    def test_file(self, config_file):
        config_file.write_text("langgraph_dev_port: 7000\n", encoding="utf-8")
        assert config_value_origin("langgraph_dev_port", 7000) == "file"
        assert config_value_origin("langgraph_dev_port", "7000") == "file"

    def test_flag_over_the_file(self, config_file):
        config_file.write_text("langgraph_dev_port: 7000\n", encoding="utf-8")
        assert config_value_origin("langgraph_dev_port", 9000) == "cli"

    def test_default(self):
        default = ScribeConfig().langgraph_dev_port
        assert config_value_origin("langgraph_dev_port", default) == "default"

    def test_workdir_compares_expanded_paths(self, config_file):
        config_file.write_text("default_workdir: ~/research\n", encoding="utf-8")
        expanded = os.path.expanduser("~/research")
        assert config_value_origin("default_workdir", expanded) == "file"

    def test_workdir_from_the_current_folder_is_the_default(self):
        assert config_value_origin("default_workdir", os.getcwd()) == "default"

    def test_workdir_flag(self, tmp_path):
        assert config_value_origin("default_workdir", str(tmp_path)) == "cli"


class TestRecording:
    def test_apply_config_to_env_records_file_copies(self, config_file):
        config_file.write_text("openai_api_key: sk-file\n", encoding="utf-8")
        apply_config_to_env(get_effective_config())
        assert os.environ["OPENAI_API_KEY"] == "sk-file"
        assert env_origin("OPENAI_API_KEY") == {"source": "file"}

    def test_the_users_export_is_not_recorded(self, config_file, monkeypatch):
        config_file.write_text("openai_api_key: sk-file\n", encoding="utf-8")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-shell")
        apply_config_to_env(get_effective_config())
        assert env_origin("OPENAI_API_KEY") is None

    def test_a_flag_override_is_cli(self):
        apply_config_to_env(get_effective_config({"dangerous_mode": True}))
        assert env_origin("METASCI_SCRIBE_DANGEROUS_MODE") == {"source": "cli"}
        # Turned off again: the variable is removed, and so is its record.
        apply_config_to_env(ScribeConfig())
        assert env_origin("METASCI_SCRIBE_DANGEROUS_MODE") is None

    def test_dotenv_records_its_path(self, tmp_path, monkeypatch):
        _use_dotenv(tmp_path, monkeypatch)
        get_effective_config()
        assert env_origin("TAVILY_API_KEY") == {
            "source": "dotenv",
            "path": str(tmp_path / ".env"),
        }


def test_the_server_reads_and_removes_the_record():
    """Read once at import; removed so the agent's commands never inherit it."""
    code = (
        "import os, json\n"
        "from metasci_scribe.config import settings\n"
        "print(json.dumps([os.environ.get(settings.ENV_ORIGINS_VAR), "
        "settings.env_origin('OPENAI_API_KEY')]))"
    )
    env = dict(os.environ)
    env[settings.ENV_ORIGINS_VAR] = json.dumps({"OPENAI_API_KEY": {"source": "file"}})
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout
    assert json.loads(out.strip().splitlines()[-1]) == [None, {"source": "file"}]


class TestServerRecord:
    """What start_langgraph_dev records for the workspace, port and host."""

    def test_a_dotenv_port_is_not_taken(self, tmp_path, monkeypatch):
        """A workspace ``.env`` sets no ``METASCI_SCRIBE_*`` key, so a port
        there is neither used nor recorded; the port the launcher resolved is
        recorded as its own value."""
        from metasci_scribe.langgraph_dev.manager import _server_env_origins

        dotenv = tmp_path / ".env"
        dotenv.write_text("METASCI_SCRIBE_LANGGRAPH_DEV_PORT=6297\n", encoding="utf-8")
        monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(dotenv))
        monkeypatch.delenv("METASCI_SCRIBE_LANGGRAPH_DEV_PORT", raising=False)
        config = get_effective_config()
        assert config.langgraph_dev_port == ScribeConfig().langgraph_dev_port
        assert env_origin("METASCI_SCRIBE_LANGGRAPH_DEV_PORT") is None
        origins = _server_env_origins(str(tmp_path), "6297", "127.0.0.1")
        assert origins["METASCI_SCRIBE_LANGGRAPH_DEV_PORT"] == {"source": "cli"}

    def test_a_symlinked_workdir_reads_file(self, config_file, tmp_path):
        from metasci_scribe.langgraph_dev.manager import _server_env_origins

        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        try:
            link.symlink_to(real, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are not available")
        config_file.write_text(
            json.dumps({"default_workdir": str(link)}), encoding="utf-8"
        )
        origins = _server_env_origins(str(real.resolve()), "6174", "127.0.0.1")
        assert origins["METASCI_SCRIBE_WORKSPACE_DIR"] == {"source": "file"}


def test_an_unreadable_config_file_does_not_stop_a_start(config_file):
    config_file.mkdir()
    apply_config_to_env(get_effective_config())


def test_apply_config_to_env_reads_the_file_once(config_file, monkeypatch):
    from metasci_scribe.config import store

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_file.write_text(
        "openai_api_key: sk-file-1\nanthropic_api_key: sk-ant-file-2\n",
        encoding="utf-8",
    )
    config = get_effective_config()
    reads = []
    real = store.read_config_mapping
    monkeypatch.setattr(store, "read_config_mapping", lambda: reads.append(1) or real())
    apply_config_to_env(config)
    assert len(reads) == 1
    for field in ("openai_api_key", "anthropic_api_key"):
        assert env_origin(_ENV_MAPPINGS[field]) == {"source": "file"}
