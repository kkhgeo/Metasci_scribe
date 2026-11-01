"""The agent's ``execute`` backends get the research environment overrides."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import metasci_scribe.agent as es_mod
from metasci_scribe.setup import research_env

_PROBE = {"METASCI_SCRIBE_SHELL_PROBE": "venv"}


@pytest.fixture
def overrides(monkeypatch):
    monkeypatch.setattr(research_env, "research_env_overrides", lambda: dict(_PROBE))


def test_default_backend_gets_the_overrides(overrides, workspace):
    backend = es_mod._get_default_backend(workspace)
    assert backend.default._env["METASCI_SCRIBE_SHELL_PROBE"] == "venv"


def test_cli_agent_backend_gets_the_overrides(overrides, tmp_path, workspace):
    backends = []

    def fake_build_kwargs(backend, *args, **kwargs):
        backends.append(backend)
        return {"name": "x"}

    cfg = MagicMock()
    cfg.sandbox_execute_timeout = 300
    cfg.recursion_limit = 100
    with (
        patch("deepagents.create_deep_agent", return_value=MagicMock()),
        patch.object(es_mod, "_apply_env_from_config"),
        patch.object(es_mod, "_get_default_middleware", return_value=[]),
        patch.object(
            es_mod, "load_mcp_and_build_kwargs", side_effect=fake_build_kwargs
        ),
    ):
        es_mod.create_cli_agent(
            work_dir=str(tmp_path),
            config=cfg,
            chat_model=MagicMock(),
            workspace=workspace,
        )
    assert backends[0].default._env["METASCI_SCRIBE_SHELL_PROBE"] == "venv"
