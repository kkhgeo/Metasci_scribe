"""The writing-focused surface: three sub-agents, two providers, nothing in the background.

Metasci Scribe is a terminal paper-writing agent. Memory workers, async
sub-agents, the scheduler and skill synthesis are off by default, so a plain
start never spawns a ``langgraph dev`` server. The removed commands,
middleware, sub-agents and providers must not come back by accident.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metasci_scribe.agent import SUBAGENTS_CONFIG
from metasci_scribe.config import ScribeConfig
from metasci_scribe.config.provider_keys import PROVIDERS

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "metasci_scribe"

REMOVED_MODULES = [
    "metasci_scribe.commands.implementation.schedule",
    "metasci_scribe.commands.implementation.experts",
    "metasci_scribe.commands.implementation.autoskills",
    "metasci_scribe.commands.implementation.model_fallback",
    "metasci_scribe.middleware.scheduler",
    "metasci_scribe.middleware.active_team",
    "metasci_scribe.middleware.expert_async_subagent",
    "metasci_scribe.middleware.model_fallback",
    "metasci_scribe.subagents.expert_container",
    "metasci_scribe.subagents.expert_container_async",
    "metasci_scribe.llm.chatgpt_oauth",
    "metasci_scribe.llm.deepseek",
    "metasci_scribe.llm.ollama_discovery",
    "metasci_scribe.tools.search",
    "metasci_scribe.langgraph_dev.settings_api",
    "metasci_scribe.langgraph_dev.config_baseline",
    "metasci_scribe.local_secret",
]

REMOVED_DEPENDENCIES = [
    "langchain-deepseek",
    "langchain-nvidia-ai-endpoints",
    "langchain-google-genai",
    "langchain-ollama",
    "langchain-openrouter",
    "openrouter",
    "tavily-python",
    "textual",
]


# --- defaults ---------------------------------------------------------------


def test_memory_and_async_off_by_default():
    c = ScribeConfig()
    assert not any(
        [
            c.memory_profile_enabled,
            c.memory_observations_enabled,
            c.memory_workers_enabled,
            c.enable_async_subagents,
        ]
    )


def test_scheduler_and_skill_synthesis_off_by_default():
    c = ScribeConfig()
    assert not c.enable_scheduler
    assert not c.memory_skill_synthesis_enabled


def test_default_provider_is_openai_with_an_openai_model():
    from metasci_scribe.llm.registry import resolve_provider

    c = ScribeConfig()
    assert c.provider == "openai"
    assert resolve_provider(c.model, c.provider) == "openai"


# --- sub-agents -------------------------------------------------------------


def test_only_three_subagents():
    names = {p.stem for p in SUBAGENTS_CONFIG.glob("*.yaml")}
    assert names == {"planner", "research", "writing"}


def test_subagent_tools_all_resolve(caplog):
    """No sub-agent names a tool the agent no longer has (``tavily_search``)."""
    from metasci_scribe.tools import think_tool
    from metasci_scribe.utils import load_subagents, resolve_subagent_tools

    subs = load_subagents(SUBAGENTS_CONFIG)
    assert {s["name"] for s in subs} == {
        "planner-agent",
        "research-agent",
        "writing-agent",
    }
    with caplog.at_level(logging.WARNING, logger="metasci_scribe.utils"):
        for sub in subs:
            resolve_subagent_tools(sub, {"think_tool": think_tool})
    assert not [r for r in caplog.records if "not in registry" in r.getMessage()]


def test_tavily_search_is_not_a_tool():
    import metasci_scribe
    import metasci_scribe.tools as tools

    assert not hasattr(tools, "tavily_search")
    assert not hasattr(tools, "fetch_webpage_content")
    assert "tavily_search" not in getattr(metasci_scribe, "__all__", [])


# --- providers --------------------------------------------------------------


def test_only_supported_providers():
    assert set(PROVIDERS) == {"openai", "anthropic"}


def test_registry_and_wizard_know_only_the_supported_providers():
    from metasci_scribe.config.onboard.constants import VALID_PROVIDERS
    from metasci_scribe.llm import registry

    assert {p for _, _, p in registry._MODEL_ENTRIES} == {"openai", "anthropic"}
    assert set(VALID_PROVIDERS) == {"openai", "anthropic"}
    assert not hasattr(registry, "_OPENAI_ROUTED_PROVIDERS")
    assert not hasattr(registry, "_ANTHROPIC_ROUTED_PROVIDERS")


def test_other_providers_are_rejected_by_config_validation():
    from metasci_scribe.config.settings import validate_setting

    for provider in ("google-genai", "openrouter", "deepseek", "ollama"):
        with pytest.raises(ValueError, match="unknown provider"):
            validate_setting("provider", provider)
    assert validate_setting("provider", "openai")[0] == "openai"
    assert validate_setting("provider", "anthropic")[0] == "anthropic"


def test_openai_auth_mode_seam():
    """``api_key`` is the default; ``chatgpt_plan`` is reserved for a later task."""
    assert ScribeConfig().openai_auth_mode == "api_key"
    assert ScribeConfig(openai_auth_mode="chatgpt_plan").openai_auth_mode == (
        "chatgpt_plan"
    )
    from metasci_scribe.config.settings import validate_setting

    assert validate_setting("openai_auth_mode", "chatgpt_plan")[0] == "chatgpt_plan"
    with pytest.raises(ValueError, match="api_key"):
        validate_setting("openai_auth_mode", "oauth")


def test_chatgpt_plan_is_not_usable_yet(monkeypatch):
    from metasci_scribe.llm.models import get_chat_model

    monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")
    with pytest.raises(ValueError, match="chatgpt_plan"):
        get_chat_model("gpt-6-sol", provider="openai")


def test_openai_model_builds_from_an_api_key(monkeypatch):
    from metasci_scribe.llm.models import get_chat_model

    monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "api_key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = get_chat_model("gpt-6-sol", provider="openai")
    assert type(model).__name__ == "ChatOpenAI"


# --- removed code and dependencies -------------------------------------------


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_module_is_gone(module):
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize(
    "name",
    [
        "code",
        "debug",
        "data_analysis",
        "scheduler",
    ],
)
def test_subagent_yaml_is_gone(name):
    assert not (SUBAGENTS_CONFIG / f"{name}.yaml").exists()


def test_removed_dependencies_are_not_declared():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        re.split(r"[<>=!~\[; ]", dep, maxsplit=1)[0].lower()
        for dep in data["project"]["dependencies"]
    }
    assert not declared & set(REMOVED_DEPENDENCIES)
    # The memory code still needs the langgraph server stack.
    assert {"langgraph-api", "langgraph-cli"} <= declared


def test_no_source_mentions_removed_names():
    forbidden = re.compile(
        r"openrouter|chatgpt_oauth|EvoAsyncSubAgentMiddleware|EvoChatDeepSeek|"
        r"expert_container|tavily|OPENROUTER_DEFAULT",
        re.IGNORECASE,
    )
    hits = []
    for path in PACKAGE_DIR.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if forbidden.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not hits, "\n".join(hits[:20])


def test_oauth_auth_mode_is_not_a_value_anywhere():
    hits = []
    for path in PACKAGE_DIR.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"""auth_mode\W+["']oauth["']|["']oauth["']\s*\)""", line):
                hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not hits, "\n".join(hits[:20])


# --- the background server ---------------------------------------------------


def test_defaults_need_no_langgraph_dev_server():
    from metasci_scribe.langgraph_dev.manager import needs_langgraph_dev

    assert needs_langgraph_dev(ScribeConfig()) is False


@pytest.mark.parametrize(
    "flag",
    ["enable_async_subagents", "enable_scheduler", "memory_skill_synthesis_enabled"],
)
def test_each_background_feature_still_asks_for_the_server(flag):
    from metasci_scribe.langgraph_dev.manager import needs_langgraph_dev

    assert needs_langgraph_dev(ScribeConfig(**{flag: True})) is True


def test_startup_with_defaults_spawns_no_server(tmp_path, monkeypatch):
    """The CLI start path asks ``ensure_langgraph_dev`` for a server; with the
    defaults it must not reach the function that spawns ``langgraph dev``, and
    it must not announce a server either."""
    from metasci_scribe.cli import commands
    from metasci_scribe.langgraph_dev import manager
    from metasci_scribe.paths import SessionDirs, Workspace

    monkeypatch.setattr(
        manager, "RUNTIME", manager.LanggraphRuntimePaths.for_directory(tmp_path)
    )
    spawn = MagicMock(name="start_langgraph_dev")
    popen = MagicMock(name="Popen")
    with (
        patch.object(manager, "start_langgraph_dev", spawn),
        patch.object(manager.subprocess, "Popen", popen),
        patch.object(manager, "_ensure_langgraph_dev_locked", spawn),
        patch.object(commands.console, "status") as status,
    ):
        dirs = SessionDirs(Workspace(tmp_path), None)
        commands._ensure_async_subagent_server(
            ScribeConfig(), dirs=dirs, backend="local"
        )
        assert manager.ensure_langgraph_dev(ScribeConfig(), tmp_path) is None
    spawn.assert_not_called()
    popen.assert_not_called()
    status.assert_not_called()
    assert manager.is_async_subagents_available() is False


def test_startup_with_a_background_feature_asks_for_the_server(tmp_path, monkeypatch):
    """The other side of the gate: turning a feature on reaches the spawn
    function (patched here; no server is ever started by a test)."""
    from metasci_scribe.cli import commands
    from metasci_scribe.langgraph_dev import manager
    from metasci_scribe.paths import SessionDirs, Workspace

    monkeypatch.setattr(
        manager, "RUNTIME", manager.LanggraphRuntimePaths.for_directory(tmp_path)
    )
    locked = MagicMock(name="_ensure_langgraph_dev_locked", return_value=None)
    with (
        patch.object(manager, "_ensure_langgraph_dev_locked", locked),
        patch.object(manager.subprocess, "Popen") as popen,
        patch.object(commands.console, "status"),
    ):
        dirs = SessionDirs(Workspace(tmp_path), None)
        commands._ensure_async_subagent_server(
            ScribeConfig(enable_async_subagents=True), dirs=dirs, backend="local"
        )
    locked.assert_called_once()
    popen.assert_not_called()


# --- the served app ----------------------------------------------------------


def test_served_app_keeps_only_the_background_process_route():
    from metasci_scribe.langgraph_dev import http

    paths = {route.path for route in http.app.routes}
    assert paths == {"/api/bg_process_status"}


def test_langgraph_json_has_no_webui_cors_and_only_kept_graphs():
    manifest = json.loads(
        (PACKAGE_DIR / "langgraph_dev" / "langgraph.json").read_text(encoding="utf-8")
    )
    assert "cors" not in manifest.get("http", {})
    assert set(manifest["graphs"]) == {
        "metasci-scribe",
        "writing-agent",
        "evomemory-subagent-worker",
        "evomemory-turn-worker",
        "evomemory-observation-linker",
        "evomemory-autoskills",
    }


def test_boundary_has_no_cors_settings():
    from metasci_scribe.langgraph_dev import boundary

    assert not hasattr(boundary, "cors_config")
    assert "CORS_CONFIG" not in boundary.server_env("127.0.0.1", "")
    assert boundary.BOUNDARY_ENV in boundary.server_env("127.0.0.1", "")
