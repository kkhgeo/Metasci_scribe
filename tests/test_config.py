"""Tests for Metasci Scribe configuration management."""

import os
from pathlib import Path

import pytest
import yaml

from metasci_scribe import paths
from metasci_scribe.config import (
    GatewaySurface,
    MemoryControls,
    MemoryObservationTarget,
    MemoryObservationWriter,
    MemorySkillSynthesisCadence,
    MemorySkillSynthesisMode,
    ScribeConfig,
    apply_config_to_env,
    get_config_dir,
    get_config_path,
    get_config_value,
    get_effective_config,
    list_config,
    load_config,
    reset_config,
    resolve_gateway_backend,
    save_config,
    set_config_value,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    """Use a temporary directory for config during tests."""
    config_dir = tmp_path / "metasci-scribe"
    monkeypatch.setattr(paths, "CONFIG_DIR", config_dir)
    # Also clear any API keys from environment
    for key in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "ATLASCLOUD_API_KEY",
        "TAVILY_API_KEY",
        "METASCI_SCRIBE_DEFAULT_MODE",
        "METASCI_SCRIBE_WORKSPACE_DIR",
        "METASCI_SCRIBE_UI_BACKEND",
        "METASCI_SCRIBE_GATEWAY_BACKEND",
        "METASCI_SCRIBE_GATEWAY_BACKEND_SINGLE_SHOT",
        "METASCI_SCRIBE_GATEWAY_BACKEND_INTERACTIVE",
        "METASCI_SCRIBE_GATEWAY_BACKEND_STANDALONE",
        "METASCI_SCRIBE_MEMORY_PROFILE_ENABLED",
        "METASCI_SCRIBE_MEMORY_OBSERVATIONS_ENABLED",
        "METASCI_SCRIBE_MEMORY_OBSERVATION_WRITER",
        "METASCI_SCRIBE_MEMORY_WORKERS_ENABLED",
        "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_ENABLED",
        "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_MODE",
        "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_CADENCE",
        "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_TIME",
        "METASCI_SCRIBE_AUXILIARY_MODEL",
        "METASCI_SCRIBE_AUXILIARY_PROVIDER",
        "METASCI_SCRIBE_OPENROUTER_ANTHROPIC_PROMPT_CACHE",
        "METASCI_SCRIBE_OPENROUTER_HTTP_REFERER",
        "METASCI_SCRIBE_OPENROUTER_APP_TITLE",
        "METASCI_SCRIBE_OPENROUTER_APP_CATEGORIES",
        "METASCI_SCRIBE_DANGEROUS_MODE",
        "METASCI_SCRIBE_MAX_CACHED_FILES",
        "METASCI_SCRIBE_PROACTIVE_IDLE_MINUTES",
        "METASCI_SCRIBE_PROACTIVE_QUIET_HOURS",
        "METASCI_SCRIBE_PROACTIVE_TIMEZONE",
    ]:
        monkeypatch.delenv(key, raising=False)
    return config_dir


@pytest.fixture
def clean_env(monkeypatch):
    """Remove environment variables that affect config (but keep temp config dir)."""
    for key in [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "ATLASCLOUD_API_KEY",
        "TAVILY_API_KEY",
        "METASCI_SCRIBE_DEFAULT_MODE",
        "METASCI_SCRIBE_WORKSPACE_DIR",
        "METASCI_SCRIBE_UI_BACKEND",
        "METASCI_SCRIBE_GATEWAY_BACKEND",
        "METASCI_SCRIBE_GATEWAY_BACKEND_SINGLE_SHOT",
        "METASCI_SCRIBE_GATEWAY_BACKEND_INTERACTIVE",
        "METASCI_SCRIBE_GATEWAY_BACKEND_STANDALONE",
        "METASCI_SCRIBE_MEMORY_PROFILE_ENABLED",
        "METASCI_SCRIBE_MEMORY_OBSERVATIONS_ENABLED",
        "METASCI_SCRIBE_MEMORY_OBSERVATION_WRITER",
        "METASCI_SCRIBE_MEMORY_WORKERS_ENABLED",
        "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_ENABLED",
        "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_MODE",
        "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_CADENCE",
        "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_TIME",
        "METASCI_SCRIBE_AUXILIARY_MODEL",
        "METASCI_SCRIBE_AUXILIARY_PROVIDER",
        "METASCI_SCRIBE_OPENROUTER_ANTHROPIC_PROMPT_CACHE",
        "METASCI_SCRIBE_OPENROUTER_HTTP_REFERER",
        "METASCI_SCRIBE_OPENROUTER_APP_TITLE",
        "METASCI_SCRIBE_OPENROUTER_APP_CATEGORIES",
        "METASCI_SCRIBE_DANGEROUS_MODE",
        "METASCI_SCRIBE_MAX_CACHED_FILES",
        "METASCI_SCRIBE_PROACTIVE_IDLE_MINUTES",
        "METASCI_SCRIBE_PROACTIVE_QUIET_HOURS",
        "METASCI_SCRIBE_PROACTIVE_TIMEZONE",
    ]:
        monkeypatch.delenv(key, raising=False)


# =============================================================================
# Test ScribeConfig dataclass
# =============================================================================


class TestScribeConfig:
    def test_default_values(self):
        """Test that default values are set correctly."""
        config = ScribeConfig()

        assert config.anthropic_api_key == ""
        assert config.openai_api_key == ""
        assert config.provider == "openai"
        assert config.model == "gpt-6.1-sol"
        assert config.default_mode == "daemon"
        assert config.default_workdir == ""
        assert config.show_thinking is True
        assert config.ui_backend == "cli"
        assert config.log_level == "warning"
        assert config.reasoning_effort == ""
        assert config.memory_profile_enabled is False
        assert config.memory_observations_enabled is False
        assert config.memory_observation_writer == MemoryObservationWriter.ALL
        assert config.memory_workers_enabled is False
        assert config.memory_skill_synthesis_enabled is False
        assert config.memory_skill_synthesis_mode == MemorySkillSynthesisMode.REVIEW
        assert (
            config.memory_skill_synthesis_cadence == MemorySkillSynthesisCadence.WEEKLY
        )
        assert config.memory_skill_synthesis_time == "03:00"
        assert config.enable_async_subagents is False
        assert config.enable_scheduler is False

    def test_bind_hosts_default_to_loopback(self):
        """The server stays off the network until asked.

        The backend is an unauthenticated API whose agent can run shell
        commands, so it is not a safe default to publish;
        ``--host 0.0.0.0`` / ``config set`` opts in to LAN exposure.
        """
        config = ScribeConfig()

        assert config.langgraph_dev_host == "127.0.0.1"

    @pytest.mark.parametrize(
        ("field", "default"),
        [("langgraph_dev_host", "127.0.0.1")],
    )
    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_bind_host_falls_back_to_default(self, field, blank, default):
        """A blank host would reach socket.bind() verbatim and surface as an
        opaque gaierror; it degrades to the field's own default instead."""
        config = ScribeConfig(**{field: blank})
        assert getattr(config, field) == default

    @pytest.mark.parametrize(
        ("field", "value"),
        [("langgraph_dev_host", "0.0.0.0")],
    )
    def test_bind_host_opt_out_is_preserved(self, field, value):
        """The LAN escape hatch — widening the bind to the wildcard — must
        survive normalization untouched."""
        config = ScribeConfig(**{field: value})
        assert getattr(config, field) == value

    @pytest.mark.parametrize("field", ["langgraph_dev_host"])
    def test_bind_host_is_stripped(self, field):
        config = ScribeConfig(**{field: "  0.0.0.0  "})
        assert getattr(config, field) == "0.0.0.0"

    def test_openai_auth_mode_default(self):
        """Test that openai_auth_mode defaults to api_key."""
        config = ScribeConfig()
        assert config.openai_auth_mode == "api_key"

    def test_openai_auth_mode_set(self):
        """chatgpt_plan is a reserved value; anything else falls back."""
        config = ScribeConfig(openai_auth_mode="chatgpt_plan")
        assert config.openai_auth_mode == "chatgpt_plan"
        assert ScribeConfig(openai_auth_mode="oauth").openai_auth_mode == "api_key"

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = ScribeConfig(
            anthropic_api_key="sk-ant-test",
            provider="openai",
            model="gpt-4o",
            default_mode="run",
        )

        assert config.anthropic_api_key == "sk-ant-test"
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.default_mode == "run"

    def test_dangerous_mode_default(self):
        """dangerous_mode defaults off and does not force auto_approve."""
        config = ScribeConfig()
        assert config.dangerous_mode is False
        assert config.auto_approve is False

    def test_dangerous_mode_implies_auto_approve(self):
        """Enabling dangerous_mode forces auto_approve via __post_init__."""
        config = ScribeConfig(dangerous_mode=True)
        assert config.dangerous_mode is True
        assert config.auto_approve is True

    def test_literal_field_invalid_value_falls_back_with_warning(self, caplog):
        """A typo'd Literal (config file / direct construction) never lands.

        ``load_config`` does not coerce file values, so without this
        normalization an out-of-set value like ``langgraph-server`` (hyphen)
        would flow verbatim into mode selection and silently select the
        local gateway path in the manager.
        """
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="metasci_scribe.config.settings"):
            config = ScribeConfig(gateway_backend="langgraph-server")
        assert config.gateway_backend == "local"
        assert "Invalid gateway_backend" in caplog.text

    def test_literal_field_valid_value_passes_through(self):
        config = ScribeConfig(gateway_backend="langgraph_server")
        assert config.gateway_backend == "langgraph_server"

    def test_gateway_backend_per_surface_defaults_split(self):
        """Global server flips the fixed-workspace surfaces, not the CLI.

        single-shot/standalone default to ``inherit`` (so a global
        ``langgraph_server`` moves them); the interactive CLI defaults to
        ``local``, so the global flag can't move it (the server path is lossy
        there until #413/#454 close).
        """
        config = ScribeConfig(gateway_backend="langgraph_server")
        for surface in (
            GatewaySurface.SINGLE_SHOT,
            GatewaySurface.STANDALONE,
        ):
            assert resolve_gateway_backend(config, surface) == "langgraph_server"
        assert resolve_gateway_backend(config, GatewaySurface.INTERACTIVE) == "local"

    def test_gateway_backend_per_surface_override_wins(self):
        """A per-surface override resolves independently of the global flag."""
        config = ScribeConfig(
            gateway_backend="local",
            gateway_backend_single_shot="langgraph_server",
        )
        assert (
            resolve_gateway_backend(config, GatewaySurface.SINGLE_SHOT)
            == "langgraph_server"
        )
        # INTERACTIVE defaults to local, and the global is local too.
        assert resolve_gateway_backend(config, GatewaySurface.INTERACTIVE) == "local"

    def test_gateway_backend_cli_explicit_server_opt_in(self):
        """The CLI reaches the server backend only via an explicit override.

        Its default is ``local``, so opting in takes a per-surface
        ``langgraph_server`` (not the global flag); ``inherit`` restores
        following the global.
        """
        config = ScribeConfig(
            gateway_backend="local",
            gateway_backend_interactive="langgraph_server",
        )
        assert (
            resolve_gateway_backend(config, GatewaySurface.INTERACTIVE)
            == "langgraph_server"
        )
        config.gateway_backend_interactive = "inherit"
        assert resolve_gateway_backend(config, GatewaySurface.INTERACTIVE) == "local"

    def test_gateway_backend_per_surface_invalid_value_coerces_to_inherit(self, caplog):
        """A typo'd per-surface override degrades to inherit, not to a raw value."""
        import logging as _logging

        with caplog.at_level(_logging.WARNING, logger="metasci_scribe.config.settings"):
            config = ScribeConfig(
                gateway_backend="langgraph_server",
                gateway_backend_single_shot="langgraph-server",  # hyphen typo
            )
        assert config.gateway_backend_single_shot == "inherit"
        assert "Invalid gateway_backend_single_shot" in caplog.text
        # Inherit means the (valid) global still applies.
        assert (
            resolve_gateway_backend(config, GatewaySurface.SINGLE_SHOT)
            == "langgraph_server"
        )

    def test_ui_backend_legacy_aliases_and_case_fold(self):
        """Legacy ``ui_backend`` spellings survive Literal normalization.

        Earlier onboarding wrote ``rich`` into ``config.yaml``. The Literal
        check must fold case and honour that alias. The retired ``tui`` and
        ``webui`` values (and ``textual``) fall back to ``cli`` with a warning.
        """
        assert ScribeConfig(ui_backend="rich").ui_backend == "cli"
        assert ScribeConfig(ui_backend="CLI").ui_backend == "cli"
        assert ScribeConfig(ui_backend="Rich").ui_backend == "cli"
        for retired in ("tui", "webui", "textual"):
            assert ScribeConfig(ui_backend=retired).ui_backend == "cli"


# =============================================================================
# Test config path functions
# =============================================================================


class TestConfigPaths:
    def test_get_config_dir_ignores_a_later_xdg_change(self, monkeypatch, tmp_path):
        """Read once at startup: a workspace ``.env`` loaded later, which can
        set ``XDG_CONFIG_HOME``, does not move the config folder mid-run."""
        before = get_config_dir()
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "elsewhere"))
        assert get_config_dir() == before

    def test_get_config_path(self, temp_config_dir):
        """Test config file path."""
        config_path = get_config_path()
        assert config_path == temp_config_dir / "config.yaml"


# =============================================================================
# Test load/save/reset
# =============================================================================


class TestLoadSaveReset:
    def test_blank_yaml_values_fall_back_to_defaults(self, temp_config_dir):
        """A hand-edited ``key:`` loads as None; it must not reach the config."""
        from metasci_scribe.config import get_config_path

        path = get_config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "provider: anthropic\nanthropic_base_url:\nlanggraph_dev_port:\n"
        )

        config = load_config()

        assert config.provider == "anthropic"
        assert config.anthropic_base_url == ""
        assert config.langgraph_dev_port == ScribeConfig().langgraph_dev_port

    def test_load_returns_defaults_when_no_file(self, temp_config_dir, clean_env):
        """Test that load returns defaults when config file doesn't exist."""
        config = load_config()
        assert config.provider == "openai"
        assert config.model == "gpt-6.1-sol"

    def test_save_creates_file(self, temp_config_dir, clean_env):
        """Test that save creates the config file."""
        config = ScribeConfig(provider="openai", model="gpt-4o")
        save_config(config)

        config_path = get_config_path()
        assert config_path.exists()

        with open(config_path) as f:
            data = yaml.safe_load(f)
        assert data["provider"] == "openai"
        assert data["model"] == "gpt-4o"

    def test_save_restricts_config_permissions(self, temp_config_dir, clean_env):
        """Config file permissions should not depend on the process umask."""
        original_umask = os.umask(0)
        try:
            save_config(ScribeConfig(anthropic_api_key="test-key"))
        finally:
            os.umask(original_umask)

        config_path = get_config_path()
        if os.name == "nt":
            assert config_path.exists()
            # Windows reports pseudo-permission bits, so we don't test them here.
            return

        assert config_path.parent.stat().st_mode & 0o777 == 0o700
        assert config_path.stat().st_mode & 0o777 == 0o600

    def test_load_reads_saved_config(self, temp_config_dir, clean_env):
        """Test that load reads previously saved config."""
        original = ScribeConfig(
            anthropic_api_key="test-key",
            provider="openai",
        )
        save_config(original)

        loaded = load_config()
        assert loaded.anthropic_api_key == "test-key"
        assert loaded.provider == "openai"

    def test_load_reads_manually_edited_utf8_config(self, temp_config_dir, clean_env):
        """Config loading handles localized content from manually edited YAML."""
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "provider: openai\nmodel: gpt-4o\ndefault_workdir: /tmp/café\n",
            encoding="utf-8",
        )

        loaded = load_config()

        assert loaded.provider == "openai"
        assert loaded.model == "gpt-4o"
        assert loaded.default_workdir == "/tmp/café"

    def test_save_writes_utf8_config(self, temp_config_dir, clean_env):
        """Config saving writes localized content with UTF-8 encoding."""
        save_config(ScribeConfig(default_workdir="/tmp/résumé"))

        config_path = get_config_path()
        raw = config_path.read_text(encoding="utf-8")
        assert "default_workdir: /tmp/résumé" in raw
        assert load_config().default_workdir == "/tmp/résumé"

    def test_reset_deletes_config_file(self, temp_config_dir, clean_env):
        """Test that reset deletes the config file."""
        config = ScribeConfig(provider="openai")
        save_config(config)

        config_path = get_config_path()
        assert config_path.exists()

        reset_config()
        assert not config_path.exists()

    def test_reset_no_file_is_safe(self, temp_config_dir, clean_env):
        """Test that reset is safe when no config file exists."""
        reset_config()  # Should not raise

    def test_load_ignores_invalid_fields(self, temp_config_dir, clean_env):
        """Test that load ignores unknown fields in config file."""
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            yaml.safe_dump(
                {
                    "provider": "openai",
                    "unknown_field": "should_be_ignored",
                    "another_bad": 123,
                },
                f,
            )

        config = load_config()
        assert config.provider == "openai"
        assert not hasattr(config, "unknown_field")


# =============================================================================
# Test get/set single values
# =============================================================================


class TestGetSetValues:
    def test_get_config_value(self, temp_config_dir, clean_env):
        """Test getting a single config value."""
        config = ScribeConfig(model="gpt-4o-mini")
        save_config(config)

        assert get_config_value("model") == "gpt-4o-mini"

    def test_get_config_value_invalid_key(self, temp_config_dir, clean_env):
        """Test getting an invalid key returns None."""
        assert get_config_value("nonexistent_key") is None

    def test_set_config_value(self, temp_config_dir, clean_env):
        """Test setting a single config value."""
        save_config(ScribeConfig())

        result = set_config_value("model", "gpt-4o")
        assert result is True

        config = load_config()
        assert config.model == "gpt-4o"

    def test_set_config_value_invalid_key(self, temp_config_dir, clean_env):
        """Test setting an invalid key returns False."""
        result = set_config_value("nonexistent_key", "value")
        assert result is False

    def test_set_config_value_type_coercion_bool(self, temp_config_dir, clean_env):
        """Test that boolean values are coerced correctly."""
        save_config(ScribeConfig())

        set_config_value("show_thinking", "false")
        assert get_config_value("show_thinking") is False

        set_config_value("show_thinking", "1")
        assert get_config_value("show_thinking") is True

        set_config_value("show_thinking", "yes")
        assert get_config_value("show_thinking") is True

    def test_set_memory_observation_writer_validates_value(
        self, temp_config_dir, clean_env
    ):
        """Observation writer mode accepts only the supported product controls."""
        save_config(ScribeConfig(memory_observation_writer=MemoryObservationWriter.ALL))

        assert set_config_value("memory_observation_writer", "worker") is True
        assert get_config_value("memory_observation_writer") == "worker"
        assert set_config_value("memory_observation_writer", "AGENT") is True
        assert get_config_value("memory_observation_writer") == "agent"
        assert set_config_value("memory_observation_writer", "subagent") is False
        assert get_config_value("memory_observation_writer") == "agent"

    def test_memory_controls_observation_writer_targets(self):
        """MemoryControls centralizes observation writer target semantics."""
        worker_controls = MemoryControls.from_config(
            ScribeConfig(
                memory_profile_enabled=False,
                memory_observations_enabled=True,
                memory_observation_writer=MemoryObservationWriter.WORKER,
                memory_workers_enabled=True,
            )
        )
        all_controls = MemoryControls.from_config(
            ScribeConfig(
                memory_profile_enabled=False,
                memory_observations_enabled=True,
                memory_observation_writer=MemoryObservationWriter.ALL,
                memory_workers_enabled=True,
            )
        )

        assert worker_controls.observation_tool_enabled(
            MemoryObservationTarget.TURN_WORKER
        )
        assert worker_controls.worker_needed(MemoryObservationTarget.TURN_WORKER)
        assert worker_controls.observation_tool_enabled(
            MemoryObservationTarget.SUBAGENT_WORKER
        )
        assert not worker_controls.observation_tool_enabled(
            MemoryObservationTarget.AGENT
        )
        assert all_controls.worker_needed(MemoryObservationTarget.TURN_WORKER)
        assert all_controls.observation_tool_enabled(MemoryObservationTarget.AGENT)
        assert all_controls.observation_tool_enabled(
            MemoryObservationTarget.TURN_WORKER
        )
        assert all_controls.observation_tool_enabled(
            MemoryObservationTarget.SUBAGENT_WORKER
        )

    def test_set_memory_skill_synthesis_values_validate(
        self, temp_config_dir, clean_env
    ):
        save_config(ScribeConfig())

        assert set_config_value("memory_skill_synthesis_mode", "auto") is True
        assert get_config_value("memory_skill_synthesis_mode") == "auto"
        assert set_config_value("memory_skill_synthesis_mode", "always") is False
        assert get_config_value("memory_skill_synthesis_mode") == "auto"

        assert set_config_value("memory_skill_synthesis_cadence", "nightly") is True
        assert get_config_value("memory_skill_synthesis_cadence") == "nightly"
        assert set_config_value("memory_skill_synthesis_cadence", "hourly") is False
        assert get_config_value("memory_skill_synthesis_cadence") == "nightly"

        assert set_config_value("memory_skill_synthesis_time", "3:05") is True
        assert get_config_value("memory_skill_synthesis_time") == "03:05"
        assert set_config_value("memory_skill_synthesis_time", "24:00") is False
        assert get_config_value("memory_skill_synthesis_time") == "03:05"

        loaded = load_config()
        assert loaded.memory_skill_synthesis_mode == MemorySkillSynthesisMode.AUTO
        assert (
            loaded.memory_skill_synthesis_cadence == MemorySkillSynthesisCadence.NIGHTLY
        )

    def test_list_config(self, temp_config_dir, clean_env):
        """Test listing all config values."""
        config = ScribeConfig(provider="openai", model="gpt-4o")
        save_config(config)

        all_config = list_config()
        assert isinstance(all_config, dict)
        assert all_config["provider"] == "openai"
        assert all_config["model"] == "gpt-4o"


# =============================================================================
# Test priority chain (CLI > env > file > default)
# =============================================================================


class TestPriorityChain:
    def test_defaults_when_nothing_set(self, temp_config_dir, clean_env, monkeypatch):
        """Test defaults are used when nothing is configured."""
        # Ensure no env vars affect the test
        for key in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "TAVILY_API_KEY",
            "METASCI_SCRIBE_DEFAULT_MODE",
            "METASCI_SCRIBE_WORKSPACE_DIR",
            "METASCI_SCRIBE_UI_BACKEND",
            "METASCI_SCRIBE_GATEWAY_BACKEND",
            "METASCI_SCRIBE_GATEWAY_BACKEND_SINGLE_SHOT",
            "METASCI_SCRIBE_GATEWAY_BACKEND_INTERACTIVE",
            "METASCI_SCRIBE_GATEWAY_BACKEND_STANDALONE",
        ]:
            monkeypatch.delenv(key, raising=False)
        config = get_effective_config()
        assert config.provider == "openai"
        assert config.default_mode == "daemon"

    def test_file_overrides_defaults(self, temp_config_dir, clean_env):
        """Test file config overrides defaults."""
        save_config(ScribeConfig(provider="openai"))

        config = get_effective_config()
        assert config.provider == "openai"

    def test_env_overrides_file(self, temp_config_dir, monkeypatch):
        """Test environment variables override file config."""
        save_config(ScribeConfig(default_mode="run"))
        monkeypatch.setenv("METASCI_SCRIBE_DEFAULT_MODE", "daemon")

        config = get_effective_config()
        assert config.default_mode == "daemon"

    def test_cli_overrides_env(self, temp_config_dir, monkeypatch):
        """Test CLI arguments override environment variables."""
        monkeypatch.setenv("METASCI_SCRIBE_DEFAULT_MODE", "daemon")

        config = get_effective_config(cli_overrides={"default_mode": "run"})
        assert config.default_mode == "run"

    def test_cli_overrides_file(self, temp_config_dir, clean_env):
        """Test CLI arguments override file config."""
        save_config(ScribeConfig(model="gpt-4o"))

        config = get_effective_config(cli_overrides={"model": "claude-opus-4-8"})
        assert config.model == "claude-opus-4-8"

    def test_env_ui_backend_retired_value_is_ignored(
        self, temp_config_dir, monkeypatch
    ):
        """The retired ``tui`` spelling is rejected, leaving the CLI in place."""
        save_config(ScribeConfig(ui_backend="cli"))
        monkeypatch.setenv("METASCI_SCRIBE_UI_BACKEND", "tui")
        config = get_effective_config()
        assert config.ui_backend == "cli"

    def test_env_ui_backend_legacy_alias_override(self, temp_config_dir, monkeypatch):
        """A legacy env spelling resolves through the same alias map.

        Without case-folding and alias support in ``_coerce_value`` the
        override raises and is swallowed by ``get_effective_config``, so
        ``METASCI_SCRIBE_UI_BACKEND=rich`` is silently ignored.
        """
        save_config(ScribeConfig(ui_backend="cli"))
        monkeypatch.setenv("METASCI_SCRIBE_UI_BACKEND", "rich")
        config = get_effective_config()
        assert config.ui_backend == "cli"

    def test_env_gateway_backend_invalid_value_falls_back_to_local(
        self, temp_config_dir, monkeypatch
    ):
        """An out-of-set env value is skipped, leaving the default in place.

        ``_coerce_value`` rejects values outside the Literal's allowed set,
        so the env-override loop skips the malformed entry instead of
        storing it verbatim — consistent with how malformed env values for
        other field types already degrade to defaults.
        """
        monkeypatch.setenv("METASCI_SCRIBE_GATEWAY_BACKEND", "langgraph-server")
        config = get_effective_config()
        assert config.gateway_backend == "local"

    def test_env_gateway_backend_per_surface_override(
        self, temp_config_dir, monkeypatch
    ):
        """Each per-surface backend has its own env var, resolving independently."""
        monkeypatch.setenv(
            "METASCI_SCRIBE_GATEWAY_BACKEND_SINGLE_SHOT", "langgraph_server"
        )
        config = get_effective_config()
        assert config.gateway_backend_single_shot == "langgraph_server"
        assert config.gateway_backend_interactive == "local"  # CLI default
        assert (
            resolve_gateway_backend(config, GatewaySurface.SINGLE_SHOT)
            == "langgraph_server"
        )
        assert resolve_gateway_backend(config, GatewaySurface.INTERACTIVE) == "local"

    def test_env_log_level_override(self, temp_config_dir, monkeypatch):
        """Log level can be selected via environment variable."""
        save_config(ScribeConfig(log_level="warning"))
        monkeypatch.setenv("METASCI_SCRIBE_LOG_LEVEL", "DEBUG")
        config = get_effective_config()
        assert config.log_level == "DEBUG"

    def test_env_reasoning_effort_override(self, temp_config_dir, monkeypatch):
        """Reasoning effort can be selected via environment variable."""
        save_config(ScribeConfig(reasoning_effort="medium"))
        monkeypatch.setenv("METASCI_SCRIBE_REASONING_EFFORT", "high")
        config = get_effective_config()
        assert config.reasoning_effort == "high"

    def test_env_memory_controls_override(self, temp_config_dir, monkeypatch):
        """Memory controls can be selected via environment variables."""
        save_config(
            ScribeConfig(
                memory_profile_enabled=True,
                memory_observations_enabled=True,
                memory_observation_writer=MemoryObservationWriter.ALL,
                memory_workers_enabled=True,
            )
        )
        monkeypatch.setenv("METASCI_SCRIBE_MEMORY_PROFILE_ENABLED", "false")
        monkeypatch.setenv("METASCI_SCRIBE_MEMORY_OBSERVATIONS_ENABLED", "false")
        monkeypatch.setenv("METASCI_SCRIBE_MEMORY_OBSERVATION_WRITER", "worker")
        monkeypatch.setenv("METASCI_SCRIBE_MEMORY_WORKERS_ENABLED", "false")

        config = get_effective_config()
        assert config.memory_profile_enabled is False
        assert config.memory_observations_enabled is False
        assert config.memory_observation_writer == MemoryObservationWriter.WORKER
        assert config.memory_workers_enabled is False

    def test_sandbox_execute_timeout_default(self, temp_config_dir, clean_env):
        """Sandbox execute timeout defaults to 300 seconds."""
        assert ScribeConfig().sandbox_execute_timeout == 300
        assert get_effective_config().sandbox_execute_timeout == 300

    def test_env_sandbox_execute_timeout_override(self, temp_config_dir, monkeypatch):
        """Sandbox execute timeout can be set via env var and coerces to int."""
        monkeypatch.setenv("METASCI_SCRIBE_SANDBOX_EXECUTE_TIMEOUT", "600")
        config = get_effective_config()
        assert config.sandbox_execute_timeout == 600
        assert isinstance(config.sandbox_execute_timeout, int)

    def test_sandbox_execute_timeout_invalid_falls_back(self):
        """Non-positive / non-int values fall back to the default (would
        otherwise crash CustomSandboxBackend construction at startup)."""
        assert ScribeConfig(sandbox_execute_timeout=0).sandbox_execute_timeout == 300
        assert ScribeConfig(sandbox_execute_timeout=-5).sandbox_execute_timeout == 300
        assert (
            ScribeConfig(sandbox_execute_timeout="abc").sandbox_execute_timeout == 300
        )

    def test_backend_allowed_origins_yaml_list_normalized_to_csv(self):
        config = ScribeConfig(
            backend_allowed_origins=["https://a.example", "https://b.example"]
        )
        assert config.backend_allowed_origins == "https://a.example,https://b.example"

    def test_shell_allow_list_yaml_list_normalized_to_csv(self):
        """A YAML list spelling survives ``__post_init__`` as a list; normalize
        to CSV so the policy resolver's ``.split`` never raises AttributeError."""
        assert ScribeConfig(shell_allow_list=["ls", "cat"]).shell_allow_list == (
            "ls,cat"
        )
        assert ScribeConfig(shell_allow_list=("ls",)).shell_allow_list == "ls"
        # The normal comma-separated string is left untouched.
        assert ScribeConfig(shell_allow_list="ls,cat").shell_allow_list == ("ls,cat")
        assert ScribeConfig(sandbox_execute_timeout=True).sandbox_execute_timeout == 300

    def test_set_sandbox_execute_timeout_rejects_invalid(
        self, temp_config_dir, clean_env
    ):
        """set_config_value must reject (not silently persist) a non-positive timeout."""
        save_config(ScribeConfig(sandbox_execute_timeout=120))
        assert set_config_value("sandbox_execute_timeout", 0) is False
        assert set_config_value("sandbox_execute_timeout", -5) is False
        # bool is an int subclass; reject it before coercion turns True into 1.
        assert set_config_value("sandbox_execute_timeout", True) is False
        # The earlier valid value is untouched on disk.
        assert get_config_value("sandbox_execute_timeout") == 120
        # A valid value still goes through.
        assert set_config_value("sandbox_execute_timeout", 600) is True
        assert get_config_value("sandbox_execute_timeout") == 600

    def test_observation_cache_max_files_default(self, temp_config_dir, clean_env):
        """Observation cache cap defaults to 2048."""
        assert ScribeConfig().memory_observation_cache_max_files == 2048
        assert get_effective_config().memory_observation_cache_max_files == 2048

    def test_env_observation_cache_max_files_override(
        self, temp_config_dir, monkeypatch
    ):
        """Observation cache cap can be set via env var."""
        monkeypatch.setenv("METASCI_SCRIBE_MAX_CACHED_FILES", "16")
        config = get_effective_config()
        assert config.memory_observation_cache_max_files == 16

    def test_observation_cache_max_files_invalid_falls_back(self):
        """Non-positive / non-int values fall back to the default."""
        assert (
            ScribeConfig(
                memory_observation_cache_max_files=0
            ).memory_observation_cache_max_files
            == 2048
        )
        assert (
            ScribeConfig(
                memory_observation_cache_max_files=-1
            ).memory_observation_cache_max_files
            == 2048
        )
        assert (
            ScribeConfig(
                memory_observation_cache_max_files=True
            ).memory_observation_cache_max_files
            == 2048
        )

    def test_env_api_key_override(self, temp_config_dir, monkeypatch):
        """Test API keys from env override file."""
        save_config(ScribeConfig(anthropic_api_key="file-key"))
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

        config = get_effective_config()
        assert config.anthropic_api_key == "env-key"

    def test_env_openai_auth_mode_override(self, temp_config_dir, monkeypatch):
        """Test openai_auth_mode from env overrides file."""
        save_config(ScribeConfig(openai_auth_mode="api_key"))
        monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")

        config = get_effective_config()
        assert config.openai_auth_mode == "chatgpt_plan"


# =============================================================================
# Test apply_config_to_env
# =============================================================================


class TestApplyConfigToEnv:
    def test_applies_api_keys_when_not_set(self, clean_env):
        """Test that API keys are applied to env when not already set."""
        config = ScribeConfig(
            anthropic_api_key="config-ant-key",
            openai_api_key="config-oai-key",
        )

        apply_config_to_env(config)

        assert os.environ.get("ANTHROPIC_API_KEY") == "config-ant-key"
        assert os.environ.get("OPENAI_API_KEY") == "config-oai-key"

    def test_does_not_override_existing_env(self, monkeypatch):
        """Test that existing env vars are not overridden."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "existing-key")

        config = ScribeConfig(anthropic_api_key="config-key")
        apply_config_to_env(config)

        assert os.environ.get("ANTHROPIC_API_KEY") == "existing-key"

    def test_empty_config_keys_not_applied(self, clean_env):
        """Test that empty config keys don't create env vars."""
        config = ScribeConfig()  # All empty
        apply_config_to_env(config)

        assert os.environ.get("ANTHROPIC_API_KEY") is None
        assert os.environ.get("OPENAI_API_KEY") is None

    def test_dangerous_mode_round_trips_to_env(self, clean_env, monkeypatch):
        """dangerous_mode set via CLI override must survive a fresh re-read.

        Regression: a --dangerous CLI flag is never persisted to file/env, so a
        fresh get_effective_config() (warning banner, run_in_background, the
        langgraph dev subprocess) would read it back as False while the backend
        is already unconfined. apply_config_to_env round-trips it to env.
        """
        from metasci_scribe.config import get_effective_config

        monkeypatch.delenv("METASCI_SCRIBE_DANGEROUS_MODE", raising=False)
        cfg = get_effective_config({"dangerous_mode": True})
        assert cfg.dangerous_mode is True

        apply_config_to_env(cfg)
        assert os.environ.get("METASCI_SCRIBE_DANGEROUS_MODE") == "true"
        # The fresh, no-override read now agrees with the backend.
        assert get_effective_config().dangerous_mode is True

    def test_dangerous_mode_not_applied_when_off(self, clean_env, monkeypatch):
        """dangerous_mode=False must not write the env var."""
        monkeypatch.delenv("METASCI_SCRIBE_DANGEROUS_MODE", raising=False)
        apply_config_to_env(ScribeConfig())
        assert os.environ.get("METASCI_SCRIBE_DANGEROUS_MODE") is None

    def test_dangerous_mode_off_clears_stale_env(self, clean_env, monkeypatch):
        """Re-applying a non-dangerous config clears a previously-set env var.

        Regression: the round-trip used to be set-only, so once dangerous mode
        was applied in a process it could never be lowered — leaking unconfined
        access into later non-dangerous reads.
        """
        monkeypatch.delenv("METASCI_SCRIBE_DANGEROUS_MODE", raising=False)
        apply_config_to_env(ScribeConfig(dangerous_mode=True))
        assert os.environ.get("METASCI_SCRIBE_DANGEROUS_MODE") == "true"
        apply_config_to_env(ScribeConfig())  # dangerous off
        assert os.environ.get("METASCI_SCRIBE_DANGEROUS_MODE") is None

    def test_openai_auth_mode_chatgpt_plan_is_exported(self, clean_env):
        apply_config_to_env(ScribeConfig(openai_auth_mode="chatgpt_plan"))

        assert os.environ["METASCI_SCRIBE_OPENAI_AUTH_MODE"] == "chatgpt_plan"

    def test_openai_auth_mode_api_key_replaces_a_stale_value(
        self, clean_env, monkeypatch
    ):
        monkeypatch.setenv("METASCI_SCRIBE_OPENAI_AUTH_MODE", "chatgpt_plan")

        apply_config_to_env(ScribeConfig(openai_auth_mode="api_key"))

        assert os.environ["METASCI_SCRIBE_OPENAI_AUTH_MODE"] == "api_key"

    def test_cli_override_reaches_a_fresh_resolve(self, clean_env, temp_config_dir):
        """The langgraph dev child re-resolves config from file + env only."""
        save_config(ScribeConfig(openai_auth_mode="oauth"))
        resolved = get_effective_config({"openai_auth_mode": "api_key"})

        apply_config_to_env(resolved)

        assert get_effective_config().openai_auth_mode == "api_key"


class TestAuxiliaryModelConfig:
    """auxiliary_model / auxiliary_provider config fields (plain str, optional)."""

    def test_defaults_empty(self):
        cfg = ScribeConfig()
        assert cfg.auxiliary_model == ""
        assert cfg.auxiliary_provider == ""

    def test_save_and_load_round_trip(self, temp_config_dir, clean_env):
        save_config(
            ScribeConfig(
                auxiliary_model="claude-haiku-4-5",
                auxiliary_provider="anthropic",
            )
        )
        loaded = load_config()
        assert loaded.auxiliary_model == "claude-haiku-4-5"
        assert loaded.auxiliary_provider == "anthropic"

    def test_get_set_value(self, temp_config_dir, clean_env):
        save_config(ScribeConfig())
        assert set_config_value("auxiliary_model", "gpt-6-luna") is True
        assert set_config_value("auxiliary_provider", "openai") is True
        assert get_config_value("auxiliary_model") == "gpt-6-luna"
        assert get_config_value("auxiliary_provider") == "openai"

    def test_env_overrides_file(self, temp_config_dir, monkeypatch):
        save_config(ScribeConfig(auxiliary_model="claude-haiku-4-5"))
        monkeypatch.setenv("METASCI_SCRIBE_AUXILIARY_MODEL", "gpt-5.5")
        monkeypatch.setenv("METASCI_SCRIBE_AUXILIARY_PROVIDER", "openai")

        config = get_effective_config()
        assert config.auxiliary_model == "gpt-5.5"
        assert config.auxiliary_provider == "openai"


def test_scheduler_config_defaults_and_env(monkeypatch):
    from metasci_scribe.config.settings import ScribeConfig, get_effective_config

    c = ScribeConfig()
    assert c.enable_scheduler is False
    assert c.scheduler_default_timezone == ""
    assert c.memory_skill_synthesis_enabled is False
    assert c.memory_skill_synthesis_mode == MemorySkillSynthesisMode.REVIEW
    assert c.memory_skill_synthesis_cadence == MemorySkillSynthesisCadence.WEEKLY
    assert c.memory_skill_synthesis_time == "03:00"

    monkeypatch.setenv("METASCI_SCRIBE_ENABLE_SCHEDULER", "true")
    eff = get_effective_config({})
    assert eff.enable_scheduler is True

    monkeypatch.setenv("METASCI_SCRIBE_SCHEDULER_DEFAULT_TIMEZONE", "America/New_York")
    monkeypatch.setenv("METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_ENABLED", "true")
    monkeypatch.setenv("METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_MODE", "auto")
    monkeypatch.setenv("METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_CADENCE", "monthly")
    monkeypatch.setenv("METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_TIME", "4:30")
    eff2 = get_effective_config({})
    assert eff2.scheduler_default_timezone == "America/New_York"
    assert eff2.memory_skill_synthesis_enabled is True
    assert eff2.memory_skill_synthesis_mode == MemorySkillSynthesisMode.AUTO
    assert eff2.memory_skill_synthesis_cadence == MemorySkillSynthesisCadence.MONTHLY
    assert eff2.memory_skill_synthesis_time == "04:30"


# =============================================================================
# Dotenv isolation (issue #322)
# =============================================================================


class TestDotenvIsolation:
    def test_env_file_not_leaked_into_process_env(self, tmp_path, monkeypatch):
        """A .env in cwd must not leak into os.environ during tests.

        Without the suite-wide ``_isolate_dotenv`` fixture,
        ``get_effective_config`` loads the developer's real .env with
        ``override=True``; an empty-valued line like ``MINIMAX_BASE_URL=``
        then poisons ``os.environ.get(key, default)`` lookups for every
        test that runs afterwards in the same process.
        """
        repro_dir = tmp_path / "repro"
        repro_dir.mkdir()
        (repro_dir / ".env").write_text("MINIMAX_BASE_URL=\n")
        monkeypatch.chdir(repro_dir)
        monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

        get_effective_config()

        assert "MINIMAX_BASE_URL" not in os.environ

    def test_env_example_template_parses_to_blank_values(self):
        """Inline comments would be parsed as values ("KEY=  # note" -> "# note")."""
        from dotenv import dotenv_values

        template = Path(__file__).resolve().parents[1] / ".env.example"
        values = dotenv_values(template)

        assert values
        assert {k: v for k, v in values.items() if v} == {}

    def test_copied_template_does_not_shadow_config_keys(
        self, temp_config_dir, tmp_path, monkeypatch
    ):
        """Copying the template and filling one key must leave config-file keys intact."""
        template = Path(__file__).resolve().parents[1] / ".env.example"
        env_file = tmp_path / ".env"
        env_file.write_text(
            template.read_text().replace(
                "ANTHROPIC_API_KEY=\n", "ANTHROPIC_API_KEY=sk-ant-filled\n"
            )
        )
        monkeypatch.setattr(
            "metasci_scribe.config.settings.find_dotenv",
            lambda *args, **kwargs: str(env_file),
        )
        from dotenv import dotenv_values

        # setenv (unlike delenv on an absent key) registers a restore, so the
        # merge's os.environ writes are undone after the test.
        for key in dotenv_values(env_file):
            monkeypatch.setenv(key, "")
        save_config(ScribeConfig(openai_api_key="sk-from-config"))

        config = get_effective_config()

        assert config.anthropic_api_key == "sk-ant-filled"
        assert config.openai_api_key == "sk-from-config"
        assert os.environ.get("MINIMAX_BASE_URL", "") == ""

    def test_dotenv_credentials_still_merge_without_oauth(
        self, temp_config_dir, tmp_path, monkeypatch
    ):
        """Without the placeholder, ``.env`` credentials keep winning as before."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-from-dotenv\n")
        monkeypatch.setattr(
            "metasci_scribe.config.settings.find_dotenv",
            lambda *args, **kwargs: str(env_file),
        )
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-shell")

        get_effective_config()

        assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv"

    def test_parent_env_wins_over_dotenv_for_mapped_keys(
        self, temp_config_dir, tmp_path, monkeypatch
    ):
        """Parent-process env values for ``METASCI_SCRIBE_*`` keys must not be
        shadowed by a workspace ``.env``. The CLI propagates the resolved bind
        port via ``METASCI_SCRIBE_LANGGRAPH_DEV_PORT`` on the
        subprocess env; without the prefix-based merge, a workspace ``.env``
        with the same key would clobber it and every self-loop async task
        would target the wrong port.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("METASCI_SCRIBE_LANGGRAPH_DEV_PORT=9999\n")
        monkeypatch.setattr(
            "metasci_scribe.config.settings.find_dotenv",
            lambda *args, **kwargs: str(env_file),
        )
        monkeypatch.setenv("METASCI_SCRIBE_LANGGRAPH_DEV_PORT", "6606")

        config = get_effective_config()

        assert config.langgraph_dev_port == 6606
        assert os.environ["METASCI_SCRIBE_LANGGRAPH_DEV_PORT"] == "6606"

    @pytest.mark.parametrize(
        ("field", "env_var", "value"),
        [
            ("langgraph_dev_host", "METASCI_SCRIBE_LANGGRAPH_DEV_HOST", "0.0.0.0"),
        ],
    )
    def test_bind_hosts_are_env_overridable(
        self, temp_config_dir, monkeypatch, field, env_var, value
    ):
        """``start_langgraph_dev`` propagates the resolved bind host to the
        subprocess through ``METASCI_SCRIBE_LANGGRAPH_DEV_HOST``, so the host
        field must be declared in ``_ENV_MAPPINGS`` or the subprocess would
        fall back to whatever the config file says."""
        monkeypatch.setenv(env_var, value)

        config = get_effective_config()

        assert getattr(config, field) == value

    def test_dotenv_wins_over_shell_for_third_party_api_keys(
        self, temp_config_dir, tmp_path, monkeypatch
    ):
        """Third-party API keys (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
        ...) are conventionally set per-project via a workspace ``.env`` to
        shadow whatever global key sits in the shell (e.g. ``~/.bashrc``).
        The prefix-based ``.env`` merge must not disturb that: only
        ``METASCI_SCRIBE_*`` keys are treated as parent-process-authoritative.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=workspace-key\n")
        monkeypatch.setattr(
            "metasci_scribe.config.settings.find_dotenv",
            lambda *args, **kwargs: str(env_file),
        )
        monkeypatch.setenv("OPENAI_API_KEY", "shell-key")

        config = get_effective_config()

        assert config.openai_api_key == "workspace-key"
        assert os.environ["OPENAI_API_KEY"] == "workspace-key"

    def test_snapshot_covers_scribe_keys_not_in_env_mappings(
        self, temp_config_dir, tmp_path, monkeypatch
    ):
        """The snapshot must protect every ``METASCI_SCRIBE_*`` key in the shell,
        not just the ones declared in ``_ENV_MAPPINGS``. Concrete case: the
        langgraph_dev manager sets ``METASCI_SCRIBE_SERVER_PROCESS`` on the
        subprocess env to mark it as the server, but
        that key is read directly via ``os.environ.get(...)`` and never goes
        through ``get_effective_config`` — so it never made it into
        ``_ENV_MAPPINGS``. Without the prefix-based snapshot, a workspace
        ``.env`` with ``METASCI_SCRIBE_SERVER_PROCESS=`` could clobber the
        parent-injected value and silently disable async subagents.
        """
        env_file = tmp_path / ".env"
        env_file.write_text("METASCI_SCRIBE_SERVER_PROCESS=\n")
        monkeypatch.setattr(
            "metasci_scribe.config.settings.find_dotenv",
            lambda *args, **kwargs: str(env_file),
        )
        monkeypatch.setenv("METASCI_SCRIBE_SERVER_PROCESS", "1")

        get_effective_config()

        assert os.environ["METASCI_SCRIBE_SERVER_PROCESS"] == "1"

    def test_a_workspace_dotenv_never_sets_a_scribe_key(
        self, temp_config_dir, tmp_path, monkeypatch
    ):
        """``METASCI_SCRIBE_*`` settings come from the user's config and shell
        only. A workspace ``.env`` is a file the agent can be steered to
        write, so even with the shell key unset or empty it sets nothing:
        it could otherwise raise the agent's own mode
        (``policy.dotenv_key_allowed``).
        """
        env_file = tmp_path / ".env"
        env_file.write_text(
            "METASCI_SCRIBE_LANGGRAPH_DEV_PORT=6606\n"
            "METASCI_SCRIBE_DANGEROUS_MODE=true\n"
        )
        monkeypatch.setattr(
            "metasci_scribe.config.settings.find_dotenv",
            lambda *args, **kwargs: str(env_file),
        )
        monkeypatch.setenv("METASCI_SCRIBE_LANGGRAPH_DEV_PORT", "")
        monkeypatch.delenv("METASCI_SCRIBE_DANGEROUS_MODE", raising=False)

        config = get_effective_config()

        assert config.langgraph_dev_port == ScribeConfig().langgraph_dev_port
        assert config.dangerous_mode is False
        assert os.environ["METASCI_SCRIBE_LANGGRAPH_DEV_PORT"] == ""
        assert "METASCI_SCRIBE_DANGEROUS_MODE" not in os.environ


def test_backend_allowed_origins_ignores_the_environment(temp_config_dir, monkeypatch):
    """A workspace ``.env`` must not widen the backend's browser boundary."""
    monkeypatch.setenv("METASCI_SCRIBE_BACKEND_ALLOWED_ORIGINS", "https://evil.example")
    monkeypatch.setenv("BACKEND_ALLOWED_ORIGINS", "https://evil.example")
    assert get_effective_config().backend_allowed_origins == ""
