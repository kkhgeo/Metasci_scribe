"""Tests for Metasci Scribe onboarding wizard."""

import os
import subprocess
from contextlib import contextmanager
from unittest.mock import MagicMock, Mock, patch

import pytest

from metasci_scribe.config import ScribeConfig
from metasci_scribe.config.onboard.style import (
    CONFIRM_STYLE,
    WIZARD_STYLE,
)
from metasci_scribe.config.onboard.validators import (
    ChoiceValidator,
    IntegerValidator,
)
from metasci_scribe.config.onboard.wizard import (
    STEPS,
    _skills_workspace,
    render_progress,
)
from metasci_scribe.paths import Workspace


@contextmanager
def _patch_all_questionary(mock_q):
    """Patch ``questionary`` in every onboard submodule that imports it.

    The wizard delegates to step functions across multiple submodules
    (steps/helpers/style/wizard), and each submodule binds its own
    ``questionary`` name at import time. Patching just one location wouldn't
    intercept calls made from the others.

    Also forces ``_check_npx`` to True so the skills-step easter egg never
    probes the host: on runners where npx is missing/slow, ``_ensure_npx``
    would otherwise fire an extra ``questionary.confirm`` and shift the
    mocked side_effect sequences (platform-dependent StopIteration).
    """
    with (
        patch("metasci_scribe.config.onboard.wizard.questionary", mock_q),
        patch("metasci_scribe.config.onboard.steps.questionary", mock_q),
        patch("metasci_scribe.config.onboard.helpers.questionary", mock_q),
        patch("metasci_scribe.config.onboard.style.questionary", mock_q),
        patch("metasci_scribe.config.onboard.helpers._check_npx", return_value=True),
    ):
        yield mock_q


# =============================================================================
# Test STEPS and WIZARD_STYLE constants
# =============================================================================


class TestConstants:
    def test_steps_has_eleven_items(self):
        """Test that STEPS contains exactly 11 steps."""
        assert len(STEPS) == 11
        assert STEPS == [
            "UI",
            "LangGraph Port",
            "Provider",
            "API Key",
            "Model",
            "Auxiliary Model",
            "Workspace",
            "Thinking",
            "Skills",
            "MCP Servers",
            "LaTeX",
        ]

    def test_wizard_style_is_style_instance(self):
        """Test that WIZARD_STYLE is a prompt_toolkit Style."""
        from prompt_toolkit.styles import Style

        assert isinstance(WIZARD_STYLE, Style)

    def test_confirm_style_is_style_instance(self):
        """Test that CONFIRM_STYLE is a prompt_toolkit Style."""
        from prompt_toolkit.styles import Style

        assert isinstance(CONFIRM_STYLE, Style)

    def test_confirm_style_differs_from_wizard(self):
        """Test that CONFIRM_STYLE has a different qmark color (orange)."""
        assert CONFIRM_STYLE is not WIZARD_STYLE


class TestSharedConstantsAlignment:
    """Drift guard: the canonical valid-value sets in
    ``metasci_scribe.config.onboard.constants`` must match the actual
    ``Choice(value=...)`` ids built by the interactive step functions in
    ``steps.py``. Without this, adding a new provider to one file but not
    the other would silently break either CLI flag validation or the
    interactive picker.
    """

    def test_provider_constants_match_step_choices(self):
        """Every value in `_step_provider`'s Choice list must be in
        ``VALID_PROVIDERS`` — and vice versa."""
        from unittest.mock import MagicMock, patch

        from metasci_scribe.config.onboard.constants import VALID_PROVIDERS
        from metasci_scribe.config.onboard.steps import _step_provider

        # Intercept questionary.select to capture the choices list before
        # any user prompt happens.
        captured = {}

        def _capture(*args, **kwargs):
            captured["choices"] = kwargs.get("choices") or (
                args[1] if len(args) > 1 else []
            )
            fake = MagicMock()
            fake.ask.return_value = "anthropic"
            return fake

        with patch(
            "metasci_scribe.config.onboard.steps.questionary.select",
            side_effect=_capture,
        ):
            _step_provider(ScribeConfig())

        actual_provider_ids = {c.value for c in captured["choices"]}
        assert actual_provider_ids == set(VALID_PROVIDERS), (
            "VALID_PROVIDERS in constants.py drifted from _step_provider's "
            f"choices. Only-in-constants: {set(VALID_PROVIDERS) - actual_provider_ids}; "
            f"only-in-choices: {actual_provider_ids - set(VALID_PROVIDERS)}"
        )

    def test_workspace_mode_constants_match_step_choices(self):
        from metasci_scribe.config.onboard.constants import VALID_WORKSPACE_MODES

        assert VALID_WORKSPACE_MODES == frozenset({"daemon", "run"})

    def test_valid_providers_aligns_with_provider_table(self):
        """The provider table in ``provider_keys`` has exactly one row per
        provider in ``VALID_PROVIDERS``: an unknown provider is an error, not
        an OpenAI fallback, so a missing row would break that provider."""
        from metasci_scribe.config.onboard.constants import VALID_PROVIDERS
        from metasci_scribe.config.provider_keys import PROVIDERS

        assert set(PROVIDERS) == set(VALID_PROVIDERS)

    def test_provider_key_fields_exist_on_config(self):
        """Every key and base URL field in the table is a config field."""
        from dataclasses import fields

        from metasci_scribe.config.provider_keys import PROVIDERS

        names = {f.name for f in fields(ScribeConfig)}
        for provider, spec in PROVIDERS.items():
            for attr in (spec.key_field, spec.base_url_field):
                assert attr is None or attr in names, (provider, attr)

    def test_provider_key_info_rejects_unknown_provider(self):
        from metasci_scribe.config.onboard.helpers import _provider_key_info

        with pytest.raises(KeyError):
            _provider_key_info(ScribeConfig(), "no-such-provider")


# =============================================================================
# Test render_progress
# =============================================================================


class TestRenderProgress:
    def test_renders_first_step_active(self):
        """Test that first step is shown as active."""
        panel = render_progress(current_step=0, completed=set())
        # Panel should contain the title
        assert panel.title is not None
        # The renderable content should contain step names
        content_str = str(panel.renderable)
        assert "Provider" in content_str

    def test_renders_completed_steps(self):
        """Test that completed steps are marked differently."""
        panel = render_progress(current_step=2, completed={0, 1})
        content_str = str(panel.renderable)
        # All step names should be present
        for step in STEPS:
            assert step in content_str

    def test_panel_has_title(self):
        """Test that the panel has the expected title."""
        panel = render_progress(current_step=0, completed=set())
        assert "Metasci Scribe Setup" in str(panel.title)

    def test_panel_has_blue_border(self):
        """Test that the panel has a blue border style."""
        panel = render_progress(current_step=0, completed=set())
        assert panel.border_style == "blue"


# =============================================================================
# Test Validators
# =============================================================================


class TestIntegerValidator:
    def test_accepts_valid_integer(self):
        """Test that valid integers are accepted."""
        validator = IntegerValidator(min_value=1, max_value=10)

        class Doc:
            text = "5"

        # Should not raise
        validator.validate(Doc())

    def test_accepts_empty_for_default(self):
        """Test that empty string is accepted for using default."""
        validator = IntegerValidator(min_value=1, max_value=10)

        class Doc:
            text = ""

        # Should not raise
        validator.validate(Doc())

    def test_rejects_non_integer(self):
        """Test that non-integers are rejected."""
        from prompt_toolkit.validation import ValidationError

        validator = IntegerValidator(min_value=1, max_value=10)

        class Doc:
            text = "abc"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(Doc())
        assert "valid integer" in str(exc_info.value.message)

    def test_rejects_below_min(self):
        """Test that values below min are rejected."""
        from prompt_toolkit.validation import ValidationError

        validator = IntegerValidator(min_value=5, max_value=10)

        class Doc:
            text = "3"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(Doc())
        assert "between" in str(exc_info.value.message)

    def test_rejects_above_max(self):
        """Test that values above max are rejected."""
        from prompt_toolkit.validation import ValidationError

        validator = IntegerValidator(min_value=1, max_value=5)

        class Doc:
            text = "10"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(Doc())
        assert "between" in str(exc_info.value.message)


class TestChoiceValidator:
    def test_accepts_valid_choice(self):
        """Test that valid choices are accepted."""
        validator = ChoiceValidator(choices=["apple", "banana", "cherry"])

        class Doc:
            text = "banana"

        # Should not raise
        validator.validate(Doc())

    def test_accepts_case_insensitive(self):
        """Test that choices are case-insensitive."""
        validator = ChoiceValidator(choices=["Apple", "Banana"])

        class Doc:
            text = "APPLE"

        # Should not raise
        validator.validate(Doc())

    def test_accepts_empty_when_allowed(self):
        """Test that empty is accepted when allow_empty=True."""
        validator = ChoiceValidator(choices=["a", "b"], allow_empty=True)

        class Doc:
            text = ""

        # Should not raise
        validator.validate(Doc())

    def test_rejects_empty_when_not_allowed(self):
        """Test that empty is rejected when allow_empty=False."""
        from prompt_toolkit.validation import ValidationError

        validator = ChoiceValidator(choices=["a", "b"], allow_empty=False)

        class Doc:
            text = ""

        with pytest.raises(ValidationError):
            validator.validate(Doc())

    def test_rejects_invalid_choice(self):
        """Test that invalid choices are rejected."""
        from prompt_toolkit.validation import ValidationError

        validator = ChoiceValidator(choices=["a", "b"])

        class Doc:
            text = "c"

        with pytest.raises(ValidationError) as exc_info:
            validator.validate(Doc())
        assert "one of" in str(exc_info.value.message)


# =============================================================================
# Test Step Functions (Mocked questionary)
# =============================================================================


class TestStepPortsRenderConfiguredHost:
    """The wizard's confirmation lines used to hard-code ``127.0.0.1`` /
    ``localhost``, which lies once a bind host is pinned to a real interface.
    They now render whatever the configured host resolves to."""

    def _capture(self, step, config, answer=""):
        printed: list[str] = []
        with (
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps.console") as mock_console,
            patch(
                "metasci_scribe.langgraph_dev.manager._is_port_occupied",
                lambda *_a, **_kw: False,
            ),
            patch(
                "metasci_scribe.langgraph_dev.manager.is_langgraph_dev_running",
                lambda *_a, **_kw: False,
            ),
        ):
            mock_q.text.return_value.ask.return_value = answer
            mock_console.print.side_effect = lambda *a, **k: printed.append(str(a[0]))
            step(config)
        return "\n".join(printed)

    def test_langgraph_dev_step_shows_pinned_interface(self):
        from metasci_scribe.config.onboard.steps import _step_langgraph_dev_port

        config = ScribeConfig(
            enable_async_subagents=True,
            langgraph_dev_port=6174,
            langgraph_dev_host="192.168.1.5",
        )
        assert "http://192.168.1.5:6174" in self._capture(
            _step_langgraph_dev_port, config
        )

    def test_langgraph_dev_step_shows_loopback_for_wildcard(self):
        """A wildcard bind is reported as loopback — that is the address this
        machine's own browser opens."""
        from metasci_scribe.config.onboard.steps import _step_langgraph_dev_port

        config = ScribeConfig(
            enable_async_subagents=True,
            langgraph_dev_port=6174,
            langgraph_dev_host="0.0.0.0",
        )
        assert "http://127.0.0.1:6174" in self._capture(
            _step_langgraph_dev_port, config
        )


class TestStepProvider:
    def test_returns_selected_provider(self):
        """Test that _step_provider returns selected provider."""
        from metasci_scribe.config.onboard.steps import _step_provider

        config = ScribeConfig()

        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "anthropic"
            result = _step_provider(config)

        assert result == "anthropic"
        mock_q.select.assert_called_once()

    def test_default_value_and_label_override(self):
        """default_value preselects a provider (re-run co-pilot default) and
        label customizes the prompt text."""
        from metasci_scribe.config.onboard.steps import _step_provider

        config = ScribeConfig(provider="anthropic")
        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "openai"
            _step_provider(config, label="co-pilot", default_value="openai")

        call = mock_q.select.call_args
        assert call.kwargs["default"] == "openai"  # override, not config.provider
        assert "co-pilot" in call.args[0]

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test that _step_provider raises KeyboardInterrupt on cancel."""
        from metasci_scribe.config.onboard.steps import _step_provider

        config = ScribeConfig()

        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_provider(config)


class TestStepReasoningEffort:
    def _run(self, provider, stored, picked="high"):
        from metasci_scribe.config.onboard import steps as onboard_steps

        question = MagicMock()
        question.ask.return_value = picked
        with patch(
            "metasci_scribe.config.onboard.steps.questionary.select",
            return_value=question,
        ) as mock_select:
            result = onboard_steps._step_reasoning_effort(
                ScribeConfig(reasoning_effort=stored), provider
            )
        call = mock_select.call_args
        return result, [c.value for c in call.kwargs["choices"]], call.kwargs["default"]

    def test_openai_offers_the_levels_its_current_models_accept(self):
        result, values, default = self._run("openai", "", picked="max")

        assert result == "max"
        assert values == ["", "low", "medium", "high", "xhigh", "max"]
        assert default == ""

    @pytest.mark.parametrize(
        ("provider", "stored", "default"),
        [
            ("openai", "max", "max"),
            ("openai", "", ""),
            ("openai", "minimal", ""),  # carried over from another provider
        ],
    )
    def test_highlights_the_stored_level_only_when_it_is_offered(
        self, provider, stored, default
    ):
        assert self._run(provider, stored)[2] == default

    def test_every_offered_level_is_sent_unchanged(self):
        from metasci_scribe.llm.models import _reasoning_effort

        _, values, _ = self._run("openai", "")

        with patch.dict("os.environ"):
            for value in filter(None, values):
                os.environ["METASCI_SCRIBE_REASONING_EFFORT"] = value
                assert _reasoning_effort("high") == value

    @pytest.mark.parametrize(
        ("provider", "asked"),
        [("openai", True), ("anthropic", False), ("", False)],
    )
    def test_only_providers_with_a_level_list_are_asked(self, provider, asked):
        from metasci_scribe.config.onboard import steps as onboard_steps

        assert onboard_steps._asks_reasoning_effort(provider) is asked

    def test_model_section_asks_openai_users_for_a_level(self):
        from metasci_scribe.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("metasci_scribe.config.onboard.wizard.load_config") as mock_load,
            patch("metasci_scribe.config.onboard.wizard.save_config") as mock_save,
            patch("metasci_scribe.config.onboard.wizard.console"),
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.config.onboard.helpers.console"),
        ):
            mock_load.return_value = ScribeConfig(
                provider="openai", model="gpt-6-sol", openai_api_key="sk-test"
            )
            mock_q.select.return_value.ask.side_effect = [
                "gpt-6-luna",  # Model
                "max",  # Reasoning effort
            ]
            mock_q.confirm.return_value.ask.side_effect = [True]  # Save config

            result = run_onboard(skip_validation=True, only_sections={"model"})

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.model == "gpt-6-luna"
        assert final_config.reasoning_effort == "max"


class TestStepModel:
    def test_returns_selected_model(self):
        """Test that _step_model returns selected model."""
        from metasci_scribe.config.onboard.steps import _step_model

        config = ScribeConfig()

        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "claude-sonnet-4-6"
            result = _step_model(config, "anthropic")

        assert result == "claude-sonnet-4-6"

    def test_main_model_not_in_provider_list_defaults_to_first(self):
        """Reset/main flow: a config.model that isn't in the chosen provider's
        list (e.g. provider switched to anthropic) defaults to that
        provider's first model, NOT the custom 'Type a model name...' entry."""
        from metasci_scribe.config.onboard.steps import _step_model
        from metasci_scribe.llm.models import get_models_for_provider

        config = ScribeConfig(model="gpt-6-sol")
        entries = get_models_for_provider("anthropic")
        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = entries[0][0]
            _step_model(config, "anthropic")

        default = mock_q.select.call_args.kwargs["default"]
        assert default == entries[0][0]
        assert default != "__custom__"

    def test_removed_model_defaults_to_the_default_model(self):
        """A saved model the provider no longer lists preselects the default
        model, not the provider's first (most expensive) entry."""
        from metasci_scribe.config.onboard.steps import _step_model
        from metasci_scribe.llm import DEFAULT_MODEL

        config = ScribeConfig(model="gpt-4o")
        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = DEFAULT_MODEL
            _step_model(config, "openai")

        assert mock_q.select.call_args.kwargs["default"] == DEFAULT_MODEL

    def test_custom_default_value_preselects_and_prefills(self):
        """Co-pilot re-run: a saved custom (non-registry) model preselects and
        prefills the 'Type a model name...' entry."""
        from metasci_scribe.config.onboard.steps import _step_model

        config = ScribeConfig()
        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "__custom__"
            mock_q.text.return_value.ask.return_value = "my-private/model"
            result = _step_model(config, "openai", default_value="my-private/model")

        assert mock_q.select.call_args.kwargs["default"] == "__custom__"
        assert mock_q.text.call_args.kwargs["default"] == "my-private/model"
        assert result == "my-private/model"

    def test_raises_keyboard_interrupt_on_cancel(self):
        """Test that _step_model raises KeyboardInterrupt on cancel."""
        from metasci_scribe.config.onboard.steps import _step_model

        config = ScribeConfig()

        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_model(config, "anthropic")


class TestStepWorkspace:
    def test_returns_daemon_mode(self):
        """Test workspace step returns selected mode."""
        from metasci_scribe.config.onboard.steps import _step_workspace

        config = ScribeConfig()

        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "daemon"
            result = _step_workspace(config)

        assert result == "daemon"

    def test_returns_run_mode(self):
        """Test workspace step returns run mode."""
        from metasci_scribe.config.onboard.steps import _step_workspace

        config = ScribeConfig()

        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = "run"
            result = _step_workspace(config)

        assert result == "run"


class TestPromptAndValidateApiKey:
    def test_keep_existing_key_still_validates(self):
        """Pressing Enter to keep current key should validate the existing key."""
        from metasci_scribe.config.onboard.helpers import _prompt_and_validate_api_key

        validate_fn = Mock(return_value=(True, "Valid"))

        with (
            patch("metasci_scribe.config.onboard.helpers.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.helpers.console"),
        ):
            mock_q.password.return_value.ask.return_value = ""  # keep existing
            result = _prompt_and_validate_api_key(
                "Enter key:",
                current="existing-key",
                validate_fn=validate_fn,
                skip_validation=False,
            )

        assert result is None  # None means "keep existing, don't overwrite"
        validate_fn.assert_called_once_with("existing-key")

    def test_new_key_still_validates(self):
        """Entering a new key should still run validation."""
        from metasci_scribe.config.onboard.helpers import _prompt_and_validate_api_key

        validate_fn = Mock(return_value=(True, "valid"))

        with patch("metasci_scribe.config.onboard.helpers.questionary") as mock_q:
            mock_q.password.return_value.ask.return_value = "new-key"
            result = _prompt_and_validate_api_key(
                "Enter key:",
                current="old-key",
                validate_fn=validate_fn,
                skip_validation=False,
            )

        assert result == "new-key"
        validate_fn.assert_called_once_with("new-key")


class TestSkillsWorkspace:
    """Onboarding checks skills where the CLI starts: env override, config, cwd."""

    def test_uses_default_workdir(self, monkeypatch, tmp_path):
        monkeypatch.delenv("METASCI_SCRIBE_WORKSPACE_DIR", raising=False)
        config = ScribeConfig(default_workdir=str(tmp_path / "cfg"))
        assert _skills_workspace(config) == Workspace(tmp_path / "cfg")

    def test_prefers_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("METASCI_SCRIBE_WORKSPACE_DIR", str(tmp_path / "env"))
        config = ScribeConfig(default_workdir=str(tmp_path / "cfg"))
        assert _skills_workspace(config) == Workspace(tmp_path / "env")

    def test_falls_back_to_cwd(self, monkeypatch, tmp_path):
        monkeypatch.delenv("METASCI_SCRIBE_WORKSPACE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert _skills_workspace(ScribeConfig()) == Workspace(tmp_path)


class TestStepSkills:
    def test_returns_empty_when_none_selected(self, workspace):
        """Test skills step returns empty list when user selects nothing."""
        from metasci_scribe.config.onboard.steps import _step_skills

        with (
            patch("metasci_scribe.config.onboard.style.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps.console"),
            # Empty selection triggers the npx easter-egg check; without this
            # mock the real questionary.confirm in _ensure_npx fires on hosts
            # where npx is missing/slow (NoConsoleScreenBufferError on
            # headless Windows CI).
            patch("metasci_scribe.config.onboard.steps._ensure_npx", return_value=True),
        ):
            mock_q.checkbox.return_value.ask.return_value = []
            result = _step_skills(workspace)

        assert result == []

    def test_installs_selected_skills(self, workspace):
        """Test skills step installs selected skills and returns sources."""
        from metasci_scribe.config.onboard.steps import (
            _RECOMMENDED_SKILLS,
            _step_skills,
        )

        source = _RECOMMENDED_SKILLS[0]["source"]

        with (
            patch("metasci_scribe.config.onboard.style.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.tools.skills_manager.install_skill") as mock_install,
        ):
            mock_q.checkbox.return_value.ask.return_value = [source]
            mock_install.return_value = {"success": True, "name": "test"}
            result = _step_skills(workspace)

        assert result == [source]
        mock_install.assert_called_once_with(source, workspace=workspace)

    def test_handles_install_failure(self, workspace):
        """Test skills step handles installation errors gracefully."""
        from metasci_scribe.config.onboard.steps import (
            _RECOMMENDED_SKILLS,
            _step_skills,
        )

        source = _RECOMMENDED_SKILLS[0]["source"]

        with (
            patch("metasci_scribe.config.onboard.style.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.tools.skills_manager.install_skill") as mock_install,
        ):
            mock_q.checkbox.return_value.ask.return_value = [source]
            mock_install.side_effect = Exception("network error")
            result = _step_skills(workspace)

        assert result == []

    def test_raises_keyboard_interrupt_on_cancel(self, workspace):
        """Test skills step raises KeyboardInterrupt on cancel."""
        from metasci_scribe.config.onboard.steps import _step_skills

        with patch("metasci_scribe.config.onboard.style.questionary") as mock_q:
            mock_q.checkbox.return_value.ask.return_value = None
            with pytest.raises(KeyboardInterrupt):
                _step_skills(workspace)

    def test_detects_pack_via_manifest(self, tmp_path, workspace):
        """A pack source recorded in the manifest is detected as installed
        even when none of the unpacked child dir names match the source."""
        from metasci_scribe.config.onboard.steps import (
            _RECOMMENDED_SKILLS,
            _step_skills,
        )

        # Recreate the EvoSkills-style layout: child dirs in GLOBAL_SKILLS_DIR
        # whose names share nothing with the pack source URL.
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        for child in ("paper-writing", "evo-memory", "research-survey"):
            (global_dir / child).mkdir()
            (global_dir / child / "SKILL.md").write_text(
                f"---\nname: {child}\ndescription: x\n---\n"
            )
        pack_source = _RECOMMENDED_SKILLS[0]["source"]
        (global_dir / ".installed.yaml").write_text(
            f"paper-writing:\n  source: {pack_source}\n"
            f"evo-memory:\n  source: {pack_source}\n"
            f"research-survey:\n  source: {pack_source}\n"
        )
        workspace.skills_dir.mkdir()

        captured: list = []

        def _capture(choices, _msg, **_kw):
            captured.extend(choices)
            return []

        with (
            patch("metasci_scribe.paths.GLOBAL_SKILLS_DIR", global_dir),
            patch(
                "metasci_scribe.config.onboard.steps._checkbox_ask",
                side_effect=_capture,
            ),
            patch("metasci_scribe.config.onboard.steps.console"),
            # _capture returns [] (empty selection), which would otherwise
            # reach the real _ensure_npx — see test_returns_empty_when_none_selected.
            patch("metasci_scribe.config.onboard.steps._ensure_npx", return_value=True),
        ):
            _step_skills(workspace)

        pack_choice = next(c for c in captured if c.value == pack_source)
        # Detection signal: an "installed" hint appears in the choice title,
        # but the choice stays selectable so users can re-sync (important for
        # packs where one child was deleted but the rest keep the entry).
        title_text = "".join(seg[1] for seg in pack_choice.title)
        assert "installed" in title_text, (
            "EvoSkills pack should be marked as installed via the manifest "
            "even though no child dir name matches the pack source string"
        )
        assert not pack_choice.disabled, (
            "Installed packs must remain selectable so users can re-sync them"
        )

    def test_surfaces_update_available_when_upstream_moved(self, tmp_path, workspace):
        """When a pack records an install-time commit and `git ls-remote`
        reports a different SHA, the choice label should call out the update."""
        from metasci_scribe.config.onboard.steps import (
            _RECOMMENDED_SKILLS,
            _step_skills,
        )

        pack_source = _RECOMMENDED_SKILLS[0]["source"]
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        (global_dir / "paper-writing").mkdir()
        (global_dir / "paper-writing" / "SKILL.md").write_text(
            "---\nname: paper-writing\ndescription: x\n---\n"
        )
        # Recorded install-time commit.
        (global_dir / ".installed.yaml").write_text(
            f"paper-writing:\n  source: {pack_source}\n  commit: aaa111\n"
        )
        workspace.skills_dir.mkdir()

        captured: list = []

        def _capture(choices, _msg, **_kw):
            captured.extend(choices)
            return []

        with (
            patch("metasci_scribe.paths.GLOBAL_SKILLS_DIR", global_dir),
            # Upstream moved past the recorded commit.
            patch(
                "metasci_scribe.tools.skills_manager.resolve_remote_head",
                return_value="bbb222",
            ),
            patch(
                "metasci_scribe.config.onboard.steps._checkbox_ask",
                side_effect=_capture,
            ),
            patch("metasci_scribe.config.onboard.steps.console"),
            # _capture returns [] (empty selection), which would otherwise
            # reach the real _ensure_npx — see test_returns_empty_when_none_selected.
            patch("metasci_scribe.config.onboard.steps._ensure_npx", return_value=True),
        ):
            _step_skills(workspace)

        pack_choice = next(c for c in captured if c.value == pack_source)
        title_text = "".join(seg[1] for seg in pack_choice.title)
        assert "update available" in title_text, (
            f"expected an 'update available' hint in the label, got: {title_text!r}"
        )

    def test_no_update_hint_when_remote_check_fails(self, tmp_path, workspace):
        """If `git ls-remote` returns None (offline, timeout, etc.), the label
        must fall back to plain 'installed' — never falsely claim 'update'."""
        from metasci_scribe.config.onboard.steps import (
            _RECOMMENDED_SKILLS,
            _step_skills,
        )

        pack_source = _RECOMMENDED_SKILLS[0]["source"]
        global_dir = tmp_path / "global"
        global_dir.mkdir()
        (global_dir / "paper-writing").mkdir()
        (global_dir / "paper-writing" / "SKILL.md").write_text(
            "---\nname: paper-writing\ndescription: x\n---\n"
        )
        (global_dir / ".installed.yaml").write_text(
            f"paper-writing:\n  source: {pack_source}\n  commit: aaa111\n"
        )
        workspace.skills_dir.mkdir()

        captured: list = []

        def _capture(choices, _msg, **_kw):
            captured.extend(choices)
            return []

        with (
            patch("metasci_scribe.paths.GLOBAL_SKILLS_DIR", global_dir),
            patch(
                "metasci_scribe.tools.skills_manager.resolve_remote_head",
                return_value=None,
            ),
            patch(
                "metasci_scribe.config.onboard.steps._checkbox_ask",
                side_effect=_capture,
            ),
            patch("metasci_scribe.config.onboard.steps.console"),
            # _capture returns [] (empty selection), which would otherwise
            # reach the real _ensure_npx — see test_returns_empty_when_none_selected.
            patch("metasci_scribe.config.onboard.steps._ensure_npx", return_value=True),
        ):
            _step_skills(workspace)

        pack_choice = next(c for c in captured if c.value == pack_source)
        title_text = "".join(seg[1] for seg in pack_choice.title)
        assert "update available" not in title_text, (
            f"must not claim 'update available' when remote check fails: {title_text!r}"
        )
        assert "installed" in title_text


class TestStepMcpServersNpxFailure:
    def _make_test_servers(self):
        from metasci_scribe.mcp.registry import MCPServerEntry

        return [
            MCPServerEntry(
                name="npx-server",
                label="NPX Server",
                tags=["onboarding"],
                command="npx",
                args=["-y", "test-server"],
            ),
            MCPServerEntry(
                name="url-server",
                label="URL Server",
                tags=["onboarding"],
                transport="streamable_http",
                url="https://example.com/mcp",
            ),
        ]

    def test_npx_failure_skips_npx_servers(self):
        """When _ensure_npx returns False, npx-dependent servers must be skipped."""
        from metasci_scribe.config.onboard.steps import _step_mcp_servers

        servers = self._make_test_servers()

        with (
            patch(
                "metasci_scribe.mcp.registry.fetch_marketplace_index",
                return_value=servers,
            ),
            patch(
                "metasci_scribe.config.onboard.steps._checkbox_ask",
                return_value=["npx-server", "url-server"],
            ),
            patch(
                "metasci_scribe.config.onboard.steps._ensure_npx", return_value=False
            ),
            patch(
                "metasci_scribe.config.onboard.helpers._check_npx", return_value=False
            ),
            patch("metasci_scribe.mcp.client._load_user_config", return_value={}),
            patch("metasci_scribe.mcp.client.add_mcp_server") as mock_add,
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            result = _step_mcp_servers()

        # The npx server must NOT have been added
        added_names = [call.args[0] for call in mock_add.call_args_list]
        assert "npx-server" not in added_names
        # The URL server should still be added
        assert "url-server" in added_names
        assert "url-server" in result
        assert "npx-server" not in result

    def test_npx_failure_returns_empty_when_all_npx(self):
        """When all selected servers are npx-based and npx fails, return []."""
        from metasci_scribe.config.onboard.steps import _step_mcp_servers

        servers = self._make_test_servers()
        npx_names = [s.name for s in servers if s.command == "npx"]

        with (
            patch(
                "metasci_scribe.mcp.registry.fetch_marketplace_index",
                return_value=servers,
            ),
            patch(
                "metasci_scribe.config.onboard.steps._checkbox_ask",
                return_value=npx_names,
            ),
            patch(
                "metasci_scribe.config.onboard.steps._ensure_npx", return_value=False
            ),
            patch(
                "metasci_scribe.config.onboard.helpers._check_npx", return_value=False
            ),
            patch("metasci_scribe.mcp.client._load_user_config", return_value={}),
            patch("metasci_scribe.mcp.client.add_mcp_server") as mock_add,
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            result = _step_mcp_servers()

        assert result == []
        mock_add.assert_not_called()

    @staticmethod
    def _render_fetch_failure(exc):
        """Run the step with a failing index fetch through a real Rich console."""
        import io

        from rich.console import Console

        from metasci_scribe.config.onboard.steps import _step_mcp_servers

        out = io.StringIO()
        with (
            patch(
                "metasci_scribe.mcp.registry.fetch_marketplace_index", side_effect=exc
            ),
            patch(
                "metasci_scribe.config.onboard.steps.console",
                Console(file=out, width=500, color_system=None),
            ),
        ):
            result = _step_mcp_servers()
        return result, out.getvalue()

    def test_index_failure_prints_the_reason(self):
        from metasci_scribe.git_cli import GitNotFoundError

        result, output = self._render_fetch_failure(GitNotFoundError())

        assert result == []
        assert "git was not found on PATH." in output
        assert "GitNotFoundError" not in output

    def test_index_failure_reason_is_not_read_as_markup(self):
        result, output = self._render_fetch_failure(
            RuntimeError("git clone failed: fatal: [/b] bad")
        )

        assert result == []
        assert "git clone failed: fatal: [/b] bad" in output


class TestStepThinking:
    def test_returns_show_thinking(self):
        """Test thinking step returns selected value."""
        from metasci_scribe.config.onboard.steps import _step_thinking

        config = ScribeConfig()

        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = True
            result = _step_thinking(config)

        assert result is True

    def test_returns_false_when_off(self):
        """Test thinking step returns False when user selects Off."""
        from metasci_scribe.config.onboard.steps import _step_thinking

        config = ScribeConfig(show_thinking=False)

        with patch("metasci_scribe.config.onboard.steps.questionary") as mock_q:
            mock_q.select.return_value.ask.return_value = False
            result = _step_thinking(config)

        assert result is False


# =============================================================================
# Test run_onboard (Integration-like test with mocked questionary)
# =============================================================================


class TestRunOnboard:
    @pytest.fixture(autouse=True)
    def _no_mcp_marketplace(self, monkeypatch):
        """The MCP step clones the marketplace from GitHub; its own tests
        above cover it with the clone stubbed."""
        monkeypatch.setattr(
            "metasci_scribe.config.onboard.wizard._step_mcp_servers", lambda: None
        )

    def test_returns_true_on_save(self):
        """Test that run_onboard returns True when config is saved."""
        from metasci_scribe.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("metasci_scribe.config.onboard.wizard.load_config") as mock_load,
            patch("metasci_scribe.config.onboard.wizard.save_config") as mock_save,
            patch("metasci_scribe.config.onboard.wizard.console"),
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.config.onboard.helpers.console"),
            patch("metasci_scribe.config.onboard.wizard._step_tinytex"),
        ):
            # Setup mock config
            mock_load.return_value = ScribeConfig()

            # Mock all questionary calls — order matches the wizard's select
            # sequence: Provider → Model →
            # Workspace → Thinking.
            mock_q.select.return_value.ask.side_effect = [
                "anthropic",  # Provider
                "claude-sonnet-4-6",  # Model
                "skip",  # Auxiliary: Skip (single driver)
                "daemon",  # Workspace mode
                True,  # Show thinking
            ]
            mock_q.password.return_value.ask.side_effect = [
                "",  # Provider API key (keep current)
            ]
            mock_q.confirm.return_value.ask.side_effect = [
                True,  # Save config
            ]
            mock_q.text.return_value.ask.side_effect = [
                "",  # Workspace directory (empty = use cwd)
            ]
            mock_q.checkbox.return_value.ask.return_value = []  # Skills: skip

            result = run_onboard(skip_validation=True)

        assert result is True
        # Config is autosaved between phases + once at the end.
        mock_save.assert_called()
        # Verify the final saved config has the values we picked — guards
        # against silent shift bugs where a missing side_effect entry causes
        # downstream prompts to consume the wrong values.
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.provider == "anthropic"
        assert final_config.model == "claude-sonnet-4-6"
        assert final_config.ui_backend == "cli"
        assert final_config.default_mode == "daemon"

    def test_auxiliary_model_enabled_collects_provider_and_key(self):
        """Enabling the auxiliary step stores its provider, model, and the
        chosen provider's API key (a different company than the main agent)."""
        from metasci_scribe.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("metasci_scribe.config.onboard.wizard.load_config") as mock_load,
            patch("metasci_scribe.config.onboard.wizard.save_config") as mock_save,
            patch("metasci_scribe.config.onboard.wizard.console"),
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.config.onboard.helpers.console"),
            patch("metasci_scribe.config.onboard.wizard._step_tinytex"),
        ):
            mock_load.return_value = ScribeConfig()

            mock_q.select.return_value.ask.side_effect = [
                "anthropic",  # Provider
                "claude-sonnet-4-6",  # Model
                "assemble",  # Auxiliary: Assemble
                "openai",  # Auxiliary provider (a different company)
                "gpt-6-luna",  # Auxiliary model
                "daemon",  # Workspace mode
                True,  # Show thinking
            ]
            mock_q.password.return_value.ask.side_effect = [
                "",  # Main provider API key (keep current)
                "sk-aux-openai",  # Auxiliary provider API key
            ]
            mock_q.confirm.return_value.ask.side_effect = [
                True,  # Save config
            ]
            mock_q.text.return_value.ask.side_effect = [
                "",  # Workspace directory
            ]
            mock_q.checkbox.return_value.ask.return_value = []  # Skills: skip

            result = run_onboard(skip_validation=True)

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "openai"
        assert final_config.auxiliary_model == "gpt-6-luna"
        assert final_config.openai_auth_mode == "api_key"
        # The auxiliary provider's key is stored in its per-provider field.
        assert final_config.openai_api_key == "sk-aux-openai"
        # Main agent is untouched.
        assert final_config.provider == "anthropic"
        assert final_config.model == "claude-sonnet-4-6"

    def test_auxiliary_same_provider_reuses_main_credentials(self):
        """Same-provider co-pilot should not imply separate credentials exist."""
        from metasci_scribe.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("metasci_scribe.config.onboard.wizard.load_config") as mock_load,
            patch("metasci_scribe.config.onboard.wizard.save_config") as mock_save,
            patch("metasci_scribe.config.onboard.wizard.console"),
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            mock_load.return_value = ScribeConfig(
                provider="openai",
                model="gpt-6-luna",
                openai_api_key="sk-main-openai",
            )
            mock_q.select.return_value.ask.side_effect = [
                "assemble",  # Auxiliary: Assemble
                "openai",  # Same provider as the main model
                "gpt-6-luna",  # Auxiliary model
            ]
            mock_q.confirm.return_value.ask.side_effect = [True]  # Save config

            result = run_onboard(
                skip_validation=True, only_sections={"auxiliary_model"}
            )

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "openai"
        assert final_config.auxiliary_model == "gpt-6-luna"
        assert final_config.openai_api_key == "sk-main-openai"
        mock_q.password.assert_not_called()
        assert mock_q.select.return_value.ask.call_count == 3

    def test_auxiliary_same_provider_prompts_when_shared_key_missing(self):
        """Same-provider reuse should not hide a missing shared API key."""
        from metasci_scribe.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("metasci_scribe.config.onboard.wizard.load_config") as mock_load,
            patch("metasci_scribe.config.onboard.wizard.save_config") as mock_save,
            patch("metasci_scribe.config.onboard.wizard.console"),
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.config.onboard.helpers.console"),
        ):
            mock_load.return_value = ScribeConfig(
                provider="openai",
                model="gpt-6-luna",
                openai_auth_mode="api_key",
                openai_api_key="",
            )
            mock_q.select.return_value.ask.side_effect = [
                "assemble",  # Auxiliary: Assemble
                "openai",  # Same provider as the main model
                "gpt-6-luna",  # Auxiliary model
            ]
            mock_q.password.return_value.ask.side_effect = [
                "sk-shared-openai",
            ]
            mock_q.confirm.return_value.ask.side_effect = [True]  # Save config

            result = run_onboard(
                skip_validation=True, only_sections={"auxiliary_model"}
            )

        assert result is True
        final_config = mock_save.call_args_list[-1].args[0]
        assert final_config.auxiliary_provider == "openai"
        assert final_config.auxiliary_model == "gpt-6-luna"
        assert final_config.openai_auth_mode == "api_key"
        assert final_config.openai_api_key == "sk-shared-openai"
        mock_q.password.assert_called_once()
        assert mock_q.select.return_value.ask.call_count == 3

    def test_returns_false_on_cancel(self):
        """Test that run_onboard returns False when cancelled."""
        from metasci_scribe.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("metasci_scribe.config.onboard.wizard.load_config") as mock_load,
            patch("metasci_scribe.config.onboard.wizard.console"),
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.config.onboard.helpers.console"),
            patch("metasci_scribe.config.onboard.wizard._step_tinytex"),
        ):
            mock_load.return_value = ScribeConfig()

            # First selection returns None (Ctrl+C)
            mock_q.select.return_value.ask.return_value = None

            result = run_onboard(skip_validation=True)

        assert result is False

    def test_returns_false_when_not_saving(self):
        """Test that run_onboard returns False when user declines to save."""
        from metasci_scribe.config.onboard.wizard import run_onboard

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch("metasci_scribe.config.onboard.wizard.load_config") as mock_load,
            patch("metasci_scribe.config.onboard.wizard.save_config") as mock_save,
            patch("metasci_scribe.config.onboard.wizard.console"),
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.config.onboard.helpers.console"),
            patch("metasci_scribe.config.onboard.wizard._step_tinytex"),
        ):
            mock_load.return_value = ScribeConfig()

            mock_q.select.return_value.ask.side_effect = [
                "anthropic",  # Provider
                "claude-sonnet-4-6",  # Model
                "skip",  # Auxiliary: Skip (single driver)
                "daemon",  # Workspace mode
                True,  # Show thinking
            ]
            mock_q.password.return_value.ask.side_effect = ["", ""]
            mock_q.confirm.return_value.ask.side_effect = [
                False,  # Save config - NO
            ]
            mock_q.text.return_value.ask.side_effect = [
                "",  # Workspace directory (empty = use cwd)
            ]
            mock_q.checkbox.return_value.ask.return_value = []  # Skills: skip

            result = run_onboard(skip_validation=True)

        assert result is False
        # The wizard autosaves between phases AND writes the original
        # snapshot back when the user declines — so save_config IS called.
        mock_save.assert_called()

    def test_reset_then_no_restores_pre_wizard_config(self, tmp_path):
        """Regression: Reset → No must restore the original user file,
        NOT overwrite it with ``ScribeConfig()`` defaults.

        Reproduces a bug where the snapshot was refreshed after Reset, so
        declining the save silently wiped the user's previous settings.
        Verifies the raw-bytes revert path introduced for CodeRabbit's
        "file absence on cancel" comment.
        """
        from metasci_scribe.config.onboard.wizard import run_onboard
        from metasci_scribe.config.settings import save_config

        config_file = tmp_path / "config.yaml"
        existing = ScribeConfig(
            provider="openai",
            model="gpt-5",
            openai_api_key="sk-existing",
            ui_backend="cli",
        )

        mock_q = MagicMock()
        with patch(
            "metasci_scribe.config.settings.get_config_path", return_value=config_file
        ):
            save_config(existing)
            # Append a comment that ``save_config`` would strip on
            # re-serialization. The test then proves revert restored the
            # EXACT bytes (not just rewrote the snapshot via save_config
            # and got lucky because logical content matched).
            with config_file.open("ab") as f:
                f.write(b"\n# user manual comment - must survive revert\n")
            original_bytes = config_file.read_bytes()

            with (
                _patch_all_questionary(mock_q),
                patch(
                    "metasci_scribe.config.onboard.wizard.get_config_path",
                    return_value=config_file,
                ),
                patch("metasci_scribe.config.onboard.wizard.console"),
                patch("metasci_scribe.config.onboard.steps.console"),
                patch("metasci_scribe.config.onboard.helpers.console"),
                patch("metasci_scribe.config.onboard.wizard._step_tinytex"),
            ):
                mock_q.select.return_value.ask.side_effect = [
                    "reset",  # Keep/Modify/Reset → Reset
                    "anthropic",
                    "claude-sonnet-4-6",
                    "skip",  # Auxiliary: Skip (single driver)
                    "daemon",
                    True,
                ]
                mock_q.password.return_value.ask.side_effect = ["", ""]
                mock_q.confirm.return_value.ask.side_effect = [
                    False,  # Save? = No
                ]
                mock_q.text.return_value.ask.side_effect = [""]
                mock_q.checkbox.return_value.ask.return_value = []

                result = run_onboard(skip_validation=True)

        assert result is False
        assert config_file.exists(), "file should still exist after revert"
        assert config_file.read_bytes() == original_bytes, (
            "revert must restore EXACT pre-wizard bytes, not autosaved state"
        )

    def test_revert_removes_file_when_none_existed(self, tmp_path):
        """Brand-new user (no config.yaml) declines the final save: the file
        created by autosaves should be deleted to match the original state.
        """
        from metasci_scribe.config.onboard.wizard import run_onboard

        config_file = tmp_path / "config.yaml"
        assert not config_file.exists()

        mock_q = MagicMock()
        with (
            _patch_all_questionary(mock_q),
            patch(
                "metasci_scribe.config.settings.get_config_path",
                return_value=config_file,
            ),
            patch(
                "metasci_scribe.config.onboard.wizard.get_config_path",
                return_value=config_file,
            ),
            patch("metasci_scribe.config.onboard.wizard.console"),
            patch("metasci_scribe.config.onboard.steps.console"),
            patch("metasci_scribe.config.onboard.helpers.console"),
            patch("metasci_scribe.config.onboard.wizard._step_tinytex"),
        ):
            mock_q.select.return_value.ask.side_effect = [
                "anthropic",
                "claude-sonnet-4-6",
                "skip",  # Auxiliary: Skip (single driver)
                "daemon",
                True,
            ]
            mock_q.password.return_value.ask.side_effect = ["", ""]
            mock_q.confirm.return_value.ask.side_effect = [
                False,  # Save? = No
            ]
            mock_q.text.return_value.ask.side_effect = [""]
            mock_q.checkbox.return_value.ask.return_value = []

            result = run_onboard(skip_validation=True)

        assert result is False
        assert not config_file.exists(), (
            "revert must delete file that was autosaved into existence "
            "during the wizard run"
        )


class TestOnboardCliErrorPresentation:
    """The CLI must turn the wizard's non-interactive ``RuntimeError`` signals
    into a clean message + exit code 1, not a raw Python traceback."""

    def test_runtime_error_becomes_clean_typer_exit(self):
        import typer

        from metasci_scribe.cli.commands import _run_onboard_cli

        with (
            patch(
                "metasci_scribe.config.onboard.run_onboard",
                side_effect=RuntimeError("--api-key rejected by validator: nope"),
            ),
            patch("metasci_scribe.cli.commands.console") as mock_console,
        ):
            with pytest.raises(typer.Exit) as exc_info:
                _run_onboard_cli(skip_validation=False)

        assert exc_info.value.exit_code == 1
        # The error message reaches the user via console, not a traceback.
        printed = " ".join(str(c.args[0]) for c in mock_console.print.call_args_list)
        assert "rejected by validator" in printed


# =============================================================================
# Test TinyTeX helpers
# =============================================================================


class TestCheckLatexComponents:
    """Tests for _check_latex_components()."""

    def test_all_available(self):
        """All three components found → all True."""
        from metasci_scribe.config.onboard.helpers import _check_latex_components

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.return_value = "/usr/local/bin/cmd"
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=0)
            result = _check_latex_components()
            assert result == {"pdflatex": True, "latexmk": True, "tlmgr": True}

    def test_only_pdflatex(self):
        """Only pdflatex available."""
        from metasci_scribe.config.onboard.helpers import _check_latex_components

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/local/bin/pdflatex" if cmd == "pdflatex" else None
            )
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=0)
            result = _check_latex_components()
            assert result == {
                "pdflatex": True,
                "latexmk": False,
                "tlmgr": False,
            }

    def test_none_available(self):
        """Nothing found → all False."""
        from metasci_scribe.config.onboard.helpers import _check_latex_components

        with patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh:
            mock_sh.which.return_value = None
            result = _check_latex_components()
            assert result == {
                "pdflatex": False,
                "latexmk": False,
                "tlmgr": False,
            }


class TestAutoInstallLatexmk:
    """Tests for _auto_install_latexmk()."""

    def test_success(self):
        """tlmgr install latexmk succeeds."""
        from metasci_scribe.config.onboard.helpers import _auto_install_latexmk

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
            patch("metasci_scribe.config.onboard.helpers.console") as mock_con,
        ):
            mock_sh.which.side_effect = lambda cmd: f"/usr/local/bin/{cmd}"
            mock_sp.run.return_value = Mock(returncode=0)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            _auto_install_latexmk()
            success_printed = any(
                "latexmk installed" in str(c) for c in mock_con.print.call_args_list
            )
            assert success_printed

    def test_tlmgr_not_found(self):
        """tlmgr not on PATH → does nothing."""
        from metasci_scribe.config.onboard.helpers import _auto_install_latexmk

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
            patch("metasci_scribe.config.onboard.helpers.console"),
        ):
            mock_sh.which.return_value = None
            _auto_install_latexmk()
            mock_sp.run.assert_not_called()

    def test_install_fails(self):
        """tlmgr install returns nonzero → warns."""
        from metasci_scribe.config.onboard.helpers import _auto_install_latexmk

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
            patch("metasci_scribe.config.onboard.helpers.console") as mock_con,
        ):
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/local/bin/tlmgr" if cmd == "tlmgr" else None
            )
            mock_sp.run.return_value = Mock(returncode=1)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            _auto_install_latexmk()
            warn_printed = any(
                "Failed" in str(c) for c in mock_con.print.call_args_list
            )
            assert warn_printed


class TestCheckTinytex:
    """Tests for _check_tinytex()."""

    def test_found_pdflatex(self):
        """pdflatex found and working → True."""
        from metasci_scribe.config.onboard.helpers import _check_tinytex

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.return_value = "/usr/local/bin/pdflatex"
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=0)
            assert _check_tinytex() is True

    def test_tlmgr_only_not_enough(self):
        """pdflatex missing but tlmgr found → False (pdflatex is required)."""
        from metasci_scribe.config.onboard.helpers import _check_tinytex

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/local/bin/tlmgr" if cmd == "tlmgr" else None
            )
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=0)
            assert _check_tinytex() is False

    def test_not_found(self):
        """Neither pdflatex nor tlmgr found → False."""
        from metasci_scribe.config.onboard.helpers import _check_tinytex

        with patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh:
            mock_sh.which.return_value = None
            assert _check_tinytex() is False

    def test_version_timeout(self):
        """Command found but --version times out → False."""
        from metasci_scribe.config.onboard.helpers import _check_tinytex

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sh.which.return_value = "/usr/local/bin/pdflatex"
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.side_effect = subprocess.TimeoutExpired("pdflatex", 10)
            assert _check_tinytex() is False

    def test_version_nonzero(self):
        """Command found but --version returns nonzero → False."""
        from metasci_scribe.config.onboard.helpers import _check_tinytex

        with (
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
        ):
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            mock_sp.run.return_value = Mock(returncode=1)
            # pdflatex fails, tlmgr not found
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/local/bin/pdflatex" if cmd == "pdflatex" else None
            )
            assert _check_tinytex() is False


class TestDetectTinytexInstallMethod:
    """Tests for _detect_tinytex_install_method()."""

    def test_macos_with_curl(self):
        """macOS with curl → curl method."""
        from metasci_scribe.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("metasci_scribe.config.onboard.helpers.sys") as mock_sys,
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "darwin"
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/bin/curl" if cmd == "curl" else None
            )
            method, command = _detect_tinytex_install_method()
            assert method == "curl"
            assert "install-bin-unix.sh" in command

    def test_linux_wget_fallback(self):
        """Linux without curl, with wget → wget method."""
        from metasci_scribe.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("metasci_scribe.config.onboard.helpers.sys") as mock_sys,
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "linux"
            mock_sh.which.side_effect = lambda cmd: (
                "/usr/bin/wget" if cmd == "wget" else None
            )
            method, command = _detect_tinytex_install_method()
            assert method == "wget"
            assert "install-bin-unix.sh" in command

    def test_windows_choco(self):
        """Windows with choco → choco method."""
        from metasci_scribe.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("metasci_scribe.config.onboard.helpers.sys") as mock_sys,
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "win32"
            mock_sh.which.side_effect = lambda cmd: (
                "C:\\choco\\choco.exe" if cmd == "choco" else None
            )
            method, command = _detect_tinytex_install_method()
            assert method == "choco"
            assert "tinytex" in command

    def test_windows_scoop(self):
        """Windows with scoop (no choco) → scoop method."""
        from metasci_scribe.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("metasci_scribe.config.onboard.helpers.sys") as mock_sys,
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "win32"
            mock_sh.which.side_effect = lambda cmd: (
                "C:\\scoop\\scoop.exe" if cmd == "scoop" else None
            )
            method, command = _detect_tinytex_install_method()
            assert method == "scoop"
            assert "tinytex" in command

    def test_no_tools(self):
        """No tools available → manual method."""
        from metasci_scribe.config.onboard.helpers import _detect_tinytex_install_method

        with (
            patch("metasci_scribe.config.onboard.helpers.sys") as mock_sys,
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sys.platform = "linux"
            mock_sh.which.return_value = None
            method, command = _detect_tinytex_install_method()
            assert method == "manual"
            assert "yihui.org" in command


class TestInstallTinytex:
    """Tests for _install_tinytex()."""

    def test_curl_install_success(self):
        """curl install succeeds → True."""
        from metasci_scribe.config.onboard.helpers import _install_tinytex

        with patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp:
            mock_sp.run.return_value = Mock(returncode=0)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            assert _install_tinytex("curl", "curl -sL ... | sh") is True
            mock_sp.run.assert_called_once()
            # Verify shell=True was used for pipe commands
            _, kwargs = mock_sp.run.call_args
            assert kwargs.get("shell") is True

    def test_curl_install_timeout(self):
        """curl install times out → False."""
        from metasci_scribe.config.onboard.helpers import _install_tinytex

        with (
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
            patch("metasci_scribe.config.onboard.helpers.console"),
        ):
            mock_sp.run.side_effect = subprocess.TimeoutExpired("curl", 300)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            assert _install_tinytex("curl", "curl -sL ... | sh") is False

    def test_choco_install_success(self):
        """choco install succeeds → True."""
        from metasci_scribe.config.onboard.helpers import _install_tinytex

        with (
            patch("metasci_scribe.config.onboard.helpers.subprocess") as mock_sp,
            patch("metasci_scribe.config.onboard.helpers.shutil") as mock_sh,
        ):
            mock_sh.which.return_value = "C:\\choco\\choco.exe"
            mock_sp.run.return_value = Mock(returncode=0)
            mock_sp.TimeoutExpired = subprocess.TimeoutExpired
            assert _install_tinytex("choco", "choco install tinytex -y") is True

    def test_manual_returns_false(self):
        """manual method → False immediately."""
        from metasci_scribe.config.onboard.helpers import _install_tinytex

        assert _install_tinytex("manual", "https://yihui.org/tinytex/") is False


class TestStepTinytex:
    """Tests for _step_tinytex()."""

    def test_user_declines_prepare(self):
        """User says No to 'Prepare LaTeX environment?' → skipped."""
        from metasci_scribe.config.onboard.steps import _step_tinytex

        with (
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps._print_step_skipped") as mock_ps,
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = False
            _step_tinytex()
            mock_ps.assert_called_once_with("LaTeX", "skipped")

    def test_already_installed_all_components(self):
        """User says Yes, all components available → prints detailed status."""
        from metasci_scribe.config.onboard.steps import _step_tinytex

        with (
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch(
                "metasci_scribe.config.onboard.steps._check_latex_components",
                return_value={
                    "pdflatex": True,
                    "latexmk": True,
                    "tlmgr": True,
                },
            ),
            patch(
                "metasci_scribe.config.onboard.steps._print_latex_status"
            ) as mock_status,
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            _step_tinytex()
            mock_status.assert_called_once()

    def test_already_installed_missing_latexmk(self):
        """pdflatex + tlmgr present but latexmk missing → auto-installs."""
        from metasci_scribe.config.onboard.steps import _step_tinytex

        with (
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch(
                "metasci_scribe.config.onboard.steps._check_latex_components",
                return_value={
                    "pdflatex": True,
                    "latexmk": False,
                    "tlmgr": True,
                },
            ),
            patch("metasci_scribe.config.onboard.steps._print_latex_status"),
            patch(
                "metasci_scribe.config.onboard.steps._auto_install_latexmk"
            ) as mock_auto,
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            _step_tinytex()
            mock_auto.assert_called_once()

    def test_user_installs_successfully(self):
        """Yes → not found → confirms install → succeeds → re-check passes."""
        from metasci_scribe.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        all_true = {"pdflatex": True, "latexmk": True, "tlmgr": True}
        with (
            patch(
                "metasci_scribe.config.onboard.steps._check_latex_components",
                side_effect=[all_false, all_true],
            ),
            patch(
                "metasci_scribe.config.onboard.steps._detect_tinytex_install_method",
                return_value=("curl", "curl ... | sh"),
            ),
            patch(
                "metasci_scribe.config.onboard.steps._install_tinytex",
                return_value=True,
            ),
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps._print_step_result") as mock_pr,
            patch("metasci_scribe.config.onboard.steps._print_latex_status"),
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            mock_q.confirm.return_value.ask.return_value = True
            _step_tinytex()
            mock_pr.assert_called_once_with("LaTeX", "TinyTeX installed")

    def test_user_declines_install(self):
        """Yes to prepare → not found → declines install → skipped."""
        from metasci_scribe.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        with (
            patch(
                "metasci_scribe.config.onboard.steps._check_latex_components",
                return_value=all_false,
            ),
            patch(
                "metasci_scribe.config.onboard.steps._detect_tinytex_install_method",
                return_value=("curl", "curl ... | sh"),
            ),
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps._print_step_skipped") as mock_ps,
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            mock_q.confirm.return_value.ask.return_value = False
            _step_tinytex()
            mock_ps.assert_called_once_with("LaTeX", "skipped")

    def test_install_fails(self):
        """Yes → not found → confirms install → install fails."""
        from metasci_scribe.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        with (
            patch(
                "metasci_scribe.config.onboard.steps._check_latex_components",
                return_value=all_false,
            ),
            patch(
                "metasci_scribe.config.onboard.steps._detect_tinytex_install_method",
                return_value=("curl", "curl ... | sh"),
            ),
            patch(
                "metasci_scribe.config.onboard.steps._install_tinytex",
                return_value=False,
            ),
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps._print_step_result") as mock_pr,
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            mock_q.confirm.return_value.ask.return_value = True
            _step_tinytex()
            mock_pr.assert_called_once_with(
                "LaTeX", "installation failed", success=False
            )

    def test_installed_but_not_in_path(self):
        """Install succeeds but pdflatex not yet in PATH → warns user."""
        from metasci_scribe.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        with (
            patch(
                "metasci_scribe.config.onboard.steps._check_latex_components",
                side_effect=[all_false, all_false],
            ),
            patch(
                "metasci_scribe.config.onboard.steps._detect_tinytex_install_method",
                return_value=("curl", "curl ... | sh"),
            ),
            patch(
                "metasci_scribe.config.onboard.steps._install_tinytex",
                return_value=True,
            ),
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps._print_step_result") as mock_pr,
            patch("metasci_scribe.config.onboard.steps.console") as mock_con,
        ):
            mock_q.select.return_value.ask.return_value = True
            mock_q.confirm.return_value.ask.return_value = True
            _step_tinytex()
            path_warning_printed = any(
                "PATH" in str(call) for call in mock_con.print.call_args_list
            )
            assert path_warning_printed
            mock_pr.assert_called_once_with(
                "LaTeX", "installed (restart terminal for PATH)"
            )

    def test_manual_method(self):
        """Yes to prepare → not found → manual method → prints URL, no install prompt."""
        from metasci_scribe.config.onboard.steps import _step_tinytex

        all_false = {"pdflatex": False, "latexmk": False, "tlmgr": False}
        with (
            patch(
                "metasci_scribe.config.onboard.steps._check_latex_components",
                return_value=all_false,
            ),
            patch(
                "metasci_scribe.config.onboard.steps._detect_tinytex_install_method",
                return_value=("manual", "https://yihui.org/tinytex/"),
            ),
            patch("metasci_scribe.config.onboard.steps.questionary") as mock_q,
            patch("metasci_scribe.config.onboard.steps._print_step_skipped") as mock_ps,
            patch("metasci_scribe.config.onboard.steps.console"),
        ):
            mock_q.select.return_value.ask.return_value = True
            _step_tinytex()
            mock_ps.assert_called_once_with("LaTeX", "manual install needed")


class TestEnsureNpx:
    """_ensure_npx installs the private Node via ensure_node after confirmation."""

    def _run(self, *, npx_after: bool, confirm, ensure=None):
        from metasci_scribe.config.onboard import helpers
        from metasci_scribe.setup import node as setup_node

        checks = iter([False, npx_after])
        ensure = ensure or MagicMock()
        with (
            patch.object(helpers, "_check_npx", side_effect=lambda: next(checks)),
            patch.object(helpers.questionary, "confirm") as mock_confirm,
            patch.object(setup_node, "ensure_node", ensure),
            patch.object(setup_node, "activate_runtime") as mock_activate,
        ):
            mock_confirm.return_value.ask.return_value = confirm
            result = helpers._ensure_npx("test reason")
        return result, ensure, mock_activate

    def test_confirmed_install_makes_npx_available(self):
        result, ensure, activate = self._run(npx_after=True, confirm=True)
        assert result is True
        ensure.assert_called_once()
        assert callable(ensure.call_args.kwargs["progress"])
        activate.assert_called_once_with()

    def test_declined_install_does_nothing(self):
        result, ensure, activate = self._run(npx_after=False, confirm=False)
        assert result is False
        ensure.assert_not_called()
        activate.assert_not_called()

    def test_failed_install_returns_false(self):
        from metasci_scribe.setup.protocol import StageError

        ensure = MagicMock(side_effect=StageError("download_failed", "offline"))
        result, _, activate = self._run(npx_after=False, confirm=True, ensure=ensure)
        assert result is False
        activate.assert_not_called()

    def test_cancelled_prompt_raises_keyboard_interrupt(self):
        with pytest.raises(KeyboardInterrupt):
            self._run(npx_after=False, confirm=None)


class TestMirrorIsNotAnOnboardingSetting:
    def test_config_with_only_the_mirror_set_is_a_fresh_config(self):
        from metasci_scribe.config.onboard.wizard import _config_has_meaningful_settings

        assert not _config_has_meaningful_settings(ScribeConfig(mirror="cn"))
        assert _config_has_meaningful_settings(ScribeConfig(model="other"))

    def test_reset_keeps_the_mirror(self):
        from metasci_scribe.config.onboard import wizard

        with patch.object(wizard.questionary, "select") as select:
            select.return_value.ask.return_value = "reset"
            _sections, config = wizard._open_existing_config_prompt(
                ScribeConfig(mirror="cn", model="other")
            )
        assert (config.mirror, config.model) == ("cn", ScribeConfig().model)
