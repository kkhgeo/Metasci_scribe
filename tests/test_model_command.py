"""Tests for the /model command and extract_model_and_provider helper."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestExtractModelAndProvider:
    """Unit tests for the argument parser helper."""

    def test_known_model_no_provider(self):
        from metasci_scribe.commands.implementation.model import (
            extract_model_and_provider,
        )

        name, prov = extract_model_and_provider(["claude-sonnet-5-5"])
        assert name == "claude-sonnet-5-5"
        assert prov == "anthropic"

    def test_known_model_with_provider_override(self):
        from metasci_scribe.commands.implementation.model import (
            extract_model_and_provider,
        )

        name, prov = extract_model_and_provider(["claude-sonnet-5-5", "anthropic"])
        assert name == "claude-sonnet-5-5"
        assert prov == "anthropic"

    def test_unsupported_provider_override_is_refused(self):
        from metasci_scribe.commands.implementation.model import (
            extract_model_and_provider,
        )

        with pytest.raises(ValueError, match="Unknown provider"):
            extract_model_and_provider(["claude-sonnet-5-5", "openrouter"])

    def test_short_name_stays_on_current_provider(self):
        """/model <name> keeps the provider that serves the name."""
        from metasci_scribe.commands.implementation.model import (
            extract_model_and_provider,
        )

        assert extract_model_and_provider(["gpt-6-sol"], "openai") == (
            "gpt-6-sol",
            "openai",
        )
        assert extract_model_and_provider(["claude-opus-5-5"], "anthropic") == (
            "claude-opus-5-5",
            "anthropic",
        )

    def test_unknown_model_no_provider_raises(self):
        from metasci_scribe.commands.implementation.model import (
            extract_model_and_provider,
        )

        with pytest.raises(ValueError, match="Unknown model"):
            extract_model_and_provider(["nonexistent-model-xyz"])

    def test_unknown_model_with_provider_still_raises(self):
        from metasci_scribe.commands.implementation.model import (
            extract_model_and_provider,
        )

        # Unknown models are always rejected, even with an explicit provider
        with pytest.raises(ValueError, match="Unknown model"):
            extract_model_and_provider(["my-custom-model", "custom-openai"])


class TestModelCommandUnknownModel:
    """Verify error message for unknown models."""

    async def test_unknown_model_shows_error(self):
        from metasci_scribe.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui

        with patch(
            "metasci_scribe.agent._ensure_config",
            return_value=cfg,
        ):
            await cmd.execute(ctx, ["nonexistent-model-xyz"])

        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Unknown model" in call_args[0][0]
        assert call_args[1]["style"] == "red"


class TestModelCommandPickerCancelled:
    """Verify no-op when the interactive picker is cancelled."""

    async def test_picker_returns_none(self):
        from metasci_scribe.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        ui.wait_for_model_pick = AsyncMock(return_value=None)
        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui

        with patch(
            "metasci_scribe.agent._ensure_config",
            return_value=cfg,
        ):
            await cmd.execute(ctx, [])

        # No model switch should have happened
        ui.append_system.assert_not_called()


class TestModelCommandSwitch:
    """Verify a successful model switch updates config and rebuilds agent."""

    async def test_switch_known_model(self):
        from metasci_scribe.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")
        new_agent = MagicMock()

        ctx = MagicMock()
        ctx.ui = ui
        ctx.work_dir = Path("/tmp/test")
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "metasci_scribe.agent._ensure_config",
                return_value=cfg,
            ),
            patch("metasci_scribe.agent._build_chat_model"),
            patch("metasci_scribe.agent.set_active_config") as set_cfg,
            patch("metasci_scribe.agent.set_chat_model_instance"),
            patch(
                "metasci_scribe.cli.agent._load_agent",
                return_value=new_agent,
            ),
        ):
            await cmd.execute(ctx, ["claude-opus-5-5"])

        # The switch is committed via set_active_config(temp_cfg), not by
        # mutating the original cfg object in place.
        set_cfg.assert_called_once()
        committed = set_cfg.call_args[0][0]
        assert committed.model == "claude-opus-5-5"
        assert committed.provider == "anthropic"

        # Agent should be replaced on context
        assert ctx.agent == new_agent

        # Success message shown
        ui.append_system.assert_called_once()
        msg = ui.append_system.call_args[0][0]
        assert "claude-opus-5-5" in msg
        assert "anthropic" in msg

    async def test_switch_with_save_flag(self):
        from metasci_scribe.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.work_dir = Path("/tmp/test")
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "metasci_scribe.agent._ensure_config",
                return_value=cfg,
            ),
            patch("metasci_scribe.agent._build_chat_model"),
            patch("metasci_scribe.agent.set_active_config"),
            patch("metasci_scribe.agent.set_chat_model_instance"),
            patch(
                "metasci_scribe.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch(
                "metasci_scribe.config.settings.set_config_values",
                return_value=("rev", {}),
            ) as mock_save,
        ):
            await cmd.execute(ctx, ["claude-opus-5-5", "--save"])

        # Config file should be updated, model and provider in one write
        mock_save.assert_called_once_with(
            {"model": "claude-opus-5-5", "provider": "anthropic"}
        )

        # Success message should mention save
        msg = ui.append_system.call_args[0][0]
        assert "saved to config" in msg

    async def test_switch_without_save_flag_does_not_persist(self):
        from metasci_scribe.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.work_dir = Path("/tmp/test")
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "metasci_scribe.agent._ensure_config",
                return_value=cfg,
            ),
            patch("metasci_scribe.agent._build_chat_model"),
            patch("metasci_scribe.agent.set_active_config"),
            patch("metasci_scribe.agent.set_chat_model_instance"),
            patch(
                "metasci_scribe.cli.agent._load_agent",
                return_value=MagicMock(),
            ),
            patch("metasci_scribe.config.settings.set_config_value") as mock_save,
        ):
            await cmd.execute(ctx, ["claude-opus-5-5"])

        # Config file should NOT be updated
        mock_save.assert_not_called()

        # Message should not mention save
        msg = ui.append_system.call_args[0][0]
        assert "saved to config" not in msg

    async def _run_switch_no_rebuild(self, graph_gateway):
        """Run /model against a server-executing gateway; return (ui, load_agent)."""
        from metasci_scribe.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.work_dir = Path("/tmp/test")
        ctx.checkpointer = MagicMock()
        ctx.graph_gateway = graph_gateway

        with (
            patch("metasci_scribe.agent._ensure_config", return_value=cfg),
            patch("metasci_scribe.agent._build_chat_model"),
            patch("metasci_scribe.agent.set_active_config") as set_cfg,
            patch("metasci_scribe.agent.set_chat_model_instance"),
            patch("metasci_scribe.cli.agent._load_agent") as load_agent,
        ):
            await cmd.execute(ctx, ["claude-opus-5-5"])

        return ui, set_cfg, load_agent

    async def test_switch_on_composite_backend_skips_rebuild(self):
        """Composite backend (executor is the server): switch config, no rebuild.

        Regression for testing ``isinstance(ctx.graph_gateway, ...)`` directly:
        the composite is not a LangGraphServerGateway, so /model rebuilt the
        unused in-process agent. It must reach through ``execute_gateway``.
        """
        from metasci_scribe.gateway.composite import CompositeGraphGateway
        from metasci_scribe.gateway.server import LangGraphServerGateway

        composite = CompositeGraphGateway(
            read=MagicMock(),
            execute=LangGraphServerGateway(thread_store=MagicMock()),
        )
        ui, set_cfg, load_agent = await self._run_switch_no_rebuild(composite)

        load_agent.assert_not_called()
        set_cfg.assert_called_once()
        assert set_cfg.call_args[0][0].model == "claude-opus-5-5"
        assert "next run" in ui.append_system.call_args[0][0]

    async def test_switch_on_plain_server_backend_skips_rebuild(self):
        """A uniform server gateway is its own executor: still no rebuild."""
        from metasci_scribe.gateway.server import LangGraphServerGateway

        server = LangGraphServerGateway(thread_store=MagicMock())
        ui, _set_cfg, load_agent = await self._run_switch_no_rebuild(server)

        load_agent.assert_not_called()
        assert "next run" in ui.append_system.call_args[0][0]


class TestModelCommandFailure:
    """Verify error handling when chat-model construction raises."""

    async def test_build_chat_model_error(self):
        from metasci_scribe.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.work_dir = Path("/tmp/test")
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "metasci_scribe.agent._ensure_config",
                return_value=cfg,
            ),
            patch(
                "metasci_scribe.agent._build_chat_model",
                side_effect=RuntimeError("API key missing"),
            ) as mock_build,
        ):
            await cmd.execute(ctx, ["claude-opus-5-5"])

        mock_build.assert_called_once()
        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Failed to switch model" in call_args[0][0]
        assert call_args[1]["style"] == "red"


@pytest.fixture
def evo_module_state():
    """Snapshot and restore ``metasci_scribe.agent`` module globals.

    The chat-model cache tests mutate ``_chat_model`` / ``_chat_model_key``
    / ``_config`` / ``_scribe_agent`` directly.  This fixture
    guarantees all four are restored — even if a test body grows an early
    return — so later tests in the suite see a clean module state.
    """
    import metasci_scribe.agent as mod

    snapshot = (
        mod._chat_model,
        mod._chat_model_key,
        mod._config,
        mod._scribe_agent,
    )
    try:
        yield mod
    finally:
        (
            mod._chat_model,
            mod._chat_model_key,
            mod._config,
            mod._scribe_agent,
        ) = snapshot


class TestEnsureChatModelCacheInvalidation:
    """Regression tests for issue #179: /model switch lagged by one step.

    Root cause: ``_ensure_chat_model()`` returned a cached ``_chat_model``
    without checking whether ``cfg.model`` / ``cfg.provider`` had changed
    since the cache was populated. ``ModelCommand._apply_model`` builds
    a new agent *before* ``set_chat_model`` runs, so the new agent was
    bound to the *previous* cached model.

    Fix: track a ``(model, provider)`` key alongside ``_chat_model`` and
    rebuild on mismatch.
    """

    def test_cache_rebuilds_when_config_model_changes(self, evo_module_state):
        """After cfg.model changes, _ensure_chat_model must return a new instance."""
        mod = evo_module_state

        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")
        m1 = MagicMock(name="model-1")
        m2 = MagicMock(name="model-2")

        mod._chat_model = None
        mod._chat_model_key = None
        mod._config = cfg
        mod._scribe_agent = None

        with patch("metasci_scribe.llm.get_chat_model", side_effect=[m1, m2]) as gm:
            first = mod._ensure_chat_model()
            assert first is m1
            # Same config → cache hit, no rebuild.
            again = mod._ensure_chat_model()
            assert again is m1
            assert gm.call_count == 1

            # Simulate /model switch writing the new choice into cfg.
            cfg.model = "minimax-m2.7"
            cfg.provider = "openrouter"

            second = mod._ensure_chat_model()
            # Must be the NEW model instance, not the cached one.
            assert second is m2
            assert second is not first
            assert gm.call_count == 2
            # Second call used the new model name + provider.
            _, kwargs = gm.call_args
            assert kwargs == {"model": "minimax-m2.7", "provider": "openrouter"}

    def test_cache_rebuilds_when_only_provider_changes(self, evo_module_state):
        """Same model name, different provider (openrouter vs anthropic) must rebuild."""
        mod = evo_module_state

        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")
        m1 = MagicMock(name="anthropic-model")
        m2 = MagicMock(name="openrouter-model")

        mod._chat_model = None
        mod._chat_model_key = None
        mod._config = cfg
        mod._scribe_agent = None

        with patch("metasci_scribe.llm.get_chat_model", side_effect=[m1, m2]) as gm:
            assert mod._ensure_chat_model() is m1
            cfg.provider = "openrouter"
            assert mod._ensure_chat_model() is m2
            assert gm.call_count == 2

    def test_set_chat_model_updates_key(self, evo_module_state):
        """set_chat_model must keep _chat_model_key in sync to avoid
        an extra rebuild on the very next _ensure_chat_model() call."""
        mod = evo_module_state

        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")
        m_set = MagicMock(name="explicit-set-model")

        mod._chat_model = None
        mod._chat_model_key = None
        mod._config = cfg
        mod._scribe_agent = None

        with patch("metasci_scribe.llm.get_chat_model", return_value=m_set) as gm:
            mod.set_chat_model("minimax-m2.7", provider="openrouter")
            assert mod._chat_model is m_set
            assert mod._chat_model_key == ("minimax-m2.7", "openrouter")

            # Align cfg to what set_chat_model was called with.
            cfg.model = "minimax-m2.7"
            cfg.provider = "openrouter"
            # Now _ensure_chat_model should NOT rebuild (key matches cfg).
            assert mod._ensure_chat_model() is m_set
            assert gm.call_count == 1

    def test_set_chat_model_is_no_op_when_key_already_matches(self, evo_module_state):
        """set_chat_model must NOT rebuild when the cache already holds the
        requested (model, provider).

        Under the /model flow, ``_load_agent`` already rebuilt ``_chat_model``
        via ``_ensure_chat_model`` before ``set_chat_model`` is reached, so
        the subsequent set should be idempotent — reusing the same Python
        instance (and thus the same underlying HTTP client) that ``ctx.agent``
        is already bound to.
        """
        mod = evo_module_state

        existing = MagicMock(name="existing-model")
        mod._chat_model = existing
        mod._chat_model_key = ("minimax-m2.7", "openrouter")
        mod._config = SimpleNamespace(model="minimax-m2.7", provider="openrouter")
        mod._scribe_agent = None

        with patch("metasci_scribe.llm.get_chat_model") as gm:
            returned = mod.set_chat_model("minimax-m2.7", provider="openrouter")

        # Fast path: returned the EXISTING instance, no rebuild.
        assert returned is existing
        assert mod._chat_model is existing
        gm.assert_not_called()


class TestApplyModelIntegration:
    """End-to-end regression for #179 + #183: `_apply_model` must produce an
    agent bound to the NEW model, threaded in via the pure ``create_cli_agent``
    path.

    Exercises the real chain:
        ``_apply_model → _build_chat_model → _load_agent(chat_model=...) →
        set_active_config / set_chat_model_instance``

    Only ``_load_agent`` is faked (to avoid spinning up deepagents, MCP tools,
    middleware, and subagent YAML); it binds the ``chat_model`` it receives.
    ``get_chat_model`` returns a distinct sentinel per ``(model, provider)``
    pair so we can assert on identity.
    """

    async def test_new_agent_is_bound_to_newly_selected_model(self, evo_module_state):
        from metasci_scribe.commands.implementation.model import ModelCommand
        from metasci_scribe.config.settings import ScribeConfig

        mod = evo_module_state

        sentinels: dict[tuple[str, str | None], MagicMock] = {}

        def _fake_get_chat_model(model, provider=None):
            key = (model, provider)
            if key not in sentinels:
                sentinels[key] = MagicMock(name=f"chat_model[{model}|{provider}]")
                sentinels[key]._bound_model = model
                sentinels[key]._bound_provider = provider
            return sentinels[key]

        def _fake_load_agent(
            work_dir=None,
            checkpointer=None,
            config=None,
            chat_model=None,
            *,
            workspace=None,
            on_mcp_progress=None,
            events=None,
        ):
            # The pure path threads the freshly built chat model in; bind it
            # directly instead of re-deriving via _ensure_chat_model.
            agent = MagicMock(name="fake-agent")
            agent._bound_model = chat_model
            return agent

        cfg = ScribeConfig(model="claude-sonnet-5-5", provider="anthropic")
        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.work_dir = Path("/tmp/test_integration")
        ctx.checkpointer = None

        # Prime: _chat_model already holds the OLD (default) model —
        # this is the state that caused the off-by-one in production.
        mod._config = cfg
        mod._chat_model = _fake_get_chat_model("claude-sonnet-5-5", "anthropic")
        mod._chat_model_key = ("claude-sonnet-5-5", "anthropic")
        mod._scribe_agent = None
        old_model = mod._chat_model

        with (
            patch(
                "metasci_scribe.llm.get_chat_model",
                side_effect=_fake_get_chat_model,
            ),
            patch(
                "metasci_scribe.cli.agent._load_agent",
                side_effect=_fake_load_agent,
            ),
        ):
            cmd = ModelCommand()
            await cmd._apply_model(ctx, "minimax-m2.7", "openrouter")

        # The agent produced by _apply_model must be bound to the
        # NEWLY requested model, threaded in via chat_model=.
        assert ctx.agent._bound_model is not old_model
        assert ctx.agent._bound_model._bound_model == "minimax-m2.7"
        assert ctx.agent._bound_model._bound_provider == "openrouter"

        # Global state reflects the switch end-to-end, committed via the setters.
        assert mod._chat_model_key == ("minimax-m2.7", "openrouter")
        assert mod._chat_model is sentinels[("minimax-m2.7", "openrouter")]
        # The switch is applied to the LIVE cfg in place and ``_config`` stays
        # bound to that same object, so callers holding the active config by
        # reference (e.g. serve's ``agent_holder["config"]``) observe the swap.
        assert mod._config is cfg
        assert cfg.model == "minimax-m2.7"
        assert cfg.provider == "openrouter"

        # User-visible success message.
        msg = ctx.ui.append_system.call_args[0][0]
        assert "minimax-m2.7" in msg
        assert "openrouter" in msg


class TestApplyModelServerBackend:
    """On the langgraph-server backend, /model must NOT rebuild the local
    agent: runs execute server-side and resolve the model per run from
    ``configurable.model``. The switch mutates the live config (the per-run
    channel's source) and validates the model, nothing more."""

    def _server_gateway(self):
        from metasci_scribe.gateway.server import (
            LangGraphServerGateway,
            LangGraphServerThreadStore,
        )
        from tests.fakes.common import FakeLangGraphClient, FakeLangGraphThreadsClient

        return LangGraphServerGateway(
            LangGraphServerThreadStore(
                client=FakeLangGraphClient(FakeLangGraphThreadsClient())
            )
        )

    async def test_server_backend_skips_agent_rebuild(self):
        from metasci_scribe.commands.implementation.model import ModelCommand
        from metasci_scribe.config.settings import ScribeConfig

        cmd = ModelCommand()
        cfg = ScribeConfig(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.work_dir = Path("/tmp/test")
        ctx.checkpointer = None
        ctx.graph_gateway = self._server_gateway()

        with (
            patch("metasci_scribe.agent._ensure_config", return_value=cfg),
            patch("metasci_scribe.agent._build_chat_model") as build,
            patch("metasci_scribe.agent.set_active_config"),
            patch("metasci_scribe.agent.set_chat_model_instance"),
            patch("metasci_scribe.cli.agent._load_agent") as load_agent,
        ):
            await cmd._apply_model(ctx, "minimax-m2.7", "openrouter")

        # Model still validated (fail-early on a bad pick)...
        assert build.call_count == 1
        # ...but no local agent rebuild — the server resolves per run.
        load_agent.assert_not_called()
        # Live config mutated in place: the per-run channel picks it up.
        assert cfg.model == "minimax-m2.7"
        assert cfg.provider == "openrouter"
        # Message names the apply semantics for the server backend.
        msg = ctx.ui.append_system.call_args[0][0]
        assert "minimax-m2.7" in msg
        assert "next run" in msg

    async def test_server_backend_failed_validation_leaves_session_untouched(self):
        from metasci_scribe.commands.implementation.model import ModelCommand
        from metasci_scribe.config.settings import ScribeConfig

        cmd = ModelCommand()
        cfg = ScribeConfig(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.graph_gateway = self._server_gateway()

        with (
            patch("metasci_scribe.agent._ensure_config", return_value=cfg),
            patch(
                "metasci_scribe.agent._build_chat_model",
                side_effect=ValueError("no such model"),
            ),
            patch("metasci_scribe.agent.set_active_config") as set_cfg,
            patch("metasci_scribe.cli.agent._load_agent") as load_agent,
        ):
            await cmd._apply_model(ctx, "broken-model", "openrouter")

        assert cfg.model == "claude-sonnet-5-5"
        set_cfg.assert_not_called()
        load_agent.assert_not_called()
        assert "Failed to switch model" in ctx.ui.append_system.call_args[0][0]


class TestApplyModelPreservesConfigByReference:
    """Regression for the din0s review on #267: serve mode (and any long-lived
    caller) holds the active config object by reference via
    ``agent_holder["config"]``.  The pure-path commit must apply the switch to
    that LIVE object in place — not rebind ``_config`` to a fresh ``temp_cfg``
    copy — otherwise a later workspace-changing ``/resume`` reloads the agent
    from the stale startup config and silently reverts the ``/model`` switch.

    Critically this must hold across *repeated* switches: the earlier
    "also mutate cfg but still rebind to temp_cfg" remedy only survives one
    switch (the held object stops being the active ``_config`` after the first).
    """

    async def test_held_config_reference_tracks_repeated_switches(
        self, evo_module_state
    ):
        from metasci_scribe.commands.implementation.model import ModelCommand
        from metasci_scribe.config.settings import ScribeConfig

        mod = evo_module_state

        def _fake_get_chat_model(model, provider=None):
            m = MagicMock(name=f"chat_model[{model}|{provider}]")
            m._bound_model = model
            return m

        def _fake_load_agent(
            work_dir=None,
            checkpointer=None,
            config=None,
            chat_model=None,
            *,
            workspace=None,
            on_mcp_progress=None,
            events=None,
        ):
            return MagicMock(name="fake-agent")

        cfg = ScribeConfig(model="claude-sonnet-5-5", provider="anthropic")
        mod._config = cfg
        mod._chat_model = None
        mod._chat_model_key = None
        mod._scribe_agent = None

        # Simulate serve capturing the startup config object once (commands.py:
        # ``agent_holder = {... "config": config}``) and never re-reading it.
        agent_holder = {"config": cfg}

        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.work_dir = Path("/tmp/test_byref")
        ctx.checkpointer = None

        cmd = ModelCommand()
        with (
            patch(
                "metasci_scribe.llm.get_chat_model",
                side_effect=_fake_get_chat_model,
            ),
            patch(
                "metasci_scribe.cli.agent._load_agent",
                side_effect=_fake_load_agent,
            ),
        ):
            for model, provider in [
                ("claude-opus-5-5", "anthropic"),
                ("minimax-m2.7", "openrouter"),
                ("claude-sonnet-5-5", "anthropic"),
            ]:
                await cmd._apply_model(ctx, model, provider)
                # The held reference must reflect the LATEST switch on every
                # iteration — not just the first — and stay the active config.
                assert agent_holder["config"].model == model
                assert agent_holder["config"].provider == provider
                assert mod._config is agent_holder["config"]


class TestModelCommandLoadAgentFailure:
    """Verify the transactional ordering: when ``_load_agent`` raises,
    nothing downstream (``set_chat_model``, ``cfg`` mutation,
    ``set_config_value``) should happen.

    This is the core guarantee of the refactor that established
    "build agent first, commit state only on success". Without this test
    the ordering could silently regress (e.g. if ``_apply_model`` were
    reordered to call ``set_chat_model`` first)."""

    async def test_load_agent_error_is_transactional(self):
        from metasci_scribe.commands.implementation.model import ModelCommand

        cmd = ModelCommand()
        ui = MagicMock()
        ui.supports_interactive = True
        cfg = SimpleNamespace(model="claude-sonnet-5-5", provider="anthropic")

        ctx = MagicMock()
        ctx.ui = ui
        ctx.work_dir = Path("/tmp/test")
        ctx.checkpointer = MagicMock()

        with (
            patch(
                "metasci_scribe.agent._ensure_config",
                return_value=cfg,
            ),
            patch("metasci_scribe.agent._build_chat_model"),
            patch(
                "metasci_scribe.cli.agent._load_agent",
                side_effect=RuntimeError("agent build failed"),
            ) as mock_load,
            patch(
                "metasci_scribe.agent.set_active_config",
            ) as mock_set_cfg,
            patch(
                "metasci_scribe.agent.set_chat_model_instance",
            ) as mock_set_model,
            patch(
                "metasci_scribe.config.settings.set_config_value",
            ) as mock_save,
        ):
            # Pass ``--save`` to strengthen the assertion: if the ordering
            # ever regresses, ``set_config_value`` would be called with
            # stale data.
            await cmd.execute(ctx, ["claude-opus-5-5", "--save"])

        # _load_agent was attempted (transactional first step).
        mock_load.assert_called_once()
        # Commit setters and downstream side-effects must NOT have happened.
        mock_set_cfg.assert_not_called()
        mock_set_model.assert_not_called()
        mock_save.assert_not_called()
        # Config must be untouched.
        assert cfg.model == "claude-sonnet-5-5"
        assert cfg.provider == "anthropic"
        # User sees a red error message.
        ui.append_system.assert_called_once()
        call_args = ui.append_system.call_args
        assert "Failed to switch model" in call_args[0][0]
        assert call_args[1]["style"] == "red"


class TestApplyModelLoadAgentFailureTransactional:
    """Regression for #183: when agent construction fails, the session stays on
    the original model with no snapshot/restore.

    Because ``create_cli_agent(config=..., chat_model=...)`` is now pure — it
    writes none of the four config/model globals — and ``_apply_model`` commits
    only after a successful build, a failing ``_load_agent`` leaves all four
    globals untouched. This replaces the old snapshot/restore rollback test.

    Complements :class:`TestModelCommandLoadAgentFailure`, which asserts the
    downstream setters never run on failure.
    """

    async def test_globals_unchanged_when_load_agent_raises(self, evo_module_state):
        from metasci_scribe.commands.implementation.model import ModelCommand
        from metasci_scribe.config.settings import ScribeConfig

        mod = evo_module_state

        def _fake_load_agent(
            work_dir=None,
            checkpointer=None,
            config=None,
            chat_model=None,
            *,
            workspace=None,
            on_mcp_progress=None,
            events=None,
        ):
            # The pure path writes no globals; mimic a failure partway through
            # agent wiring (middleware build, deepagents, MCP reconnect, ...).
            raise RuntimeError("middleware build failed")

        cfg = ScribeConfig(model="claude-sonnet-5-5", provider="anthropic")
        old_model = MagicMock(name="old-model")
        old_agent = MagicMock(name="old-default-agent")

        mod._config = cfg
        mod._chat_model = old_model
        mod._chat_model_key = ("claude-sonnet-5-5", "anthropic")
        mod._scribe_agent = old_agent

        ctx = MagicMock()
        ctx.ui = MagicMock()
        ctx.ui.supports_interactive = True
        ctx.work_dir = Path("/tmp/test_rollback")
        ctx.checkpointer = None

        with (
            patch(
                "metasci_scribe.agent._build_chat_model",
                return_value=MagicMock(name="new-model"),
            ),
            patch(
                "metasci_scribe.cli.agent._load_agent",
                side_effect=_fake_load_agent,
            ),
        ):
            cmd = ModelCommand()
            await cmd._apply_model(ctx, "minimax-m2.7", "openrouter")

        # All four globals are unchanged — nothing was committed.
        assert mod._config is cfg
        assert mod._chat_model is old_model
        assert mod._chat_model_key == ("claude-sonnet-5-5", "anthropic")
        assert mod._scribe_agent is old_agent
        # The ``cfg`` object itself was not mutated.
        assert cfg.model == "claude-sonnet-5-5"
        assert cfg.provider == "anthropic"
        # User sees an error message.
        msg = ctx.ui.append_system.call_args[0][0]
        assert "Failed to switch model" in msg
