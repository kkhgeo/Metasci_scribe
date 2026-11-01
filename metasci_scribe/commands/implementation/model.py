from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from typing import ClassVar

from ..base import Argument, Command, CommandContext
from ..manager import manager

_AUTH_MODE_ENV = "METASCI_SCRIBE_OPENAI_AUTH_MODE"


def _stage_auth_mode(temp_cfg, auth_mode: str | None) -> Callable[[], None]:
    """Make ``auth_mode`` the OpenAI route for a trial build; return the undo.

    ``get_chat_model`` reads the route from the environment, not from the
    config, so a switch that changes the route must set it before the build and
    put it back if the build fails. A committed switch is written by
    ``set_active_config``.
    """
    if auth_mode is None:
        return lambda: None
    previous = os.environ.get(_AUTH_MODE_ENV)
    temp_cfg.openai_auth_mode = auth_mode
    os.environ[_AUTH_MODE_ENV] = auth_mode

    def undo() -> None:
        if previous is None:
            os.environ.pop(_AUTH_MODE_ENV, None)
        else:
            os.environ[_AUTH_MODE_ENV] = previous

    return undo


async def _plan_catalog(ctx: CommandContext) -> list[tuple[str, str]] | None:
    """The signed-in account's models, or ``None`` after saying why not.

    The list comes from the account, not from a built-in table: what a plan
    can use depends on the plan. A :class:`PlanStop` is shown with its own
    message, which names the code and request ID support asks for.
    """
    from ...llm import chatgpt_plan as plan_llm
    from ...llm.errors import PlanAction, PlanStop

    try:
        catalog = await asyncio.to_thread(plan_llm.list_models)
    except PlanStop as stop:
        style = "yellow" if stop.action is PlanAction.RETRY else "red"
        ctx.ui.append_system(str(stop), style=style)
        return None
    if not catalog:
        ctx.ui.append_system(
            "Your ChatGPT plan lists no models for this app.", style="red"
        )
        return None
    return catalog


async def _save_model(ctx: CommandContext, model_name: str, provider: str) -> bool:
    """Save model and provider to the config file in one write.

    Runs in a thread because the write may wait for the config lock. Reports
    a failure in the UI and returns ``False``.
    """
    from ...config.settings import SettingsError, set_config_values
    from ...config.store import ConfigError

    try:
        await asyncio.to_thread(
            set_config_values, {"model": model_name, "provider": provider}
        )
    except (SettingsError, ConfigError) as e:
        ctx.ui.append_system(f"Could not save to config: {e}", style="red")
        return False
    return True


def extract_model_and_provider(
    args: list[str], current_provider: str | None = None
) -> tuple[str, str]:
    """Parse model name and provider from argument list.

    Args:
        args: Non-empty argument list (model_name [provider]).
        current_provider: Kept when it also serves the named model.

    Returns:
        ``(model_name, provider)`` tuple.

    Raises:
        ValueError: If the model is not in the registry, or the provider is
            not one Metasci Scribe supports.
    """
    from ...config.provider_keys import PROVIDERS
    from ...llm.models import MODELS
    from ...llm.registry import resolve_provider

    model_name = args[0]
    provider_override = args[1] if len(args) > 1 else None

    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'")

    if provider_override:
        if provider_override not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider_override}'")
        provider = provider_override
    else:
        provider = resolve_provider(model_name, current_provider)

    return model_name, provider


class ModelCommand(Command):
    """Switch the LLM model for the current session."""

    name = "/model"
    description = "Switch model (--save to persist)"
    category = "Model"
    # ``--save`` is parsed manually in ``execute`` via ``"--save" in args``;
    # ``type=bool`` below is declarative metadata, not enforced by the manager.
    arguments: ClassVar[list[Argument]] = [
        Argument(
            name="model_name",
            type=str,
            description="Model short name (e.g. gpt-6.1-sol). Opens picker if omitted.",
            required=False,
        ),
        Argument(
            name="--save",
            type=bool,
            description="Save the choice to config file",
            required=False,
        ),
    ]

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...agent import _ensure_config
        from ...llm.models import list_model_picker_entries

        cfg = _ensure_config()
        current_model = cfg.model
        current_provider = cfg.provider
        on_plan = getattr(cfg, "openai_auth_mode", "api_key") == "chatgpt_plan"

        # Parse --save flag
        save = "--save" in args
        args = [a for a in args if a != "--save"]

        if args:
            target = await self._target(ctx, args, current_provider, on_plan)
            if target is None:
                return
            model_name, provider = target
            await self._apply_model(ctx, model_name, provider, save=save)
            return

        # Interactive picker
        if not ctx.ui.supports_interactive:
            ctx.ui.append_system(
                "Usage: /model <name> [provider] [--save]",
                style="yellow",
            )
            return

        entries = list_model_picker_entries()
        if on_plan:
            catalog = await _plan_catalog(ctx)
            if catalog is None:
                return
            # OpenAI models are the account's; the Claude ones keep their own key.
            entries = [(slug, slug, "openai") for slug, _name in catalog] + [
                entry for entry in entries if entry[2] != "openai"
            ]

        result = await ctx.ui.wait_for_model_pick(
            entries,
            current_model=current_model,
            current_provider=current_provider,
        )
        if result is None:
            return

        name, provider = result
        await self._apply_model(ctx, name, provider, save=save)

    async def _target(
        self,
        ctx: CommandContext,
        args: list[str],
        current_provider: str | None,
        on_plan: bool,
    ) -> tuple[str, str] | None:
        """``(model, provider)`` for ``/model <name> [provider]``, or ``None``
        after saying why not.

        On the ChatGPT plan an OpenAI name must be one the account lists; a
        Claude name is checked against the built-in table as before.
        """
        from ...llm.registry import _MODEL_ENTRIES

        name = args[0]
        override = args[1] if len(args) > 1 else None
        claude_name = any(n == name and p == "anthropic" for n, _, p in _MODEL_ENTRIES)
        if on_plan and override in (None, "openai") and not claude_name:
            catalog = await _plan_catalog(ctx)
            if catalog is None:
                return None
            if name not in {slug for slug, _name in catalog}:
                ctx.ui.append_system(
                    f"'{name}' is not in your ChatGPT plan's model list. "
                    "Use /model to see it.",
                    style="red",
                )
                return None
            return name, "openai"
        try:
            return extract_model_and_provider(args, current_provider)
        except ValueError:
            ctx.ui.append_system(
                f"Unknown model '{name}'. Use /model to browse available models.",
                style="red",
            )
            return None

    async def _apply_model(
        self,
        ctx: CommandContext,
        model_name: str,
        provider: str,
        *,
        save: bool = False,
        auth_mode: str | None = None,
    ) -> bool:
        """Switch the session to ``model_name``; ``True`` when it switched.

        ``auth_mode`` also moves the OpenAI route (``/login`` passes
        ``chatgpt_plan``): the trial build runs on it, a failure puts the old
        one back, and a success commits it with the model.
        """
        import asyncio
        import copy

        from ...agent import (
            _build_chat_model,
            _ensure_config,
            set_active_config,
            set_chat_model_instance,
        )
        from ...cli.agent import _load_agent
        from ...runtime import AsyncRuntime

        cfg = _ensure_config()

        # Server backend: runs execute in the langgraph dev process, which
        # resolves the model per run from ``configurable.model`` /
        # ``configurable.model_provider`` (ConfigurableModelMiddleware). The
        # expensive local agent rebuild is pointless there — mutate the live
        # config (the per-run channel's source) and validate the model, but
        # skip the rebuild. Local backend keeps the rebuild path unchanged.
        # Test the *execution* side: on the composite backend ctx.graph_gateway
        # is a CompositeGraphGateway whose executor is the server gateway, so
        # reach through ``execute_gateway`` (a plain server/local gateway is its
        # own executor).
        from ...gateway.server import LangGraphServerGateway

        execute_gateway = getattr(
            ctx.graph_gateway, "execute_gateway", ctx.graph_gateway
        )
        if isinstance(execute_gateway, LangGraphServerGateway):
            temp_cfg = copy.copy(cfg)
            temp_cfg.model = model_name
            temp_cfg.provider = provider
            undo_auth_mode = _stage_auth_mode(temp_cfg, auth_mode)
            try:
                # Validation only — _build_chat_model does not mutate the
                # cached config/model globals, so a failure below leaves the
                # session untouched.
                new_chat_model = _build_chat_model(temp_cfg)
            except Exception as e:
                undo_auth_mode()
                ctx.ui.append_system(f"Failed to switch model: {e}", style="red")
                return False
            cfg.model = model_name
            cfg.provider = provider
            if auth_mode is not None:
                cfg.openai_auth_mode = auth_mode
            set_active_config(cfg)
            # Keep the local chat-model cache consistent too — cheap, and any
            # in-process graph use (e.g. background extraction) then matches.
            set_chat_model_instance(new_chat_model, (model_name, provider))
            if save:
                save = await _save_model(ctx, model_name, provider)
            update_model_fn = getattr(ctx.ui, "update_status_after_model_change", None)
            if callable(update_model_fn):
                update_model_fn(model_name, provider)
            saved_note = " (saved to config)" if save else ""
            ctx.ui.append_system(
                f"Switched to {model_name} ({provider}){saved_note} — "
                f"applies from the next run",
                style="green",
            )
            return True

        # Build a temporary config + its chat model and verify the agent can be
        # built before committing anything. ``create_cli_agent(config=...,
        # chat_model=...)`` is pure (issue #183) — it writes none of the cached
        # config/model module globals — so a failure below leaves the session
        # on the original model with no snapshot/restore needed.
        temp_cfg = copy.copy(cfg)
        temp_cfg.model = model_name
        temp_cfg.provider = provider
        undo_auth_mode = _stage_auth_mode(temp_cfg, auth_mode)

        # Re-thread the session's frontend event sink so the rebuilt agent's
        # middleware keeps driving the tool-selection widget
        # after a /model switch (the sink lives on the gateway, not the agent).
        events = ctx.graph_gateway.events

        try:
            new_chat_model = _build_chat_model(temp_cfg)
            load_kwargs = {
                "work_dir": str(ctx.work_dir),
                "workspace": ctx.workspace,
                "checkpointer": ctx.checkpointer,
                "config": temp_cfg,
                "chat_model": new_chat_model,
                "events": events,
            }
            async_runtime = getattr(ctx, "async_runtime", None)
            if isinstance(async_runtime, AsyncRuntime):
                load_kwargs["runtime"] = async_runtime
            new_agent = await asyncio.to_thread(
                _load_agent,
                **load_kwargs,
            )
        except Exception as e:
            undo_auth_mode()
            ctx.ui.append_system(f"Failed to switch model: {e}", style="red")
            return False

        # Agent built with no global mutation — commit the switch atomically.
        # These are pure assignments and cannot fail, so the session can never
        # be left half-switched. Apply the switch to the LIVE ``cfg`` in place
        # (the active config object) instead of rebinding ``_config`` to the
        # fresh ``temp_cfg`` — callers that hold the active config by reference
        # (e.g. a workspace-changing ``/resume`` reload) must observe the new model/provider. The verify
        # build above used the ``temp_cfg`` copy, so a failed build never reaches
        # here and the live ``cfg`` stays untouched (failure still no-ops).
        cfg.model = model_name
        cfg.provider = provider
        if auth_mode is not None:
            cfg.openai_auth_mode = auth_mode
        set_active_config(cfg)
        set_chat_model_instance(new_chat_model, (model_name, provider))
        ctx.agent = new_agent

        # Persist to config file if --save was given
        if save:
            save = await _save_model(ctx, model_name, provider)

        # Keep the session runtime pointing at the new agent.
        if ctx.session_runtime is not None and ctx.session_runtime.agent is not None:
            ctx.session_runtime.agent = new_agent

        # Update status bar if available
        update_model_fn = getattr(ctx.ui, "update_status_after_model_change", None)
        if callable(update_model_fn):
            update_model_fn(model_name, provider)

        saved_note = " (saved to config)" if save else ""
        ctx.ui.append_system(
            f"Switched to {model_name} ({provider}){saved_note}", style="green"
        )
        return True


manager.register(ModelCommand())
