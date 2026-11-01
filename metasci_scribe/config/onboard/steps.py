"""Individual wizard step functions.

Each ``_step_*`` prompts the user for one logical decision and returns the
chosen value.
"""

from __future__ import annotations

from pathlib import Path

import questionary
from prompt_toolkit.formatted_text import FormattedText
from questionary import Choice
from rich.markup import escape

from ...llm import DEFAULT_MODEL, get_models_for_provider
from ...paths import Workspace
from ..settings import ScribeConfig
from .helpers import (
    _auto_install_latexmk,
    _check_latex_components,
    _detect_tinytex_install_method,
    _ensure_npx,
    _install_tinytex,
    _print_latex_status,
    _prompt_and_validate_api_key,
    _provider_key_info,
)
from .style import (
    QMARK,
    WIZARD_STYLE,
    _checkbox_ask,
    _print_step_result,
    _print_step_skipped,
    console,
)


def _step_langgraph_dev_port(config: ScribeConfig) -> int:
    """Step 0.5: Choose the local TCP port for the langgraph dev subprocess.

    metasci-scribe auto-starts a ``langgraph dev`` server in the background to host
    deployed sub-agents (writing-agent) when ``enable_async_subagents`` is True. This step lets the user pick a free
    port, with a live conflict check on the configured default.

    Returns the chosen port; caller assigns it to ``config.langgraph_dev_port``.
    """
    if not getattr(config, "enable_async_subagents", False):
        # User has async disabled — port is irrelevant, no prompt.
        return getattr(config, "langgraph_dev_port", 6174)

    from ...langgraph_dev.manager import _is_port_occupied, is_langgraph_dev_running

    current_port = getattr(config, "langgraph_dev_port", 6174)
    current_occupied = _is_port_occupied(current_port)
    if current_occupied and is_langgraph_dev_running(port=current_port):
        # Another metasci-scribe shell is already serving on this port — reuse, don't
        # force the user to renumber.
        current_occupied = False

    # Bake the live status into the prompt label so the user sees it WITH
    # the question, not as a side-effect line that prints before input.
    # Single set of parens, no nesting.
    if current_occupied:
        prompt_label = (
            f"Enter port for Metasci Scribe server "
            f"(Current: {current_port}, occupied, pick another):"
        )
    else:
        prompt_label = (
            f"Enter port for Metasci Scribe server "
            f"(Current: {current_port}, available, Enter to keep):"
        )

    def valid_port(value: str) -> bool:
        if not value:
            # Allow keeping the default only if it's actually free; otherwise
            # require the user to pick something else.
            return not current_occupied
        try:
            port = int(value)
        except (ValueError, TypeError):
            return False
        if not (1024 < port < 65536):
            return False
        # Reject user-typed ports that are already occupied UNLESS the
        # occupier is our own langgraph dev (e.g., another metasci-scribe shell) —
        # in that case the runtime will reuse it.
        if not _is_port_occupied(port):
            return True
        return is_langgraph_dev_running(port=port)

    raw = questionary.text(
        prompt_label,
        validate=valid_port,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()

    if raw is None:
        raise KeyboardInterrupt()

    port = int(raw) if raw else current_port

    # Final probe — warn (don't fail) if the chosen port is still occupied
    # by something OTHER than our own langgraph dev. Reuse of an existing
    # metasci-scribe server on that port is fine. They can always change later via:
    # metasci-scribe config set langgraph_dev_port <port>
    if _is_port_occupied(port) and not is_langgraph_dev_running(port=port):
        console.print(
            f"  [yellow]⚠ Port {port} is occupied. metasci-scribe may fail to start its "
            f"server. Free the port or change later with: "
            f"metasci-scribe config set langgraph_dev_port <other-port>[/yellow]"
        )
    else:
        # Render the address the configured bind actually produces rather than
        # a hard-coded loopback URL — the two diverge once langgraph_dev_host
        # is pinned to a specific interface.
        from ...langgraph_dev.manager import _base_url

        host = getattr(config, "langgraph_dev_host", "")
        console.print(
            f"  [green]✓ Metasci Scribe will run on {_base_url(port, host)}[/green]"
        )
    return port


def _step_provider(
    config: ScribeConfig,
    *,
    label: str | None = None,
    default_value: str | None = None,
    allow_chatgpt: bool = False,
) -> str:
    """Step 1: Select LLM provider.

    Args:
        config: Current configuration.
        label: Optional role label (e.g. "co-pilot") to clarify which model this
            provider is for. When omitted, the generic main-model prompt is used.
        default_value: Preselect this provider instead of ``config.provider``
            (e.g. the auxiliary provider when configuring the co-pilot).
        allow_chatgpt: Offer "Continue with ChatGPT" as the first choice, with
            the API keys as the other way in. Only the main model has it: the
            co-pilot follows the OpenAI route the session already uses.

    Returns:
        Selected provider name, or ``"chatgpt_plan"`` for the ChatGPT sign-in.
    """
    from ...auth.notices import CONTINUE_LABEL

    choices = [
        Choice(title="OpenAI (GPT models — API key)", value="openai"),
        Choice(title="Anthropic (Claude models — API key)", value="anthropic"),
    ]
    preferred = default_value or config.provider
    if allow_chatgpt:
        choices.insert(
            0,
            Choice(title=CONTINUE_LABEL, value="chatgpt_plan"),
        )
        on_plan = (
            config.provider == "openai" and config.openai_auth_mode == "chatgpt_plan"
        )
        has_key = bool(config.openai_api_key or config.anthropic_api_key)
        if not default_value and (on_plan or not has_key):
            preferred = "chatgpt_plan"

    # Set default based on current config (or an explicit override).
    valid_providers = {c.value for c in choices}
    default = preferred if preferred in valid_providers else "openai"

    provider = questionary.select(
        f"Select {label} provider:" if label else "Select your LLM provider:",
        choices=choices,
        default=default,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if provider is None:
        raise KeyboardInterrupt()

    return provider


def _step_chatgpt_signin():
    """Step 1b: sign in with ChatGPT in the browser.

    Returns the credentials when the plan can be used, else ``None`` after
    saying why, and the wizard offers the choice again. Ctrl+C ends the wait
    and the wizard, like at any other step.
    """
    from ...auth import chatgpt_plan as plan_auth
    from ...auth.notices import (
        LOGIN_WARNING,
        USAGE_URL,
        WELCOME,
        describe_auth_error,
        plan_off_lines,
    )

    for line in LOGIN_WARNING:
        console.print(f"  [yellow]{escape(line)}[/yellow]")
    console.print(
        "  [dim]Opening your browser to continue with ChatGPT. "
        "Press Ctrl+C to cancel.[/dim]"
    )
    try:
        creds = plan_auth.login()
    except plan_auth.ChatGPTAuthError as exc:
        message, style = describe_auth_error(
            exc, retry="Pick Continue with ChatGPT to try again."
        )
        console.print(f"  [{style}]{escape(message)}[/{style}]")
        return None

    who = f"Signed in as {creds.email}" if creds.email else "Signed in to ChatGPT"
    if not creds.plan_enabled:
        headline, advice = plan_off_lines(who)
        console.print(f"  [yellow]{escape(headline)}[/yellow]")
        console.print(f"  [dim]{escape(advice)}[/dim]")
        return None
    console.print(f"  [green]✓ {escape(who)}. ChatGPT plan usage is on.[/green]")
    if plan_auth.claim_plan_welcome():
        console.print(f"  [green]{escape(WELCOME[0])}[/green]")
        console.print(f"  [dim]{escape(WELCOME[1])}[/dim]")
    console.print(f"  [dim]Manage usage: {USAGE_URL}[/dim]")
    return creds


def _plan_model_entries() -> list[tuple[str, str]] | None:
    """The signed-in account's models for the picker, or ``None`` to use the
    built-in list. Best effort: a catalog that cannot be read is no reason to
    stop the wizard, and the model can be changed later with ``/model``."""
    from ...llm import chatgpt_plan as plan_llm

    try:
        catalog = plan_llm.list_models()
    except Exception as exc:
        console.print(
            "  [yellow]Could not read your ChatGPT plan's model list: "
            f"{escape(str(exc))}[/yellow]"
        )
        console.print("  [dim]Showing the built-in list instead.[/dim]")
        return None
    return catalog or None


def _step_provider_api_key(
    config: ScribeConfig,
    provider: str,
    skip_validation: bool = False,
) -> str | None:
    """Step 2: Enter API key for the selected provider.

    Args:
        config: Current configuration.
        provider: Selected provider name.
        skip_validation: Skip API key validation.

    Returns:
        New API key or None if unchanged.
    """
    key_name, current, validate_fn = _provider_key_info(config, provider)

    hint = f"Current: ***{current[-4:]}" if current else "Not set"
    prompt_text = f"Enter {key_name} API key ({hint}, Enter to keep):"

    return _prompt_and_validate_api_key(
        prompt_text,
        current,
        validate_fn,
        skip_validation,
    )


def _step_model(
    config: ScribeConfig,
    provider: str,
    *,
    label: str | None = None,
    default_value: str | None = None,
    entries: list[tuple[str, str]] | None = None,
) -> str:
    """Step 3: Select model for the provider.

    Args:
        config: Current configuration.
        provider: Selected provider name.
        label: Optional role label (e.g. "co-pilot") for the prompt. When omitted,
            the generic main-model prompt is used.
        default_value: Preselect this model instead of ``config.model`` (e.g. the
            auxiliary model when configuring the co-pilot).
        entries: ``(name, hint)`` pairs to offer instead of the provider's
            built-in models, such as the signed-in account's own list.

    Returns:
        Selected model name.
    """
    model_prompt = f"Select {label} model:" if label else "Select model:"
    model_default = default_value or config.model
    # Get models for the selected provider
    if entries is None:
        entries = get_models_for_provider(provider)

    if not entries:
        # Unknown provider: direct text input.
        # Keep prompting until a non-empty model name is provided — saving an
        # empty string here leaves the first request broken with an opaque
        # "model required" error from the provider SDK.
        while True:
            model = questionary.text(
                "Model name:",
                style=WIZARD_STYLE,
                qmark=QMARK,
                placeholder=FormattedText([("fg:#858585", " e.g. owner/model-name")]),
                default=model_default or "",
            ).ask()
            if model is None:
                raise KeyboardInterrupt()
            model = model.strip()
            if model:
                return model
            console.print(
                "  [yellow]Model name cannot be empty. Press Ctrl+C to cancel.[/yellow]"
            )

    provider_models = [name for name, _ in entries]

    # Create choices with model IDs as hints
    _CUSTOM_SENTINEL = "__custom__"
    choices = []
    for name, model_id in entries:
        choices.append(Choice(title=f"{name} ({model_id})", value=name))
    choices.append(Choice(title="Type a model name...", value=_CUSTOM_SENTINEL))

    # Determine default. An explicit ``default_value`` override (e.g. a saved
    # co-pilot model on a re-run) that isn't a registry model is a custom name:
    # preselect "Type a model name..." and prefill it. A plain ``config.model``
    # that just isn't in the current provider's list (e.g. the provider was
    # changed, or the model was removed) falls back to the default model when
    # the provider lists it, else the first model, NOT the custom entry.
    custom_default = (
        default_value if default_value and default_value not in provider_models else ""
    )
    if model_default in provider_models:
        default = model_default
    elif custom_default:
        default = _CUSTOM_SENTINEL
    elif DEFAULT_MODEL in provider_models:
        default = DEFAULT_MODEL
    else:
        default = provider_models[0]

    selected = questionary.select(
        model_prompt,
        choices=choices,
        default=default,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if selected is None:
        raise KeyboardInterrupt()

    if selected != _CUSTOM_SENTINEL:
        return selected

    model = questionary.text(
        "Model name:",
        default=custom_default,
        style=WIZARD_STYLE,
        qmark=QMARK,
        placeholder=FormattedText([("fg:#858585", " e.g. owner/model-name")]),
    ).ask()
    if model is None:
        raise KeyboardInterrupt()
    model = model.strip()
    if not model:
        model = provider_models[0]
        console.print(f"  [dim]Using default: {model}[/dim]")
    return model


def _step_auxiliary_enable(config: ScribeConfig) -> bool:
    """Step 3.25: Choose whether to assemble a co-pilot (auxiliary) model.

    The co-pilot runs background/helper LLM calls — EvoMemory (memory workers)
    and the main agent's tool selector — so it can be a cheaper/faster model.
    Returns True when the user picks "Assemble"; the caller then runs the
    provider/key/model pickers. Returns False to keep the pilot (main model)
    everywhere.
    """
    console.print(
        "  [dim]A cheaper/faster co-pilot for EvoMemory (memory workers).[/dim]"
    )
    choice = questionary.select(
        "Co-pilot (auxiliary model):",
        choices=[
            Choice(
                title="Skip — single pilot (main model handles everything)",
                value="skip",
            ),
            Choice(
                title="Assemble a co-pilot — separate cheaper/faster model",
                value="assemble",
            ),
        ],
        default="assemble" if config.auxiliary_model else "skip",
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()
    if choice is None:
        raise KeyboardInterrupt()
    return choice == "assemble"


# Levels offered per provider. "" stores nothing, so each model keeps its own
# default.
_REASONING_EFFORT_CHOICES: dict[str, list[tuple[str, str]]] = {
    "openai": [
        ("", "default — Metasci Scribe's level for each model (recommended)"),
        ("low", "low    — fast responses with lighter reasoning"),
        ("medium", "medium — balances speed and reasoning depth"),
        ("high", "high   — greater reasoning depth"),
        ("xhigh", "xhigh  — extra high reasoning depth"),
        ("max", "max    — deepest reasoning; uses your quota fastest"),
    ],
}


def _asks_reasoning_effort(provider: str) -> bool:
    """Whether onboarding offers a reasoning level for ``provider``."""
    return provider in _REASONING_EFFORT_CHOICES


def _step_reasoning_effort(config: ScribeConfig, provider: str) -> str:
    """Step 3.5: Pick the reasoning effort level for OpenAI.

    Args:
        config: Current configuration.
        provider: The selected provider; it decides which levels are offered.

    Returns:
        Selected reasoning effort level.
    """
    levels = _REASONING_EFFORT_CHOICES[provider]
    effort_choices = [Choice(title=title, value=value) for value, title in levels]

    # A stored level (kept from another provider, or set by hand) may not be offered here.
    current = config.reasoning_effort
    offered = {value for value, _ in levels}
    if current not in offered:
        current = "" if "" in offered else "high"
    effort = questionary.select(
        "Select reasoning effort level:",
        choices=effort_choices,
        default=current,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if effort is None:
        raise KeyboardInterrupt()

    return effort


def _step_thinking(config: ScribeConfig) -> bool:
    """Step 6: Configure thinking panel visibility.

    Args:
        config: Current configuration.

    Returns:
        Whether to show thinking panels in the interface.
    """
    thinking_choices = [
        Choice(title="On (show model reasoning)", value=True),
        Choice(title="Off (hide model reasoning)", value=False),
    ]

    show_thinking = questionary.select(
        "Show thinking panel?",
        choices=thinking_choices,
        default=config.show_thinking,
        style=WIZARD_STYLE,
        qmark=QMARK,
        use_indicator=True,
    ).ask()

    if show_thinking is None:
        raise KeyboardInterrupt()

    return show_thinking


_RECOMMENDED_SKILLS = [
    # ── Third-party (K-Dense) ──
    {
        "label": "Scientific Skills  (143 research & experiment skills, third party by K-Dense)",
        "source": "K-Dense-AI/scientific-agent-skills@skills",
    },
    {
        "label": "Scientific Writer  (27 writing, review & presentation skills, third party by K-Dense)",
        "source": "K-Dense-AI/claude-scientific-writer@skills",
    },
    # ── Third-party (Orchestra Research) ──
    {
        "label": "AI Research Skills  (98 skills for training, evaluation, deployment, etc., third party by Orchestra Research)",
        "source": "Orchestra-Research/AI-Research-SKILLs",
    },
    # ── Third-party (Google DeepMind) ──
    {
        "label": "Science Skills  (37 genomics, structural-biology & literature skills, third party by Google DeepMind)",
        "source": "google-deepmind/science-skills@skills",
    },
    # ── Third-party (Anthropic) ──
    {
        "label": "Anthropic Skills  (co-authoring, design, etc., third party by Anthropic)",
        "source": "anthropics/skills@skills",
    },
    # ── Third-party (HuggingFace) ──
    {
        "label": "HuggingFace Skills  (dataset creation, model training & evaluation, third party by HuggingFace)",
        "source": "huggingface/skills@skills",
    },
    # ── Third-party (NVIDIA BioNeMo) ──
    {
        "label": "BioNeMo Skills  (31 protein folding, docking, generative chemistry & genomics skills, third party by NVIDIA)",
        "source": "NVIDIA-BioNeMo/bionemo-agent-toolkit@plugins/bionemo-agent-toolkit/skills",
    },
]


def _step_tinytex() -> None:
    """Step 9: Prepare LaTeX environment (TinyTeX).

    Asks the user whether they want to set up LaTeX for paper compilation.
    If yes, checks for an existing installation and offers to install TinyTeX
    when none is found.  The agent can auto-install missing LaTeX packages at
    runtime via ``tlmgr``, so only the base TinyTeX is needed here.
    """
    latex_choices = [
        Choice(title="No need (skip LaTeX setup)", value=False),
        Choice(title="Install now (TinyTeX compiler)", value=True),
    ]
    prepare = questionary.select(
        "LaTeX environment (needed to compile .tex → .pdf):",
        choices=latex_choices,
        default=False,
        style=WIZARD_STYLE,
        qmark=QMARK,
    ).ask()

    if prepare is None:
        raise KeyboardInterrupt()

    if not prepare:
        _print_step_skipped("LaTeX", "skipped")
        console.print(
            "  [dim]Install later:"
            ' curl -sL "https://yihui.org/tinytex/install-bin-unix.sh" | sh[/dim]'
        )
        return

    # User wants LaTeX — check existing installation
    console.print("  [dim]Checking LaTeX environment...[/dim]")

    components = _check_latex_components()

    if components["pdflatex"]:
        # Already installed — show detailed status
        _print_latex_status(components)
        # Auto-fix missing latexmk if tlmgr is available
        if not components["latexmk"] and components["tlmgr"]:
            _auto_install_latexmk()
        return

    # Not installed — detect install method and offer
    console.print("  [yellow]✗ pdflatex not found[/yellow]")
    method, command = _detect_tinytex_install_method()

    if method == "manual":
        _print_step_skipped("LaTeX", "manual install needed")
        console.print(f"  [dim]Install TinyTeX: {command}[/dim]")
        return

    install = questionary.confirm(
        f"Install TinyTeX via {method}?",
        default=True,
        style=WIZARD_STYLE,
        qmark=f"  {QMARK}",
    ).ask()

    if install is None:
        raise KeyboardInterrupt()

    if not install:
        _print_step_skipped("LaTeX", "skipped")
        console.print(f"  [dim]Install later: {command}[/dim]")
        return

    console.print("  [dim]Installing TinyTeX (this may take a minute)...[/dim]")
    if _install_tinytex(method, command):
        post = _check_latex_components()
        if post["pdflatex"]:
            _print_latex_status(post)
            _print_step_result("LaTeX", "TinyTeX installed")
        else:
            console.print("  [green]✓ TinyTeX installed[/green]")
            console.print(
                "  [yellow]⚠ Restart your terminal"
                " for pdflatex to appear in PATH[/yellow]"
            )
            _print_step_result("LaTeX", "installed (restart terminal for PATH)")
    else:
        console.print(f"  [dim]Install manually: {command}[/dim]")
        _print_step_result("LaTeX", "installation failed", success=False)


def _step_skills(workspace: Workspace) -> list[str]:
    """Step 7: Optionally install recommended skills.

    Shows checkbox first. Already-installed skills are shown as disabled
    so users don't accidentally reinstall them. If user selects nothing,
    checks npx as an easter egg — confirms skill discovery is available,
    or offers to install Node.js if missing.

    Returns:
        List of skill sources that were selected (empty if skipped).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from ...paths import GLOBAL_SKILLS_DIR
    from ...tools.skills_manager import installed_provenance, resolve_remote_head

    # Collect installed-skill dir names across both tiers. The dir-name match
    # is a best-effort fallback for legacy installs without manifests; the
    # provenance map below is the authoritative signal (handles packs that
    # explode into many child directories with unrelated names).
    installed_names: set[str] = set()
    for skills_dir in (workspace.skills_dir, Path(GLOBAL_SKILLS_DIR)):
        if skills_dir.exists():
            installed_names.update(e.name for e in skills_dir.iterdir() if e.is_dir())
    provenance = installed_provenance(workspace=workspace)
    installed_src = set(provenance)

    def _hint_name(source: str) -> str:
        """Derive expected skill directory name from source URL."""
        if "@" in source and "://" not in source:
            return source.split("@", 1)[1].strip()
        return source.rstrip("/").rsplit("/", 1)[-1]

    def _is_installed(source: str) -> bool:
        return source in installed_src or _hint_name(source) in installed_names

    # For installed packs with a stored commit, ask GitHub if upstream has
    # moved. Bounded parallel calls so onboard stays snappy; failures are
    # silently treated as "unknown" (label falls back to plain "installed").
    sources_to_check = [
        skill["source"]
        for skill in _RECOMMENDED_SKILLS
        if _is_installed(skill["source"])
        and provenance.get(skill["source"], {}).get("commit")
    ]
    upstream_heads: dict[str, str | None] = {}
    if sources_to_check:
        with ThreadPoolExecutor(max_workers=min(6, len(sources_to_check))) as ex:
            futures = {
                ex.submit(resolve_remote_head, src): src for src in sources_to_check
            }
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    upstream_heads[src] = fut.result()
                except Exception:
                    upstream_heads[src] = None

    def _has_update(source: str) -> bool:
        head = upstream_heads.get(source)
        recorded = provenance.get(source, {}).get("commit")
        return bool(head and recorded and head != recorded)

    choices = []
    for skill in _RECOMMENDED_SKILLS:
        src = skill["source"]
        if _is_installed(src):
            hint = (
                "  (installed — update available, re-select to sync)"
                if _has_update(src)
                else "  (installed — re-select to sync)"
            )
            choices.append(
                Choice(
                    title=[
                        ("", skill["label"]),
                        ("class:instruction", hint),
                    ],
                    value=src,
                )
            )
        else:
            choices.append(Choice(title=skill["label"], value=src))

    all_installed = all(_is_installed(skill["source"]) for skill in _RECOMMENDED_SKILLS)
    has_updates = any(_has_update(skill["source"]) for skill in _RECOMMENDED_SKILLS)
    if all_installed:
        if has_updates:
            console.print(
                "  [green]✓ All recommended skills installed; "
                "[yellow]updates available[/yellow] — re-select any to sync.[/green]"
            )
        else:
            console.print(
                "  [green]✓ All recommended skills are already installed "
                "and up to date.[/green]"
            )
            # No updates AND nothing new to install — nothing useful the
            # picker can do.
            return []

    prompt_label = (
        "Re-select installed skills to sync, or pick new ones:"
        if all_installed
        else "Install or Sync predefined skills:"
    )
    selected = _checkbox_ask(choices, prompt_label)

    if selected is None:
        raise KeyboardInterrupt()

    if not selected:
        # Verify skill discovery environment
        console.print("  [dim]Checking skill discovery environment...[/dim]")
        has_npx = _ensure_npx("skill discovery requires Node.js")
        if has_npx:
            _print_step_skipped("Skills", "none selected — good choice!")
            console.print("  [green]✓ npx found — skill discovery available[/green]")
            console.print(
                "  [yellow bold]* Less is more[/yellow bold] [dim](Metasci Scribe can discover and install skills on its own)[/dim]"
            )
        else:
            _print_step_skipped("Skills", "none selected")

        return []

    from ...tools.skills_manager import install_skill

    installed = []
    for source in selected:
        label = next(s["label"] for s in _RECOMMENDED_SKILLS if s["source"] == source)
        try:
            result = install_skill(source, workspace=workspace)
            if result.get("success"):
                _print_step_result("Skill", label)
                installed.append(source)
            else:
                _print_step_result(
                    "Skill", f"{label} — {result.get('error', 'failed')}", success=False
                )
        except Exception as e:
            _print_step_result("Skill", f"{label} — {e}", success=False)

    return installed


def _step_mcp_servers() -> list[str]:
    """Step 8: Optionally install recommended MCP servers.

    Shows a checkbox list of recommended servers. Already-configured servers
    are shown as disabled so users don't accidentally override them.
    Selected ones are added to the user MCP config via ``install_mcp_server()``.

    Handles env-key prompts, pip package installs, and URL-based servers.

    Returns:
        List of server names that were installed.
    """
    from ...mcp.client import _load_user_config
    from ...mcp.registry import fetch_marketplace_index, install_mcp_server

    try:
        all_servers = fetch_marketplace_index()
    except Exception as exc:
        from rich.markup import escape

        # The message carries git's stderr; unescaped, `[...]` in it would be
        # read as Rich markup.
        console.print(
            "  [yellow]\u26a0 Could not fetch MCP marketplace index: "
            f"{escape(str(exc))}[/yellow]"
        )
        console.print(
            "  [yellow]Skipping MCP setup \u2014 "
            "you can re-run with [bold]metasci-scribe configure mcp[/bold] later.[/yellow]"
        )
        return []
    servers = [s for s in all_servers if "onboarding" in s.tags]
    existing_config = _load_user_config()

    choices = []
    for srv in servers:
        if srv.name in existing_config:
            choices.append(
                Choice(
                    title=[
                        ("", srv.label),
                        ("class:instruction", "  (already configured)"),
                    ],
                    value=srv.name,
                    disabled=True,
                )
            )
        else:
            choices.append(Choice(title=srv.label, value=srv.name))

    # Only declare "all configured" when there ARE recommended servers AND
    # every one is already in the user's config \u2014 distinct from "marketplace
    # returned nothing" (a transient failure) which is handled above.
    if servers and all(srv.name in existing_config for srv in servers):
        console.print(
            "[green]\u2713 All recommended MCP servers are already configured.[/green]"
        )
        return []
    if not servers:
        console.print(
            "  [dim]No recommended MCP servers are tagged for onboarding right now.[/dim]"
        )
        return []

    selected = _checkbox_ask(choices, "Install recommended MCP servers:")

    if selected is None:
        raise KeyboardInterrupt()

    if not selected:
        _print_step_skipped("MCP Servers", "none selected")
        console.print(
            "  [dim]Add later with: metasci-scribe mcp add <name> <command> [--env-ref KEY] -- [args][/dim]"
        )
        return []

    # Check if any selected servers require npx
    needs_npx = any(srv.command == "npx" for srv in servers if srv.name in selected)
    if needs_npx:
        if not _ensure_npx("some MCP servers require Node.js"):
            npx_servers = {
                srv.name
                for srv in servers
                if srv.name in selected and srv.command == "npx"
            }
            selected = [s for s in selected if s not in npx_servers]
            if npx_servers:
                console.print(
                    f"  [yellow]\u26a0 Skipping {', '.join(sorted(npx_servers))} (npx not available)[/yellow]"
                )
            if not selected:
                return []

    installed = []
    for name in selected:
        srv = next(s for s in servers if s.name == name)
        try:
            if install_mcp_server(srv):
                _print_step_result("MCP", f"{name}")
                installed.append(name)
            else:
                _print_step_result(
                    "MCP", f"{name} — installation failed", success=False
                )
        except Exception as e:
            _print_step_result("MCP", f"{name} — {e}", success=False)

    return installed
