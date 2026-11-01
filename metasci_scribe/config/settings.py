"""Configuration management for Metasci Scribe.

Handles loading, saving, and merging configuration from multiple sources.
See :func:`get_effective_config` for the authoritative priority chain —
``METASCI_SCRIBE_*`` shell values and third-party keys are treated
asymmetrically with respect to workspace ``.env`` handling.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml
from dotenv import dotenv_values, find_dotenv

from .. import paths
from . import store

# Tools that run shell commands and need manual HITL approval (subject to
# shell_allow_list). Single source of truth for every interrupt consumer
# (stream/display.py, stream/interaction.py) — keep aligned with the agent's
# `interrupt_on` set in agent.py.
HITL_SHELL_TOOLS = ("execute", "run_in_background")

# Armed non-shell destructive tools must always prompt — no allow-list carve-outs
# (their args carry paths, not commands). Keep aligned with HITL_INTERRUPT_ON.
HITL_ALWAYS_PROMPT_TOOLS = ("delete",)


class MemoryObservationTarget(StrEnum):
    """Runtime locations that can receive `record_observation`."""

    AGENT = "agent"
    TURN_WORKER = "turn_worker"
    SUBAGENT_WORKER = "subagent_worker"


class MemoryObservationWriter(StrEnum):
    """Configured observation-writing policy."""

    OFF = "off"
    AGENT = "agent"
    WORKER = "worker"
    ALL = "all"

    def enables(self, target: MemoryObservationTarget) -> bool:
        match self:
            case MemoryObservationWriter.OFF:
                return False
            case MemoryObservationWriter.AGENT:
                return target == MemoryObservationTarget.AGENT
            case MemoryObservationWriter.WORKER:
                return target in (
                    MemoryObservationTarget.TURN_WORKER,
                    MemoryObservationTarget.SUBAGENT_WORKER,
                )
            case MemoryObservationWriter.ALL:
                return target in (
                    MemoryObservationTarget.AGENT,
                    MemoryObservationTarget.TURN_WORKER,
                    MemoryObservationTarget.SUBAGENT_WORKER,
                )


class MemorySkillSynthesisMode(StrEnum):
    """Configured AutoSkills approval behavior."""

    REVIEW = "review"
    AUTO = "auto"


class MemorySkillSynthesisCadence(StrEnum):
    """Preset cadence for the built-in AutoSkills schedule."""

    NIGHTLY = "nightly"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


DEFAULT_MEMORY_OBSERVATION_WRITER = MemoryObservationWriter.ALL
DEFAULT_MEMORY_SKILL_SYNTHESIS_MODE = MemorySkillSynthesisMode.REVIEW
DEFAULT_MEMORY_SKILL_SYNTHESIS_CADENCE = MemorySkillSynthesisCadence.WEEKLY
DEFAULT_MEMORY_SKILL_SYNTHESIS_TIME = "03:00"


def _normalize_hhmm(value: Any) -> str | None:
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        return None
    hour, minute = parts
    if not (hour.isdecimal() and minute.isdecimal()):
        return None
    try:
        hour_int = int(hour)
        minute_int = int(minute)
    except ValueError:
        return None
    if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
        return None
    return f"{hour_int:02d}:{minute_int:02d}"


# =============================================================================
# Configuration paths
# =============================================================================


def get_config_dir() -> Path:
    """Get the configuration directory path.

    ``$XDG_CONFIG_HOME/metasci-scribe`` when that is set, otherwise
    ``~/.config/metasci-scribe``, read once at startup (``paths.CONFIG_DIR``).
    """
    return paths.CONFIG_DIR


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return get_config_dir() / "config.yaml"


class GatewaySurface(StrEnum):
    """A production surface that selects a gateway backend.

    Each member maps to a per-surface ``gateway_backend_*`` config field; see
    :func:`resolve_gateway_backend`. Only one surface runs per process, so a
    surface resolves its backend once at startup and threads it explicitly.
    """

    SINGLE_SHOT = "single_shot"
    INTERACTIVE = "interactive"
    STANDALONE = "standalone"


# =============================================================================
# Configuration dataclass
# =============================================================================


@dataclass
class ScribeConfig:
    """Metasci Scribe configuration settings.

    Attributes:
        anthropic_api_key: Anthropic API key for Claude models.
        openai_api_key: OpenAI API key for GPT models.
        provider: Default LLM provider ('openai' or 'anthropic').
        model: Default model name (short name or full ID).
        auxiliary_provider: Provider for auxiliary_model (empty = use main provider).
        auxiliary_model: Model for memory workers + tool selector (empty = use main model).
        default_mode: Default workspace mode ('daemon' or 'run').
        default_workdir: Default workspace directory (empty = use current working directory).
        show_thinking: Whether to show thinking panels in CLI.
    """

    # API Keys
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    openai_api_key: str = ""
    # "api_key" is the only route that works today. "chatgpt_plan" is reserved
    # for the ChatGPT plan connector and is refused until that connector exists.
    openai_auth_mode: Literal["api_key", "chatgpt_plan"] = "api_key"

    # LLM Settings
    provider: str = "openai"
    model: str = "gpt-6.1-sol"
    # Transient model-call failures (5xx, 429, connection drops): how long one
    # model call keeps retrying after it first fails, in seconds; 0 = no retry.
    model_retry_wait_seconds: int = 60
    auto_mode_retry_wait_seconds: int = 1800  # unattended runs: auto_mode, scheduler
    # Optional auxiliary model for background/helper LLM calls (memory workers +
    # tool selector). Empty = fall back to the main model/provider.
    auxiliary_provider: str = ""  # empty = use main provider
    auxiliary_model: str = ""  # empty = use main model

    # Async Sub-agent Settings
    # When True, the metasci-scribe CLI auto-starts a langgraph dev subprocess
    # so any sub-agent flagged ``async: true`` in subagents/<name>.yaml runs
    # non-blocking via AsyncSubAgent. Currently affects writing-agent. Adds
    # ~10-15s to CLI startup (langgraph dev cold start, mostly MCP server
    # spawn time).
    #
    # Off by default: every sub-agent runs in-process and a plain start never
    # spawns a background server. Memory workers, skill synthesis and the
    # scheduler also ask for that server; they are off by default too.
    enable_async_subagents: bool = False

    # Port for the auto-started langgraph dev subprocess. 6174 is Kaprekar's
    # constant — a memorable Metasci-Scribe-themed default that avoids collisions
    # with common dev ports (3000/5000/8000/8080) and the langgraph CLI default
    # 2024. Override if it conflicts with another local service.
    langgraph_dev_port: int = 6174

    # Network interface the langgraph dev subprocess binds to. Loopback by
    # default — this is the unauthenticated agent API (the agent can run
    # shell), so "0.0.0.0" is opt-in and every launcher prints a PUBLIC BIND
    # banner while exposed. Internal callers *connect* via manager._probe_host,
    # so widening never redirects their traffic off-box.
    langgraph_dev_host: str = "127.0.0.1"

    # --- Scheduled tasks (cron) ---
    # Master switch for the scheduler. Off by default.
    enable_scheduler: bool = False
    # Default IANA timezone for cron schedules created without an explicit tz.
    # Empty string => the host's local IANA zone (resolved via tzlocal), falling
    # back to UTC if it can't be determined; set e.g. "Europe/London" to pin one.
    scheduler_default_timezone: str = ""

    # Whether langgraph dev persists its runtime state to .langgraph_api/ next
    # to the subprocess cwd. True (default) keeps async-task, scheduler, and
    # Store API state across subprocess restarts — useful for future
    # cross-session async, cron, and Store features. Set False to suppress
    # writes (workspace stays cleaner; state is in-memory only and lost on
    # CLI exit). Metasci Scribe's main thread persistence uses sessions.db
    # regardless of this setting.
    langgraph_dev_file_persistence: bool = True

    # Concurrency: how many runs each langgraph dev worker processes in parallel.
    # 10 is the langgraph dev recommended default and works well on a typical
    # dev machine. Lower it (e.g., 4) on memory-constrained or low-core
    # machines if multiple async sub-agents in flight cause noticeable
    # slowdown.
    langgraph_dev_jobs_per_worker: int = 10

    # Keep the auto-started langgraph dev subprocess running after the CLI
    # exits. The next `metasci-scribe` start in the same workspace reuses it instantly
    # instead of paying the cold boot (~15s). Starting in a DIFFERENT workspace
    # raises WorkspaceMismatchError with the leftover server's pid — stop it
    # manually (the server is pinned to one workspace per process). Known
    # limitation: changing langgraph_dev_port/host while a keepalive server
    # runs orphans its records — run `metasci-scribe server stop` before switching.
    langgraph_dev_keepalive: bool = False

    # Which backend serves the CLI/TUI's graph runs: the in-process local
    # gateway (default) or the langgraph dev server via LangGraphServerGateway.
    # Server-gateway-migration rollback flag: flipping back to "local" restores
    # the in-process path without code changes. The CLI keeps building its
    # in-process agent (and its MCP sessions) until a surface actually cuts
    # over to the server gateway - skipping that init lands with the surface
    # cutover, not here.
    gateway_backend: Literal["local", "langgraph_server"] = "local"

    # Per-surface overrides of ``gateway_backend``. Each production surface can
    # pick its own backend; ``"inherit"`` defers to the global ``gateway_backend``
    # above. The fixed-workspace surfaces (single-shot, standalone) default
    # to ``inherit``, so the global flag moves them. The interactive CLI
    # defaults to ``local`` instead: the server path is still lossy there (the
    # per-session workspace is not applied server-side, #413, and per-run config
    # is dropped on resume, #454), so a global ``langgraph_server`` must not move
    # them - reaching the server backend on those surfaces takes an explicit
    # ``langgraph_server`` here. Resolve with :func:`resolve_gateway_backend`;
    # never read these fields directly for a routing decision.
    gateway_backend_single_shot: Literal["inherit", "local", "langgraph_server"] = (
        "inherit"
    )
    gateway_backend_interactive: Literal["inherit", "local", "langgraph_server"] = (
        "local"
    )
    gateway_backend_standalone: Literal["inherit", "local", "langgraph_server"] = (
        "inherit"
    )

    # Max LangGraph super-steps (LLM call / tool call / sub-agent delegation
    # each count as 1) before raising GraphRecursionError. Resets on every
    # ``agent.invoke()`` — i.e., this is per-turn, NOT per-conversation. For
    # long conversations the relevant mechanisms are checkpointer persistence
    # (sessions.db), ContextEditingMiddleware (window management), and
    # EvoMemoryMiddleware (cross-turn memory).
    #
    # 1,000,000 is "effectively unlimited" — typical research turns use
    # 200-1000 steps; reaching 1M would cost ~$10K in tokens, by which point
    # rate limits, context overflow, or API quota errors would trip first.
    # Lower (e.g., 5000) if you want a tighter safety net against runaway loops.
    recursion_limit: int = 1_000_000

    # Memory Settings
    # Profile memory injects and maintains `/memories/profile/...` files.
    memory_profile_enabled: bool = False
    # Observation memory indexes `/memories/observations/...` and adds
    # observation-read guidance/context. Writes require this switch plus an
    # allowed `memory_observation_writer` role below.
    memory_observations_enabled: bool = False
    # Which observation-writing path receives the `record_observation` tool:
    # "off" disables writes; "agent" means live agents; "worker" means
    # post-run memory workers; "all" means live agents and post-run memory
    # workers.
    memory_observation_writer: MemoryObservationWriter = (
        DEFAULT_MEMORY_OBSERVATION_WRITER
    )
    # Post-turn and post-subagent memory workers. Disable for no-background-memory
    # controls while still allowing live agents to read configured memory.
    memory_workers_enabled: bool = False
    # Slow EvoMemory maintenance that periodically scans observation clusters
    # and drafts reusable skills.
    memory_skill_synthesis_enabled: bool = False
    memory_skill_synthesis_mode: MemorySkillSynthesisMode = (
        DEFAULT_MEMORY_SKILL_SYNTHESIS_MODE
    )
    memory_skill_synthesis_cadence: MemorySkillSynthesisCadence = (
        DEFAULT_MEMORY_SKILL_SYNTHESIS_CADENCE
    )
    memory_skill_synthesis_time: str = DEFAULT_MEMORY_SKILL_SYNTHESIS_TIME
    # Max number of parsed observation files kept in the process-wide parse
    # cache. Each entry holds one parsed document keyed on the file path; at
    # the end of a call the LRU trims down to max(cap, entries touched by
    # the call), so an active store larger than the cap temporarily exceeds
    # it instead of thrashing. 2048 is generous for the single-workspace
    # deploy model; raise for a long-running server that cycles through many
    # large workspaces.
    memory_observation_cache_max_files: int = 2048

    # Workspace Settings
    default_mode: Literal["daemon", "run"] = "daemon"
    default_workdir: str = ""

    # UI Settings
    show_thinking: bool = True
    # The terminal conversation is the only screen. An old config file that
    # holds "tui" or "webui" reads as "cli" (with a warning).
    ui_backend: Literal["cli"] = "cli"
    # Download source for `metasci-scribe setup` and on-demand installs (Node.js).
    # "cn" uses mainland China mirrors; set by `metasci-scribe setup --cn`.
    mirror: Literal["default", "cn"] = "default"
    log_level: str = "warning"
    # Empty means use the provider/model default. A non-empty value is an
    # explicit user override exported as METASCI_SCRIBE_REASONING_EFFORT.
    reasoning_effort: str = ""

    # HITL (Human-in-the-Loop) Settings
    auto_approve: bool = False  # Auto-approve all tool executions without prompting
    auto_mode: bool = False  # Run unattended: imply auto_approve and disable ask_user
    shell_allow_list: str = ""  # Comma-separated shell command prefixes to auto-approve
    # Comma-separated origins (``https://host[:port]``) whose pages may use the
    # backend besides this machine's own. Read from config.yaml only: no
    # environment variable sets it, so a workspace ``.env`` cannot widen it.
    backend_allowed_origins: str = ""

    # Dangerous mode: real-filesystem access (no workspace confinement). The agent
    # operates on real absolute paths anywhere on disk; the privileged-command
    # blocklist (sudo/chmod/dd/...) still applies. Implies auto_approve.
    dangerous_mode: bool = False

    # Agent features
    enable_ask_user: bool = True  # Enable ask_user tool for agent-initiated questions

    # CodeInterpreterMiddleware (PTC — Parallel Tool Calls) tuning
    # The PTC allowlist itself is hardcoded in
    # ``metasci_scribe/middleware/code_interpreter.py`` as a load-bearing safety
    # decision (excludes ``execute`` so PTC can't bypass HITL approval,
    # excludes ``write_file``/``edit_file`` because batched writes have no
    # benefit). Only the resource budget knobs are user-tunable.
    code_interpreter_timeout: float = 60.0  # seconds per JS eval
    code_interpreter_max_result_chars: int = 10000  # truncate large JSON results

    # Default per-command timeout (seconds) for the sandbox `execute` tool.
    # Only the default — the agent can still override per command up to the
    # deepagents max_execute_timeout cap (3600s).
    sandbox_execute_timeout: int = 300

    # Checkpoint pruning (sessions.db retention per (thread_id, checkpoint_ns))
    # Safety net for runaway conversations. Under DeltaChannel (deepagents 0.6+)
    # normal usage produces linear growth, so this default is set well above
    # any realistic conversation length (~180-450 turns of dialogue) while
    # still capping legacy bloat at upgrade time. 0 disables ongoing pruning
    # entirely; the one-time legacy migration sweep still runs.
    checkpoint_keep_per_thread: int = 1000

    # OpenAI API mode - "" = auto, "true" = force Responses, "false" = force Completions
    use_responses_api: str = ""

    def __post_init__(self) -> None:
        # A non-positive or non-int sandbox_execute_timeout (e.g. a hand-edited
        # config file value — load_config does not coerce file values — or a
        # 0/negative env value) would raise inside CustomSandboxBackend.__init__
        # and crash agent/CLI startup. Fall back to the default instead, matching
        # how malformed env values already degrade to defaults.
        t = self.sandbox_execute_timeout
        if not isinstance(t, int) or isinstance(t, bool) or t <= 0:
            logging.getLogger(__name__).warning(
                "Invalid sandbox_execute_timeout %r; falling back to 300.", t
            )
            self.sandbox_execute_timeout = 300

        # A retry budget that is not a whole number of seconds (a hand-edited
        # ``60s``) goes back to its default; ``0`` or less switches the retry off.
        for name in ("model_retry_wait_seconds", "auto_mode_retry_wait_seconds"):
            budget = getattr(self, name)
            if not isinstance(budget, int) or isinstance(budget, bool):
                default = type(self).__dataclass_fields__[name].default
                logging.getLogger(__name__).warning(
                    "Invalid %s %r; falling back to %d.", name, budget, default
                )
                setattr(self, name, default)

        # A non-positive cache cap would evict every file entry immediately,
        # defeating the cache entirely.
        cap = self.memory_observation_cache_max_files
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            logging.getLogger(__name__).warning(
                "Invalid memory_observation_cache_max_files %r; falling back to 2048.",
                cap,
            )
            self.memory_observation_cache_max_files = 2048

        # shell_allow_list is typed as a comma-separated string, but a YAML list
        # spelling (``shell_allow_list: [ls, cat]``) survives here as a list.
        # The policy resolver calls ``.split`` on it, so normalise to CSV once at
        # the source rather than guarding every consumer.
        if isinstance(self.shell_allow_list, list | tuple):
            self.shell_allow_list = ",".join(str(s) for s in self.shell_allow_list)
        if isinstance(self.backend_allowed_origins, list | tuple):
            self.backend_allowed_origins = ",".join(
                str(s) for s in self.backend_allowed_origins
            )

        # auto_mode and dangerous_mode both imply auto_approve regardless of
        # source (CLI, env, config file, direct construction) — done here so the
        # "unattended → zero prompts" contract holds even when either is set via
        # `config set` or a config file rather than a CLI flag.
        if self.auto_mode or self.dangerous_mode:
            self.auto_approve = True

        _normalize_str_enum_fields(self)
        _normalize_literal_fields(self)

        # Bind hosts reach socket.bind() / the langgraph CLI verbatim, where a
        # stray-whitespace or empty value surfaces as an opaque gaierror at
        # startup. Normalize to the field's own default instead.
        for _host_field, _host_default in (("langgraph_dev_host", "127.0.0.1"),):
            _host = getattr(self, _host_field, _host_default)
            _host = _host.strip() if isinstance(_host, str) else ""
            setattr(self, _host_field, _host or _host_default)

        synthesis_time = _normalize_hhmm(self.memory_skill_synthesis_time)
        if synthesis_time is None:
            logging.getLogger(__name__).warning(
                "Invalid memory_skill_synthesis_time %r; falling back to %s.",
                self.memory_skill_synthesis_time,
                DEFAULT_MEMORY_SKILL_SYNTHESIS_TIME,
            )
            self.memory_skill_synthesis_time = DEFAULT_MEMORY_SKILL_SYNTHESIS_TIME
        else:
            self.memory_skill_synthesis_time = synthesis_time


@dataclass(frozen=True)
class MemoryControls:
    """Resolved memory feature switches used by agent and worker wiring."""

    profile_enabled: bool
    observations_enabled: bool
    observation_writer: MemoryObservationWriter
    workers_enabled: bool

    @classmethod
    def from_config(cls, config: ScribeConfig) -> MemoryControls:
        return cls(
            profile_enabled=config.memory_profile_enabled,
            observations_enabled=config.memory_observations_enabled,
            observation_writer=config.memory_observation_writer,
            workers_enabled=config.memory_workers_enabled,
        )

    @property
    def memory_enabled(self) -> bool:
        return self.profile_enabled or self.observations_enabled

    def observation_tool_enabled(self, target: MemoryObservationTarget) -> bool:
        return self.observations_enabled and self.observation_writer.enables(target)

    def worker_needed(self, target: MemoryObservationTarget) -> bool:
        if not self.workers_enabled:
            return False
        match target:
            case MemoryObservationTarget.TURN_WORKER:
                return self.profile_enabled or self.observation_tool_enabled(target)
            case MemoryObservationTarget.SUBAGENT_WORKER:
                return self.profile_enabled or self.observation_tool_enabled(target)
            case MemoryObservationTarget.AGENT:
                return False


# =============================================================================
# Config file operations
# =============================================================================


def load_config() -> ScribeConfig:
    """Load configuration from file.

    Returns:
        ScribeConfig instance with values from file, or defaults if
        file doesn't exist.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return ScribeConfig()

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return config_from_mapping(data)
    except Exception:
        # On any error, return defaults
        return ScribeConfig()


def config_from_mapping(data: dict[str, Any]) -> ScribeConfig:
    """Build a config from the file's mapping, defaults for what it lacks."""
    # Filter to only valid fields; a blank ``key:`` loads as None, which no
    # field accepts, so it falls back to the default.
    valid_fields = {f.name for f in fields(ScribeConfig)}
    filtered_data = {
        k: v for k, v in data.items() if k in valid_fields and v is not None
    }
    return ScribeConfig(**filtered_data)


def save_config(config: ScribeConfig) -> None:
    """Save the whole configuration to file, atomically and under the lock.

    Raises :class:`~metasci_scribe.config.store.ConfigParseError` when the
    existing file does not parse: ``config`` then most likely holds the
    defaults :func:`load_config` fell back to, and writing it would replace
    the user's configuration. Raises
    :class:`~metasci_scribe.config.store.ConfigLockedError` while another
    process (``metasci-scribe onboard``) holds the lock.

    Args:
        config: ScribeConfig instance to save.
    """
    # Save all fields including empty API keys (users can set them via env vars instead)
    data = store.dump_config_mapping(_config_to_dict(config))
    with store.config_lock():
        store.read_config_mapping()
        store.write_config_bytes(data)


def reset_config() -> None:
    """Reset configuration to defaults by deleting the config file."""
    with store.config_lock():
        store.remove_config_file()


def _config_to_dict(config: ScribeConfig) -> dict[str, Any]:
    """Return a plain serializable config dict."""
    return {key: _plain_config_value(value) for key, value in asdict(config).items()}


# Secret fields, listed by name rather than matched on it.
API_KEY_FIELDS: frozenset[str] = frozenset({"anthropic_api_key", "openai_api_key"})
SECRET_FIELDS: frozenset[str] = API_KEY_FIELDS


def secret_hint(key: str, value: str) -> str | None:
    """The last four characters of an API key longer than eight, else ``None``.

    Passwords and tokens get no hint: it would show a large part of a
    short password.
    """
    if key in API_KEY_FIELDS and len(value) > 8:
        return "…" + value[-4:]
    return None


def display_config_value(key: str, value: Any) -> str:
    """A value for printing: secrets masked, an empty value as ``(not set)``."""
    if value == "" or value is None:
        return "(not set)"
    if key in SECRET_FIELDS:
        hint = secret_hint(key, str(value))
        return "***" + hint[1:] if hint else "***"
    return str(value)


# =============================================================================
# Config value operations
# =============================================================================

_LITERAL_ALIASES = {"rich": "cli"}
"""Legacy ``ui_backend`` spelling written by earlier onboarding versions.

Other retired values ("tui", "webui") are not aliases: a config file that holds
one falls back to the field's default, with a warning.
"""


def _match_literal(value: Any, allowed: tuple[Any, ...]) -> str | None:
    """Case-insensitive Literal match that also accepts legacy aliases."""
    text = str(value).strip().lower()
    text = _LITERAL_ALIASES.get(text, text)
    return text if text in allowed else None


def _coerce_value(value: Any, field_type: Any) -> Any:
    """Coerce a value to the expected field type.

    Args:
        value: The value to coerce.
        field_type: The target type (from dataclass field).

    Returns:
        The coerced value.

    Raises:
        ValueError: If the value cannot be coerced.
        TypeError: If the value cannot be coerced.
    """
    if _is_str_enum_type(field_type):
        return field_type(str(value).strip().lower())
    if get_origin(field_type) is Literal:
        allowed = get_args(field_type)
        text = _match_literal(value, allowed)
        if text is None:
            raise ValueError(f"expected one of {allowed}, got {value!r}")
        return text
    if field_type == "bool" or field_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    if field_type == "int" or field_type is int:
        return int(value)
    if field_type == "float" or field_type is float:
        return float(value)
    return str(value)


def _plain_config_value(value: Any) -> Any:
    """Return the persisted/user-facing representation for a config value."""
    return value.value if isinstance(value, StrEnum) else value


@lru_cache(maxsize=1)
def _config_field_types() -> dict[str, Any]:
    """Return resolved dataclass annotations for config fields."""
    return get_type_hints(ScribeConfig)


def _config_field_type(key: str, fallback: Any) -> Any:
    """Return the resolved dataclass annotation for a config field."""
    return _config_field_types().get(key, fallback)


def _is_str_enum_type(field_type: Any) -> bool:
    return isinstance(field_type, type) and issubclass(field_type, StrEnum)


def _normalize_str_enum_fields(config: ScribeConfig) -> None:
    """Normalize all StrEnum config fields, falling back to field defaults."""
    for field in fields(config):
        field_type = _config_field_type(field.name, field.type)
        if not _is_str_enum_type(field_type):
            continue

        raw_value = getattr(config, field.name)
        try:
            value = _coerce_value(raw_value, field_type)
        except (ValueError, TypeError):
            default = field.default
            logging.getLogger(__name__).warning(
                "Invalid %s %r; falling back to %s.",
                field.name,
                raw_value,
                _plain_config_value(default),
            )
            value = default
        setattr(config, field.name, value)


def _normalize_literal_fields(config: ScribeConfig) -> None:
    """Validate Literal fields, falling back to defaults on malformed values.

    ``load_config`` does not coerce file values, so a typo'd Literal in the
    config file (or in direct construction) would otherwise flow verbatim
    into mode selection — e.g. ``gateway_backend: langgraph-server`` (hyphen)
    silently selecting the local path in the manager.
    """
    for field in fields(config):
        field_type = _config_field_type(field.name, field.type)
        if get_origin(field_type) is not Literal:
            continue

        raw_value = getattr(config, field.name)
        value = _match_literal(raw_value, get_args(field_type))
        if value is None:
            value = field.default
            logging.getLogger(__name__).warning(
                "Invalid %s %r; falling back to %s.",
                field.name,
                raw_value,
                _plain_config_value(value),
            )
        setattr(config, field.name, value)


_GATEWAY_SURFACE_FIELDS: dict[GatewaySurface, str] = {
    GatewaySurface.SINGLE_SHOT: "gateway_backend_single_shot",
    GatewaySurface.INTERACTIVE: "gateway_backend_interactive",
    GatewaySurface.STANDALONE: "gateway_backend_standalone",
}


def resolve_gateway_backend(
    config: Any,
    surface: GatewaySurface,
) -> Literal["local", "langgraph_server"]:
    """Resolve the effective gateway backend for one surface.

    Returns the surface's ``gateway_backend_*`` override when it is not
    ``"inherit"``, else the global ``gateway_backend``. This is the single
    place routing decisions read the flag from; callers thread the returned
    value down (to the runtime factory and the dev-server spawn), so the
    factory and manager never re-read the global and get a different answer.
    """
    override = getattr(config, _GATEWAY_SURFACE_FIELDS[surface], "inherit")
    if override in ("local", "langgraph_server"):
        return override
    return getattr(config, "gateway_backend", "local")


def get_config_value(key: str) -> Any:
    """Get a single configuration value.

    Args:
        key: Configuration key name.

    Returns:
        The value, or None if key doesn't exist.
    """
    config = load_config()
    value = getattr(config, key, None)
    return _plain_config_value(value)


class SettingsError(ValueError):
    """One or more settings failed validation; ``errors`` maps field to reason."""

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        super().__init__(
            "; ".join(f"{key}: {reason}" for key, reason in errors.items())
        )


def validate_setting(key: str, value: Any) -> tuple[Any, str | None]:
    """Validate and coerce one setting; return ``(value, warning)``.

    The one rule set for ``metasci-scribe config set`` and the settings API. Raises ``ValueError`` with the reason when the value is not
    accepted. The returned value is the plain form written to the file.
    """
    field_info = next((f for f in fields(ScribeConfig) if f.name == key), None)
    if field_info is None:
        raise ValueError("unknown setting")
    field_type = _config_field_type(key, field_info.type)

    # __post_init__ only clamps on load, so validate here too. Reject bool before coercion
    # (_coerce_value(True, int) would turn it into 1 and slip past).
    if key == "sandbox_execute_timeout" and isinstance(value, bool):
        raise ValueError("expected a number")

    try:
        value = _coerce_value(value, field_type)
    except (ValueError, TypeError) as e:
        if get_origin(field_type) is Literal or _is_str_enum_type(field_type):
            raise ValueError(str(e)) from None
        raise ValueError(
            f"expected {getattr(field_type, '__name__', field_type)}"
        ) from None
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("must be a finite number")

    warning = None
    if key == "sandbox_execute_timeout" and value <= 0:
        raise ValueError("must be greater than 0")
    if key == "memory_skill_synthesis_time":
        value = _normalize_hhmm(value)
        if value is None:
            raise ValueError("expected a time as HH:MM")
    if key in ("provider", "auxiliary_provider") and (value or key == "provider"):
        from .provider_keys import PROVIDERS

        if value not in PROVIDERS:
            raise ValueError("unknown provider")
    if key == "default_workdir" and value:
        path = Path(os.path.expanduser(value))
        if not path.is_absolute():
            raise ValueError("must be an absolute path")
        if path.exists() and not path.is_dir():
            raise ValueError("is a file, not a folder")
        if not path.exists():
            warning = "the folder does not exist yet; it is created at start"
    return _plain_config_value(value), warning


def set_config_values(
    changes: dict[str, Any], *, expected_revision: Any = store.UNSET
) -> tuple[str | None, dict[str, str]]:
    """Validate ``changes`` and write them to the file in one locked step.

    A ``None`` value removes the setting from the file, so its default
    applies. Nothing is written when any value fails: :class:`SettingsError`
    names every failing field. Returns ``(new_revision, warnings)``.

    Raises :class:`~metasci_scribe.config.store.ConfigError` subclasses when
    the file does not parse, is locked, or changed since
    ``expected_revision`` (``None`` = "no file expected"; omitted = no check).
    """
    coerced: dict[str, Any] = {}
    warnings: dict[str, str] = {}
    errors: dict[str, str] = {}
    for key, value in changes.items():
        if value is None:
            if key in {f.name for f in fields(ScribeConfig)}:
                coerced[key] = None
            else:
                errors[key] = "unknown setting"
            continue
        try:
            coerced[key], warning = validate_setting(key, value)
        except ValueError as e:
            errors[key] = str(e)
            continue
        if warning:
            warnings[key] = warning
    if errors:
        raise SettingsError(errors)
    revision = store.update_config_mapping(coerced, expected_revision=expected_revision)
    return revision, warnings


def set_config_value(key: str, value: Any) -> bool:
    """Set a single configuration value.

    Args:
        key: Configuration key name.
        value: New value.

    Returns:
        True if successful, False if the key or value is invalid.

    Raises:
        ConfigError: the file does not parse or another process holds the
            lock (see :mod:`metasci_scribe.config.store`).
    """
    try:
        set_config_values({key: value})
    except SettingsError:
        return False
    return True


def list_config() -> dict[str, Any]:
    """List all configuration values.

    Returns:
        Dictionary of all configuration key-value pairs.
    """
    return _config_to_dict(load_config())


# =============================================================================
# Effective configuration (merging sources)
# =============================================================================

# Environment variable mappings
_ENV_MAPPINGS = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "anthropic_base_url": "ANTHROPIC_BASE_URL",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_auth_mode": "METASCI_SCRIBE_OPENAI_AUTH_MODE",
    "default_mode": "METASCI_SCRIBE_DEFAULT_MODE",
    "default_workdir": "METASCI_SCRIBE_WORKSPACE_DIR",
    "ui_backend": "METASCI_SCRIBE_UI_BACKEND",
    "log_level": "METASCI_SCRIBE_LOG_LEVEL",
    "gateway_backend": "METASCI_SCRIBE_GATEWAY_BACKEND",
    "gateway_backend_single_shot": "METASCI_SCRIBE_GATEWAY_BACKEND_SINGLE_SHOT",
    "gateway_backend_interactive": "METASCI_SCRIBE_GATEWAY_BACKEND_INTERACTIVE",
    "gateway_backend_standalone": "METASCI_SCRIBE_GATEWAY_BACKEND_STANDALONE",
    "model_retry_wait_seconds": "METASCI_SCRIBE_MODEL_RETRY_WAIT_SECONDS",
    "auto_mode_retry_wait_seconds": "METASCI_SCRIBE_AUTO_MODE_RETRY_WAIT_SECONDS",
    "auxiliary_provider": "METASCI_SCRIBE_AUXILIARY_PROVIDER",
    "auxiliary_model": "METASCI_SCRIBE_AUXILIARY_MODEL",
    "reasoning_effort": "METASCI_SCRIBE_REASONING_EFFORT",
    "dangerous_mode": "METASCI_SCRIBE_DANGEROUS_MODE",
    "use_responses_api": "METASCI_SCRIBE_USE_RESPONSES_API",
    "checkpoint_keep_per_thread": "METASCI_SCRIBE_CHECKPOINT_KEEP_PER_THREAD",
    "enable_async_subagents": "METASCI_SCRIBE_ENABLE_ASYNC_SUBAGENTS",
    "langgraph_dev_port": "METASCI_SCRIBE_LANGGRAPH_DEV_PORT",
    "langgraph_dev_host": "METASCI_SCRIBE_LANGGRAPH_DEV_HOST",
    "enable_scheduler": "METASCI_SCRIBE_ENABLE_SCHEDULER",
    "scheduler_default_timezone": "METASCI_SCRIBE_SCHEDULER_DEFAULT_TIMEZONE",
    "code_interpreter_timeout": "METASCI_SCRIBE_CODE_INTERPRETER_TIMEOUT",
    "code_interpreter_max_result_chars": "METASCI_SCRIBE_CODE_INTERPRETER_MAX_RESULT_CHARS",
    "sandbox_execute_timeout": "METASCI_SCRIBE_SANDBOX_EXECUTE_TIMEOUT",
    "langgraph_dev_file_persistence": "METASCI_SCRIBE_LANGGRAPH_DEV_FILE_PERSISTENCE",
    "langgraph_dev_jobs_per_worker": "METASCI_SCRIBE_LANGGRAPH_DEV_JOBS_PER_WORKER",
    "langgraph_dev_keepalive": "METASCI_SCRIBE_LANGGRAPH_DEV_KEEPALIVE",
    "recursion_limit": "METASCI_SCRIBE_RECURSION_LIMIT",
    "memory_profile_enabled": "METASCI_SCRIBE_MEMORY_PROFILE_ENABLED",
    "memory_observations_enabled": "METASCI_SCRIBE_MEMORY_OBSERVATIONS_ENABLED",
    "memory_observation_writer": "METASCI_SCRIBE_MEMORY_OBSERVATION_WRITER",
    "memory_workers_enabled": "METASCI_SCRIBE_MEMORY_WORKERS_ENABLED",
    "memory_skill_synthesis_enabled": "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_ENABLED",
    "memory_skill_synthesis_mode": "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_MODE",
    "memory_skill_synthesis_cadence": "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_CADENCE",
    "memory_skill_synthesis_time": "METASCI_SCRIBE_MEMORY_SKILL_SYNTHESIS_TIME",
    "memory_observation_cache_max_files": "METASCI_SCRIBE_MAX_CACHED_FILES",
}


_ENV_FIELDS: dict[str, str] = {env: field for field, env in _ENV_MAPPINGS.items()}


# =============================================================================
# Where environment values came from
# =============================================================================

# The process environment cannot tell the user's own exports from values our
# code put there: apply_config_to_env() copies saved keys, get_effective_config()
# merges a ``.env`` and flags too. Each such write is
# recorded here, by variable name: ``{"source": "file" | "default" | "cli" |
# "dotenv", "path"?: <.env path>}``. A variable with no record is
# the user's (``env``). The settings API reports it as ``sources``.
#
# The launcher passes its record to the langgraph dev server in
# ENV_ORIGINS_VAR (server environment only); this module reads and removes it
# at import, so nothing the server starts inherits it.
ENV_ORIGINS_VAR = "METASCI_SCRIBE_CONFIG_ORIGINS"
_ENV_ORIGINS: dict[str, dict[str, str]] = {}


def _load_env_origins() -> None:
    raw = os.environ.pop(ENV_ORIGINS_VAR, None)
    if not raw:
        return
    try:
        data = json.loads(raw)
    except ValueError:
        return
    if isinstance(data, dict):
        _ENV_ORIGINS.update(
            {
                str(k): {str(a): str(b) for a, b in v.items()}
                for k, v in data.items()
                if isinstance(v, dict)
            }
        )


_load_env_origins()


def record_env_origin(env_key: str, source: str, path: str | None = None) -> None:
    """Record that our code set ``env_key``, and from where."""
    origin = {"source": source}
    if path:
        origin["path"] = path
    _ENV_ORIGINS[env_key] = origin


def env_origin(env_key: str) -> dict[str, str] | None:
    """Where ``env_key`` came from, or ``None`` when our code did not set it."""
    return _ENV_ORIGINS.get(env_key)


def env_origins() -> dict[str, dict[str, str]]:
    return {key: dict(value) for key, value in _ENV_ORIGINS.items()}


def _same_value(field: str, a: Any, b: Any) -> bool:
    if field == "default_workdir":
        a, b = (
            os.path.realpath(os.path.expanduser(str(v))) if v else "" for v in (a, b)
        )
    return str(_plain_config_value(a)) == str(_plain_config_value(b))


def _saved_mapping() -> dict[str, Any]:
    """The file's mapping, or ``{}`` when it does not parse or cannot be read."""
    try:
        return store.read_config_mapping()
    except (store.ConfigError, OSError):
        return {}


def config_value_origin(
    field: str, value: Any, mapping: dict[str, Any] | None = None
) -> str:
    """``file`` when the file holds ``value`` for ``field``, ``default`` when
    the file lacks the field and ``value`` is its default, else ``cli``: the
    value can then only have come from a command-line flag.

    ``mapping`` is the file's mapping when the caller has already read it.
    """
    if mapping is None:
        mapping = _saved_mapping()
    saved = mapping.get(field)
    if saved is not None and saved != "":
        return "file" if _same_value(field, saved, value) else "cli"
    default = next(f.default for f in fields(ScribeConfig) if f.name == field)
    if field == "default_workdir":
        # An empty default_workdir means "the current folder".
        return "default" if _same_value(field, os.getcwd(), value) else "cli"
    return "default" if _same_value(field, default, value) else "cli"


def get_effective_config(
    cli_overrides: dict[str, Any] | None = None,
) -> ScribeConfig:
    """Get effective configuration by merging all sources.

    Priority (highest to lowest):
        1. CLI arguments (``cli_overrides``)
        2. Parent-process environment variables for any ``METASCI_SCRIBE_*`` key
        3. ``.env`` file at (or above) the current working directory
        4. Parent-process environment variables for everything else
           (third-party API keys / base URLs, plus arbitrary unmapped keys)
        5. Config file (``~/.config/metasci-scribe/config.yaml``)
        6. Dataclass defaults

    Rows 2 and 4 differ because ``.env`` values need different treatment
    for our own namespaced config knobs vs third-party credentials.
    Third-party keys (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ...)
    follow the industry convention that ``.env`` is the per-project
    credential store; extending shell-wins to them would silently flip
    a workspace key back to a global ``.bashrc`` key. Our own
    ``METASCI_SCRIBE_*`` keys are the opposite: an explicit CLI/parent-
    process value (e.g. the bind port the CLI hands to the langgraph dev
    subprocess) must not be shadowed by a
    workspace ``.env``. We implement this by reading ``.env`` into a
    dict via ``dotenv_values`` (no ``os.environ`` mutation), then
    writing third-party keys unconditionally and ``METASCI_SCRIBE_*`` keys
    only when the shell doesn't already have a non-empty value.

    Tradeoff: ``OPENAI_API_KEY=xxx metasci-scribe ...`` inline overrides
    still lose to a workspace ``.env`` containing ``OPENAI_API_KEY``,
    because the merge writes third-party keys from ``.env``
    unconditionally. Users who need to override a ``.env``-defined
    credential inline must edit or unset the ``.env`` entry.

    Args:
        cli_overrides: Dictionary of CLI argument overrides.

    Returns:
        ScribeConfig with merged values.
    """
    # Merge workspace ``.env`` into ``os.environ`` without going through
    # ``load_dotenv``. The previous snapshot → ``load_dotenv`` → restore
    # sequence was a read-modify-write on ``os.environ`` that could race with
    # concurrent ``get_effective_config`` calls in the langgraph dev subprocess
    # (per-request threads in ``langgraph_dev/http.py``, ``sessions.py``
    # checkpoint writes, memory workers): one thread's mid-flight ``.env``
    # value could be re-captured by another as "parent env" and then restored
    # last, promoting the ``.env`` value into the snapshot permanently.
    #
    # ``dotenv_values`` returns a dict without touching ``os.environ``, so the
    # merge below is a pure write sequence and idempotent under interleaving.
    # Third-party keys keep ``.env``-wins (industry convention).
    # ``METASCI_SCRIBE_*`` keys are our own namespaced config knobs where
    # CLI/parent-process intent should stay authoritative — write from ``.env``
    # only when the shell doesn't already have a non-empty value. Treating an
    # empty shell value as "unset" matches the ``if env_value:`` truthy check
    # in the ``_ENV_MAPPINGS`` loop below; without this, an empty parent export
    # would silently regress vs main by falling through to file/defaults.
    dotenv_path = find_dotenv(usecwd=True)
    dotenv_map = dotenv_values(dotenv_path) if dotenv_path else {}
    for env_key, env_value in dotenv_map.items():
        if env_value is None:
            continue  # bare ``FOO`` without ``=`` — nothing to write
        if env_key.startswith("METASCI_SCRIBE_"):
            if not os.environ.get(env_key):
                os.environ[env_key] = env_value
                record_env_origin(env_key, "dotenv", dotenv_path)
        else:
            os.environ[env_key] = env_value
            record_env_origin(env_key, "dotenv", dotenv_path)

    # Start with file config (includes defaults for missing values)
    config = load_config()
    data = _config_to_dict(config)

    # Apply environment variable overrides
    for config_key, env_key in _ENV_MAPPINGS.items():
        env_value = os.environ.get(env_key)
        if env_value:
            field_info = next(f for f in fields(ScribeConfig) if f.name == config_key)
            try:
                data[config_key] = _coerce_value(
                    env_value,
                    _config_field_type(config_key, field_info.type),
                )
            except (ValueError, TypeError):
                pass

    # Apply CLI overrides (highest priority)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and key in data:
                data[key] = value

    return ScribeConfig(**data)


def apply_config_to_env(config: ScribeConfig) -> None:
    """Apply config API keys to environment variables if not already set.

    This allows the config file to provide API keys that downstream
    libraries (like langchain-anthropic) can pick up. Every variable it
    sets is recorded as coming from the file, the defaults or a flag (see
    :func:`record_env_origin`).

    Args:
        config: Configuration to apply.
    """
    before = dict(os.environ)
    _apply_config_to_env(config)
    mapping = None  # read once, when the first variable needs it
    for env_key, field in _ENV_FIELDS.items():
        value = os.environ.get(env_key)
        if value is None:
            _ENV_ORIGINS.pop(env_key, None)
        elif value != before.get(env_key):
            if mapping is None:
                mapping = _saved_mapping()
            record_env_origin(
                env_key, config_value_origin(field, getattr(config, field), mapping)
            )


def _apply_config_to_env(config: ScribeConfig) -> None:
    if config.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key
    if config.anthropic_base_url and not os.environ.get("ANTHROPIC_BASE_URL"):
        os.environ["ANTHROPIC_BASE_URL"] = config.anthropic_base_url
    if config.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    if config.reasoning_effort and not os.environ.get(
        "METASCI_SCRIBE_REASONING_EFFORT"
    ):
        os.environ["METASCI_SCRIBE_REASONING_EFFORT"] = config.reasoning_effort
    # Written for both values: get_chat_model reads it, and the langgraph dev
    # subprocess re-resolves config on its own.
    os.environ["METASCI_SCRIBE_OPENAI_AUTH_MODE"] = config.openai_auth_mode
    # Round-trip dangerous_mode to env so it survives a fresh get_effective_config()
    # (warning banner, run_in_background) and is inherited by the langgraph dev
    # subprocess — otherwise a --dangerous CLI flag (not persisted to file/env)
    # is invisible to those consumers while the backend is already unconfined.
    # Bidirectional: clear it when off so a re-apply with a lower config (or a
    # stale value) can't leave the process stuck in dangerous mode.
    if config.dangerous_mode:
        os.environ["METASCI_SCRIBE_DANGEROUS_MODE"] = "true"
    else:
        os.environ.pop("METASCI_SCRIBE_DANGEROUS_MODE", None)
    if config.use_responses_api and not os.environ.get(
        "METASCI_SCRIBE_USE_RESPONSES_API"
    ):
        os.environ["METASCI_SCRIBE_USE_RESPONSES_API"] = config.use_responses_api
