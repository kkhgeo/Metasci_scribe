"""Shared session status bar helpers for CLI frontends."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

from ..llm.context_window import (
    DEFAULT_CONTEXT_WINDOW_FALLBACK,
    resolve_context_window,
)
from ..memory.worker_activity import (
    MemoryWorkerStatusSnapshot,
    ObservationLinkerStatusSnapshot,
    memory_worker_status,
    observation_linker_status,
)

if TYPE_CHECKING:
    from ..gateway import GraphGateway

_FALLBACK_CONTEXT_WINDOW = DEFAULT_CONTEXT_WINDOW_FALLBACK
STATUS_BAR_BG = "#171a20"
STATUS_TEXT = "#cbd5e1"
STATUS_STRONG = "#e5e7eb"
STATUS_DIM = "#7c8594"
STATUS_GOOD = "#5fcf8b"
STATUS_WARN = "#d7b45a"
STATUS_BAD = "#d08c61"
STATUS_CRITICAL = "#d86f6f"
STATUS_HINT_IDLE = "#8b9bb0"
STATUS_HINT_BUSY = "#f0c36a"
STATUS_HINT_WRITING = "#7eb8e0"

# Braille spinner frames used by the CLI bottom toolbar
# to animate the "Loading MCP tools" indicator.
SPINNER_FRAMES = "\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"


@dataclass(slots=True)
class SessionStatusSnapshot:
    """Current session metrics shown in the persistent status bar."""

    model_full: str
    model_short: str
    context_tokens: int
    context_window: int
    context_percent: int
    context_source: str = "estimated"


def _percent_from_context(context_tokens: int, context_window: int) -> int:
    """Convert token counts into a clamped percent value."""
    if context_window <= 0:
        return 0
    return max(0, min(100, round((context_tokens / context_window) * 100)))


def _get_default_chat_model() -> Any:
    """Resolve the default chat model lazily to avoid import cycles."""
    from ..agent import _ensure_chat_model

    return _ensure_chat_model()


def _resolve_model_name(model_name: str | None, model_obj: Any | None) -> str:
    """Best-effort model name resolution for display."""
    if model_name:
        return str(model_name)
    if model_obj is None:
        model_obj = _get_default_chat_model()
    for attr in ("model_name", "model", "name"):
        value = getattr(model_obj, attr, None)
        if value:
            return str(value)
    return "unknown"


def _resolve_context_window(model_obj: Any | None) -> int:
    """Resolve the model context window with a safe fallback."""
    if model_obj is None:
        model_obj = _get_default_chat_model()
    return resolve_context_window(model_obj, fallback=_FALLBACK_CONTEXT_WINDOW)


def shorten_model_name(model_name: str, max_len: int = 26) -> str:
    """Shorten provider-prefixed model names for compact display."""
    short = (model_name or "unknown").split("/")[-1]
    if len(short) > max_len:
        return f"{short[: max_len - 3]}..."
    return short


def format_token_count_compact(value: int) -> str:
    """Format large token counts into a compact human-readable form."""
    abs_value = abs(int(value))
    if abs_value >= 1_000_000:
        num = float(value) / 1_000_000
        suffix = "M"
    elif abs_value >= 1_000:
        num = float(value) / 1_000
        suffix = "K"
    else:
        return str(value)

    if num == int(num):
        return f"{int(num)}{suffix}"
    return f"{num:.1f}{suffix}"


def format_duration_compact(started_at: datetime, now: datetime | None = None) -> str:
    """Format elapsed wall time into a compact duration string."""
    current = now or datetime.now()
    seconds = max(0, int((current - started_at).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h"
    days = hours // 24
    return f"{days}d"


def status_style_name(percent_used: int) -> str:
    """Map utilization percent to shared status bar style buckets."""
    if percent_used >= 95:
        return "critical"
    if percent_used > 80:
        return "bad"
    if percent_used >= 50:
        return "warn"
    return "good"


def build_context_bar(percent_left: int, width: int = 10) -> str:
    """Build a compact visual bar; the filled part is the context left."""
    safe_percent = max(0, min(100, int(percent_left)))
    filled = round((safe_percent / 100) * width)
    body = ("█" * filled) + ("░" * max(0, width - filled))
    return f"[{body}]"


@dataclass(frozen=True, slots=True)
class RunProgress:
    """Paragraph counts of the autonomous run, from ``run.json``."""

    done: int
    total: int
    blocked: int = 0


# run.json path -> ((mtime_ns, size), progress read at that version)
_run_progress_cache: dict[Path, tuple[tuple[int, int], RunProgress | None]] = {}


def read_run_progress(root: str | Path) -> RunProgress | None:
    """Paragraph progress from ``<root>/.metasci-scribe/run.json``, or None.

    The file is read again only when its modification time or size changes,
    so the status bar can ask on every refresh. A missing, unreadable or
    invalid file gives None, and the status bar leaves the field out.
    """
    from ..paths import Workspace

    root = Path(root)
    path = Workspace(root).state_dir / "run.json"
    try:
        stat = path.stat()
    except OSError:
        _run_progress_cache.pop(path, None)
        return None
    version = (stat.st_mtime_ns, stat.st_size)
    cached = _run_progress_cache.get(path)
    if cached is not None and cached[0] == version:
        return cached[1]

    progress: RunProgress | None = None
    try:
        from ..run import RunState

        run_state = RunState.load(root)
        if run_state is not None and run_state.paragraphs:
            progress = RunProgress(
                done=run_state.count("done"),
                total=len(run_state.paragraphs),
                blocked=run_state.count("blocked"),
            )
    except Exception:
        progress = None
    _run_progress_cache[path] = (version, progress)
    return progress


def _display_width(text: str) -> int:
    try:
        from prompt_toolkit.utils import get_cwidth

        return get_cwidth(text or "")
    except Exception:
        return len(text or "")


def trim_status_text(text: str, max_width: int) -> str:
    """Trim status-bar content to fit a single terminal row."""
    if max_width <= 0:
        return ""
    if _display_width(text) <= max_width:
        return text

    ellipsis = "..."
    ellipsis_width = _display_width(ellipsis)
    if max_width <= ellipsis_width:
        return ellipsis[:max_width]

    out: list[str] = []
    width = 0
    for ch in text:
        ch_width = _display_width(ch)
        if width + ch_width + ellipsis_width > max_width:
            break
        out.append(ch)
        width += ch_width
    return "".join(out).rstrip() + ellipsis


def get_memory_worker_status() -> MemoryWorkerStatusSnapshot | None:
    """Read completed EvoMemory save counts without making rendering fail."""
    try:
        return memory_worker_status()
    except Exception:
        return None


def get_observation_linker_status() -> ObservationLinkerStatusSnapshot | None:
    """Read active observation-linker status without making rendering fail."""
    try:
        return observation_linker_status()
    except Exception:
        return None


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def _memory_activity_label(
    *,
    worker_status: MemoryWorkerStatusSnapshot | None,
    linker_status: ObservationLinkerStatusSnapshot | None,
) -> str:
    parts: list[str] = []
    if worker_status is not None and worker_status.is_running:
        parts.append("🧠")
    if linker_status is not None and linker_status.is_running:
        parts.append("🔗")

    saved: list[str] = []
    if worker_status is not None:
        if worker_status.profile_updates:
            saved.append(_plural(worker_status.profile_updates, "profile edit"))
        if worker_status.observations_recorded:
            saved.append(_plural(worker_status.observations_recorded, "observation"))
    if saved:
        parts.append(f"Saved {', '.join(saved)}")

    if linker_status is not None and linker_status.relations_linked:
        parts.append(
            f"Created {_plural(linker_status.relations_linked, 'memory link')}"
        )

    return " ".join(parts)


PLAN_LABEL = "Using ChatGPT plan"

_Fragments = list[tuple[str, str]]


@dataclass(frozen=True, slots=True)
class _Field:
    """One status-bar field. A field with ``priority`` None is never left out;
    one that ``joins_next`` is followed by a space instead of a separator."""

    priority: int | None
    fragments: _Fragments
    joins_next: bool = False


def _fragments_width(frags: _Fragments) -> int:
    return sum(_display_width(text) for _, text in frags)


def _join_fields(fields: list[_Field], separator: str) -> _Fragments:
    frags: _Fragments = [("class:status-bar", " ")]
    for i, item in enumerate(fields):
        if i > 0:
            gap = " " if fields[i - 1].joins_next else separator
            frags.append(("class:status-bar-dim", gap))
        frags.extend(item.fragments)
    frags.append(("class:status-bar", " "))
    return frags


def _fit_fields(fields: list[_Field], width: int, separator: str) -> _Fragments:
    """Join *fields* into one row of *width* columns.

    While the row is too wide, the field with the lowest priority is left out.
    The last optional field and the fields without a priority stay, trimmed if
    they must.
    """
    fields = list(fields)
    while _fragments_width(_join_fields(fields, separator)) > width:
        optional = [item for item in fields if item.priority is not None]
        if len(optional) <= 1:
            break
        fields.remove(min(optional, key=lambda item: item.priority or 0))

    frags = _join_fields(fields, separator)
    if _fragments_width(frags) <= width:
        return frags
    # Still too wide: trim the optional text and keep the required fields whole.
    required = [item for item in fields if item.priority is None]
    rest = "".join(
        text
        for item in fields
        if item.priority is not None
        for _, text in item.fragments
    )
    kept = _join_fields(required, separator)[:-1] if required else []
    room = width - _fragments_width(kept) - (_display_width(separator) if kept else 0)
    if not kept:
        return [("class:status-bar", trim_status_text(f" {rest}", width))]
    if not rest or room < 4:
        plain = "".join(text for _, text in kept)
        return [("class:status-bar-strong", trim_status_text(plain, width))]
    return [
        *kept,
        ("class:status-bar-dim", separator),
        ("class:status-bar", trim_status_text(rest, room)),
    ]


def build_status_fragments(
    snapshot: SessionStatusSnapshot,
    started_at: datetime,
    width: int,
    *,
    rubric_badge: str | None = None,
    folder: str | None = None,
    on_plan: bool = False,
    run_progress: RunProgress | None = None,
) -> list[tuple[str, str]]:
    """Build prompt_toolkit formatted-text fragments for the status bar.

    The fields: the paper *folder* name, the model, ``Using ChatGPT plan``
    while *on_plan*, the paragraph progress of the autonomous run, the context
    left, the tokens in use, the session time, ``rubric_badge`` (``on`` /
    ``next`` / ``on+next``, or ``goal: <status>``) and memory activity. When
    the row is too narrow, the least important fields go first. The plan label
    is never left out: it says how requests are being paid for.
    """
    percent_used = snapshot.context_percent
    percent_left = 100 - max(0, min(100, percent_used))
    bucket = f"class:status-bar-{status_style_name(percent_used)}"
    left_label = f"{percent_left}% left"

    tokens_label = (
        f"{format_token_count_compact(snapshot.context_tokens)}/"
        f"{format_token_count_compact(snapshot.context_window)}"
    )
    duration_label = format_duration_compact(started_at, now=datetime.now())

    # Priority: higher stays longer. None is never left out.
    fields: list[_Field] = []
    if folder:
        fields.append(_Field(4, [("class:status-bar", folder)]))
    fields.append(_Field(7, [("class:status-bar-strong", snapshot.model_short)]))
    if on_plan:
        fields.append(_Field(None, [("class:status-bar-good", PLAN_LABEL)]))
    if run_progress is not None:
        progress: _Fragments = [
            ("class:status-bar", f"{run_progress.done}/{run_progress.total} done")
        ]
        if run_progress.blocked:
            progress.append(
                ("class:status-bar-warn", f", {run_progress.blocked} blocked")
            )
        fields.append(_Field(5, progress))
    fields.append(
        _Field(3, [(bucket, build_context_bar(percent_left))], joins_next=True)
    )
    fields.append(_Field(6, [(bucket, left_label)]))
    fields.append(_Field(1, [("class:status-bar-dim", tokens_label)]))
    fields.append(_Field(2, [("class:status-bar-dim", duration_label)]))
    if rubric_badge and width >= 76:
        badge = (
            rubric_badge
            if rubric_badge.startswith("goal: ")
            else f"rubric: {rubric_badge}"
        )
        fields.append(_Field(3, [("class:status-bar-strong", badge)]))
    memory_label = _memory_activity_label(
        worker_status=get_memory_worker_status(),
        linker_status=get_observation_linker_status(),
    )
    if memory_label:
        fields.append(_Field(3, [("class:status-bar-warn", memory_label)]))

    separator = " │ " if width >= 76 else " · "
    return _fit_fields(fields, width, separator)


def build_status_text(
    snapshot: SessionStatusSnapshot,
    started_at: datetime,
    width: int,
    *,
    rubric_badge: str | None = None,
    folder: str | None = None,
    on_plan: bool = False,
    run_progress: RunProgress | None = None,
):
    """Build a Rich Text object for the persistent status bar."""
    from rich.text import Text

    rich_styles = {
        "status-bar": f"on {STATUS_BAR_BG} {STATUS_TEXT}",
        "status-bar-strong": f"on {STATUS_BAR_BG} {STATUS_STRONG} bold",
        "status-bar-dim": f"on {STATUS_BAR_BG} {STATUS_DIM}",
        "status-bar-good": f"on {STATUS_BAR_BG} {STATUS_GOOD} bold",
        "status-bar-warn": f"on {STATUS_BAR_BG} {STATUS_WARN} bold",
        "status-bar-bad": f"on {STATUS_BAR_BG} {STATUS_BAD} bold",
        "status-bar-critical": f"on {STATUS_BAR_BG} {STATUS_CRITICAL} bold",
    }
    text = Text(no_wrap=True, overflow="crop")
    for style, content in build_status_fragments(
        snapshot,
        started_at,
        width,
        rubric_badge=rubric_badge,
        folder=folder,
        on_plan=on_plan,
        run_progress=run_progress,
    ):
        rich_style = rich_styles.get(
            style.removeprefix("class:"),
            f"on {STATUS_BAR_BG} {STATUS_TEXT}",
        )
        text.append(content, style=rich_style)
    return text


def make_empty_status_snapshot(
    model_name: str | None = None, model_obj: Any | None = None
) -> SessionStatusSnapshot:
    """Build a placeholder snapshot before async context counting completes."""
    resolved_name = _resolve_model_name(model_name, model_obj)
    window = _resolve_context_window(model_obj)
    return SessionStatusSnapshot(
        model_full=resolved_name,
        model_short=shorten_model_name(resolved_name),
        context_tokens=0,
        context_window=window,
        context_percent=0,
        context_source="estimated",
    )


def make_usage_status_snapshot(
    input_tokens: int,
    *,
    model_name: str | None = None,
    model_obj: Any | None = None,
) -> SessionStatusSnapshot:
    """Build a snapshot from the last real model input usage."""
    resolved_name = _resolve_model_name(model_name, model_obj)
    window = _resolve_context_window(model_obj)
    context_tokens = max(0, int(input_tokens))
    return SessionStatusSnapshot(
        model_full=resolved_name,
        model_short=shorten_model_name(resolved_name),
        context_tokens=context_tokens,
        context_window=window,
        context_percent=_percent_from_context(context_tokens, window),
        context_source="usage",
    )


def estimate_message_tokens(
    text: str,
    *,
    message_type: str = "ai",
) -> int:
    """Estimate tokens for a single in-flight message fragment."""
    content = (text or "").strip()
    if not content:
        return 0

    try:
        if message_type == "human":
            messages = [HumanMessage(content=content)]
        else:
            messages = [AIMessage(content=content)]
        return int(count_tokens_approximately(messages))
    except Exception:
        return 0


def apply_assistant_text_to_snapshot(
    snapshot: SessionStatusSnapshot,
    assistant_text: str | None,
) -> SessionStatusSnapshot:
    """Overlay in-flight assistant output on top of a base snapshot."""
    extra_tokens = estimate_message_tokens(assistant_text or "", message_type="ai")
    if extra_tokens <= 0:
        return snapshot

    context_tokens = snapshot.context_tokens + extra_tokens
    return replace(
        snapshot,
        context_tokens=context_tokens,
        context_percent=_percent_from_context(context_tokens, snapshot.context_window),
    )


def apply_user_text_to_snapshot(
    snapshot: SessionStatusSnapshot,
    user_text: str | None,
) -> SessionStatusSnapshot:
    """Overlay pending user input on top of an existing snapshot."""
    extra_tokens = estimate_message_tokens(user_text or "", message_type="human")
    if extra_tokens <= 0:
        return snapshot

    context_tokens = snapshot.context_tokens + extra_tokens
    return replace(
        snapshot,
        context_tokens=context_tokens,
        context_percent=_percent_from_context(context_tokens, snapshot.context_window),
    )


async def build_session_status_snapshot(
    thread_id: str,
    *,
    model_name: str | None = None,
    model_obj: Any | None = None,
    pending_user_text: str | None = None,
    graph_gateway: GraphGateway,
) -> SessionStatusSnapshot:
    """Count current thread context and return a display snapshot."""
    resolved_name = _resolve_model_name(model_name, model_obj)
    window = _resolve_context_window(model_obj)
    messages = list(await graph_gateway.get_thread_messages(thread_id))

    pending = (pending_user_text or "").strip()
    if pending:
        messages.append(HumanMessage(content=pending))

    try:
        context_tokens = int(count_tokens_approximately(messages)) if messages else 0
    except Exception:
        context_tokens = 0

    percent = _percent_from_context(context_tokens, window)

    return SessionStatusSnapshot(
        model_full=resolved_name,
        model_short=shorten_model_name(resolved_name),
        context_tokens=context_tokens,
        context_window=window,
        context_percent=percent,
        context_source="estimated",
    )
