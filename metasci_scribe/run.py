"""Autonomous mode: read the outline, keep the run state, drive the agent.

The harness owns the state. The model never writes ``.metasci-scribe/run.json``;
it reports progress through the ``mark_paragraph`` tool, which checks the
paragraph ID against the state the harness wrote, and the harness decides what
the run is worth (design section 6):

* complete means every paragraph of ``outline.md`` is ``done`` and the
  section files under ``Manuscript/`` hold text, not that the model said it
  was finished;
* a ``blocked`` paragraph stops the run for a human decision;
* running again continues in the same thread while ``outline.md`` is unchanged
  (comments and blank lines do not count), and starts a new thread, keeping
  the ``done`` paragraphs, when it changed;
* a run can be limited to one section of the outline; it then writes only
  that section's paragraphs, in a thread of its own.

The model writes each paragraph into its section's file,
``Manuscript/<Section>.md``. At the end of every run the harness joins those
files into ``manuscript.md`` (:func:`metasci_scribe.manuscript.join_manuscript`).

``run_project`` is the whole body of a run. The real agent is passed in as
``invoke`` (task 14 wires it); everything here works with a fake.

Everything that can be wrong with the folder or the outline is found before
the first model call and before ``Knowledge/`` is read.

Interrupts: ``KeyboardInterrupt`` is re-raised. The harness saves the state
once, before the first model call, and the tool replaces ``run.json`` whole, so
the file on disk is always a complete state, and an interrupt saves nothing
more. The next run resumes from it. The caller turns the exception into its
own exit.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal

from filelock import FileLock, Timeout
from langchain_core.tools import BaseTool, tool

from . import paths
from .auth.chatgpt_plan import ChatGPTAuthError
from .knowledge import LOCK_WAIT_SECONDS, report_messages, sync_knowledge
from .llm.credentials import ModelNotConfigured
from .llm.errors import PlanAction, PlanStop, _redact_api_keys
from .manuscript import (
    MANUSCRIPT_DIR,
    SECTION_HEADING,
    copy_section_files,
    join_manuscript,
    read_section_texts,
    same_section,
    section_file_name,
    section_names,
)
from .paths import Workspace
from .project import NetworkPathError, require_local_path

__all__ = [
    "MAX_PASSES",
    "AgentInvoke",
    "OutlineError",
    "Paragraph",
    "RunExit",
    "RunResult",
    "RunState",
    "RunStateError",
    "make_mark_paragraph_tool",
    "open_markers",
    "outline_digest",
    "parse_paragraphs",
    "run_project",
]

MAX_PASSES = 3  # model calls per run, the first included
PENDING, DONE, BLOCKED = "pending", "done", "blocked"
_STATUSES = (PENDING, DONE, BLOCKED)
_MAX_REASON = 500
_MAX_ERROR_TEXT = 300
_SHOWN_IDS = 12

_HEADING = re.compile(
    r"^###[ \t]+(?P<id>[A-Z]+\d+)[ \t]*·[ \t]*"
    r"(?:(?P<num>[①-⑳㉑-㊿]|\d+)(?![^\s]))?[ \t]*(?P<title>.*)$"
)
_ID_LIKE = re.compile(r"^###[ \t]+[A-Za-z]+\d+(?![A-Za-z0-9])")
_SECTION = SECTION_HEADING
_EVIDENCE = re.compile(r"\[evidence needed\]", re.IGNORECASE)
_MARKER = "◇"


# ---------------------------------------------------------------- the outline


class OutlineError(ValueError):
    """``outline.md`` cannot be read as a map. ``line`` is 1-based, or None."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(f"line {line}: {message}" if line else message)
        self.line = line
        self.detail = message


@dataclass(frozen=True)
class Paragraph:
    id: str
    section: str
    title: str


def _normalize_newlines(text: str) -> str:
    return text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")


def _mask_comments(text: str) -> str:
    """*text* with every ``<!-- -->`` comment blanked, newlines kept.

    Line numbers therefore stay those of the file. An unclosed comment would
    hide the rest of the outline silently, so it is an error.
    """
    out: list[str] = []
    pos = 0
    while True:
        start = text.find("<!--", pos)
        if start < 0:
            out.append(text[pos:])
            return "".join(out)
        end = text.find("-->", start + 4)
        if end < 0:
            raise OutlineError(
                "a comment opened with <!-- is never closed with -->",
                text.count("\n", 0, start) + 1,
            )
        out.append(text[pos:start])
        out.append(re.sub(r"[^\n]", " ", text[start : end + 3]))
        pos = end + 3


def parse_paragraphs(text: str) -> list[Paragraph]:
    """The paragraphs of an outline, in file order.

    A paragraph heading reads ``### R1 · ①``: the ID (capital letters and a
    number) is the key. The circled number is the reading order within a
    section and repeats from section to section, so it is not used. Comments
    are skipped. Other ``###`` headings are ignored. Raises ``OutlineError``,
    with the line, for a heading that starts like an ID but is not written
    that way, a repeated ID, or an unclosed comment.
    """
    lines = _mask_comments(_normalize_newlines(text)).split("\n")
    paragraphs: list[Paragraph] = []
    first_line: dict[str, int] = {}
    section = ""
    for number, line in enumerate(lines, start=1):
        if m := _SECTION.match(line):
            section = m.group("name")
            continue
        if not _ID_LIKE.match(line):
            continue
        m = _HEADING.match(line)
        if m is None:
            raise OutlineError(
                f"paragraph heading {line.strip()!r} must look like '### R1 · ①' "
                "(capital letters and a number, a dot, then the reading number)",
                number,
            )
        pid = m.group("id")
        if pid in first_line:
            raise OutlineError(
                f"paragraph ID {pid} is used twice (first on line {first_line[pid]})",
                number,
            )
        first_line[pid] = number
        title = m.group("title").replace(_MARKER, " ")
        paragraphs.append(Paragraph(pid, section, " ".join(title.split())))
    return paragraphs


def open_markers(text: str, section: str | None = None) -> list[int]:
    """1-based numbers of the lines that hold an open-question mark (◇).

    Marks inside comments do not count: the template explains the mark there.
    With *section*, only the lines before the first section (they concern the
    whole paper) and that section's own lines count.
    """
    lines = _mask_comments(_normalize_newlines(text)).split("\n")
    marks: list[int] = []
    current: str | None = None
    for n, line in enumerate(lines, start=1):
        if m := _SECTION.match(line):
            current = m.group("name")
        if _MARKER not in line:
            continue
        if section is None or current is None or same_section(current, section):
            marks.append(n)
    return marks


_SHOWN_MARKS = 8


def open_markers_message(text: str, marks: list[int]) -> str:
    """What stops a run: each line with a ``◇`` (see ``open_markers``), where
    it is and what it says, so the author can name it in chat."""
    lines = _normalize_newlines(text).split("\n")
    where: dict[int, str] = {}
    current = ""
    for n, line in enumerate(lines, start=1):
        if m := _SECTION.match(line):
            current = m.group("name")
        where[n] = current
    shown = []
    for n in marks[:_SHOWN_MARKS]:
        place = f"section {_short(where[n])[:40]}" if where[n] else "whole paper"
        shown.append(f"  line {n} ({place}): {_short(lines[n - 1].strip())[:100]}")
    if len(marks) > _SHOWN_MARKS:
        shown.append(f"  ... and {len(marks) - _SHOWN_MARKS} more")
    first = marks[0]
    return (
        f"outline.md still has {len(marks)} open choice(s) (◇) that stop this run:\n"
        + "\n".join(shown)
        + "\nSettle them in chat, naming the line (for example: "
        f'"settle the ◇ on line {first} of outline.md"), then run again.'
    )


def outline_digest(text: str) -> str:
    """A hash of what the outline says, without comments, blank lines or
    trailing spaces. A change that is only a comment keeps the digest, so it
    does not start a new thread."""
    lines = _mask_comments(_normalize_newlines(text)).split("\n")
    body = "\n".join(s for s in (line.rstrip() for line in lines) if s.strip())
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


# ------------------------------------------------------------------ the state


class RunStateError(ValueError):
    """``run.json`` exists but cannot be used. It is never replaced silently."""


def _state_path(root: Path) -> Path:
    return Workspace(root).state_dir / "run.json"


def _replace(tmp: str, target: Path) -> None:
    """``os.replace``, retried a few times: on Windows a virus scanner or a
    reader holding the target open makes it fail for a moment."""
    for attempt in range(6):
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


@dataclass
class RunState:
    """Where a run stands. ``paragraphs`` maps ID to pending, done or blocked, in
    outline order; ``reasons`` holds why a paragraph is blocked."""

    thread_id: str
    outline_digest: str
    paragraphs: dict[str, str]
    reasons: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, root: Path) -> RunState | None:
        """The saved state, or None when there is none.

        Raises ``RunStateError`` for a file that cannot be read or is not a
        state, so a damaged file is not mistaken for a fresh start.
        """
        path = _state_path(root)
        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except (OSError, UnicodeDecodeError) as exc:
            raise RunStateError(f"{path} cannot be read: {exc}") from exc
        try:
            data = json.loads(raw)
            thread_id = data["thread_id"]
            digest = data["outline_digest"]
            paragraphs = data["paragraphs"]
            reasons = data.get("reasons", {})
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise RunStateError(
                f"{path} is not a run state ({type(exc).__name__})"
            ) from exc
        if (
            not isinstance(thread_id, str)
            or not thread_id
            or not isinstance(digest, str)
            or not isinstance(paragraphs, dict)
            or not isinstance(reasons, dict)
            or not all(
                isinstance(k, str) and v in _STATUSES for k, v in paragraphs.items()
            )
            or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in reasons.items()
            )
        ):
            raise RunStateError(f"{path} is not a valid run state")
        return cls(thread_id, digest, dict(paragraphs), dict(reasons))

    def save(self, root: Path) -> None:
        """Write the state whole, or leave the old file as it was.

        The text goes to a temporary file next to ``run.json``, is flushed to
        disk, and replaces ``run.json`` in one step.
        """
        path = _state_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "version": 1,
                "thread_id": self.thread_id,
                "outline_digest": self.outline_digest,
                "paragraphs": self.paragraphs,
                "reasons": self.reasons,
            },
            ensure_ascii=False,
            indent=2,
        )
        fd, tmp = tempfile.mkstemp(prefix=".run.json.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(payload + "\n")
                f.flush()
                os.fsync(f.fileno())
            _replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def count(self, status: str) -> int:
        return sum(1 for s in self.paragraphs.values() if s == status)


def _reconcile(
    saved: RunState | None, paragraphs: list[Paragraph], digest: str
) -> RunState:
    """The state for this run. Same outline: same thread. Changed outline: a new
    thread, the ``done`` paragraphs that remain kept, removed IDs dropped, new
    ones pending. A blocked paragraph is tried again, since running again is
    the author's answer to the block."""
    ids = [p.id for p in paragraphs]
    if saved is None:
        return RunState(
            f"run-{uuid.uuid4().hex}", digest, dict.fromkeys(ids, PENDING), {}
        )
    thread_id = (
        saved.thread_id if saved.outline_digest == digest else f"run-{uuid.uuid4().hex}"
    )
    statuses = {i: (DONE if saved.paragraphs.get(i) == DONE else PENDING) for i in ids}
    return RunState(thread_id, digest, statuses, {})


# --------------------------------------------------------------- mark_paragraph

_MARK_LOCK = threading.Lock()

# ------------------------------------------------------- the manuscript guard
#
# ``write_file`` replaces a whole file, and only the prompt tells the model to
# keep the paragraphs already finished. So the harness keeps copies of the
# section files and checks that no line of the text it started a pass with, or
# had at the last ``done`` mark, has disappeared from them. Lines are compared
# stripped and as a set over all section files: a paragraph moved, even to
# another section, is fine, a missing one is not.

_BACKUPS_KEPT = 3
_BACKUP_NAME = re.compile(r"^manuscript\.pass-(\d+)$")
MARK_SNAPSHOT = "manuscript.mark"  # folder, as of the last paragraph marked done
LOST_MARKER = "manuscript.lost"  # written by the tool when it sees a loss


def _all_text(folder: Path) -> str:
    """The text of every section file in *folder*, one after the other."""
    return "\n".join(read_section_texts(folder).values())


def _missing_lines(before: str, after: str) -> list[str]:
    """Non-blank lines of *before* (stripped, first occurrence) that *after*
    no longer has."""
    kept = {line.strip() for line in after.splitlines()}
    seen: set[str] = set()
    out: list[str] = []
    for line in before.splitlines():
        text = line.strip()
        if text and text not in kept and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _backup_manuscript(root: Path, state_dir: Path) -> tuple[Path | None, str]:
    """Copy the section files to the folder ``manuscript.pass-N`` (N counts up
    across runs), keep the newest ``_BACKUPS_KEPT``, and return the copy and
    its text.

    No section file, nothing to lose: ``(None, "")``. Raises ``OSError``.
    """
    if not read_section_texts(root / MANUSCRIPT_DIR):
        return None, ""
    before = _all_text(root / MANUSCRIPT_DIR)
    numbers = {
        int(m.group(1)): f
        for f in state_dir.iterdir()
        if (m := _BACKUP_NAME.match(f.name)) and f.is_dir()
    }
    target = state_dir / f"manuscript.pass-{max(numbers, default=0) + 1}"
    copy_section_files(root, target)
    numbers[int(_BACKUP_NAME.match(target.name).group(1))] = target
    for number in sorted(numbers)[:-_BACKUPS_KEPT]:
        shutil.rmtree(numbers[number], ignore_errors=True)
    return target, before


def _clear_marks(state_dir: Path) -> None:
    shutil.rmtree(state_dir / MARK_SNAPSHOT, ignore_errors=True)
    with contextlib.suppress(OSError):
        (state_dir / LOST_MARKER).unlink()


def _lost_since(
    root: Path, state_dir: Path, before: str, backup: Path | None
) -> tuple[list[str], Path | None]:
    """Lines lost from the section files during a pass, and the copy that still
    has them. Empty list: nothing was lost."""
    after = _all_text(root / MANUSCRIPT_DIR)
    snapshot = state_dir / MARK_SNAPSHOT
    if (state_dir / LOST_MARKER).exists():
        # The tool saw it mid-pass; what is missing now is not the point.
        return _missing_lines(_all_text(snapshot), after) or ["(text)"], snapshot
    if missing := _missing_lines(before, after):
        return missing, backup
    if missing := _missing_lines(_all_text(snapshot), after):
        return missing, snapshot
    return [], None


def _mark_lost_text(root: Path, state_dir: Path) -> bool:
    """Whether the section files lack text they had at the previous ``done``
    mark. Leaves a marker for the harness when they do."""
    snapshot = state_dir / MARK_SNAPSHOT
    if not snapshot.is_dir():
        return False
    if not _missing_lines(_all_text(snapshot), _all_text(root / MANUSCRIPT_DIR)):
        return False
    with contextlib.suppress(OSError):
        (state_dir / LOST_MARKER).write_text(MARK_SNAPSHOT, encoding="utf-8")
    return True


def _take_mark_snapshot(root: Path, state_dir: Path) -> None:
    with contextlib.suppress(OSError):
        copy_section_files(root, state_dir / MARK_SNAPSHOT)


def _mark_progress(root: Path, state: RunState, pid: str) -> str:
    """How far the section of *pid* is, then the whole outline: a run of one
    section ("Results 1/12 done") must not read as "1/14" of something."""
    whole = f"{state.count(DONE)}/{len(state.paragraphs)}"
    try:
        paragraphs = parse_paragraphs((root / "outline.md").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, OutlineError):
        return f"{whole} done"
    section = next((p.section for p in paragraphs if p.id == pid), "")
    ids = [
        p.id for p in paragraphs if p.section == section and p.id in state.paragraphs
    ]
    if not section or len(ids) == len(state.paragraphs):
        return f"{whole} done"
    done = sum(state.paragraphs[i] == DONE for i in ids)
    return f"{_short(section)[:40]} {done}/{len(ids)} done; whole outline {whole}"


def make_mark_paragraph_tool(root: Path) -> BaseTool:
    """The ``mark_paragraph`` tool of the run in *root*.

    The autonomous agent alone gets it (a chat session's tool list never
    includes it). It checks the ID against the state the harness saved, takes
    two statuses only, and replaces ``run.json`` whole. Marking the same thing
    twice changes nothing. Its replies are short.
    """
    root = Path(root)

    @tool(parse_docstring=True)
    def mark_paragraph(
        paragraph_id: str, status: Literal["done", "blocked"], reason: str = ""
    ) -> str:
        """Record the state of one outline paragraph in this run.

        Call it with "done" right after the paragraph is written into its
        section's file under Manuscript/. Call it with "blocked" and a reason
        when the sources do not support the paragraph's claim or the outline
        would have to change; then stop working.

        Args:
            paragraph_id: The ID from outline.md, for example "R1".
            status: "done" when the paragraph is in its section file, "blocked" when you cannot write it.
            reason: Required with "blocked": one sentence on what is missing or must be decided.
        """
        pid = _clean(paragraph_id)
        why = _clean(reason)[:_MAX_REASON]
        with _MARK_LOCK:
            try:
                state = RunState.load(root)
            except RunStateError:
                return "Rejected: the run state cannot be read."
            if state is None:
                return "No active run: nothing was recorded."
            if pid not in state.paragraphs:
                known = list(state.paragraphs)
                shown = ", ".join(known[:_SHOWN_IDS]) + (
                    " ..." if len(known) > _SHOWN_IDS else ""
                )
                return f"Unknown paragraph ID {pid!r}. Known IDs: {shown}."
            if status == BLOCKED and not why:
                return "Rejected: a blocked paragraph needs a reason."
            state_dir = Workspace(root).state_dir
            if _mark_lost_text(root, state_dir):
                return (
                    "Rejected: the files in Manuscript/ lost text that was there at "
                    "the last mark. The run will stop; do not rewrite finished paragraphs."
                )
            if state.paragraphs[pid] == status and (
                status == DONE or state.reasons.get(pid) == why
            ):
                return f"{pid} is already {status}."
            state.paragraphs[pid] = status
            if status == BLOCKED:
                state.reasons[pid] = why
            else:
                state.reasons.pop(pid, None)
            try:
                state.save(root)
            except (OSError, UnicodeError) as exc:
                return f"Rejected: could not save ({type(exc).__name__})."
            if status == DONE:
                _take_mark_snapshot(root, state_dir)
            return f"{pid} marked {status} ({_mark_progress(root, state, pid)})."

    mark_paragraph.handle_validation_error = lambda _exc: (
        "Rejected: give paragraph_id and a status of 'done' or 'blocked'."
    )
    return mark_paragraph


# ------------------------------------------------------------------- the run


class RunExit(IntEnum):
    OK = 0
    FAILED = 2
    USAGE_LIMIT = 3
    NEEDS_DECISION = 4


@dataclass
class RunResult:
    exit: RunExit
    done: int
    total: int
    evidence_needed: int
    message: str


AgentInvoke = Callable[[str, str], None]
"""``invoke(thread_id, prompt)``: one autonomous agent turn in that thread."""

_FIRST_PROMPT = """\
You are running unattended. Nobody can answer questions, so decide and keep going.

Write the manuscript draft following outline.md, paragraph by paragraph, in the
outline's order. Each section of the outline has its own file under Manuscript/;
save each paragraph in the file named after it below. Read program.md and
outline.md first.

Paragraphs still to write, in order:
{todo}

{kept}Rules:
- Use only this project's own material (Knowledge/, .metasci-scribe/extracted/, Figures/, Tables/). Never invent a number or a citation.
- Where the material gives no evidence, write [evidence needed] in the text and keep going. This is not a failure.
- A section file holds that section's paragraphs only, in outline order, with no section heading; create it if it does not exist.
- After each paragraph is in its file, call mark_paragraph(paragraph_id, "done").
- If the material does not support a paragraph's claim, or the outline would have to change, call mark_paragraph(paragraph_id, "blocked", reason) with one sentence of reason, then stop. Do not edit outline.md.
- Edit only {files}. Never edit manuscript.md (the app joins it from Manuscript/), and never write under .metasci-scribe/.
"""

_NEXT_PROMPT = """\
You stopped, but these paragraphs are not marked done yet:
{todo}

Continue with them under the same rules. Write each one into the file named after
it, then call mark_paragraph(paragraph_id, "done"). If a paragraph cannot be written from
the project's material, call mark_paragraph(paragraph_id, "blocked", reason) and stop.
"""


_KEPT = """\
The section files already hold these finished paragraphs: {ids}.
Keep them exactly as they are; do not rewrite or reorder them. Add each pending
paragraph where the outline puts it.

"""


def _file_of(p: Paragraph) -> str:
    return f"{MANUSCRIPT_DIR}/{section_file_name(p.section or MANUSCRIPT_DIR)}"


def _prompt(
    first: bool, state: RunState, titles: dict[str, Paragraph], targets: list[str]
) -> str:
    todo = "\n".join(
        f"- {i}  {titles[i].section}: {titles[i].title}".rstrip(": ")
        + f" -> {_file_of(titles[i])}"
        for i in targets
        if state.paragraphs.get(i) == PENDING
    )
    if not first:
        return _NEXT_PROMPT.format(todo=todo)
    done = [i for i in targets if state.paragraphs.get(i) == DONE]
    kept = _KEPT.format(ids=", ".join(done)) if done else ""
    files = sorted({_file_of(titles[i]) for i in targets})
    return _FIRST_PROMPT.format(todo=todo, kept=kept, files=", ".join(files))


def _targets_text(root: Path, titles: dict[str, Paragraph], targets: list[str]) -> str:
    """The text of the section files this run writes."""
    texts = []
    for name in sorted({_file_of(titles[i]) for i in targets}):
        with contextlib.suppress(OSError):
            texts.append((root / name).read_text(encoding="utf-8", errors="replace"))
    return "\n".join(texts)


def _clean(text: str) -> str:
    """*text* on one line without control, format or surrogate characters, so
    nothing the model wrote can steer a terminal or break a file write."""
    one_line = " ".join(text.split())
    return "".join(c for c in one_line if unicodedata.category(c)[0] != "C")


def _short(value: object) -> str:
    """Text that came from outside (an error, the model), safe to print:
    cleaned, scrubbed of deployed API keys, and cut short."""
    return _redact_api_keys(_clean(str(value)))[:_MAX_ERROR_TEXT]


def _plan_stop_message(exc: PlanStop, progress: str) -> str:
    parts = [f"The ChatGPT plan stopped the run ({exc.action.name.lower()})"]
    if exc.code:
        parts.append(f"code {exc.code}")
    if exc.request_id:
        parts.append(f"request id {exc.request_id}")
    text = ", ".join(parts) + "."
    if exc.message and exc.message != exc.action.name:
        text += f" {_short(exc)}"
    if exc.action is PlanAction.RELOGIN:
        text += " Run /login to sign in again."
    return f"{text} {progress} Run again to continue."


def _section_thread(thread_id: str, section: str) -> str:
    """The thread of a run limited to *section*: derived from the outline's
    thread, so it changes with the outline, and shaped like a run thread
    (``run-<32 hex>``) so chat lists keep hiding it."""
    key = f"{thread_id}/{' '.join(section.split()).casefold()}"
    return f"run-{uuid.uuid5(uuid.NAMESPACE_URL, key).hex}"


def run_project(
    root: Path, invoke: AgentInvoke, section: str | None = None
) -> RunResult:
    """Run the paper in *root* to the end, or to the first thing that needs a person.

    Checks first, without any model cost: a local folder, ``program.md`` and
    ``outline.md`` present, an outline that parses, no open question mark.
    Then it reads ``Knowledge/`` (``sync_knowledge``), settles the state
    (see ``_reconcile``), saves it, and calls ``invoke(thread_id, prompt)`` up
    to ``MAX_PASSES`` times while paragraphs are pending. See the module doc
    for interrupts. Never raises for the run's own failures; they come back as
    a ``RunResult`` with a message for the author.

    With *section* (an outline section name, case and spacing aside), only
    that section's paragraphs are written and counted, and open question
    marks elsewhere in the outline do not stop the run. Every result that got
    as far as the state ends with the section files joined into
    ``manuscript.md``.
    """
    root = Path(root)
    targets: list[str] = []
    sections: list[str] = []

    def result(
        code: RunExit, message: str, state: RunState | None = None, notes=()
    ) -> RunResult:
        done = total = evidence = 0
        lines = [message, *notes]
        if state:
            done = sum(1 for i in targets if state.paragraphs.get(i) == DONE)
            total = len(targets)
            evidence = len(_EVIDENCE.findall(_targets_text(root, titles, targets)))
            if note := join_manuscript(root, sections).note():
                lines.append(note)
        return RunResult(code, done, total, evidence, "\n".join(lines))

    # -- checks that cost nothing --------------------------------------------
    try:
        require_local_path(root)
        paths.require_workspace_root(root)
    except (NetworkPathError, paths.WorkspaceRootError) as exc:
        return result(RunExit.FAILED, str(exc))
    if not root.is_dir():
        return result(RunExit.FAILED, f"{root} is not a folder.")
    missing = [n for n in ("program.md", "outline.md") if not (root / n).is_file()]
    if missing:
        return result(
            RunExit.FAILED,
            f"{' and '.join(missing)} not found in {root}. Run /init in the paper folder first.",
        )
    try:
        outline = (root / "outline.md").read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return result(RunExit.FAILED, "outline.md is not UTF-8 text. Save it as UTF-8.")
    except OSError as exc:
        return result(RunExit.FAILED, f"outline.md cannot be read: {exc}")
    try:
        paragraphs = parse_paragraphs(outline)
    except OutlineError as exc:
        return result(RunExit.FAILED, f"outline.md, {exc}")
    if not paragraphs:
        return result(
            RunExit.FAILED,
            "outline.md has no paragraph headings such as '### R1 · ①'. "
            "Draw the map in a chat session first.",
        )
    sections[:] = section_names(outline)
    by_file: dict[str, str] = {}
    for name in sections:
        clash = by_file.setdefault(section_file_name(name).casefold(), name)
        if clash != name:
            return result(
                RunExit.FAILED,
                f"outline.md, the sections {clash!r} and {name!r} would share the file "
                f"{MANUSCRIPT_DIR}/{section_file_name(name)}. Rename one of them.",
            )
    if section is not None:
        match = next((s for s in sections if same_section(s, section)), None)
        if match is None:
            return result(
                RunExit.FAILED,
                f"outline.md has no section {_short(section)!r}. Its sections: "
                + (", ".join(sections) or "none")
                + ".",
            )
        targets[:] = [p.id for p in paragraphs if same_section(p.section, match)]
        if not targets:
            return result(
                RunExit.FAILED,
                f"The section {match!r} of outline.md has no paragraph headings "
                "such as '### R1 · ①'.",
            )
        section = match
    else:
        targets[:] = [p.id for p in paragraphs]
    marks = open_markers(outline, section)
    if marks:
        return result(RunExit.NEEDS_DECISION, open_markers_message(outline, marks))
    digest = outline_digest(outline)
    titles = {p.id: p for p in paragraphs}

    state_dir = Workspace(root).state_dir
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return result(RunExit.FAILED, f"{state_dir} cannot be used: {_short(exc)}")
    lock = FileLock(str(state_dir / "run.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return result(RunExit.FAILED, "Another run is already working on this folder.")
    except OSError as exc:
        return result(RunExit.FAILED, f"{state_dir} cannot be used: {_short(exc)}")
    try:
        # Only now, holding the lock, is the saved state the one to build on.
        _remove_stale_temps(state_dir)
        try:
            saved = RunState.load(root)
        except RunStateError as exc:
            return result(
                RunExit.FAILED,
                f"{exc}. Fix or delete .metasci-scribe/run.json; delete it to start the run over.",
            )
        return _run_locked(
            root, invoke, paragraphs, titles, digest, saved, result, targets, section
        )
    finally:
        lock.release()


def _remove_stale_temps(state_dir: Path) -> None:
    """Delete ``.run.json.*.tmp`` files that a killed save left behind. Called
    holding the run lock, so no save of this folder is in progress."""
    for stale in state_dir.glob(".run.json.*.tmp"):
        with contextlib.suppress(OSError):
            stale.unlink()


def _run_locked(
    root, invoke, paragraphs, titles, digest, saved, result, targets, section
) -> RunResult:
    # -- Knowledge -------------------------------------------------------------
    try:
        report = sync_knowledge(root, wait=LOCK_WAIT_SECONDS)
    except OSError as exc:
        return result(RunExit.FAILED, f"Knowledge/ could not be read: {exc}")
    notes = report_messages(report, root)
    if report.busy:
        return result(
            RunExit.FAILED,
            "Another Metasci Scribe is still reading the PDFs of this folder, so the "
            "run did not start. Wait for it to finish and run again.",
            notes=notes[1:],
        )

    # -- the state -------------------------------------------------------------
    state = _reconcile(saved, paragraphs, digest)
    try:
        state.save(root)
    except OSError as exc:
        return result(
            RunExit.FAILED, f".metasci-scribe/run.json cannot be written: {exc}"
        )
    thread_id = (
        state.thread_id
        if section is None
        else _section_thread(state.thread_id, section)
    )

    def count(status: str) -> int:
        return sum(1 for i in targets if state.paragraphs.get(i) == status)

    def progress() -> str:
        return f"{count(DONE)}/{len(targets)} paragraphs marked done"

    # -- the passes ------------------------------------------------------------
    state_dir = Workspace(root).state_dir
    for attempt in range(MAX_PASSES):
        if count(PENDING) == 0:
            break
        _clear_marks(state_dir)
        try:
            backup, before = _backup_manuscript(root, state_dir)
        except OSError as exc:
            return result(
                RunExit.FAILED,
                f"{MANUSCRIPT_DIR}/ could not be backed up before the pass: {_short(exc)}",
                state,
                notes,
            )
        error: Exception | None = None
        try:
            invoke(thread_id, _prompt(attempt == 0, state, titles, targets))
        except Exception as exc:  # PlanStop and the rest: handled below
            error = exc
        if error is not None:
            state = _reload_quietly(root, state)
        lost, copy = _lost_since(root, state_dir, before, backup)
        if lost:
            return result(RunExit.FAILED, _loss_message(root, lost, copy), state, notes)
        if error is not None:
            return _error_result(error, state, result, notes, progress())
        try:
            state = _reload(root, state)
        except RunStateError as exc:
            return result(
                RunExit.FAILED, f"{exc}. Delete it to start over.", None, notes
            )
        if count(BLOCKED):
            break

    # -- the verdict -----------------------------------------------------------
    total = len(targets)
    scope = "" if section is None else f" of {section}"
    blocked = [i for i in targets if state.paragraphs.get(i) == BLOCKED]
    pending = [i for i in targets if state.paragraphs.get(i) == PENDING]
    if blocked:
        lines = [f"- {i}: {_short(state.reasons.get(i, ''))}" for i in blocked]
        return result(
            RunExit.NEEDS_DECISION,
            f"{len(blocked)} paragraph(s) need your decision "
            f"({count(DONE)}/{total}{scope} marked done):\n"
            + "\n".join(lines)
            + "\nResolve them (the sources, program.md, or the outline in a chat session), "
            "then run again.",
            state,
            notes,
        )
    if pending:
        shown = ", ".join(pending[:_SHOWN_IDS]) + (
            " ..." if len(pending) > _SHOWN_IDS else ""
        )
        return result(
            RunExit.FAILED,
            f"The agent did not finish after {MAX_PASSES} passes "
            f"({count(DONE)}/{total}{scope} marked done). Still pending: {shown}. "
            "Run again to continue.",
            state,
            notes,
        )
    written = _targets_text(root, titles, targets)
    if not written.strip():
        return result(
            RunExit.FAILED,
            f"All {total} paragraphs{scope} are marked done but their files in "
            f"{MANUSCRIPT_DIR}/ are empty or missing. "
            "Delete .metasci-scribe/run.json to start the run over.",
            state,
            notes,
        )
    evidence = len(_EVIDENCE.findall(written))
    tail = (
        f" {evidence} place(s) are marked [evidence needed]; check them."
        if evidence
        else ""
    )
    return result(
        RunExit.OK,
        f"The agent marked {total} of {total} paragraphs{scope} done.{tail}\n"
        f"Nobody has checked the draft yet: read it in {MANUSCRIPT_DIR}/ or "
        "manuscript.md, and check its claims and citations against your sources, "
        "before you use it.",
        state,
        notes,
    )


def _loss_message(root: Path, lost: list[str], copy: Path | None) -> str:
    shown = "; ".join(f"“{_short(line)[:80]}”" for line in lost[:3])
    more = f" and {len(lost) - 3} more" if len(lost) > 3 else ""
    where = (
        f"The last good copy is in {copy.relative_to(root).as_posix()}/"
        if copy is not None
        else "No copy was kept"
    )
    return (
        f"The files in {MANUSCRIPT_DIR}/ lost {len(lost)} line(s) of finished text "
        f"during the pass ({shown}{more}). {where}. Copy its files over those in "
        f"{MANUSCRIPT_DIR}/, check the draft, and run again. The paragraphs marked "
        "done in this pass cannot be trusted."
    )


def _error_result(
    exc: Exception, state: RunState, result, notes, progress: str
) -> RunResult:
    """The result for an error out of ``invoke``. Only the type is shown for
    anything but a plan stop, so nothing secret in an error text can leak."""
    saved = f"Progress is saved ({progress})."
    if isinstance(exc, PlanStop):
        if exc.action is PlanAction.STOP_USAGE:
            return result(
                RunExit.USAGE_LIMIT,
                "The ChatGPT plan usage limit was reached. "
                f"{saved} Check your plan's usage in ChatGPT settings, "
                "and run again when it allows more use; the run continues where it stopped.",
                state,
                notes,
            )
        return result(RunExit.FAILED, _plan_stop_message(exc, saved), state, notes)
    if isinstance(exc, ChatGPTAuthError | ModelNotConfigured):
        return result(
            RunExit.FAILED,
            f"The model is not available ({type(exc).__name__}). Run /login, then run again. "
            f"{saved}",
            state,
            notes,
        )
    return result(
        RunExit.FAILED,
        f"The agent stopped with an unexpected error ({type(exc).__name__}). "
        f"{saved[:-1]}; run again to continue.",
        state,
        notes,
    )


def _reload(root: Path, fallback: RunState) -> RunState:
    """The state the tool left on disk (the harness itself saves only once).

    Raises ``RunStateError`` if the file was damaged meanwhile."""
    fresh = RunState.load(root)
    return fresh if fresh is not None else fallback


def _reload_quietly(root: Path, fallback: RunState) -> RunState:
    try:
        return _reload(root, fallback)
    except RunStateError:
        return fallback
