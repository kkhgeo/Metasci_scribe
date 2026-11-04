"""The manuscript on disk: one file per outline section, and the joined copy.

Each ``##`` section of ``outline.md`` has its own file,
``Manuscript/<Section>.md``, and the agent writes a paragraph into its
section's file. ``manuscript.md`` is the whole draft in outline order. The app
writes it by joining the section files (``join_manuscript``); nobody edits it.

``manuscript.md`` is never overwritten when it holds text the app did not
write: a hand edit, or a draft from before the section files. The app keeps a
hash of what it last wrote in ``.metasci-scribe/`` to tell the two apart.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .paths import Workspace

__all__ = [
    "JOINED_NAME",
    "MANUSCRIPT_DIR",
    "SECTION_HEADING",
    "JoinResult",
    "copy_section_files",
    "join_manuscript",
    "read_section_texts",
    "same_section",
    "section_file",
    "section_file_name",
    "section_names",
]

MANUSCRIPT_DIR = "Manuscript"
JOINED_NAME = "manuscript.md"
_JOINED_STAMP = "manuscript.joined.sha256"

# A section heading of outline.md: ``## Results``.
SECTION_HEADING = re.compile(r"^##[ \t]+(?P<name>\S.*?)[ \t]*#*[ \t]*$")

_COMMENT = re.compile(r"<!--.*?(?:-->|\Z)", re.DOTALL)
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{kind}{n}" for kind in ("COM", "LPT") for n in range(1, 10)
}


def _normalize(name: str) -> str:
    return " ".join(name.split()).casefold()


def same_section(a: str, b: str) -> bool:
    """Whether two section names mean the same section (case and spacing aside)."""
    return _normalize(a) == _normalize(b)


def section_names(outline: str) -> list[str]:
    """The ``##`` sections of *outline*, in order, each once. Comments are
    skipped; an unclosed one hides the rest, as the outline parser would refuse
    it anyway."""
    text = _COMMENT.sub("", outline.lstrip("﻿").replace("\r\n", "\n"))
    names: list[str] = []
    for line in text.split("\n"):
        if (m := SECTION_HEADING.match(line)) and not any(
            same_section(m.group("name"), n) for n in names
        ):
            names.append(m.group("name"))
    return names


def section_file_name(section: str) -> str:
    """The file name for *section*: the name with the characters Windows
    refuses replaced, ``.md`` added. ``Materials and Methods`` keeps its
    spaces."""
    name = " ".join(_BAD_CHARS.sub("-", section).split()).rstrip(". ")
    if not name:
        name = "Section"
    if name.split(".")[0].upper() in _RESERVED:
        name += "_"
    return f"{name}.md"


def section_file(root: Path, section: str) -> Path:
    return Path(root) / MANUSCRIPT_DIR / section_file_name(section)


def read_section_texts(folder: Path) -> dict[str, str]:
    """The ``.md`` files directly in *folder*, name to text, sorted by name.
    No folder: nothing."""
    try:
        files = sorted(p for p in Path(folder).iterdir() if p.suffix == ".md")
    except OSError:
        return {}
    texts: dict[str, str] = {}
    for path in files:
        with contextlib.suppress(OSError):
            if path.is_file():
                texts[path.name] = path.read_text(encoding="utf-8", errors="replace")
    return texts


def copy_section_files(root: Path, target: Path) -> None:
    """Replace the folder *target* with a copy of the section files. Raises
    ``OSError``."""
    target = Path(target)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for name in read_section_texts(Path(root) / MANUSCRIPT_DIR):
        shutil.copy2(Path(root) / MANUSCRIPT_DIR / name, target / name)


@dataclass
class JoinResult:
    """What ``join_manuscript`` did. ``written`` is False when the text was the
    same or ``held_back`` says why it was not written. ``unused`` lists section
    files no outline section names."""

    written: bool = False
    held_back: str = ""
    unused: list[str] = field(default_factory=list)

    def note(self) -> str:
        """One line for the author, or "" when there is nothing to say."""
        parts = []
        if self.written:
            parts.append(
                f"{JOINED_NAME} was updated from the files in {MANUSCRIPT_DIR}/."
            )
        if self.held_back:
            parts.append(self.held_back)
        if self.unused:
            parts.append(
                f"Not in {JOINED_NAME} (no such section in outline.md): "
                + ", ".join(f"{MANUSCRIPT_DIR}/{n}" for n in self.unused)
                + "."
            )
        return " ".join(parts)


def _body(text: str, section: str) -> str:
    """The section file's text without a first heading that repeats the
    section name, since the join adds its own."""
    lines = text.strip().split("\n")
    if lines and (m := re.match(r"^#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$", lines[0])):
        if same_section(m.group(1), section):
            lines = lines[1:]
    return "\n".join(lines).strip()


def joined_text(root: Path, sections: list[str]) -> tuple[str, list[str]]:
    """The joined draft, and the section files left out of it."""
    texts = read_section_texts(Path(root) / MANUSCRIPT_DIR)
    used: set[str] = set()
    parts: list[str] = []
    for section in sections:
        name = section_file_name(section)
        used.add(name)
        body = _body(texts.get(name, ""), section)
        if body:
            parts.append(f"## {section}\n\n{body}\n")
    unused = [n for n in texts if n not in used and texts[n].strip()]
    return "\n".join(parts), unused


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def join_manuscript(root: Path, sections: list[str]) -> JoinResult:
    """Write ``manuscript.md`` from the section files, in the order of
    *sections*, each under a ``## Section`` heading.

    Held back, with the reason, when ``manuscript.md`` has text the app did not
    write. Deleting the file lets the next join write it again.
    """
    root = Path(root)
    text, unused = joined_text(root, sections)
    result = JoinResult(unused=unused)
    target = root / JOINED_NAME
    stamp = Workspace(root).state_dir / _JOINED_STAMP
    try:
        current = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        current = None
    except OSError as exc:
        result.held_back = f"{JOINED_NAME} could not be read ({type(exc).__name__})."
        return result
    if current == text or (current is None and not text):
        return result
    if current is not None and current.strip():
        try:
            ours = stamp.read_text(encoding="utf-8").strip() == _digest(current)
        except OSError:
            ours = False
        if not ours:
            result.held_back = (
                f"{JOINED_NAME} was left as it is: it has text the app did not "
                f"write. Move your edits into {MANUSCRIPT_DIR}/, then delete "
                f"{JOINED_NAME} to have it joined again."
            )
            return result
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".manuscript.", suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            os.replace(tmp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        stamp.write_text(_digest(text) + "\n", encoding="utf-8")
    except OSError as exc:
        result.held_back = f"{JOINED_NAME} could not be written ({type(exc).__name__})."
        return result
    result.written = True
    return result
