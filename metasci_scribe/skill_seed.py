"""Editable copies of the writing skills in ``~/.metasci-scribe/skills``.

The app ships the skills its paragraph review uses (``scribe-writing``,
``scribe-review``, …). A skill inside the installed package is replaced by
every update, so the author could not keep a change to it. At start, each one
is copied to the author's skills folder, which the agent reads before the
shipped one:

- no copy yet: it is copied;
- a copy the author has not changed: it follows the app, replaced when the
  shipped version changes;
- a copy the author changed: it is kept, and the author is told once per
  shipped version that a newer one exists.

A copy counts as unchanged when its files hash to what was copied, recorded in
``.shipped`` inside it. Deleting a copy brings the shipped version back at the
next start.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from . import paths

SEEDED = (
    "scribe-writing",
    "scribe-writing-korean",
    "scribe-review",
    "scribe-rewriting",
    "scribe-rewriting-korean",
)
SHIPPED = Path(__file__).parent / "skills"
STAMP = ".shipped"  # hash of the shipped files this copy was made from
SEEN = ".shipped-seen"  # newest shipped hash the author was told about
_SKIP_DIRS = {"__pycache__"}


def tree_hash(folder: Path) -> str:
    """A hash of every file under *folder* (paths and bytes), without the
    stamps and Python caches."""
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        rel = path.relative_to(folder)
        if rel.name in (STAMP, SEEN) or _SKIP_DIRS & set(rel.parts):
            continue
        digest.update(rel.as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _copy(src: Path, dst: Path, shipped: str) -> None:
    """Put a fresh copy of *src* at *dst*, replacing what is there."""
    staging = dst.with_name(dst.name + ".new")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(src, staging, ignore=shutil.ignore_patterns(*_SKIP_DIRS))
    (staging / STAMP).write_text(shipped + "\n", encoding="utf-8")
    if dst.exists():
        shutil.rmtree(dst)
    staging.rename(dst)


def seed_skills(source: Path | None = None, target: Path | None = None) -> list[str]:
    """Bring the author's copies up to date; one line per thing worth saying."""
    source = SHIPPED if source is None else source
    target = paths.GLOBAL_SKILLS_DIR if target is None else target
    notes: list[str] = []
    for name in SEEDED:
        src, dst = source / name, target / name
        if not (src / "SKILL.md").is_file():
            continue
        try:
            shipped = tree_hash(src)
            if not dst.exists():
                target.mkdir(parents=True, exist_ok=True)
                _copy(src, dst, shipped)
                notes.append(f"Copied the skill {name} to {dst}; edit it there.")
                continue
            current = tree_hash(dst)
            recorded = _read(dst / STAMP)
            if current == shipped:
                if recorded != shipped:
                    (dst / STAMP).write_text(shipped + "\n", encoding="utf-8")
                continue
            if recorded is not None and current == recorded:
                _copy(src, dst, shipped)  # not edited: follows the app
                notes.append(f"Updated the skill {name} to the app's new version.")
                continue
            if recorded != shipped and _read(dst / SEEN) != shipped:
                (dst / SEEN).write_text(shipped + "\n", encoding="utf-8")
                notes.append(
                    f"The skill {name} in {dst} has your edits, so the app's newer "
                    "version was not applied. Delete that folder to get it."
                )
        except OSError as exc:
            notes.append(f"The skill {name} could not be copied to {target}: {exc}")
    return notes
