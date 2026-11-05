"""The paragraph pipeline of an autonomous run: write, review, rewrite.

The author found that a section written in one go comes out thin, so a run can
take one outline paragraph at a time through three agent turns, each in its
own fresh thread:

1. **write** — the writing skill puts the paragraph into its section file;
2. **review** — the review skill, in a context that did not write it, reports
   on it into ``Reviews/<Section>/<ID>/r<N>/review.md``;
3. **rewrite** — only when the review found a 치명 or 주요 problem, the
   rewriting skill writes a revision to ``revised.txt`` in the same folder.

The agent never edits a section file after the write turn: the harness finds
the paragraph by the anchor line ``<!-- R3 -->`` it keeps in front of it,
gives the reviewer an exact copy, checks the revision mechanically and puts it
in place itself. Everything here is pure file and text work; the turns are run
by :mod:`metasci_scribe.run`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .manuscript import PARAGRAPH_ANCHOR, section_file_name

REVIEWS_DIR = "Reviews"
SKILLS_DIR = Path(__file__).parent / "skills"  # the skills that ship with the app

# Severity headings of a review report (scribe-review's report_format.md).
_FINDING = re.compile(r"^### +\d+\. +\[(치명|주요|경미)\]", re.MULTILINE)
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_LANGUAGE = re.compile(r"^language:\s*([A-Za-z]+)", re.MULTILINE)
CHECK_TIMEOUT = 120
# The gates of verify_korean_revision.py whose ABORT refuses a revision.
_KOREAN_HARD = re.compile(
    r"^- \[ABORT\] (P1-numeric-expressions|P2-citations|P2-direct-quotes"
    r"|P2-named-spans)\b",
    re.MULTILINE,
)


# ------------------------------------------------------------------ language


def manuscript_language(program: str, *texts: str) -> str:
    """``"ko"`` or ``"en"``: ``program.md``'s ``language`` when it names one,
    else the script most of *texts* (outline, section files) is written in."""
    if m := _LANGUAGE.search(program):
        value = m.group(1).lower()
        if value in ("ko", "en"):
            return value
    sample = "".join(texts)
    return (
        "ko"
        if len(_HANGUL.findall(sample)) * 3 >= len(_LATIN.findall(sample))
        else "en"
    )


@dataclass(frozen=True)
class Skills:
    writing: str
    review: str
    rewriting: str


def skills_for(language: str) -> Skills:
    if language == "ko":
        return Skills(
            "scribe-writing-korean", "scribe-review", "scribe-rewriting-korean"
        )
    return Skills("scribe-writing", "scribe-review", "scribe-rewriting")


# ---------------------------------------------------------------- paragraphs


def _blocks(text: str) -> list[list[str]]:
    """The text split into blocks of non-blank lines."""
    out: list[list[str]] = []
    current: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.strip():
            current.append(line)
        elif current:
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out


def _anchor(block: list[str]) -> str | None:
    m = PARAGRAPH_ANCHOR.match(block[0])
    return m.group("id") if m else None


def _join(blocks: list[list[str]]) -> str:
    return "\n\n".join("\n".join(b) for b in blocks) + ("\n" if blocks else "")


def _split_anchor(blocks: list[list[str]]) -> list[list[str]]:
    """Blocks with an anchor line on its own (``<!-- R1 -->`` then a blank line)
    joined to the block it names, so each paragraph is one block."""
    out: list[list[str]] = []
    pending: list[str] | None = None
    for block in blocks:
        if pending is not None:
            block = pending + block
            pending = None
        if len(block) == 1 and _anchor(block):
            pending = block
            continue
        out.append(block)
    if pending is not None:
        out.append(pending)
    return out


def paragraph_text(section_text: str, pid: str) -> str | None:
    """The paragraph that the anchor ``<!-- pid -->`` names, without the
    anchor, or None when the file has no such anchor."""
    for block in _split_anchor(_blocks(section_text)):
        if _anchor(block) == pid:
            body = "\n".join(block[1:]).strip()
            return body or None
    return None


def anchor_new_paragraph(before: str, after: str, pid: str) -> str | None:
    """*after* with the anchor of *pid* in front of the paragraph the write turn
    added, or None when that paragraph cannot be told apart.

    An anchor the agent already wrote is kept. Otherwise the new paragraph is
    the one block of *after* that *before* does not have.
    """
    blocks = _split_anchor(_blocks(after))
    if any(_anchor(b) == pid and len(b) > 1 for b in blocks):
        return _join(blocks)
    old = {tuple(line.strip() for line in b) for b in _blocks(before)}
    new = [
        i
        for i, b in enumerate(blocks)
        if not _anchor(b) and tuple(line.strip() for line in b) not in old
    ]
    if len(new) != 1:
        return None
    blocks[new[0]] = [f"<!-- {pid} -->", *blocks[new[0]]]
    return _join(blocks)


def replace_paragraph(section_text: str, pid: str, revised: str) -> str | None:
    """*section_text* with the paragraph of *pid* replaced by *revised*, every
    other block unchanged; None when the anchor is not there."""
    blocks = _split_anchor(_blocks(section_text))
    lines = [line for line in revised.strip().split("\n") if line.strip()]
    for i, block in enumerate(blocks):
        if _anchor(block) == pid:
            blocks[i] = [block[0], *lines]
            return _join(blocks)
    return None


def single_paragraph(text: str) -> str | None:
    """*text* as one paragraph: None when it is empty or holds several."""
    blocks = [b for b in _blocks(text) if not (len(b) == 1 and _anchor(b))]
    if len(blocks) != 1:
        return None
    return "\n".join(
        line.strip() for line in blocks[0] if not PARAGRAPH_ANCHOR.match(line)
    )


# ------------------------------------------------------------------- reviews


def review_dir(root: Path, section: str, pid: str) -> Path:
    """The folder of the paragraph's reviews: ``Reviews/<Section>/<ID>``."""
    name = section_file_name(section).removesuffix(".md")
    return Path(root) / REVIEWS_DIR / name / pid


def next_round(folder: Path) -> Path:
    """``r1``, ``r2``, … : the first round folder that does not exist yet."""
    n = 1
    while (folder / f"r{n}").exists():
        n += 1
    return folder / f"r{n}"


def count_findings(review: str) -> Counter[str]:
    """How many findings of each severity a review report holds."""
    return Counter(m.group(1) for m in _FINDING.finditer(review))


def needs_rewrite(findings: Counter[str]) -> bool:
    return bool(findings["치명"] or findings["주요"])


# ------------------------------------------------------------------ checkers


@dataclass(frozen=True)
class Check:
    passed: bool
    output: str
    code: int = 0


def skill_dirs(root: Path) -> list[Path]:
    """Where a skill may be, first match wins, as the agent's ``/skills/``
    sees them: the paper's own, the author's (``~/.metasci-scribe/skills``),
    then those that ship with the app. The pipeline's skills are the
    author's own and do not ship with the public app."""
    from .paths import Workspace

    return [Workspace(root).skills_dir, paths.GLOBAL_SKILLS_DIR, SKILLS_DIR]


def find_skill(root: Path, name: str) -> Path | None:
    """The folder of skill *name*, or None when no tier has its SKILL.md."""
    for base in skill_dirs(root):
        if (base / name / "SKILL.md").is_file():
            return base / name
    return None


def missing_skills(root: Path, language: str) -> list[str]:
    """The pipeline's skills for *language* that cannot be found."""
    s = skills_for(language)
    return [
        n for n in (s.writing, s.review, s.rewriting) if find_skill(root, n) is None
    ]


def _script(root: Path, skill: str, name: str) -> Path:
    folder = find_skill(root, skill)
    return (folder or Path(skill)) / "scripts" / name


def _run_script(script: Path, *args: str) -> Check:
    """Run one of the skills' checker scripts with this interpreter.

    ``-I`` keeps the paper folder off its import path. A script that is
    missing or cannot run counts as passed with a note: the check is extra
    safety, not a reason to stop the run.
    """
    if not script.is_file():
        return Check(True, f"(no {script.name}; not checked)")
    try:
        done = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", str(script), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=CHECK_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check(True, f"({script.name} could not run: {exc}; not checked)")
    output = (done.stdout + done.stderr).strip()
    return Check(done.returncode == 0, output, done.returncode)


def check_review(
    root: Path, round_dir: Path, knowledge: Path, previous: Path | None
) -> Check:
    """scribe-review's own check of ``review.md`` against ``draft.txt``."""
    args = [str(round_dir / "draft.txt"), str(round_dir / "review.md")]
    if knowledge.is_dir() and any(knowledge.glob("*.md")):
        args += ["--knowledge", str(knowledge)]
    if previous is not None and previous.is_file():
        args += ["--previous", str(previous)]
    return _run_script(_script(root, "scribe-review", "check_review.py"), *args)


def check_revision(root: Path, round_dir: Path, language: str) -> Check:
    """The mechanical preservation check of ``revised.txt`` against
    ``draft.txt``. It fails only on what a revision must never change:
    numbers, units, quotations, citations. A changed hedge or wording is what
    the review asked for, so it is reported, not refused.

    Korean: ``verify_korean_revision.py --mode deep``; an ABORT of a content
    gate fails (numbers, citations, direct quotes, named spans). Its
    ``P1b-binding`` gate is only reported: it pairs sentences split at every
    full stop, a decimal point too ("0.95"), so it aborts on most reworded
    sentences that hold a decimal. English: ``check_preservation.py``; a FAIL
    on numbers or citations fails, on hedges, boosters, negation or names it
    is only reported.
    """
    before, after = round_dir / "draft.txt", round_dir / "revised.txt"
    if language == "ko":
        check = _run_script(
            _script(root, "scribe-rewriting-korean", "verify_korean_revision.py"),
            "--before",
            str(before),
            "--after",
            str(after),
            "--mode",
            "deep",
        )
        hard = _KOREAN_HARD.search(check.output)
        return Check(check.code != 2 or hard is None, check.output, check.code)
    check = _run_script(
        _script(root, "scribe-review", "check_preservation.py"), str(before), str(after)
    )
    hard = re.search(r"^FAIL (numbers|citations)\b", check.output, re.MULTILINE)
    return Check(hard is None, check.output, check.code)
