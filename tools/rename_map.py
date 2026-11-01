"""One-shot rename of the imported EvoScientist v0.3.8 tree to ``metasci_scribe``.

Run from the repository root::

    python tools/rename_map.py --dry-run     # report what would change
    python tools/rename_map.py               # rewrite files, then move the package

What it does
------------
1. Rewrites file contents (see ``TEXT_RULES`` / ``NAME_RULES``):

   * ``EvoScientist.EvoScientist`` module path   -> ``metasci_scribe.agent``
   * ``EvoScientistConfig``                      -> ``ScribeConfig``
   * ``EVOSCIENTIST_`` / ``EVOSCI_`` env prefix  -> ``METASCI_SCRIBE_``
   * ``~/.evoscientist``                         -> ``~/.metasci-scribe``
   * package / import paths                      -> ``metasci_scribe``
   * command and distribution name              -> ``metasci-scribe``
   * brand text                                  -> ``Metasci Scribe``

   Python files are rewritten token by token so identifiers and string
   contents get different treatment (an identifier cannot contain a hyphen).
   URLs that point at the upstream project are left alone on purpose, so
   attribution links stay correct.
2. Moves ``EvoScientist/`` to ``metasci_scribe/`` and ``EvoScientist.py`` to
   ``agent.py`` (via ``git mv`` when run inside a git work tree).

Edited by hand, outside this tool: the ``migrate_legacy_sessions_db`` removal in
``paths.py`` and its tests, one brand assertion in ``tests/test_prompts.py``,
``pyproject.toml``, ``README.md``, ``NOTICE``, and
the new ``tests/test_identity.py``.

Not touched:``docs/superpowers/``, ``spikes/``, ``.superpowers/``, ``tools/``,
``NOTICE``, ``docs/baseline.md``, ``uv.lock`` (regenerate with ``uv lock``),
``pyproject.toml`` (edited by hand), and binary files.
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_PREFIXES = (
    "docs/superpowers/",
    "spikes/",
    ".superpowers/",
    "tools/",
    ".git/",
)
SKIP_FILES = {"NOTICE", "docs/baseline.md", "uv.lock", "pyproject.toml", "LICENSE"}

# Spans that must survive untouched (upstream URLs, npm package, contact).
PROTECT = re.compile(
    r"https?://[^\s\"'`)>\]]*"
    r"|(?:ghcr\.io|github\.com|api\.star-history\.com)/[^\s\"'`)>\]]*"
    r"|pkgs/container/evoscientist"
    r"|@evoscientist(?:/|%2F)webui"
    r"|evoscientist\.ai(?:@gmail\.com)?"
    r"|mailto:[^)\s]*",
    re.IGNORECASE,
)
_PROTECT_ONLY_IF = re.compile(r"evoscientist|evosci", re.IGNORECASE)

# Ordered; each applies to string contents, comments, and non-Python files.
TEXT_RULES: list[tuple[str, str]] = [
    (r"EvoScientist\.EvoScientist\b", "metasci_scribe.agent"),
    (r"EvoScientist/EvoScientist\.py\b", "metasci_scribe/agent.py"),
    (r"(from\s+\.+)EvoScientist\b", r"\1agent"),
    (
        r"(?<=[\"'])(\.+)EvoScientist(?=[\"'])",
        r"\1agent",
    ),  # lazy-export "module" strings
    (r"\bEvoScientist\.py\b", "agent.py"),
    (r"EvoScientistConfig", "ScribeConfig"),
    (r"EVOSCIENTIST_", "METASCI_SCRIBE_"),
    (r"EVOSCI_", "METASCI_SCRIBE_"),
    (r"\bEVOSCIENTIST\b", "METASCI_SCRIBE"),
    (r"\bEVOSCI\b", "METASCI_SCRIBE"),
    (r"(?<=_)EVOSCI\b", "METASCI_SCRIBE"),  # e.g. "__EVOSCI" prefix checks
    (r"com\.evosci\.", "com.metasci_scribe."),
    (r"\$EvoSci\b", "$Scribe"),
    (r"\.evoscientist\b", ".metasci-scribe"),
    # executables and exact quoted names (graph ids, distribution name)
    (r"EvoScientist(?=\.exe\b)", "metasci-scribe"),
    (r"(?<=[\"'])EvoScientist(?=[\"'])", "metasci-scribe"),
    # package / path references
    (r"(?<=import )EvoScientist\b", "metasci_scribe"),
    (r"(?<=from )EvoScientist\b", "metasci_scribe"),
    (r"(?<=-m )EvoScientist\b", "metasci_scribe"),
    (r"\bEvoScientist(?=[./\\][A-Za-z_])", "metasci_scribe"),
    # identifiers: EvoScientist_agent -> scribe_agent, TestEvoScientistX -> TestScribeX
    (r"EvoScientist_", "scribe_"),
    (r"_EvoScientist", "_scribe"),
    (r"(?<=[A-Za-z0-9])EvoScientist|EvoScientist(?=[A-Za-z0-9])", "Scribe"),
    (r"EvoScientist(?=-)|(?<=-)EvoScientist", "Metasci-Scribe"),
    (r"\bEvoScientist(?=\[|==|>=|\{)", "metasci-scribe"),  # pip requirement strings
    (r"\bEvoScientist\b", "Metasci Scribe"),
    # lowercase
    (r"evoscientist(?=[_A-Za-z0-9])|(?<=[_A-Za-z0-9])evoscientist", "scribe"),
    (r"\bevoscientist\b", "metasci-scribe"),
    (r"evosci(?=_)|(?<=_)evosci", "scribe"),
    (r"\bevosci\b", "metasci-scribe"),
    (r"EvoSci(?=[A-Z])", "Scribe"),  # CamelCase: TestEvoSciMcpInstall
    (r"\bEvoSci\b", "metasci-scribe"),
]
_COMPILED_TEXT = [(re.compile(p), r) for p, r in TEXT_RULES]

# Bare identifiers (Python NAME tokens): cannot contain '-' or spaces.
NAME_EXACT = {
    "EvoScientist": "metasci_scribe",
    "evoscientist": "scribe",
    "evosci": "scribe",
    "EvoSci": "Scribe",
}


def rewrite_text(text: str) -> str:
    """Apply TEXT_RULES to ``text`` while shielding upstream URLs."""
    held: list[str] = []

    def hold(m: re.Match[str]) -> str:
        if not _PROTECT_ONLY_IF.search(m.group(0)):
            return m.group(0)
        held.append(m.group(0))
        return f"\x00{len(held) - 1}\x00"

    out = PROTECT.sub(hold, text)
    for pat, repl in _COMPILED_TEXT:
        out = pat.sub(repl, out)
    return re.sub(r"\x00(\d+)\x00", lambda m: held[int(m.group(1))], out)


def rewrite_name(name: str) -> str:
    if name in NAME_EXACT:
        return NAME_EXACT[name]
    return rewrite_text(name)


_FSTRING_MIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)


def rewrite_python(src: str, warnings: list[str], label: str) -> str:
    lines = io.StringIO(src, newline="\n").readlines()
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(pos: tuple[int, int]) -> int:
        return starts[pos[0] - 1] + pos[1]

    edits: list[tuple[int, int, str]] = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src, newline="\n").readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        warnings.append(f"{label}: tokenize failed ({exc}); used text rules")
        return rewrite_text(src)

    for tok in toks:
        kind = tok.type
        if kind not in (tokenize.NAME, tokenize.STRING, tokenize.COMMENT) and (
            _FSTRING_MIDDLE is None or kind != _FSTRING_MIDDLE
        ):
            continue
        lo, hi = offset(tok.start), offset(tok.end)
        piece = src[lo:hi]
        if kind == tokenize.NAME:
            new = rewrite_name(piece)
        else:
            if _FSTRING_MIDDLE is not None and kind == _FSTRING_MIDDLE:
                if piece != tok.string:
                    if re.search(r"evo", piece, re.IGNORECASE):
                        warnings.append(
                            f"{label}:{tok.start[0]}: f-string piece skipped"
                        )
                    continue
            if (
                _FSTRING_MIDDLE is not None
                and kind == _FSTRING_MIDDLE
                and src[hi : hi + 1] == "{"
            ):
                # f"EvoScientist{extras}==1.0" is a distribution name, not prose
                piece = re.sub(r"EvoScientist$", "metasci-scribe", piece)
            new = rewrite_text(piece)
        if new != src[lo:hi]:
            edits.append((lo, hi, new))

    for lo, hi, new in reversed(edits):
        src = src[:lo] + new + src[hi:]
    # ``EvoScientist.EvoScientist`` written as two NAME tokens (import / attribute chain)
    src = re.sub(r"\bmetasci_scribe\.metasci_scribe\b", "metasci_scribe.agent", src)
    # ``from .EvoScientist import x`` is NAME tokens too: the module became ``agent``
    src = re.sub(r"(from\s+\.+)metasci_scribe\b", r"\1agent", src)
    return re.sub(
        r"from metasci_scribe import metasci_scribe\b",
        "from metasci_scribe import agent",
        src,
    )


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    ).stdout.decode("utf-8")
    return [p for p in out.split("\0") if p]


def wanted(rel: str) -> bool:
    if rel in SKIP_FILES or rel.startswith(SKIP_PREFIXES):
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = ap.parse_args()

    warnings: list[str] = []
    changed = 0
    for rel in tracked_files():
        if not wanted(rel):
            continue
        path = ROOT / rel
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            src = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new = (
            rewrite_python(src, warnings, rel)
            if rel.endswith(".py")
            else rewrite_text(src)
        )
        if new != src:
            changed += 1
            if not args.dry_run:
                path.write_bytes(new.encode("utf-8"))

    print(f"{'would change' if args.dry_run else 'changed'} {changed} files")
    for w in warnings:
        print("WARN", w)

    if not args.dry_run:
        pkg_old, pkg_new = ROOT / "EvoScientist", ROOT / "metasci_scribe"
        if pkg_old.is_dir() and not pkg_new.exists():
            subprocess.run(
                ["git", "mv", "EvoScientist", "metasci_scribe"], cwd=ROOT, check=True
            )
        mod_old = pkg_new / "EvoScientist.py"
        if mod_old.exists():
            subprocess.run(
                [
                    "git",
                    "mv",
                    "metasci_scribe/EvoScientist.py",
                    "metasci_scribe/agent.py",
                ],
                cwd=ROOT,
                check=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
