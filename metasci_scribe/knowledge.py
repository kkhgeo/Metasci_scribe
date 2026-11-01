"""Text from the PDFs in ``Knowledge/``, for the agent to read.

``sync_knowledge`` walks ``Knowledge/`` (subfolders included) and writes the
text of each new or changed PDF to ``.metasci-scribe/extracted/``, with a
``[p. N]`` marker per page so a citation can point at a page. The agent reads
that folder with its ordinary file tools. They refuse every write there, so
only this module writes it. The PDF itself is never touched.

A PDF is untrusted input. Each one is read by a child process
(``_pdf_worker``) that is killed after ``MAX_SECONDS``; a file with more than
``MAX_PAGES`` pages or ``MAX_BYTES`` bytes is not read. A file that cannot be
read is reported and the others carry on. A PDF with no text (a scan) is
reported as such; there is no OCR.

What was read is kept next to the text in ``.index.json``: size and
modification time per source, and what came of it. A file whose size and time
are the same is not read again, whatever came of it (a scan or a broken file
is reported again, not retried). Output names are built from the path inside
``Knowledge/`` plus a short hash, so two files that differ only by case or
folder never meet, and no name can leave the folder.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from filelock import FileLock, Timeout

from . import policy
from .project import extract_dir

MAX_PAGES = 2000
MAX_SECONDS = 120
MAX_BYTES = 200 * 1024 * 1024
LOCK_WAIT_SECONDS = 300

INDEX_NAME = ".index.json"
LOCK_NAME = ".lock"
TMP_PREFIX = ".tmp-"
_INDEX_VERSION = 1
_SAVE_EVERY = 20
_WORKER = Path(__file__).with_name("_pdf_worker.py")

MB = 1024 * 1024
_STEM_CHARS = 60
_HASH_CHARS = 10
_SHOWN = 5
_SURROGATES = re.compile("[\ud800-\udfff]")
_BAD_CHARS = re.compile('[<>:"/\\\\|?*\x00-\x1f\x7f\ud800-\udfff]')
_RESERVED = frozenset(
    {"con", "prn", "aux", "nul", "conin$", "conout$"}
    | {f"com{n}" for n in "123456789¹²³"}
    | {f"lpt{n}" for n in "123456789¹²³"}
)


@dataclass
class SyncReport:
    """What a sync found.

    ``extracted`` are the PDFs read this time. ``unchanged`` counts the ones
    read before that did not change. ``scanned`` and ``failed`` list every
    PDF now known to hold no text or to be unreadable, including ones found
    so on an earlier run, so the person is told each time. All paths are
    those of the source PDFs.
    """

    extracted: list[Path]
    unchanged: int
    scanned: list[Path]
    failed: list[tuple[Path, str]]


@dataclass
class _Source:
    rel: str
    path: Path
    size: int
    mtime_ns: int
    outside: bool = False


# -- names -------------------------------------------------------------------


def _output_name(rel: str, taken: set[str]) -> str:
    """The extract's file name for the PDF at *rel* (a path inside Knowledge/).

    A readable part (the path without ``.pdf``, folders joined by ``__``, made
    safe for any file system) plus a hash of the exact path. *taken* holds
    the case-folded names already in use; a clash adds a counter.
    """
    shown = unicodedata.normalize("NFC", rel.replace("\\", "/"))
    if shown.casefold().endswith(".pdf"):
        shown = shown[:-4]
    parts = [p for p in shown.split("/") if p not in ("", ".", "..")]
    stem = "__".join(_BAD_CHARS.sub("_", p) for p in parts) or "_"
    stem = stem[:_STEM_CHARS].strip(" .").lstrip(".") or "_"
    if stem.split(".")[0].rstrip(" ").casefold() in _RESERVED:
        stem = "_" + stem
    digest = hashlib.sha256(rel.encode("utf-8", "surrogatepass")).hexdigest()
    name = f"{stem}.{digest[:_HASH_CHARS]}.md"
    counter = 1
    while name.casefold() in taken:
        counter += 1
        name = f"{stem}.{digest[:_HASH_CHARS]}-{counter}.md"
    return name


def _safe_out(name: object) -> bool:
    """Whether an index entry names a file this module may have written."""
    return (
        isinstance(name, str)
        and name.endswith(".md")
        and not name.startswith(".")
        and name == Path(name).name
        and "\\" not in name
        and "\x00" not in name
    )


# -- the index ---------------------------------------------------------------


def _load_index(out_dir: Path) -> dict[str, dict]:
    try:
        data = json.loads((out_dir / INDEX_NAME).read_text(encoding="utf-8"))
        files = data["files"]
        if data["version"] != _INDEX_VERSION or not isinstance(files, dict):
            return {}
        return {
            rel: entry
            for rel, entry in files.items()
            if isinstance(rel, str) and isinstance(entry, dict)
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _save_index(out_dir: Path, index: dict[str, dict]) -> None:
    tmp = out_dir / f"{TMP_PREFIX}index-{uuid.uuid4().hex}"
    try:
        tmp.write_text(
            json.dumps({"version": _INDEX_VERSION, "files": index}, indent=1),
            encoding="utf-8",
        )
        os.replace(tmp, out_dir / INDEX_NAME)
    finally:
        tmp.unlink(missing_ok=True)


def _remove_output(out_dir: Path, entry: dict | None) -> None:
    name = entry.get("out") if entry else None
    if _safe_out(name):
        try:
            (out_dir / name).unlink(missing_ok=True)
        except OSError:
            pass


# -- finding the PDFs --------------------------------------------------------


def _is_link(path: str) -> bool:
    if os.path.islink(path):
        return True
    junction = getattr(os.path, "isjunction", None)
    return bool(junction and junction(path))


def _inside(path: str, folder: str) -> bool:
    path, folder = os.path.normcase(path), os.path.normcase(folder)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def _discover(knowledge_dir: Path) -> tuple[list[_Source], bool]:
    """The PDFs under *knowledge_dir*, and whether the walk saw everything.

    Links to folders are not followed (a loop would never end). A link to a
    file outside the folder is listed with ``outside`` set, never read.
    """
    complete = True

    def on_error(_exc: OSError) -> None:
        nonlocal complete
        complete = False

    base = os.path.realpath(knowledge_dir)
    found: list[_Source] = []
    for folder, dirs, files in os.walk(knowledge_dir, onerror=on_error):
        dirs[:] = sorted(d for d in dirs if not _is_link(os.path.join(folder, d)))
        for name in sorted(files):
            if not name.casefold().endswith(".pdf"):
                continue
            full = os.path.join(folder, name)
            try:
                info = os.stat(full)
            except FileNotFoundError:
                continue
            except OSError:
                complete = False
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            rel = Path(full).relative_to(knowledge_dir).as_posix()
            outside = os.path.islink(full) and not _inside(os.path.realpath(full), base)
            found.append(
                _Source(rel, Path(full), info.st_size, info.st_mtime_ns, outside)
            )
    return found, complete


def _is_current(entry: dict | None, source: _Source, out_dir: Path) -> bool:
    if not entry or entry.get("status") not in ("ok", "scanned", "failed"):
        return False
    if entry.get("size") != source.size or entry.get("mtime_ns") != source.mtime_ns:
        return False
    if entry["status"] == "ok":
        name = entry.get("out")
        return _safe_out(name) and (out_dir / name).is_file()
    return True


# -- the child process -------------------------------------------------------


def _worker_command(pdf: Path, tmp: Path, label: str) -> list[str]:
    # -I: no user site, no PYTHON* variables, and the folder it runs in is not
    # on the import path.
    return [
        sys.executable,
        "-I",
        str(_WORKER),
        str(pdf),
        str(tmp),
        str(MAX_PAGES),
        label,
    ]


def _run_worker(
    pdf: Path, tmp: Path, label: str, cancel: threading.Event
) -> dict[str, object]:
    """Read *pdf* in a child process; the answer is the worker's JSON object.

    ``timeout`` and ``crashed`` are added here, and ``cancelled`` when *cancel*
    was set. The child is killed on a timeout or a cancel.
    """
    try:
        proc = subprocess.Popen(
            _worker_command(pdf, tmp, label),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=policy.child_python_env(os.environ),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, ValueError) as exc:
        return {"status": "unreadable", "error": f"the reader did not start ({exc})"}
    deadline = time.monotonic() + MAX_SECONDS
    while True:
        try:
            stdout, _ = proc.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired:
            if cancel.is_set():
                proc.kill()
                proc.communicate()
                return {"status": "cancelled"}
            if time.monotonic() >= deadline:
                proc.kill()
                proc.communicate()
                return {"status": "timeout"}
    lines = stdout.decode("utf-8", "replace").strip().splitlines()
    try:
        result = json.loads(lines[-1])
        if isinstance(result, dict) and isinstance(result.get("status"), str):
            return result
    except (IndexError, ValueError):
        pass
    return {
        "status": "crashed",
        "error": f"the reader stopped unexpectedly (exit code {proc.returncode})",
    }


# -- the sync ----------------------------------------------------------------


def sync_knowledge(
    root: Path,
    *,
    progress: Callable[[int, int, Path], None] | None = None,
    cancel: threading.Event | None = None,
) -> SyncReport:
    """Bring ``.metasci-scribe/extracted/`` up to date with ``Knowledge/``.

    *progress* is called as ``progress(n, total, pdf)`` before each PDF that
    needs reading (``total`` counts them all). *cancel* stops the sync between
    files and ends a read in progress; what was done is kept. A sync of the
    same folder that is already running is waited for; after
    ``LOCK_WAIT_SECONDS`` this one gives up with an empty report.

    Raises ``OSError`` when the extract folder cannot be used. A file that
    cannot be read never raises: it lands in ``SyncReport.failed``.
    """
    root = Path(root)
    knowledge_dir = root / "Knowledge"
    if not knowledge_dir.is_dir():
        return SyncReport([], 0, [], [])
    out_dir = extract_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    cancel = cancel or threading.Event()
    lock = FileLock(str(out_dir / LOCK_NAME), timeout=LOCK_WAIT_SECONDS)
    try:
        lock.acquire()
    except Timeout:
        return SyncReport([], 0, [], [])
    try:
        return _sync(knowledge_dir, out_dir, progress, cancel)
    finally:
        lock.release()


def _sync(
    knowledge_dir: Path,
    out_dir: Path,
    progress: Callable[[int, int, Path], None] | None,
    cancel: threading.Event,
) -> SyncReport:
    for leftover in out_dir.glob(f"{TMP_PREFIX}*"):
        try:
            leftover.unlink()
        except OSError:
            pass
    index = _load_index(out_dir)
    found, complete = _discover(knowledge_dir)
    todo = [s for s in found if not _is_current(index.get(s.rel), s, out_dir)]

    extracted: list[Path] = []
    transient: dict[str, str] = {}
    unsaved = 0
    try:
        for number, source in enumerate(todo, 1):
            if cancel.is_set():
                break
            if progress is not None:
                try:
                    progress(number, len(todo), source.path)
                except Exception:
                    pass
            outcome = _read_one(source, out_dir, index, cancel)
            if outcome == "cancelled":
                break
            if outcome == "ok":
                extracted.append(source.path)
            elif outcome is not None:
                transient[source.rel] = outcome
            unsaved += 1
            if unsaved >= _SAVE_EVERY:
                _save_index(out_dir, index)
                unsaved = 0
    finally:
        if complete:
            seen = {s.rel for s in found}
            for rel in [r for r in index if r not in seen]:
                _remove_output(out_dir, index.pop(rel))
        try:
            _save_index(out_dir, index)
        except OSError:
            pass

    unchanged = 0
    scanned: list[Path] = []
    failed: list[tuple[Path, str]] = []
    done = set(extracted)
    for source in found:
        entry = index.get(source.rel)
        if source.rel in transient:
            failed.append((source.path, transient[source.rel]))
        elif not entry or not _is_current(entry, source, out_dir):
            continue
        elif entry["status"] == "scanned":
            scanned.append(source.path)
        elif entry["status"] == "failed":
            failed.append((source.path, str(entry.get("error", "could not be read"))))
        elif source.path not in done:
            unchanged += 1
    return SyncReport(extracted, unchanged, scanned, failed)


def _read_one(
    source: _Source, out_dir: Path, index: dict[str, dict], cancel: threading.Event
) -> str | None:
    """Read one PDF and record what came of it in *index*.

    Returns ``"ok"`` for extracted text, ``"cancelled"``, ``None`` when there
    is nothing to report (scanned, failed for good, or gone), or a reason
    when the failure may pass (the file is locked, the disk refused): that one
    is not remembered, so the next sync tries again.
    """
    old = index.get(source.rel)
    base = {"size": source.size, "mtime_ns": source.mtime_ns}

    def settle(entry: dict | None) -> None:
        if entry is None:
            index.pop(source.rel, None)
        else:
            index[source.rel] = entry
        if not old or not entry or entry.get("out") != old.get("out"):
            _remove_output(out_dir, old)

    def fail(reason: str) -> None:
        settle({**base, "status": "failed", "error": reason})

    if source.outside:
        fail("link to a file outside Knowledge/, not read")
        return None
    if source.size > MAX_BYTES:
        fail(f"too large ({source.size // MB} MB, the limit is {MAX_BYTES // MB} MB)")
        return None

    tmp = out_dir / f"{TMP_PREFIX}{uuid.uuid4().hex}"
    label = "Knowledge/" + _SURROGATES.sub("?", source.rel)
    try:
        result = _run_worker(source.path, tmp, label, cancel)
        status = result.get("status")
        error = str(result.get("error", "could not be read"))
        if status == "cancelled":
            return "cancelled"
        if status == "missing":
            settle(None)
            return None
        if status == "timeout":
            fail(f"took longer than {MAX_SECONDS} s, stopped")
            return None
        if status in ("failed", "crashed"):
            fail(error)
            return None
        if status == "scanned":
            settle({**base, "status": "scanned"})
            return None
        if status != "ok":
            return error
        taken = {
            e["out"].casefold()
            for r, e in index.items()
            if r != source.rel and _safe_out(e.get("out"))
        }
        if old and _safe_out(old.get("out")):
            name = old["out"]
        else:
            name = _output_name(source.rel, taken)
        try:
            os.replace(tmp, out_dir / name)
        except OSError as exc:
            return f"the text could not be saved ({exc})"
        settle({**base, "status": "ok", "out": name, "pages": result.get("pages")})
        return "ok"
    finally:
        tmp.unlink(missing_ok=True)


# -- what to tell the person ---------------------------------------------------


def _name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root / "Knowledge").as_posix()
    except ValueError:
        return path.name


def _pdfs(count: int) -> str:
    return f"{count} PDF" + ("" if count == 1 else "s")


def _more(count: int) -> str:
    return f" and {count - _SHOWN} more" if count > _SHOWN else ""


def report_messages(report: SyncReport, root: Path) -> list[str]:
    """Plain-text lines that tell the person what a sync did, or none.

    Nothing is said when every PDF was read before and nothing is wrong.
    """
    lines: list[str] = []
    if report.extracted:
        lines.append(
            f"Knowledge: read the text of {_pdfs(len(report.extracted))} for the agent."
        )
    if report.scanned:
        names = ", ".join(_name(p, root) for p in report.scanned[:_SHOWN])
        lines.append(
            f"Knowledge: no text found in {_pdfs(len(report.scanned))} "
            "(scanned images? OCR is not supported, so the agent cannot read "
            f"them): {names}{_more(len(report.scanned))}"
        )
    if report.failed:
        shown = "; ".join(
            f"{_name(path, root)} ({reason})" for path, reason in report.failed[:_SHOWN]
        )
        lines.append(
            f"Knowledge: could not read {_pdfs(len(report.failed))}: "
            f"{shown}{_more(len(report.failed))}"
        )
    return lines
