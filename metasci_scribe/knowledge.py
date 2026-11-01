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
_MAX_ATTEMPTS = 1000
_IS_WINDOWS = os.name == "nt"
_PATH_LIMIT = 260
_TOO_LONG = "path too long for Windows; shorten the folder or file names"
_OUTSIDE = "link to a file outside Knowledge/, not read"
_WORKER = Path(__file__).with_name("_pdf_worker.py")

MB = 1024 * 1024
_WORKER_GRACE = 10  # the worker stops itself this long after the parent would
_STEM_CHARS = 60
_HASH_CHARS = 10
_SHOWN = 5
_OUT_NAME = re.compile(rf".+\.([0-9a-f]{{{_HASH_CHARS}}})(?:-\d+)?\.md")
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
    those of the source PDFs. ``busy`` means another sync of the same folder
    held the lock for the whole wait: nothing was done, which is not the
    same as nothing to do.
    """

    extracted: list[Path]
    unchanged: int
    scanned: list[Path]
    failed: list[tuple[Path, str]]
    busy: bool = False


@dataclass
class _Source:
    rel: str
    path: Path
    size: int
    mtime_ns: int
    problem: str | None = None


# -- names -------------------------------------------------------------------


def _digest(rel: str) -> str:
    """The hash in the output name of the PDF at *rel*."""
    return hashlib.sha256(rel.encode("utf-8", "surrogatepass")).hexdigest()[
        :_HASH_CHARS
    ]


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
    digest = _digest(rel)
    name = f"{stem}.{digest}.md"
    counter = 1
    while name.casefold() in taken:
        counter += 1
        name = f"{stem}.{digest}-{counter}.md"
    return name


def _safe_out(name: object) -> bool:
    """Whether an index entry names a file this module may have written:
    one of the names ``_output_name`` makes (a stem, a hash, ``.md``)."""
    return (
        isinstance(name, str)
        and _OUT_NAME.fullmatch(name) is not None
        and name == Path(name).name
        and not name.startswith(".")
        and "\\" not in name
        and "\x00" not in name
    )


def _owns(rel: str, name: object) -> bool:
    """Whether *name* is an output name that the PDF at *rel* could have been
    given: ours in form, with the hash of that very path."""
    if not _safe_out(name):
        return False
    match = _OUT_NAME.fullmatch(name)
    return match is not None and match.group(1) == _digest(rel)


# -- the index ---------------------------------------------------------------


def _load_index(out_dir: Path) -> dict[str, dict]:
    """The saved index, or an empty one when it is missing or damaged.

    An entry that does not look like one this module wrote is dropped, and
    its source is read again: an ``ok`` entry whose output name is not the
    one its own path gets, or that a second entry also claims; a scan or a
    failure that names an output at all. The file is data that anything with
    write access can change, so nothing in it is trusted, and a name taken
    from it is never used to delete a file unless it passed this.
    """
    try:
        data = json.loads((out_dir / INDEX_NAME).read_text(encoding="utf-8"))
        files = data["files"]
        if data["version"] != _INDEX_VERSION or not isinstance(files, dict):
            return {}
    except (OSError, ValueError, KeyError, TypeError, RecursionError):
        return {}
    index: dict[str, dict] = {}
    claimed: set[str] = set()
    for rel, entry in files.items():
        if not isinstance(rel, str) or not isinstance(entry, dict):
            continue
        if "out" in entry:
            name = entry["out"]
            if (
                entry.get("status") != "ok"
                or not _owns(rel, name)
                or name.casefold() in claimed
            ):
                continue
            claimed.add(name.casefold())
        elif entry.get("status") == "ok":
            continue
        attempts = entry.get("attempts")
        if "attempts" in entry and not (
            isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and 0 <= attempts <= _MAX_ATTEMPTS
        ):
            del entry["attempts"]
        index[rel] = entry
    return index


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


def _remove_output(out_dir: Path, rel: str, entry: dict | None) -> None:
    """Delete the output of *entry*, a source's index entry, if the name is
    one that source's path gets."""
    name = entry.get("out") if entry else None
    if _owns(rel, name):
        try:
            (out_dir / name).unlink(missing_ok=True)
        except OSError:
            pass


# -- finding the PDFs --------------------------------------------------------


_REPARSE_POINT = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT
# Reparse tags that send a path somewhere else. Cloud placeholders (OneDrive and
# the like) are reparse points too, but they are the folder itself: followed.
_LINK_TAGS = frozenset(
    {
        0xA000000C,  # IO_REPARSE_TAG_SYMLINK
        0xA0000003,  # IO_REPARSE_TAG_MOUNT_POINT (a junction)
    }
)


def _is_link(path: str) -> bool:
    """A symbolic link or, on Windows, a junction or mount point.

    ``os.path.isjunction`` exists only from Python 3.12, so the reparse tag
    is read instead (``st_reparse_tag``, Windows, 3.8+). A reparse point of
    another kind is followed; a file must still resolve inside the folder.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return True  # cannot be examined: not to be walked into
    if stat.S_ISLNK(info.st_mode):
        return True
    if not getattr(info, "st_file_attributes", 0) & _REPARSE_POINT:
        return False
    tag = getattr(info, "st_reparse_tag", None)
    return tag is None or tag in _LINK_TAGS  # no tag known: be careful


def _inside(path: str, folder: str) -> bool:
    path, folder = os.path.normcase(path), os.path.normcase(folder)
    return path == folder or path.startswith(folder.rstrip(os.sep) + os.sep)


def _discover(knowledge_dir: Path) -> tuple[list[_Source], bool]:
    """The PDFs under *knowledge_dir*, and whether the walk saw everything.

    Links and junctions to folders are not followed (a loop would never end,
    and the target may be anywhere). A file whose real path lies outside the
    folder (a link to it, or a path that goes through one) is listed with a
    ``problem`` and never read. A path the system cannot open is a problem
    too, not a file that went away.
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
                if _IS_WINDOWS and len(full) >= _PATH_LIMIT:
                    rel = Path(full).relative_to(knowledge_dir).as_posix()
                    found.append(_Source(rel, Path(full), 0, 0, _TOO_LONG))
                continue
            except OSError:
                complete = False
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            rel = Path(full).relative_to(knowledge_dir).as_posix()
            problem = None
            if not _inside(os.path.realpath(full), base):
                problem = _OUTSIDE
            found.append(
                _Source(rel, Path(full), info.st_size, info.st_mtime_ns, problem)
            )
    return found, complete


def _is_valid(entry: dict | None, source: _Source, out_dir: Path) -> bool:
    """Whether *entry* describes *source* as it is now (and its text is there)."""
    if not entry or entry.get("status") not in ("ok", "scanned", "failed"):
        return False
    if entry.get("size") != source.size or entry.get("mtime_ns") != source.mtime_ns:
        return False
    if entry["status"] == "ok":
        name = entry.get("out")
        return _safe_out(name) and (out_dir / name).is_file()
    return True


def _is_current(entry: dict | None, source: _Source, out_dir: Path) -> bool:
    """Whether *source* needs no reading: a timeout or crash is tried once more."""
    return _is_valid(entry, source, out_dir) and not entry.get("retry")


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
        str(MAX_SECONDS + _WORKER_GRACE),
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
    except (OSError, ValueError) as exc:  # UnicodeEncodeError is a ValueError
        return {
            "status": "unreadable",
            "error": f"the reader could not be started for this file name ({exc})",
        }
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
    wait: float | None = None,
) -> SyncReport:
    """Bring ``.metasci-scribe/extracted/`` up to date with ``Knowledge/``.

    *progress* is called as ``progress(n, total, pdf)`` before each PDF that
    needs reading (``total`` counts them all). *cancel* stops the sync between
    files and ends a read in progress; what was done is kept. A sync of the
    same folder that is already running is waited for, *wait* seconds
    (``LOCK_WAIT_SECONDS`` by default; ``0`` does not wait). If it is still
    running then, nothing is done and the report has ``busy`` set: a caller
    that needs the extracts, such as ``run``, treats that as a failure to
    start.

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
    lock = FileLock(str(out_dir / LOCK_NAME))
    deadline = time.monotonic() + (LOCK_WAIT_SECONDS if wait is None else wait)
    while True:
        try:
            lock.acquire(timeout=0.1, poll_interval=0.05)
            break
        except Timeout:
            if cancel.is_set():
                return SyncReport([], 0, [], [])
            if time.monotonic() >= deadline:
                return SyncReport([], 0, [], [], busy=True)
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
                _remove_output(out_dir, rel, index.pop(rel))
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
        elif not _is_valid(entry, source, out_dir):
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
            _remove_output(out_dir, source.rel, old)

    def fail(reason: str, *, again: bool = False) -> None:
        """Remember a failure. With *again* (the reader hung or died, which
        a busy machine can cause) it is tried once more on a later sync."""
        entry = {**base, "status": "failed", "error": reason}
        if again:
            same = old and (old.get("size"), old.get("mtime_ns")) == (
                source.size,
                source.mtime_ns,
            )
            before = old.get("attempts") if same else 0
            tries = (before if isinstance(before, int) else 1) + 1
            entry["attempts"] = tries
            entry["retry"] = tries < 2
        settle(entry)

    if source.problem:
        fail(source.problem)
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
            if _IS_WINDOWS and len(str(source.path)) >= _PATH_LIMIT:
                fail(_TOO_LONG)
            else:
                settle(None)
            return None
        if status == "timeout":
            fail(f"took longer than {MAX_SECONDS} s, stopped", again=True)
            return None
        if status == "crashed":
            fail(error, again=True)
            return None
        if status == "failed":
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
        if old and _owns(source.rel, old.get("out")):
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
    if report.busy:
        lines.append(
            "Knowledge: another Metasci Scribe is reading the PDFs of this "
            "folder; their text is not up to date yet."
        )
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
