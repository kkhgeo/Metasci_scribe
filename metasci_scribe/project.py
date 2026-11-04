"""The paper folder: what ``/init`` creates, and the network-path check.

A paper folder holds ``program.md`` (settings), ``outline.md`` (the map),
``Manuscript/`` (one file per outline section), ``Knowledge/``, ``Figures/``,
``Tables/`` and the app's own ``.metasci-scribe/``. ``manuscript.md``, the
joined draft, is written by the app later (:mod:`metasci_scribe.manuscript`).
``init_project`` creates what is missing and never touches what exists.

A network drive with a letter (``Z:``) works like a local one:
:mod:`metasci_scribe.drive_paths` keeps the letter when a path is resolved. A
share reached only by its UNC name (``\\\\host\\share``) does not:
``cmd.exe`` cannot use it as a working folder, and some drive tools publish a
UNC name Windows cannot open. ``is_network_path`` finds those so the app can
stop with a clear message instead of failing later with strange path errors.
"""

from __future__ import annotations

import contextlib
import os
import re
import uuid
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path, PurePath, PureWindowsPath

from . import paths
from .drive_paths import drive_letter_path
from .manuscript import MANUSCRIPT_DIR, section_file_name, section_names

_IS_WINDOWS = os.name == "nt"
_DRIVE = re.compile(r"([A-Za-z]):")

STATE_DIR_NAME = ".metasci-scribe"

NETWORK_PATH_MESSAGE = (
    "{path} is a network share without a drive letter, which Metasci Scribe "
    "does not support. Map the share to a drive letter (for example Z:) and "
    "open the folder through it, or copy the folder to a local disk."
)


class NetworkPathError(RuntimeError):
    """The folder is a network share reached without a drive letter."""


@dataclass
class InitResult:
    """What ``init_project`` did.

    ``created`` are new items, ``skipped`` existing ones of the right kind that
    were left alone, ``problems`` existing items of the wrong kind (a file where
    a folder belongs, or the reverse), which were left alone and cannot be used.
    """

    created: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    problems: list[Path] = field(default_factory=list)


class InitError(RuntimeError):
    """``init_project`` hit a file system error part way through.

    ``result`` lists what had been created before it stopped.
    """

    def __init__(self, message: str, result: InitResult) -> None:
        super().__init__(message)
        self.result = result


def is_network_path(path: str | os.PathLike[str] | PurePath) -> bool:
    """Whether *path* is a network share reached without a drive letter.

    ``\\\\host\\share\\...``, ``//host/share/...`` and ``\\\\?\\UNC\\...`` are,
    unless a mapped drive letter covers the share. A path with a drive letter
    never is, network drive or not. A relative path counts by where it lands
    from the current folder. Other platforms always answer False. The path need
    not exist.
    """
    if not (_IS_WINDOWS or isinstance(path, PureWindowsPath)):
        return False
    text = str(path).replace("/", "\\")
    low = text.lower()
    if low.startswith("\\\\?\\unc\\") or (
        text.startswith("\\\\") and not low.startswith(("\\\\?\\", "\\\\.\\"))
    ):
        return not _IS_WINDOWS or drive_letter_path(text) == text
    if low.startswith(("\\\\?\\", "\\\\.\\")):
        text = text[4:]
    if _DRIVE.match(text) or not _IS_WINDOWS or PureWindowsPath(text).is_absolute():
        return False
    # No drive: relative to the current folder, which may be a network one.
    landed = os.path.abspath(text)
    if landed.replace("/", "\\") == text:
        return False
    return is_network_path(landed)


def require_local_path(path: str | os.PathLike[str] | PurePath) -> None:
    """Raise ``NetworkPathError`` with the local-disk message for a network path."""
    if is_network_path(path):
        raise NetworkPathError(NETWORK_PATH_MESSAGE.format(path=path))


def stored_network_path(*stored: str | None) -> str | None:
    """The first of the stored folder spellings that is a network path.

    For folders read back from a stored thread. Judged by the stored text
    alone, so nothing touches the share (a dead one can hang a stat for a
    long time).
    """
    for value in stored:
        if value and is_network_path(value):
            return value
    return None


def require_local_stored(*stored: str | None) -> None:
    """Raise ``NetworkPathError`` if any stored folder is on a network."""
    bad = stored_network_path(*stored)
    if bad is not None:
        raise NetworkPathError(NETWORK_PATH_MESSAGE.format(path=bad))


def extract_dir(root: Path) -> Path:
    """Where text extracted from ``Knowledge/`` PDFs is kept."""
    return root / STATE_DIR_NAME / "extracted"


def _template(name: str) -> str:
    return (
        resources.files("metasci_scribe")
        .joinpath("templates", name)
        .read_text(encoding="utf-8")
    )


def _write_temp(path: Path, text: str) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _publish(tmp: Path, target: Path) -> None:
    """Move *tmp* to *target*, failing with ``FileExistsError`` if it exists."""
    if os.name == "nt":
        os.rename(tmp, target)  # never replaces on Windows
        return
    try:
        os.link(tmp, target)  # never replaces
    except FileExistsError:
        raise
    except OSError:
        # No hard links here (some removable file systems).
        if os.path.lexists(target):
            raise FileExistsError(str(target)) from None
        os.rename(tmp, target)
        return
    os.unlink(tmp)


def _write_new(path: Path, text: str) -> bool:
    """Create *path* with *text* unless something is already there.

    The text goes to a temporary file in the same folder first and is moved
    into place whole, so a write that fails leaves nothing at *path* for a
    later ``/init`` to mistake for the real file. A link, even a dangling one,
    counts as something there: Windows would follow it and create its target.
    """
    if os.path.lexists(path):
        return False
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        _write_temp(tmp, text)
        _publish(tmp, path)
    except FileExistsError:
        return False
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    return True


def _make_dir(path: Path) -> bool:
    try:
        path.mkdir()
    except FileExistsError:
        return False
    return True


def problem_reason(path: Path) -> str:
    """Why an existing *path* cannot serve as the layout item it is named for."""
    is_folder_item = path.name in _FOLDERS
    if is_folder_item:
        return "is a file, not a folder"
    if path.is_dir():
        return "is a folder, not a file"
    return "is not a regular file (a broken link?)"


_FOLDERS = ("Knowledge", "Figures", "Tables", MANUSCRIPT_DIR, STATE_DIR_NAME)


def init_project(root: Path) -> InitResult:
    """Create the paper folder layout in *root*, skipping what already exists.

    Raises ``NetworkPathError`` for a network folder and
    ``paths.WorkspaceRootError`` for one that holds the app's data (the home
    folder, say). Nothing is written in either case. Existing items of the
    right kind are listed in ``skipped``, existing items of the wrong kind in
    ``problems``; both are left exactly as they are. A file system error
    raises ``InitError`` carrying what had been created so far.
    """
    require_local_path(root)
    paths.require_workspace_root(root)
    root = Path(root)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"{root} is not a folder")

    result = InitResult()
    outline = _template("outline.md")
    files = {
        "program.md": _template("program.md"),
        "outline.md": outline,
    }
    target = root
    try:
        root.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            target = root / name
            if _write_new(target, text):
                result.created.append(target)
            elif target.is_file():
                result.skipped.append(target)
            else:
                result.problems.append(target)
        for name in _FOLDERS:
            target = root / name
            if _make_dir(target):
                result.created.append(target)
            elif target.is_dir():
                result.skipped.append(target)
            else:
                result.problems.append(target)
        # An empty file per section, only for the outline made just now: an
        # outline the author already had may name other sections.
        if (
            root / "outline.md" in result.created
            and root / MANUSCRIPT_DIR in result.created
        ):
            for section in section_names(outline):
                target = root / MANUSCRIPT_DIR / section_file_name(section)
                if _write_new(target, ""):
                    result.created.append(target)
    except OSError as exc:
        raise InitError(f"Could not create {target}: {exc}", result) from exc
    return result
