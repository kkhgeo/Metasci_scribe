"""The paper folder: what ``/init`` creates, and the network-path check.

A paper folder holds ``program.md`` (settings), ``outline.md`` (the map),
``manuscript.md``, ``Knowledge/``, ``Figures/``, ``Tables/`` and the app's own
``.metasci-scribe/``. ``init_project`` creates what is missing and never
touches what exists.

Network folders are not supported: ``Path.resolve()`` turns a mapped drive
such as ``Z:`` into a UNC path, which Windows path handling and the shell
cannot all deal with. ``is_network_path`` finds both spellings so the app can
stop with a clear message instead of failing later with strange path errors.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path, PurePath, PureWindowsPath

from . import paths

# GetDriveTypeW return values (winbase.h).
DRIVE_NO_ROOT_DIR = 1
DRIVE_REMOVABLE = 2
DRIVE_FIXED = 3
DRIVE_REMOTE = 4

_IS_WINDOWS = os.name == "nt"
_DRIVE = re.compile(r"([A-Za-z]):")

STATE_DIR_NAME = ".metasci-scribe"

NETWORK_PATH_MESSAGE = (
    "{path} is on a network drive or share, which Metasci Scribe does not "
    "support. Copy the paper folder to a local disk (for example C:\\papers) "
    "and open it there."
)


class NetworkPathError(RuntimeError):
    """The folder is on a network drive or share."""


@dataclass
class InitResult:
    """What ``init_project`` did: new items, and existing ones it left alone."""

    created: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)


def _drive_type(root: str) -> int:
    """``GetDriveTypeW`` for a drive root such as ``Z:\\`` (Windows only)."""
    import ctypes

    return int(ctypes.windll.kernel32.GetDriveTypeW(root))  # type: ignore[attr-defined]


def is_network_path(path: str | os.PathLike[str] | PurePath) -> bool:
    """Whether *path* is a UNC path or lies on a mapped network drive.

    ``\\\\host\\share\\...``, ``//host/share/...`` and ``\\\\?\\UNC\\...`` are
    network paths. A drive letter is asked from Windows (``DRIVE_REMOTE``). A
    relative path counts by where it lands from the current folder. Other
    platforms always answer False. The path need not exist.
    """
    if not (_IS_WINDOWS or isinstance(path, PureWindowsPath)):
        return False
    text = str(path).replace("/", "\\")
    low = text.lower()
    if low.startswith("\\\\?\\unc\\"):
        return True
    if low.startswith(("\\\\?\\", "\\\\.\\")):
        text = text[4:]
    elif text.startswith("\\\\"):
        return True
    match = _DRIVE.match(text)
    if match is None:
        if not _IS_WINDOWS or PureWindowsPath(text).is_absolute():
            return False
        # No drive: relative to the current folder, which may be a network one.
        landed = os.path.abspath(text)
        if landed.replace("/", "\\") == text:
            return False
        return is_network_path(landed)
    if not _IS_WINDOWS:
        return False
    return _drive_type(match.group(1).upper() + ":\\") == DRIVE_REMOTE


def require_local_path(path: str | os.PathLike[str] | PurePath) -> None:
    """Raise ``NetworkPathError`` with the local-disk message for a network path."""
    if is_network_path(path):
        raise NetworkPathError(NETWORK_PATH_MESSAGE.format(path=path))


def extract_dir(root: Path) -> Path:
    """Where text extracted from ``Knowledge/`` PDFs is kept."""
    return root / STATE_DIR_NAME / "extracted"


def _template(name: str) -> str:
    return (
        resources.files("metasci_scribe")
        .joinpath("templates", name)
        .read_text(encoding="utf-8")
    )


def _write_new(path: Path, text: str) -> bool:
    """Create *path* with *text* unless something is already there.

    A link, even a dangling one, counts as something there: Windows would
    follow it and create its target. Exclusive creation covers a file made
    between the check and the write.
    """
    if os.path.lexists(path):
        return False
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            f.write(text)
    except FileExistsError:
        return False
    return True


def _make_dir(path: Path) -> bool:
    try:
        path.mkdir()
    except FileExistsError:
        return False
    return True


def init_project(root: Path) -> InitResult:
    """Create the paper folder layout in *root*, skipping what already exists.

    Raises ``NetworkPathError`` for a network folder and
    ``paths.WorkspaceRootError`` for one that holds the app's data (the home
    folder, say). Nothing is written in either case. Existing files and
    folders are listed in ``skipped`` and left exactly as they are.
    """
    require_local_path(root)
    paths.require_workspace_root(root)
    root = Path(root)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"{root} is not a folder")
    root.mkdir(parents=True, exist_ok=True)

    result = InitResult()
    files = {
        "program.md": _template("program.md"),
        "outline.md": _template("outline.md"),
        "manuscript.md": "",
    }
    for name, text in files.items():
        target = root / name
        (result.created if _write_new(target, text) else result.skipped).append(target)
    for name in ("Knowledge", "Figures", "Tables", STATE_DIR_NAME):
        target = root / name
        (result.created if _make_dir(target) else result.skipped).append(target)
    return result
