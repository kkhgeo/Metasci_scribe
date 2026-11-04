"""Keep the drive letter of a mapped network drive when a path is resolved.

On Windows, ``os.path.realpath`` (and so ``Path.resolve``) turns a path on a
mapped network drive into the share's UNC name: ``Z:\\paper`` becomes
``\\\\RaiDrive-KEI\\Synology\\paper``. Some drive tools (RaiDrive among them)
publish a UNC name that Windows itself cannot open, and ``cmd.exe`` refuses
any UNC name as a working folder. The app resolves every path it guards, and
so do the file tools it builds on, so one such drive would break all of them.

``install`` wraps ``os.path.realpath`` once per process: a result under the
UNC root of a mapped drive is written with that drive letter instead. Both
sides of every comparison go through the same function, so the guards compare
like with like. Nothing changes off Windows, or for a UNC path no drive letter
maps.
"""

from __future__ import annotations

import ntpath
import os
import string
import threading
from collections.abc import Callable

__all__ = ["drive_letter_path", "install", "remote_roots"]

DRIVE_REMOTE = 4  # GetDriveTypeW

_original_realpath: Callable[..., str] | None = None
_roots: list[tuple[str, str]] | None = None  # (UNC root, casefolded) -> "Z:"
_roots_lock = threading.Lock()


def _drive_type(root: str) -> int:
    import ctypes

    return ctypes.windll.kernel32.GetDriveTypeW(root)


def _realpath(path: str) -> str:
    real = _original_realpath or ntpath.realpath
    return real(path)


def _unc_body(text: str) -> str | None:
    """``\\\\host\\share\\x`` for a UNC spelling of *text*, else None."""
    if text[:8].lower() == "\\\\?\\unc\\":
        return "\\\\" + text[8:]
    if text.startswith("\\\\") and text[:4] not in ("\\\\?\\", "\\\\.\\"):
        return text
    return None


def remote_roots() -> list[tuple[str, str]]:
    """``(UNC root, drive)`` for every mapped network drive, longest root
    first. Worked out once per process, on first need."""
    global _roots
    with _roots_lock:
        if _roots is None:
            found: list[tuple[str, str]] = []
            for letter in string.ascii_uppercase:
                drive = f"{letter}:"
                try:
                    if _drive_type(drive + "\\") != DRIVE_REMOTE:
                        continue
                    unc = _unc_body(_realpath(drive + "\\"))
                except (OSError, ValueError):
                    continue
                if unc:
                    found.append((unc.rstrip("\\"), drive))
            found.sort(key=lambda item: len(item[0]), reverse=True)
            _roots = found
        return _roots


def drive_letter_path(text: str) -> str:
    """*text* with a mapped drive letter in place of the UNC root it lies
    under. Any other text comes back as it was."""
    unc = _unc_body(text)
    if unc is None:
        return text
    low = unc.casefold()
    for root, drive in remote_roots():
        folded = root.casefold()
        if low == folded or low.startswith(folded + "\\"):
            return drive + (unc[len(root) :] or "\\")
    return text


def _realpath_with_drive_letters(path, *args, **kwargs):
    result = _original_realpath(path, *args, **kwargs)
    if isinstance(result, str) and result.startswith("\\\\"):
        return drive_letter_path(result)
    return result


def install() -> None:
    """Wrap ``os.path.realpath`` so resolved paths keep mapped drive letters.
    Once per process, Windows only."""
    global _original_realpath
    if os.name != "nt" or _original_realpath is not None:
        return
    _original_realpath = ntpath.realpath
    ntpath.realpath = _realpath_with_drive_letters
