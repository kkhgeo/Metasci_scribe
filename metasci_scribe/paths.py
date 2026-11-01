"""Path resolution utilities for Metasci Scribe runtime directories."""

from __future__ import annotations

import functools
import logging
import os
import os.path as ospath
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkspaceRootError(ValueError):
    """A folder that holds the app's own data cannot be a workspace."""


def normalize_path(path: str | Path) -> Path:
    """Return the canonical absolute form of *path*.

    Expands ``~`` and resolves symlinks, so ``/tmp/x`` and ``/private/tmp/x``
    on macOS, or two spellings of a symlinked project folder, compare equal.
    The path does not need to exist. Raises ``ValueError`` for a path with a
    NUL byte, which no platform accepts (Windows on Python 3.13 would
    otherwise resolve it without complaint).
    """
    if "\x00" in str(path):
        raise ValueError(f"Path contains a NUL byte: {path!r}")
    return Path(path).expanduser().resolve()


def file_tool_path(path: str | Path) -> str:
    """Spell a host path so deepagents' virtual-path validator accepts it.

    Windows drive paths need a leading slash (``/C:/project/file``). The
    sandbox decodes that alias; shell commands still use ``C:/project/file``.
    POSIX paths and paths that are already virtual are unchanged.
    """
    value = str(path)
    if re.match(r"^[A-Za-z]:[\\/]", value):
        return "/" + value.replace("\\", "/")
    return value


def host_file_tool_path(path: str) -> str:
    """On Windows, use forward slashes for every path and decode `/C:/...` aliases.

    Other platforms get *path* back unchanged.
    """
    if os.name == "nt":
        if re.match(r"^/[A-Za-z]:(/|$)", path):
            path = f"{path[1:3]}/{path[4:]}"
        return path.replace("\\", "/")
    return path


def _is_inside(path: str | Path, folder: str | Path) -> bool:
    """Whether *path* is *folder* or lies under it.

    By name, without regard to case, since Windows and macOS file systems
    ignore it; then by file identity, up the parents, for a spelling the name
    check cannot see: ``realpath`` keeps a ``//?/`` or ``//localhost/C$/`` prefix.
    """
    target = ospath.realpath(path)
    root = ospath.realpath(folder)
    name, root_name = target.casefold(), root.casefold()
    if name == root_name or name.startswith(root_name.rstrip(ospath.sep) + ospath.sep):
        return True
    try:
        root_stat = Path(root).stat()
    except OSError:
        return False
    candidate = Path(target)
    for ancestor in (candidate, *candidate.parents):
        try:
            if ospath.samestat(ancestor.stat(), root_stat):
                return True
        except OSError:
            continue
    return False


def is_auth_path(path: str | Path) -> bool:
    """Whether *path*, links followed, is the sign-in folder or lies inside it.

    A path that cannot be examined counts as inside.
    """
    try:
        return _is_inside(path, DATA_DIR / "auth")
    except (OSError, ValueError, RuntimeError):
        return True


def contains_auth_dir(path: str | Path) -> bool:
    """Whether *path* is the sign-in folder, lies inside it, or holds it."""
    try:
        return is_auth_path(path) or _is_inside(DATA_DIR / "auth", path)
    except (OSError, ValueError, RuntimeError):
        return True


def holds_data_dir(root: str | Path) -> bool:
    """Whether *root* is the app's data folder or a parent of it."""
    return _is_inside(DATA_DIR, root)


def require_workspace_root(root: str | Path) -> None:
    """Refuse a folder that holds the app's data, such as the home folder.

    The agent reads and writes throughout its workspace, so this keeps the
    sign-in and other app state out of its way before any guard is needed.
    Separate from :class:`Workspace`, which also names stored folders that
    are only compared; call it where a workspace is put to use.
    """
    if holds_data_dir(root):
        raise WorkspaceRootError(
            f"{root} cannot be a workspace: it contains Metasci Scribe's data "
            f"folder ({DATA_DIR}), which holds your sign-in. Open a paper "
            f"folder instead."
        )


@dataclass(frozen=True)
class Workspace:
    """A project folder: the unit that owns skills, runs, media and memory.

    Build one where a workspace enters the process (CLI startup, ``/resume``,
    a server graph build) and pass it to whatever needs it.
    Constructing a ``Workspace`` never touches the filesystem beyond resolving
    the path, and never creates the folder.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", normalize_path(self.root))

    @property
    def state_dir(self) -> Path:
        """Project state folder (``<root>/.metasci-scribe``)."""
        return self.root / ".metasci-scribe"

    @property
    def skills_dir(self) -> Path:
        """Workspace skills tier (``<root>/.metasci-scribe/skills``)."""
        return self.state_dir / "skills"

    @property
    def runs_dir(self) -> Path:
        """Parent folder of ``--mode=run`` session folders (``<root>/runs``)."""
        return self.root / "runs"

    @property
    def key(self) -> str:
        """Stable string form for storage and comparison (POSIX separators)."""
        return self.root.as_posix()


# Name ``--mode=run`` gives a run folder when the user passes no ``--name``.
RUN_NAME_FORMAT = "%Y%m%d_%H%M%S"
_RUN_NAME_SHAPE = re.compile(r"\d{8}_\d{6}")


def is_generated_run_name(name: str) -> bool:
    """True if *name* is a run folder name ``--mode=run`` generates."""
    if not _RUN_NAME_SHAPE.fullmatch(name):
        return False
    try:
        datetime.strptime(name, RUN_NAME_FORMAT)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SessionDirs:
    """Where a session lives: its workspace, and the run folder it works in.

    ``run_dir`` is set only in ``--mode=run``, where the session works in a
    folder under ``<root>/runs``. Skills, media, the memory project and crons
    always come from the workspace; the sandbox works in :attr:`work_dir`.
    """

    workspace: Workspace
    run_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.run_dir is None:
            return
        run_dir = normalize_path(self.run_dir)
        object.__setattr__(
            self, "run_dir", None if run_dir == self.workspace.root else run_dir
        )

    @property
    def work_dir(self) -> Path:
        """The folder the agent works in."""
        return self.run_dir or self.workspace.root

    def metadata(self) -> dict[str, str]:
        """Thread metadata and ``configurable`` keys for this session.

        ``run_dir`` is left out, not set to an empty string, when the session
        works in the workspace root.
        """
        fields = {"workspace_dir": self.workspace.key}
        if self.run_dir is not None:
            fields["run_dir"] = self.run_dir.as_posix()
        return fields

    @classmethod
    def from_stored(
        cls, workspace_dir: str | Path | None, run_dir: str | Path | None = None
    ) -> SessionDirs | None:
        """Read the folders a thread was stored with, normalised.

        Returns ``None`` when nothing was stored or the value cannot be
        resolved. A ``run_dir`` outside the workspace is dropped: run folders
        live under ``<root>/runs``. Resolves paths, so call it off the event
        loop on the server.
        """
        if not workspace_dir:
            return None
        try:
            dirs = cls(Workspace(workspace_dir), Path(run_dir) if run_dir else None)
        except (TypeError, ValueError, OSError, RuntimeError):
            return None
        if dirs.run_dir is not None and not dirs.run_dir.is_relative_to(
            dirs.workspace.root
        ):
            return cls(dirs.workspace)
        return dirs

    @classmethod
    def from_legacy(cls, workspace_dir: str | Path) -> SessionDirs:
        """Read a ``workspace_dir`` stored before ``run_dir`` existed.

        Run-mode sessions stored their run folder as ``workspace_dir``. A
        folder ``<root>/runs/<name>`` whose name has the form ``--mode=run``
        generates is split into the workspace and ``run_dir``. Named runs are
        left as they are, because a project folder can itself live in a
        folder called ``runs``. Only the one-time upgrade of stored values
        uses this; new values carry ``run_dir`` themselves.
        """
        path = normalize_path(workspace_dir)
        if path.parent.name == "runs" and is_generated_run_name(path.name):
            return cls(Workspace(path.parent.parent), path)
        return cls(Workspace(path))


def start_workspace_path(
    workdir: str | Path | None = None,
    default_workdir: str | Path | None = None,
) -> Path:
    """Pick the folder a process starts in: ``workdir``, then ``default_workdir``,
    then the current directory. The result is absolute but not resolved."""
    for candidate in (workdir, default_workdir):
        if candidate:
            return Path(os.path.abspath(os.path.expanduser(candidate)))
    return Path(os.getcwd())


@functools.cache
def process_workspace() -> Workspace:
    """The workspace of a process that was not given one explicitly.

    ``METASCI_SCRIBE_WORKSPACE_DIR`` when set (the langgraph dev manager sets it
    for the server subprocess), otherwise the current directory. Read once per
    process, so every graph the server builds and every route it answers agree
    on the workspace even if the environment changes later (a ``.env`` merge).
    """
    return Workspace(_env_path("METASCI_SCRIBE_WORKSPACE_DIR") or Path.cwd())


@functools.cache
def process_session_dirs() -> SessionDirs:
    """:func:`process_workspace` plus the run folder the process works in.

    The langgraph dev manager sets ``METASCI_SCRIBE_RUN_DIR`` when it starts the
    server for a ``--mode=run`` session. Read once per process, like the
    workspace.
    """
    return SessionDirs(process_workspace(), _env_path("METASCI_SCRIBE_RUN_DIR"))


def resolve_virtual_path(work_dir: str | Path, virtual_path: str) -> Path:
    """Resolve a virtual path (e.g. ``/image.png``) against *work_dir*."""
    vpath = virtual_path if virtual_path.startswith("/") else "/" + virtual_path
    return (Path(work_dir) / vpath.lstrip("/")).resolve()


def _env_path(key: str) -> Path | None:
    value = os.environ.get(key)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser()


def _shared_dir_from_env(key: str) -> Path | None:
    """Read a folder that other programs locate too from *key*, as an absolute path.

    The data and config folders are also found by the WebUI and the desktop
    app, so they follow one path rule: an unset, empty or whitespace-only
    value counts as unset; a leading ``~``, alone or followed by a path
    separator, is the home folder, and nothing else is expanded (no ``~user``,
    no environment variables); a relative path is resolved against the
    current folder.
    """
    value = os.environ.get(key)
    if value is None or not value.strip():
        return None
    if value == "~" or value.startswith(("~/", "~" + os.sep)):
        value = str(Path.home()) + value[1:]
    return Path(os.path.abspath(value))


def _data_dir() -> Path:
    """Data folder: ``$METASCI_SCRIBE_DATA_DIR``, by default ``~/.metasci-scribe``.

    This is the base for sessions.db, skills/, memories/, history — things
    that are NOT configuration but application state. Config files (config.yaml,
    mcp.yaml) live in CONFIG_DIR.
    """
    return (
        _shared_dir_from_env("METASCI_SCRIBE_DATA_DIR")
        or Path.home() / ".metasci-scribe"
    )


def _config_dir() -> Path:
    """Configuration folder: ``$XDG_CONFIG_HOME/metasci-scribe``, by default
    ``~/.config/metasci-scribe`` (config.yaml, mcp.yaml)."""
    return (
        _shared_dir_from_env("XDG_CONFIG_HOME") or Path.home() / ".config"
    ) / "metasci-scribe"


# Both folders are read once, when this module is imported: a workspace ``.env``
# loaded later does not move them, and processes this one starts get them as
# absolute paths (see ``dir_env``).
DATA_DIR: Path = _data_dir()
CONFIG_DIR: Path = _config_dir()
# Whether XDG_CONFIG_HOME was set when this process started (see ``dir_env``).
_XDG_CONFIG_HOME_SET: bool = _shared_dir_from_env("XDG_CONFIG_HOME") is not None


def dir_env(env: dict[str, str]) -> dict[str, str]:
    """Return *env* with this process's data and config folders, as absolute paths.

    For processes this one starts (the langgraph dev server, the WebUI): they
    run in other folders and load their own ``.env``, so they get the folders
    this process uses rather than resolving the variables again.
    ``XDG_CONFIG_HOME`` is passed only if this process started with it: unset
    gives the same folder, and other programs the child runs (the agent's
    shell, MCP servers) may treat a set value differently from an unset one.
    """
    env = {**env, "METASCI_SCRIBE_DATA_DIR": str(DATA_DIR)}
    if _XDG_CONFIG_HOME_SET:
        env["XDG_CONFIG_HOME"] = str(CONFIG_DIR.parent)
    else:
        env.pop("XDG_CONFIG_HOME", None)
    return env


def _global_skills_dir() -> Path:
    return DATA_DIR / "skills"


def _global_memories_dir() -> Path:
    return DATA_DIR / "memories"


# Global skills: shared across all workspaces (~/.metasci-scribe/skills/)
GLOBAL_SKILLS_DIR: Path = _global_skills_dir()

# Global memories: shared across all workspaces (~/.metasci-scribe/memories/)
GLOBAL_MEMORIES_DIR: Path = _global_memories_dir()

# Server home: files that belong to the langgraph dev server rather than to a
# workspace, such as the local secret (~/.metasci-scribe/server/).
SERVER_HOME: Path = DATA_DIR / "server"


# Memories dir: global by default, overridable via env var.
# Supports both new (METASCI_SCRIBE_MEMORIES_DIR) and old (METASCI_SCRIBE_MEMORY_DIR) env vars.
def _memories_dir_from_env() -> Path:
    return (
        _env_path("METASCI_SCRIBE_MEMORIES_DIR")
        or _env_path("METASCI_SCRIBE_MEMORY_DIR")
        or GLOBAL_MEMORIES_DIR
    )


MEMORIES_DIR: Path = _memories_dir_from_env()


def reload_env_dirs() -> None:
    """Re-read the memories-folder override from the environment.

    ``paths`` is imported before ``get_effective_config()`` merges the
    project ``.env`` into ``os.environ``, so entry points call this once
    after loading their config to pick up an override set there.
    """
    global MEMORIES_DIR
    MEMORIES_DIR = _memories_dir_from_env()


def ensure_dirs() -> None:
    """Create runtime subdirectories if they do not exist.

    Creates DATA_DIR (and MEMORIES_DIR as its subdir). Skills directories
    are created on demand by ``install_skill()`` when the user first
    installs a skill.

    Does NOT create the workspace root itself — it should already exist
    (either the user's cwd or a directory they specified).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)
