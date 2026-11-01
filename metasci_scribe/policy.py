"""What the agent may not touch, and what the app trusts from outside.

The agent reads untrusted papers, so every rule here assumes a prompt
injection is steering it. The rules live in one module so each check is read
in one place and applied the same way everywhere:

* a workspace ``.env`` may set provider credentials and endpoints, nothing
  that changes the app's own settings or how programs run
  (:func:`dotenv_key_allowed`);
* git runs for skill installs without any execution hook from the
  environment, without hooks, and over plain transports only
  (:func:`git_env`, :data:`GIT_SAFE_ARGS`, :data:`GIT_CLONE_ARGS`);
* the saved sign-in and the server secret are never readable through a tool
  (:func:`is_private_state`);
* configuration and code the app, Python, git or a shell loads at startup is
  never writable through a tool, in any mode (:func:`write_refusal`);
* a shell command or a skill install needs a person to approve it, and is
  unavailable when nobody can (:func:`call_needs_person`);
* links in skill folders are never followed out of them
  (:func:`leaves_folder`, :func:`copy_ignore`);
* the folder the app starts in is not on Python's import path
  (:func:`harden_sys_path`, :func:`child_python_env`).
"""

from __future__ import annotations

import os
import os.path as ospath
import re
import site
import stat
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from . import paths

# =============================================================================
# Workspace .env
# =============================================================================

# The names a workspace ``.env`` may set: a provider's credential or endpoint,
# by the suffixes those use. The deny lists below win over a matching suffix.
# Only an uppercase name counts: Windows looks environment variables up
# without regard to case, so ``Git_Ssh_Command`` would set ``GIT_SSH_COMMAND``.
_DOTENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_DOTENV_ALLOWED_SUFFIXES = (
    "_API_KEY",
    "_KEY",
    "_KEY_ID",
    "_TOKEN",
    "_SECRET",
    "_BASE_URL",
    "_API_BASE",
    "_API_URL",
    "_ENDPOINT",
    "_API_VERSION",
    "_ORG_ID",
    "_ORGANIZATION",
    "_PROJECT",
    "_PROJECT_ID",
    "_REGION",
)
_DOTENV_ALLOWED_NAMES = frozenset({"GOOGLE_APPLICATION_CREDENTIALS"})
# Our own settings, and everything that changes how git, Python, Node, a
# shell, a linker or an HTTP client runs.
_DOTENV_DENIED_PREFIXES = (
    "METASCI_SCRIBE_",
    "GIT_",
    "GCM_",
    "SSH_",
    "PYTHON",
    "NODE_",
    "NPM_",
    "LD_",
    "DYLD_",
    "UV_",
    "PIP_",
    "CONDA_",
    "LANGCHAIN_",
    "LANGSMITH_",
    "LANGGRAPH_",
    "JAVA_",
    "_JAVA_",
    "PERL",
    "RUBY",
    "BASH_",
    "XDG_",
)
_DOTENV_DENIED_PARTS = ("PROXY", "SSL", "CERT", "CA_BUNDLE", "PATH", "TRACING")


def dotenv_key_allowed(key: str) -> bool:
    """Whether a workspace ``.env`` may set *key* in this process."""
    if not key or not _DOTENV_NAME.match(key):
        return False
    if key.startswith(_DOTENV_DENIED_PREFIXES):
        return False
    if any(part in key for part in _DOTENV_DENIED_PARTS):
        return False
    return key in _DOTENV_ALLOWED_NAMES or key.endswith(_DOTENV_ALLOWED_SUFFIXES)


def dotenv_values_allowed(mapping: Mapping[str, str | None]) -> dict[str, str]:
    """The entries of a parsed ``.env`` that may be merged: allowed keys with a value."""
    return {
        key: value
        for key, value in mapping.items()
        if value is not None and dotenv_key_allowed(key)
    }


# =============================================================================
# git
# =============================================================================

# Environment variables git reads that name a program to run, a config source,
# a repository to operate on, or a file to write. Everything with these
# prefixes is dropped before git starts; :data:`_GIT_ENV_KEPT` names the few
# ``GIT_*`` variables that only tune HTTPS and are kept.
_GIT_ENV_DROPPED_PREFIXES = ("GIT_", "SSH_ASKPASS")
_GIT_ENV_KEPT = frozenset(
    {
        "GIT_SSL_CAINFO",
        "GIT_SSL_CAPATH",
        "GIT_SSL_VERSION",
        "GIT_HTTP_USER_AGENT",
        "GIT_HTTP_LOW_SPEED_LIMIT",
        "GIT_HTTP_LOW_SPEED_TIME",
        "GIT_REDIRECT_STDERR",
    }
)
# A hooks folder that holds nothing, on every platform git runs on.
GIT_NO_HOOKS_PATH = "/dev/null"
# For every git command: no hooks, no file-system monitor program, and only
# the transports a skill install needs (``ext::`` and remote helpers for other
# schemes run programs).
GIT_SAFE_ARGS: tuple[str, ...] = (
    "-c",
    f"core.hooksPath={GIT_NO_HOOKS_PATH}",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "protocol.allow=never",
    "-c",
    "protocol.file.allow=always",
    "-c",
    "protocol.https.allow=always",
    "-c",
    "protocol.ssh.allow=always",
)
# For ``git clone``: no template folder, which is where a clone's hooks come
# from (``post-checkout`` runs as the clone finishes).
GIT_CLONE_ARGS: tuple[str, ...] = ("--template=",)


def git_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* without the variables that would make git run a program or read
    configuration from somewhere the user did not put it."""
    kept: dict[str, str] = {}
    for key, value in env.items():
        upper = key.upper()
        if upper.startswith(_GIT_ENV_DROPPED_PREFIXES) and upper not in _GIT_ENV_KEPT:
            continue
        kept[key] = value
    return kept


# =============================================================================
# Private state: never readable through a tool
# =============================================================================

PRIVATE_STATE_ERROR = "Metasci Scribe's saved sign-in is not available to file tools."


def is_private_state(path: str | Path) -> bool:
    """Whether *path*, links followed, is the sign-in folder, the server's
    secret folder, or lies inside either. A path that cannot be examined
    counts as private."""
    try:
        return paths.is_auth_path(path) or paths._is_inside(path, paths.SERVER_HOME)
    except (OSError, ValueError, RuntimeError):
        return True


# =============================================================================
# Protected writes: configuration and code loaded at startup
# =============================================================================

# File names that are read as configuration or code by this app, Python, pip,
# git, ssh, a shell or an HTTP client when they start.
_PROTECTED_NAMES = frozenset(
    {
        ".env",
        ".installed.yaml",
        ".gitconfig",
        ".git-credentials",
        "sitecustomize.py",
        "usercustomize.py",
        "pyvenv.cfg",
        ".bashrc",
        ".bash_profile",
        ".bash_login",
        ".bash_logout",
        ".profile",
        ".zshrc",
        ".zprofile",
        ".zshenv",
        ".zlogin",
        "profile.ps1",
        "pip.ini",
        "pip.conf",
        ".pypirc",
        ".npmrc",
        ".condarc",
        ".curlrc",
        ".wgetrc",
        ".netrc",
        "_netrc",
        ".inputrc",
    }
)
_PROTECTED_NAME_PREFIXES = (".env.",)
_PROTECTED_NAME_SUFFIXES = (".pth", "_profile.ps1")
# Folders whose whole content is such configuration or code. The workspace
# state folder is handled apart: its ``skills`` tier stays under the skill
# rules.
_PROTECTED_DIR_NAMES = frozenset({".git", ".ssh", ".langgraph_api", ".vscode", ".idea"})
_STATE_DIR_NAME = ".metasci-scribe"
_STATE_SKILLS_DIR = "skills"

PROTECTED_WRITE_ERROR = (
    "{name} is configuration or code that Metasci Scribe, Python, git or the "
    "shell loads; file tools may not write it."
)


def _python_roots() -> list[str]:
    roots = {
        sys.prefix,
        sys.base_prefix,
        sys.exec_prefix,
        sys.base_exec_prefix,
        ospath.dirname(ospath.abspath(__file__)),
    }
    try:
        roots.update(site.getsitepackages())
    except (AttributeError, OSError):
        pass
    try:
        roots.add(site.getusersitepackages())
    except (AttributeError, OSError):
        pass
    return [ospath.realpath(root) for root in roots if root]


def _under(real: str, root: str) -> bool:
    root = ospath.realpath(root)
    real_cf, root_cf = real.casefold(), root.casefold()
    return real_cf == root_cf or real_cf.startswith(
        root_cf.rstrip(ospath.sep) + ospath.sep
    )


def _protected_name(name: str) -> bool:
    lowered = name.casefold()
    return (
        lowered in _PROTECTED_NAMES
        or lowered.startswith(_PROTECTED_NAME_PREFIXES)
        or lowered.endswith(_PROTECTED_NAME_SUFFIXES)
    )


def _state_dir_refuses(parts: tuple[str, ...]) -> bool:
    """A workspace ``.metasci-scribe`` folder: only its skills tier is writable."""
    lowered = [part.casefold() for part in parts]
    if _STATE_DIR_NAME not in lowered:
        return False
    index = lowered.index(_STATE_DIR_NAME)
    inside = lowered[index + 1 :]
    return not (len(inside) >= 2 and inside[0] == _STATE_SKILLS_DIR)


def _data_dir_refuses(real: str) -> bool:
    """The app's data folder: only memories and the skills tier are writable."""
    for allowed in (
        paths.MEMORIES_DIR,
        paths.GLOBAL_MEMORIES_DIR,
        paths.GLOBAL_SKILLS_DIR,
    ):
        if (
            _under(real, str(allowed))
            and real.casefold() != ospath.realpath(str(allowed)).casefold()
        ):
            return False
    return True


def write_refusal(path: str | Path) -> str | None:
    """Why a tool may not write *path*, or ``None`` when it may.

    Checked by name on the path as given and on its real path, so a link
    into a protected file is refused too; then by place: the data and config
    folders, the workspace state folder, the Python environment this app runs
    in, and folders such as ``.git`` or ``.ssh``.
    """
    try:
        given = Path(path)
        real = ospath.realpath(given)
    except (OSError, ValueError, RuntimeError):
        return PROTECTED_WRITE_ERROR.format(name="that path")
    if is_private_state(real):
        return PRIVATE_STATE_ERROR
    # The data folder is ``~/.metasci-scribe`` by default, the same name as a
    # workspace state folder; inside it the data-folder rule decides.
    in_data_dir = _under(real, str(paths.DATA_DIR))
    for parts in (given.parts, Path(real).parts):
        if not parts:
            continue
        if _protected_name(parts[-1]):
            return PROTECTED_WRITE_ERROR.format(name=parts[-1])
        folders = [part.casefold() for part in parts[:-1]]
        for folder in _PROTECTED_DIR_NAMES:
            if folder in folders:
                return PROTECTED_WRITE_ERROR.format(name=f"a file under {folder}")
        if not in_data_dir and _state_dir_refuses(parts):
            return PROTECTED_WRITE_ERROR.format(name=f"a file under {_STATE_DIR_NAME}")
    if in_data_dir and _data_dir_refuses(real):
        return PROTECTED_WRITE_ERROR.format(
            name="a file in Metasci Scribe's data folder"
        )
    if _under(real, str(paths.CONFIG_DIR)):
        return PROTECTED_WRITE_ERROR.format(
            name="a file in Metasci Scribe's config folder"
        )
    for root in _python_roots():
        if _under(real, root):
            return PROTECTED_WRITE_ERROR.format(
                name="a file in the Python environment Metasci Scribe runs in"
            )
    return None


# =============================================================================
# Calls that need a person
# =============================================================================

SKILL_TOOL = "skill_manager"
NO_PERSON_ERROR = "{name} is not available: no person is approving {what} in this run."


def skill_install_call(name: str, args: object) -> bool:
    """Whether a tool call installs or updates a skill."""
    return (
        name == SKILL_TOOL
        and isinstance(args, dict)
        and args.get("action") == "install"
    )


def call_needs_person(name: str, args: object) -> bool:
    """Whether a tool call must be shown to a person whatever grant is in force.

    A shell command runs with the user's full rights; a skill is instructions
    and scripts the agent will follow. Both can be asked for by text the agent
    read, so each one is shown, and neither exists with no person to ask.
    """
    from .config.settings import HITL_SHELL_TOOLS

    return name in HITL_SHELL_TOOLS or skill_install_call(name, args)


def hitl_armed(name: str, args: object) -> bool:
    """Whether the approval interrupt applies to this call of an armed tool.

    Every armed tool pauses, except ``skill_manager``, which pauses for an
    install only: listing or describing skills changes nothing.
    """
    return name != SKILL_TOOL or skill_install_call(name, args)


def no_person_error(name: str) -> str:
    what = "skill installs" if name == SKILL_TOOL else "commands"
    return "Error: " + NO_PERSON_ERROR.format(name=name, what=what)


# =============================================================================
# Links
# =============================================================================

_LINK_REPARSE_TAGS = (0xA000000C, 0xA0000003)  # Windows symlink, junction


def is_link(path: str | Path) -> bool:
    """A symbolic link, or a Windows junction, which ``S_ISLNK`` does not see."""
    info = os.lstat(path)
    return (
        stat.S_ISLNK(info.st_mode)
        or getattr(info, "st_reparse_tag", 0) in _LINK_REPARSE_TAGS
    )


def leaves_folder(file: str | Path, folder: str | Path) -> bool:
    """Whether *file*, links followed, lies outside *folder*, links followed.

    A path that cannot be resolved counts as leaving.
    """
    try:
        return not _under(ospath.realpath(file), ospath.realpath(folder))
    except (OSError, ValueError, RuntimeError):
        return True


def copy_ignore(
    extra: Callable[[str, list[str]], object] | None = None,
) -> Callable[[str, list[str]], set[str]]:
    """A ``shutil.copytree`` ``ignore`` that drops every link and the private
    state, plus whatever *extra* names.

    A skill comes from outside, and a link in it would copy whatever it
    names, so links are dropped, not followed.
    """

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set(extra(directory, names)) if extra else set()
        for name in names:
            entry = ospath.join(directory, name)
            try:
                if is_link(entry) or is_private_state(entry):
                    skipped.add(name)
            except OSError:
                skipped.add(name)
        return skipped

    return ignore


# =============================================================================
# Python's import path
# =============================================================================


def harden_sys_path() -> None:
    """Take the working folder off ``sys.path``.

    ``python -m metasci_scribe`` puts the folder it starts in first on the
    import path, and this app imports lazily: a module written into the
    workspace later would be imported in place of a real one. The console
    script does not add it, so nothing here relies on it.
    """
    try:
        cwd = ospath.realpath(os.getcwd())
    except OSError:
        cwd = None
    kept = []
    for entry in sys.path:
        if entry in ("", "."):
            continue
        try:
            if cwd is not None and ospath.realpath(entry) == cwd:
                continue
        except (OSError, ValueError):
            pass
        kept.append(entry)
    sys.path[:] = kept


def child_python_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* for a Python process this app starts: its working folder is not
    on its import path either (``PYTHONSAFEPATH``, Python 3.11+)."""
    return {**env, "PYTHONSAFEPATH": "1"}
