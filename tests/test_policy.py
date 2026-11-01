"""The one place that says what the agent may not touch: ``metasci_scribe.policy``.

Each check is tried through the module and, where it matters, through the real
code that consumes it (config loading, git, the file backends), with a fake
home and a fake token.
"""

from __future__ import annotations

import json
import os
import shutil
import site
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from metasci_scribe import git_cli, paths, policy
from metasci_scribe.backends import CustomSandboxBackend
from metasci_scribe.config import settings
from metasci_scribe.config.settings import get_effective_config

SECRET = "refresh-secret-value"


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "home"
    data = root / ".metasci-scribe"
    (data / "auth").mkdir(parents=True)
    (data / "auth" / "chatgpt.json").write_text(
        json.dumps({"refresh_token": SECRET}), encoding="utf-8"
    )
    (data / "server").mkdir()
    (data / "server" / "secret").write_text("server-secret", encoding="utf-8")
    config = root / ".config" / "metasci-scribe"
    config.mkdir(parents=True)
    monkeypatch.setattr(paths, "DATA_DIR", data)
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", data / "skills")
    monkeypatch.setattr(paths, "GLOBAL_MEMORIES_DIR", data / "memories")
    monkeypatch.setattr(paths, "MEMORIES_DIR", data / "memories")
    monkeypatch.setattr(paths, "SERVER_HOME", data / "server")
    monkeypatch.setattr(paths, "CONFIG_DIR", config)
    return root


@pytest.fixture
def project(home) -> Path:
    folder = home / "paper"
    folder.mkdir()
    return folder


# ------------------------------------------------------------------ .env keys


@pytest.mark.parametrize(
    "key",
    [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "MINIMAX_BASE_URL",
        "CUSTOM_OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "GITHUB_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_OPENAI_ENDPOINT",
    ],
)
def test_provider_credentials_and_endpoints_may_come_from_a_workspace_env(key):
    assert policy.dotenv_key_allowed(key)


@pytest.mark.parametrize(
    "key",
    [
        "METASCI_SCRIBE_DANGEROUS_MODE",
        "METASCI_SCRIBE_LANGGRAPH_DEV_PORT",
        "metasci_scribe_dangerous_mode",
        "GIT_SSH_COMMAND",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "Git_Ssh_Command",
        "GIT_EXEC_PATH",
        "SSH_ASKPASS",
        "PATH",
        "Path",
        "PATHEXT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONSAFEPATH",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "OPENAI_PROXY_URL",
        "LD_PRELOAD",
        "DYLD_INSERT_LIBRARIES",
        "NODE_OPTIONS",
        "NODE_PATH",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_ENDPOINT",
        "LANGCHAIN_TRACING_V2",
        "HOME",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
        "TMPDIR",
        "COMSPEC",
        "SHELL",
        "BASH_ENV",
        "UV_INDEX_URL",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "JAVA_TOOL_OPTIONS",
        "PERL5OPT",
        "",
        "NOT A KEY",
    ],
)
def test_settings_and_execution_environment_never_come_from_a_workspace_env(key):
    assert not policy.dotenv_key_allowed(key)


def test_the_dotenv_merge_takes_only_allowed_keys(tmp_path, monkeypatch):
    """The critical finding: a planted ``.env`` raised the mode and hooked git."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "METASCI_SCRIBE_DANGEROUS_MODE=true\n"
        "GIT_SSH_COMMAND=cat secret > out; exit 1; #\n"
        "GIT_CONFIG_COUNT=1\n"
        "GIT_CONFIG_KEY_0=url.evil.insteadOf\n"
        "GIT_CONFIG_VALUE_0=https://github.com/\n"
        "PYTHONPATH=/tmp/evil\n"
        "OPENAI_API_KEY=sk-from-dotenv\n"
        "OPENAI_BASE_URL=https://api.example.test/v1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "find_dotenv", lambda *a, **k: str(env_file))
    for key in (
        "METASCI_SCRIBE_DANGEROUS_MODE",
        "GIT_SSH_COMMAND",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "PYTHONPATH",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(settings, "_ENV_ORIGINS", {})

    config = get_effective_config()

    assert config.dangerous_mode is False
    assert "METASCI_SCRIBE_DANGEROUS_MODE" not in os.environ
    assert "GIT_SSH_COMMAND" not in os.environ
    assert "GIT_CONFIG_COUNT" not in os.environ
    assert "PYTHONPATH" not in os.environ
    assert os.environ["OPENAI_API_KEY"] == "sk-from-dotenv"
    assert os.environ["OPENAI_BASE_URL"] == "https://api.example.test/v1"
    assert settings.env_origin("OPENAI_API_KEY") == {
        "source": "dotenv",
        "path": str(env_file),
    }
    assert settings.env_origin("GIT_SSH_COMMAND") is None
    # The merge is the only writer: restore for the tests after this one.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


# ------------------------------------------------------------------ git env


def test_git_env_drops_every_execution_hook_and_keeps_the_rest():
    env = {
        "PATH": "/usr/bin",
        "HOME": "/home/u",
        "KEEP_ME": "yes",
        "GIT_SSH_COMMAND": "cat secret",
        "GIT_SSH": "/tmp/x",
        "GIT_ASKPASS": "/tmp/x",
        "SSH_ASKPASS": "/tmp/x",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "url.x.insteadOf",
        "GIT_CONFIG_VALUE_0": "https://github.com/",
        "GIT_CONFIG_PARAMETERS": "'core.sshCommand=x'",
        "GIT_CONFIG_GLOBAL": "/tmp/evil",
        "GIT_CONFIG_SYSTEM": "/tmp/evil",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_EXTERNAL_DIFF": "/tmp/x",
        "GIT_PROXY_COMMAND": "/tmp/x",
        "GIT_EXEC_PATH": "/tmp/x",
        "GIT_TEMPLATE_DIR": "/tmp/x",
        "GIT_EDITOR": "/tmp/x",
        "GIT_SEQUENCE_EDITOR": "/tmp/x",
        "GIT_PAGER": "/tmp/x",
        "GIT_DIR": "/tmp/x",
        "GIT_WORK_TREE": "/tmp/x",
        "GIT_INDEX_FILE": "/tmp/x",
        "GIT_OBJECT_DIRECTORY": "/tmp/x",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/tmp/x",
        "GIT_COMMON_DIR": "/tmp/x",
        "GIT_NAMESPACE": "x",
        "GIT_CEILING_DIRECTORIES": "/tmp",
        "GIT_SSL_NO_VERIFY": "1",
        "GIT_TRACE": "/tmp/trace",
        "GIT_TRACE2_EVENT": "/tmp/trace",
        "GIT_ALLOW_PROTOCOL": "ext",
        "GIT_PROTOCOL_FROM_USER": "1",
        "git_ssh_command": "lower",
        "GIT_SSL_CAINFO": "/etc/corp-ca.pem",
    }
    out = policy.git_env(env)
    assert out["PATH"] == "/usr/bin"
    assert out["HOME"] == "/home/u"
    assert out["KEEP_ME"] == "yes"
    # Only the HTTPS tuning variables survive.
    assert [k for k in out if k.upper().startswith("GIT_")] == ["GIT_SSL_CAINFO"]
    assert "SSH_ASKPASS" not in out
    assert env["GIT_SSH_COMMAND"] == "cat secret"  # the input is untouched


def test_run_git_strips_hooks_from_the_process_environment(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("GIT_SSH_COMMAND", "cat secret > out; exit 1; #")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_EXEC_PATH", "/tmp/evil")
    monkeypatch.setenv("KEEP_ME", "yes")
    proc = SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch("metasci_scribe.git_cli.subprocess.run", return_value=proc) as run:
        git_cli.run_git(["ls-remote", "https://github.com/o/r.git"], timeout=5)

    env = run.call_args.kwargs["env"]
    assert "GIT_SSH_COMMAND" not in env
    assert "GIT_CONFIG_COUNT" not in env
    assert "GIT_EXEC_PATH" not in env
    assert env["KEEP_ME"] == "yes"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    argv = run.call_args.args[0]
    assert argv[0] == "git"
    assert "core.hooksPath=" + policy.GIT_NO_HOOKS_PATH in argv
    assert "protocol.allow=never" in argv
    assert "protocol.https.allow=always" in argv
    assert "core.fsmonitor=false" in argv


def test_clone_disables_the_template_hooks(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    proc = SimpleNamespace(returncode=0, stdout="", stderr="")
    with patch("metasci_scribe.git_cli.subprocess.run", return_value=proc) as run:
        git_cli.clone_repo("owner/repo", None, "/tmp/dest")
    argv = run.call_args.args[0]
    assert "--template=" in argv
    assert argv.index("--template=") > argv.index("clone")


def _git_source_repo(root: Path) -> Path:
    src = root / "src"
    src.mkdir()
    subprocess.run(["git", "init", "-q", str(src)], check=True)
    (src / "f.txt").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(src), "add", "f.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(src),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )
    return src


def _hook_template(root: Path) -> Path:
    template = root / "tpl"
    hooks = template / "hooks"
    hooks.mkdir(parents=True)
    hook = hooks / "post-checkout"
    hook.write_text("#!/bin/sh\necho hooked > hooked.txt\n", encoding="utf-8")
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return template


@pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git")
def test_a_planted_git_environment_runs_no_hook_through_run_git(tmp_path, monkeypatch):
    """What the planted ``.env`` did: a template and ``core.hooksPath`` from the
    environment make ``git clone`` run a script. Through ``run_git`` it does not."""
    src = _git_source_repo(tmp_path)
    template = _hook_template(tmp_path)
    planted = {
        "GIT_TEMPLATE_DIR": str(template),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(template / "hooks"),
    }
    control = subprocess.run(
        ["git", "clone", "-q", str(src), str(tmp_path / "control")],
        env={**os.environ, **planted},
        capture_output=True,
        text=True,
    )
    if control.returncode != 0 or not (tmp_path / "control" / "hooked.txt").exists():
        pytest.skip(
            "this git does not run the planted hook, so there is nothing to stop"
        )

    for key, value in planted.items():
        monkeypatch.setenv(key, value)
    result = git_cli.run_git(
        ["clone", "-q", str(src), str(tmp_path / "dest")], timeout=60
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "dest" / "f.txt").is_file()
    assert not (tmp_path / "dest" / "hooked.txt").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs a real git")
def test_run_git_refuses_the_ext_transport(tmp_path):
    marker = tmp_path / "pwned"
    result = git_cli.run_git(
        ["clone", f"ext::sh -c touch% {marker.as_posix()}", str(tmp_path / "dest")],
        timeout=60,
    )
    assert result.returncode != 0
    assert not marker.exists()


# ------------------------------------------------------------------ private state


def test_private_state_is_the_sign_in_and_the_server_secret(home, project):
    data = home / ".metasci-scribe"
    assert policy.is_private_state(data / "auth" / "chatgpt.json")
    assert policy.is_private_state(data / "auth")
    assert policy.is_private_state(data / "server" / "secret")
    assert policy.is_private_state(data / "server")
    assert not policy.is_private_state(data / "memories" / "x.md")
    assert not policy.is_private_state(data / "skills")
    assert not policy.is_private_state(project / "notes.md")


def test_require_workspace_root_refuses_the_private_folders(home, project):
    data = home / ".metasci-scribe"
    paths.require_workspace_root(project)
    for folder in (data / "auth", data / "auth" / "deeper", data / "server"):
        with pytest.raises(paths.WorkspaceRootError):
            paths.require_workspace_root(folder)


# ------------------------------------------------------------------ protected writes


def _refused(path) -> bool:
    return policy.write_refusal(Path(path)) is not None


def test_config_and_code_the_app_loads_cannot_be_written(home, project):
    data = home / ".metasci-scribe"
    config = home / ".config" / "metasci-scribe"
    refused = [
        project / ".env",
        project / ".env.local",
        project / "sub" / ".env",
        home / ".env",
        project / ".metasci-scribe" / "crons.json",
        project / ".metasci-scribe" / "state.db",
        project / ".metasci-scribe" / "skills" / ".installed.yaml",
        project / ".metasci-scribe" / "skills" / "helper" / ".installed.yaml",
        project / ".langgraph_api" / "store.pckl",
        data / "sessions.db",
        data / "auth" / "chatgpt.json",
        data / "server" / "secret",
        data / "host_id",
        data / "envs" / "default" / "Lib" / "site-packages" / "x.py",
        data / "skills" / ".installed.yaml",
        data / "tools" / "git.json",
        config / "config.yaml",
        config / "mcp.yaml",
        home / ".gitconfig",
        home / ".git-credentials",
        project / ".git" / "config",
        project / ".git" / "hooks" / "pre-commit",
        home / ".ssh" / "config",
        home / ".ssh" / "authorized_keys",
        project / "sitecustomize.py",
        project / "usercustomize.py",
        project / "easy.pth",
        project / "pyvenv.cfg",
        home / ".bashrc",
        home / ".bash_profile",
        home / ".profile",
        home / ".zshrc",
        home / "Documents" / "WindowsPowerShell" / "Microsoft.PowerShell_profile.ps1",
        home / "pip.ini",
        home / ".config" / "pip" / "pip.conf",
        home / ".pypirc",
        home / ".npmrc",
        home / ".condarc",
        home / ".curlrc",
        home / ".wgetrc",
        home / ".netrc",
        home / ".inputrc",
        Path(sys.prefix) / "Lib" / "site-packages" / "anything.py",
        Path(sys.prefix) / "anything.txt",
        Path(sys.base_prefix) / "anything.txt",
        Path(policy.__file__).with_name("anything.py"),
        Path(policy.__file__),
    ]
    for entry in [*site.getsitepackages(), site.getusersitepackages()]:
        refused.append(Path(entry) / "planted.py")
    allowed = [
        project / "paper.tex",
        project / "figures" / "f1.py",
        project / ".metasci-scribe" / "skills" / "helper" / "SKILL.md",
        project / ".metasci-scribe" / "skills" / "helper" / "scripts" / "run.py",
        project / "env.md",
        project / "environment.yml",
        project / "config.yaml",
        project / "notes" / "git" / "intro.md",
        project / ".github" / "workflows" / "ci.yml",
        project / "my.pth.md",
        data / "memories" / "profile" / "me.md",
        data / "skills" / "helper" / "SKILL.md",
    ]
    assert [p for p in refused if not _refused(p)] == []
    assert [p for p in allowed if _refused(p)] == []


def test_write_refusal_follows_links_to_a_protected_file(home, project):
    link = project / "notes.txt"
    try:
        link.symlink_to(project / ".env")
    except OSError:
        pytest.skip("this account may not create symlinks")
    assert _refused(link)


def test_file_tools_refuse_the_protected_files_in_every_mode(home, project):
    (project / ".env").write_text("OPENAI_API_KEY=old\n", encoding="utf-8")
    (project / ".metasci-scribe").mkdir()
    default = CustomSandboxBackend(root_dir=str(project), virtual_mode=True)
    dangerous = CustomSandboxBackend(root_dir=str(project), dangerous=True)
    cases = {
        default: [
            "/.env",
            "/.env.local",
            "/sub/.env",
            "/.metasci-scribe/crons.json",
            "/.metasci-scribe/skills/.installed.yaml",
            "/.git/hooks/post-checkout",
            "/sitecustomize.py",
            "/.langgraph_api/x.pckl",
        ],
        dangerous: [
            str(project / ".env"),
            str(home / ".gitconfig"),
            str(home / ".ssh" / "config"),
            str(home / ".config" / "metasci-scribe" / "mcp.yaml"),
            str(home / ".metasci-scribe" / "sessions.db"),
            str(Path(sys.prefix) / "Lib" / "site-packages" / "sitecustomize.py"),
            str(Path(policy.__file__).with_name("planted.py")),
        ],
    }
    for backend, keys in cases.items():
        for key in keys:
            assert backend.write(key, "GIT_SSH_COMMAND=x\n").error, key
            assert backend.edit(key, "old", "new").error, key
            assert backend.upload_files([(key, b"x")])[0].error, key
            assert backend.delete(key).error, key
    assert (project / ".env").read_text(encoding="utf-8") == "OPENAI_API_KEY=old\n"
    assert not (project / ".git").exists()
    assert not (home / ".gitconfig").exists()
    assert not (Path(policy.__file__).with_name("planted.py")).exists()
    # Ordinary files still work, in both modes.
    assert not default.write("/paper.tex", "x").error
    assert not dangerous.write(str(project / "notes.md"), "x").error
    assert not default.write("/.metasci-scribe/skills/helper/SKILL.md", "x").error


def test_the_server_secret_cannot_be_read_by_file_tools(home, project):
    backend = CustomSandboxBackend(root_dir=str(project), dangerous=True)
    secret = home / ".metasci-scribe" / "server" / "secret"
    try:
        result = backend.read(str(secret))
    except ValueError:
        return
    assert result.error
    assert "server-secret" not in str(result)
    listing = backend.ls(str(home / ".metasci-scribe"))
    assert not [
        e for e in listing.entries or [] if e["path"].rstrip("/").endswith("server")
    ]


# ------------------------------------------------------------------ gated calls


def test_shell_and_skill_installs_need_a_person():
    assert policy.call_needs_person("execute", {"command": "ls"})
    assert policy.call_needs_person("run_in_background", {"command": "ls"})
    assert policy.call_needs_person(
        "skill_manager", {"action": "install", "source": "o/r"}
    )
    assert not policy.call_needs_person("skill_manager", {"action": "list"})
    assert not policy.call_needs_person("skill_manager", {"action": "browse"})
    assert not policy.call_needs_person(
        "skill_manager", {"action": "info", "name": "x"}
    )
    assert not policy.call_needs_person("skill_manager", {})
    assert not policy.call_needs_person("skill_manager", None)
    assert not policy.call_needs_person("read_file", {"file_path": "/x"})
    assert not policy.call_needs_person("delete", {"path": "/x"})


# ------------------------------------------------------------------ links


def test_leaves_folder(home, project):
    skill = project / "helper"
    skill.mkdir()
    (skill / "SKILL.md").write_text("x", encoding="utf-8")
    assert not policy.leaves_folder(skill / "SKILL.md", skill)
    assert not policy.leaves_folder(skill / "sub" / "x.md", skill)
    assert policy.leaves_folder(project / "SKILL.md", skill)
    link = skill / "EXPERT.md"
    try:
        link.symlink_to(home / ".metasci-scribe" / "auth" / "chatgpt.json")
    except OSError:
        pytest.skip("this account may not create symlinks")
    assert policy.leaves_folder(link, skill)


# ------------------------------------------------------------------ sys.path


def test_harden_sys_path_drops_the_working_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = list(sys.path)
    monkeypatch.setattr(sys, "path", ["", ".", str(tmp_path), os.getcwd(), *original])
    policy.harden_sys_path()
    assert "" not in sys.path
    assert "." not in sys.path
    assert str(tmp_path) not in sys.path
    for entry in original:
        if entry not in ("", ".", str(tmp_path), os.getcwd()):
            assert entry in sys.path


def test_the_cli_entry_hardens_sys_path_before_importing_anything():
    source = (
        Path(policy.__file__)
        .with_name("cli")
        .joinpath("__init__.py")
        .read_text(encoding="utf-8")
    )
    body = source[source.index("def main(") :]
    assert "harden_sys_path()" in body
    assert body.index("harden_sys_path()") < body.index("from ..config import")


def test_child_pythons_get_a_safe_path():
    env = policy.child_python_env({"PATH": "/usr/bin"})
    assert env["PYTHONSAFEPATH"] == "1"
    assert env["PATH"] == "/usr/bin"


def test_the_server_package_takes_the_workspace_off_sys_path(tmp_path):
    """``langgraph dev`` appends its working folder, the workspace, to
    ``sys.path`` before importing our graphs; importing them takes it off
    again, in the server process only."""
    code = (
        "import os, sys\n"
        "sys.path.append(os.getcwd())\n"
        "import metasci_scribe.langgraph_dev\n"
        "print(any(os.path.realpath(p) == os.path.realpath(os.getcwd())"
        " for p in sys.path))\n"
    )
    for marker, expected in (("1", "False"), (None, "True")):
        env = {
            k: v for k, v in os.environ.items() if k != "METASCI_SCRIBE_SERVER_PROCESS"
        }
        if marker:
            env["METASCI_SCRIBE_SERVER_PROCESS"] = marker
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
        assert out.strip().splitlines()[-1] == expected, (marker, out)


def test_project_id_git_runs_without_hooks(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import patch

    from metasci_scribe.memory import project

    monkeypatch.setenv("GIT_SSH_COMMAND", "cat secret")
    proc = SimpleNamespace(returncode=0, stdout="/repo\n", stderr="")
    with patch(
        "metasci_scribe.memory.project.subprocess.run", return_value=proc
    ) as run:
        assert project._run_git(["rev-parse", "--show-toplevel"], Path(".")) == "/repo"
    argv = run.call_args.args[0]
    assert argv[0] == "git"
    assert "core.hooksPath=" + policy.GIT_NO_HOOKS_PATH in argv
    assert "GIT_SSH_COMMAND" not in run.call_args.kwargs["env"]
