"""Identity of the renamed package: name, data dir, env prefix, entry point."""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "metasci_scribe"


def test_package_identity():
    import metasci_scribe
    import metasci_scribe.paths as p

    assert p.DATA_DIR.name == ".metasci-scribe"
    assert metasci_scribe.__name__ == "metasci_scribe"


def test_env_prefix_replaced():
    """Project-specific variables use METASCI_SCRIBE_; none keep the old prefix.

    ``_ENV_MAPPINGS`` also lists vendor variables such as ``ANTHROPIC_API_KEY``,
    which are not ours to rename, so only the project-specific ones are checked.
    """
    from metasci_scribe.config import settings

    values = list(settings._ENV_MAPPINGS.values())
    assert not [v for v in values if "EVOSCI" in v.upper()]
    # Every value is ours (METASCI_SCRIBE_*) or a vendor credential/endpoint.
    stray = [
        v
        for v in values
        if not v.startswith("METASCI_SCRIBE_")
        and not v.endswith(("_API_KEY", "_BASE_URL"))
    ]
    assert not stray, stray
    ours = [v for v in values if v.startswith("METASCI_SCRIBE_")]
    assert "METASCI_SCRIBE_ANTHROPIC_AUTH_MODE" in ours
    assert "METASCI_SCRIBE_OPENAI_AUTH_MODE" in ours


def test_default_dirs_follow_the_new_home(tmp_path):
    """With no override set, data and config land under the new names.

    DATA_DIR and CONFIG_DIR are read at import time, so this runs in a fresh
    interpreter with HOME/USERPROFILE pointed at an empty folder.
    """
    home = tmp_path / "home"
    home.mkdir()
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(("METASCI_SCRIBE_", "EVOSCIENTIST_"))
        and k.upper() != "XDG_CONFIG_HOME"
    }
    env.update(HOME=str(home), USERPROFILE=str(home), PYTHONUTF8="1")
    code = (
        "import json, pathlib;"
        "from metasci_scribe import paths;"
        "print(json.dumps([str(pathlib.Path.home()),"
        " str(paths.DATA_DIR), str(paths.CONFIG_DIR)]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    seen_home, data_dir, config_dir = (Path(p) for p in json.loads(out))
    assert seen_home == home
    assert data_dir == seen_home / ".metasci-scribe"
    assert config_dir.name == "metasci-scribe"
    assert config_dir == seen_home / ".config" / "metasci-scribe"
    for path in (data_dir, config_dir):
        assert "evoscientist" not in str(path).lower()


def test_no_old_prefix_left_in_package_source():
    pattern = re.compile(r"EVOSCI(ENTIST)?_|\.evoscientist|\bEvoScientist")
    assert PACKAGE_DIR.is_dir()
    # Links and owner/repo coordinates of the upstream project stay as credit.
    url = re.compile(r"https?://\S+|EvoScientist/Evo(?:Skills|Scientist)\b(?!\.py)")
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        text = url.sub("", path.read_text(encoding="utf-8"))
        if pattern.search(text):
            offenders.append(str(path.relative_to(PACKAGE_DIR)))
    assert not offenders, offenders


def test_config_class_renamed():
    from metasci_scribe.config import ScribeConfig

    assert ScribeConfig().__class__.__name__ == "ScribeConfig"
    assert not hasattr(
        importlib.import_module("metasci_scribe.config"), "EvoScientistConfig"
    )


def test_agent_module_renamed():
    agent = importlib.import_module("metasci_scribe.agent")
    assert agent.__name__ == "metasci_scribe.agent"
    assert not (PACKAGE_DIR / "EvoScientist.py").exists()


def test_single_cli_entry_point():
    scripts = {
        ep.name: ep.value
        for ep in metadata.distribution("metasci-scribe").entry_points
        if ep.group == "console_scripts"
    }
    assert scripts == {"metasci-scribe": "metasci_scribe.cli:main"}


def test_legacy_session_migration_removed():
    import metasci_scribe.paths as p

    assert not hasattr(p, "migrate_legacy_sessions_db")
