"""Identity of the renamed package: name, data dir, env prefix, entry point."""

from __future__ import annotations

import importlib
import re
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
    ours = [v for v in values if v.startswith("METASCI_SCRIBE_")]
    assert "METASCI_SCRIBE_ANTHROPIC_AUTH_MODE" in ours
    assert "METASCI_SCRIBE_OPENAI_AUTH_MODE" in ours


def test_no_old_prefix_left_in_package_source():
    pattern = re.compile(r"EVOSCI(ENTIST)?_|\.evoscientist|\bEvoScientist")
    assert PACKAGE_DIR.is_dir()
    url = re.compile(r"https?://\S+")  # links to the upstream project stay as credit
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
