"""Tests for ``metasci-scribe --version``."""

from __future__ import annotations

import sys
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from metasci_scribe.cli._app import app


@pytest.fixture
def scribe(monkeypatch):
    """Invoke the CLI with ``sys.argv`` set as a real run would set it."""
    import metasci_scribe.config as config_mod

    # A fall-through to the main callback would load the developer's config
    # and start the agent; fail inside the test instead.
    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda *_a, **_k: pytest.fail("the main callback ran past --version"),
    )
    runner = CliRunner()

    def invoke(*argv):
        monkeypatch.setattr(sys, "argv", ["metasci-scribe", *argv])
        return runner.invoke(app, list(argv))

    return invoke


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_prints_plain_text(scribe, flag):
    result = scribe(flag)
    assert result.exit_code == 0
    assert result.stdout == f"Metasci Scribe {version('metasci-scribe')}\n"


@pytest.mark.parametrize("argv", [("--version", "--json"), ("--json",)])
def test_json_flag_no_longer_exists(scribe, argv):
    """The JSON form served the desktop app, which this project does not have."""
    result = scribe(*argv)
    assert result.exit_code == 2
