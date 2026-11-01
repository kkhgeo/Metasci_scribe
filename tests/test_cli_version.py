"""Tests for ``metasci-scribe --version`` and ``metasci-scribe --version --json``."""

from __future__ import annotations

import json
import sys
from importlib.metadata import version

import pytest
from typer.testing import CliRunner

from metasci_scribe.cli._app import app
from metasci_scribe.setup import webui
from metasci_scribe.setup.protocol import PROTOCOL


@pytest.fixture
def scribe(monkeypatch):
    """Invoke the CLI with ``sys.argv`` set as a real run would set it: the
    eager ``--version`` callback reads the raw arguments."""
    import metasci_scribe.config as config_mod
    from metasci_scribe.stream.console import console

    monkeypatch.delenv(webui.COMPAT_ENV, raising=False)
    # A fall-through to the main callback would load the developer's config
    # and start whatever UI it names; fail inside the test instead.
    monkeypatch.setattr(
        config_mod,
        "get_effective_config",
        lambda *_a, **_k: pytest.fail("the main callback ran past --version/--json"),
    )
    # --version --json redirects the shared console; undo it after the test.
    monkeypatch.setattr(console, "_file", None)
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


@pytest.mark.parametrize(
    "argv", [("--version", "--json"), ("--json", "--version"), ("-V", "--json")]
)
def test_version_json_in_either_order(scribe, argv):
    result = scribe(*argv)
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "version": version("metasci-scribe"),
        "protocol": PROTOCOL,
        "webui_compat": webui.WEBUI_COMPAT,
    }


def test_version_json_reports_the_range_in_effect(scribe, monkeypatch):
    monkeypatch.setenv(webui.COMPAT_ENV, ">=0.2,<0.4")
    monkeypatch.setattr(webui, "_warned_override", None)
    result = scribe("--version", "--json")
    assert result.exit_code == 0
    assert json.loads(result.stdout)["webui_compat"] == ">=0.2,<0.4"


def test_json_without_version_exits_2(scribe):
    result = scribe("--json")
    assert result.exit_code == 2
    assert result.stdout == ""
    assert "--version" in result.stderr
