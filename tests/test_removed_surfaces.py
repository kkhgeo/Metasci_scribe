"""The TUI, WebUI and operations surfaces are gone; nothing may reach for them.

Metasci Scribe is a terminal agent with one conversation screen
(``cli/interactive.py``). It does not update itself, install a proxy for a
chat subscription, host a browser front-end or run as a service.
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from dataclasses import fields
from pathlib import Path
from typing import Literal

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = ROOT / "metasci_scribe"

REMOVED_MODULES = [
    "metasci_scribe.cli.tui_interactive",
    "metasci_scribe.cli.tui_runtime",
    "metasci_scribe.cli.tui_backends",
    "metasci_scribe.cli.widgets",
    "metasci_scribe.cli.service_cmd",
    "metasci_scribe.ccproxy_manager",
    "metasci_scribe.stt",
    "metasci_scribe.proactive",
    "metasci_scribe.middleware.proactive_mode",
    "metasci_scribe.commands.implementation.proactive",
    "metasci_scribe.updater",
    "metasci_scribe.update_helper",
    "metasci_scribe.update_check",
    "metasci_scribe.uv_tool",
    "metasci_scribe.deploy",
    "metasci_scribe.setup.webui",
]

# Names no source line may mention, imports and prose alike. ``tui_runtime``
# and friends are matched as whole words so ``tui_`` in an unrelated name
# stays legal.
_FORBIDDEN_WORDS = re.compile(
    r"\b(?:tui_interactive|tui_runtime|tui_backends|service_cmd|ccproxy|"
    r"ccproxy_manager|proactive_mode|update_check|update_helper|uv_tool|"
    r"run_textual_interactive|run_webui)\b"
)
_FORBIDDEN_IMPORTS = re.compile(
    r"(?:^|\s)(?:from\s+[\w.]*\b(?:textual|updater|proactive|deploy|stt|widgets)\b"
    r"[\w.]*\s+import\b"
    r"|from\s+[\w.]*\s+import\s+[^\n#]*\b(?:textual|updater|proactive|deploy|stt|widgets)\b"
    r"|import\s+[\w.]*\b(?:textual|updater|proactive|deploy|stt|widgets)\b)",
    re.MULTILINE,
)


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_module_is_gone(module):
    assert importlib.util.find_spec(module) is None


def test_removed_files_and_scripts_are_gone():
    for rel in (
        "metasci_scribe/cli/widgets",
        "metasci_scribe/proactive",
        "metasci_scribe/deploy",
        "metasci_scribe/setup/webui.py",
        "scripts/install.sh",
        "scripts/install.ps1",
    ):
        assert not (ROOT / rel).exists(), rel
    assert list((PACKAGE_DIR / "cli").glob("tui_*.py")) == []


def test_textual_is_not_a_dependency():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = list(data["project"].get("dependencies", []))
    for extra in data["project"].get("optional-dependencies", {}).values():
        declared.extend(extra)
    for group in data.get("dependency-groups", {}).values():
        declared.extend(item for item in group if isinstance(item, str))
    names = [re.split(r"[<>=!~\[; ]", spec, maxsplit=1)[0].lower() for spec in declared]
    assert "textual" not in names
    assert "ccproxy-api" not in names


def test_no_source_file_mentions_removed_surfaces():
    offenders: list[str] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in (_FORBIDDEN_WORDS, _FORBIDDEN_IMPORTS):
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(
                    f"{path.relative_to(PACKAGE_DIR)}:{line_no}: {match.group().strip()}"
                )
    assert not offenders, "removed surfaces still referenced:\n" + "\n".join(offenders)


def test_ui_backend_is_cli_only():
    from metasci_scribe.config.settings import ScribeConfig

    assert ScribeConfig().ui_backend == "cli"
    (field,) = (f for f in fields(ScribeConfig) if f.name == "ui_backend")
    assert "Literal['cli']" in str(field.type) or 'Literal["cli"]' in str(field.type)


@pytest.mark.parametrize("old", ["tui", "webui", "textual", "rich", "nonsense"])
def test_old_ui_backend_values_read_as_cli(old):
    """A config file written by the upstream project still loads."""
    from metasci_scribe.config.settings import ScribeConfig, config_from_mapping

    assert config_from_mapping({"ui_backend": old}).ui_backend == "cli"
    assert ScribeConfig(ui_backend=old).ui_backend == "cli"


def test_config_set_rejects_removed_ui_backend():
    from metasci_scribe.config.settings import _coerce_value

    with pytest.raises(ValueError, match="expected one of"):
        _coerce_value("tui", Literal["cli"])


def test_config_has_no_removed_settings():
    from metasci_scribe.config.settings import ScribeConfig

    names = {f.name for f in fields(ScribeConfig)}
    leftovers = sorted(
        n
        for n in names
        if n.startswith(("webui_", "ccproxy_", "proactive_", "stt_", "update_"))
        or n in {"anthropic_auth_mode", "auto_update", "check_updates"}
    )
    assert leftovers == []
    # ChatGPT sign-in is the one supported subscription route.
    assert "openai_auth_mode" in names


def test_cli_exposes_no_removed_command():
    from typer.testing import CliRunner

    from metasci_scribe.cli import app

    runner = CliRunner()
    for command in (["deploy"], ["update"], ["service", "install"], ["serve"]):
        result = runner.invoke(app, command)
        assert result.exit_code == 2, command


def test_slash_commands_have_no_removed_entries():
    from metasci_scribe.commands import manager

    names = {name for name, _ in manager.list_commands()}
    assert "/proactive" not in names
    assert "/update" not in names
    # The rubric and goal workflows stay.
    assert {"/rubric", "/goal"} <= names
