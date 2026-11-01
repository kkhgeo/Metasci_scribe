"""The messaging-channel subsystem is gone; nothing may import it."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "metasci_scribe"

# Any import of a ``channels`` module, absolute or relative, top-level or lazy:
#   from metasci_scribe.channels import x / from ..channels.bus import y
#   from . import channels / import metasci_scribe.channels.base
_CHANNELS_IMPORT = re.compile(
    r"(?:^|\s)(?:from\s+[\w.]*\bchannels\b[\w.]*\s+import\b"
    r"|from\s+[\w.]*\s+import\s+[^\n#]*\bchannels\b"
    r"|import\s+[\w.]*\bchannels\b)",
    re.MULTILINE,
)
_LEGACY_MODULES = re.compile(
    r"\b(?:cli\.channel(?:_sends)?|channel_ui|commands\.implementation\.channel)\b"
    r"|(?:from|import)\s+[\w.]*\bchannel_sends\b"
)


def test_channels_package_is_gone():
    assert importlib.util.find_spec("metasci_scribe.channels") is None
    assert not (PACKAGE_DIR / "channels").exists()


def test_channel_modules_are_gone():
    for rel in (
        "cli/channel.py",
        "cli/channel_sends.py",
        "commands/channel_ui.py",
        "commands/implementation/channel.py",
        "config/onboard/channels.py",
    ):
        assert not (PACKAGE_DIR / rel).exists(), rel


def test_no_source_file_imports_channels():
    offenders: list[str] = []
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for pattern in (_CHANNELS_IMPORT, _LEGACY_MODULES):
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(
                    f"{path.relative_to(PACKAGE_DIR)}:{line_no}: {match.group().strip()}"
                )
    assert not offenders, "channel imports remain:\n" + "\n".join(offenders)


def test_shared_interaction_modules_live_in_stream():
    from metasci_scribe.stream.formatter import _format_todo_list
    from metasci_scribe.stream.hitl_budget import (
        channel_response_with_budget_stop,
        hitl_budget_stop,
    )
    from metasci_scribe.stream.interaction import ApprovalOutcome

    assert callable(_format_todo_list)
    assert callable(hitl_budget_stop)
    assert callable(channel_response_with_budget_stop)
    assert ApprovalOutcome is not None


def test_cli_exposes_no_channel_surface():
    from typer.testing import CliRunner

    from metasci_scribe.cli import app

    runner = CliRunner()
    for command in (["serve"], ["channel", "setup"], ["configure", "channels"]):
        result = runner.invoke(app, command)
        assert result.exit_code == 2, command


def test_config_has_no_channel_settings():
    from dataclasses import fields

    from metasci_scribe.config.settings import ScribeConfig

    prefixes = (
        "channel_",
        "imessage_",
        "telegram_",
        "discord_",
        "slack_",
        "feishu_",
        "wechat_",
        "dingtalk_",
        "email_",
        "qq_",
        "signal_",
        "stt_",
    )
    leftovers = [f.name for f in fields(ScribeConfig) if f.name.startswith(prefixes)]
    assert leftovers == []
