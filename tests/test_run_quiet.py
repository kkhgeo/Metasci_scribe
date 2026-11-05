"""An autonomous run prints no notice that does not concern it."""

from __future__ import annotations

import subprocess
import sys

_CHECK = """
import warnings
from metasci_scribe.cli import quiet_known_warnings
quiet_known_warnings()
from langchain_core._api import LangChainBetaWarning
warnings.warn("The v3 streaming protocol on Pregel is experimental.", LangChainBetaWarning)
warnings.warn("some other beta feature", LangChainBetaWarning)
"""


def test_the_v3_streaming_notice_is_hidden_and_other_beta_notices_are_not():
    # A fresh interpreter: langchain_core is first imported there, as in /run.
    out = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", _CHECK],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "v3 streaming protocol" not in out.stderr
    assert "some other beta feature" in out.stderr


def test_an_unattended_build_does_not_log_the_shell_hint(monkeypatch):
    from metasci_scribe import agent_shell
    from metasci_scribe.setup import research_env

    logged: list[bool] = []
    monkeypatch.setattr(agent_shell, "log_setup_hint", lambda: logged.append(True))
    research_env.research_env_overrides(log_hint=False)
    assert logged == []
    research_env.research_env_overrides()
    assert logged == [True]
