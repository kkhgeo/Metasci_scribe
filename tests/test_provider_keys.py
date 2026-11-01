"""Tests for metasci_scribe.config.provider_keys (key checks and provider table)."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

from metasci_scribe.config import provider_keys
from metasci_scribe.config.provider_keys import KeyCheck, check_provider_key


def _mock_client(real, handler, http=httpx):
    """Wrap an SDK client class so its requests go to ``handler``.

    The Anthropic SDK is built on ``httpx2``; pass it as ``http`` there.
    """
    return lambda **kw: real(
        **kw, http_client=http.Client(transport=http.MockTransport(handler))
    )


class TestResult:
    @pytest.mark.parametrize(
        ("status", "ok"),
        [
            ("valid", True),
            ("invalid", False),
            ("inconclusive", None),
            ("unavailable", None),
            ("empty", None),
        ],
    )
    def test_ok_is_set_only_for_conclusive_results(self, status, ok):
        assert KeyCheck(status, "m").ok is ok

    @pytest.mark.parametrize(
        ("error", "status", "message"),
        [
            ("Error code: 401", "invalid", "Invalid API key"),
            (
                "Connection refused",
                "inconclusive",
                "Validation inconclusive — transient error, try again later",
            ),
            ("something odd", "inconclusive", "Error: something odd"),
        ],
    )
    def test_exceptions_are_classified(self, error, status, message):
        result = provider_keys._classify_validation_error(RuntimeError(error))
        assert (result.status, result.message) == (status, message)

    def test_empty_key_is_empty_not_valid(self):
        result = check_provider_key("openai", "")
        assert result.status == "empty"
        assert result.ok is None

    def test_unknown_provider_raises(self):
        with pytest.raises(KeyError):
            check_provider_key("no-such-provider", "k")


class TestWizardAdapter:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("valid", True),
            ("empty", True),
            ("invalid", False),
            ("inconclusive", False),
            ("unavailable", False),
        ],
    )
    def test_as_pair(self, status, expected):
        from metasci_scribe.config.onboard.validators import as_pair

        assert as_pair(KeyCheck(status, "msg")) == (expected, "msg")

    def test_validator_follows_a_patched_check(self):
        from metasci_scribe.config.onboard.validators import validate_openai_key

        with patch.object(
            provider_keys, "check_openai_key", return_value=KeyCheck("invalid", "no")
        ):
            assert validate_openai_key("k") == (False, "no")


class TestTimeouts:
    def test_sdk_clients_get_timeout_and_no_retries(self):
        import anthropic

        with patch("anthropic.Anthropic") as client_cls:
            client_cls.return_value.models.list.side_effect = (
                anthropic.AuthenticationError(
                    "bad", response=MagicMock(status_code=401), body=None
                )
            )
            result = provider_keys.check_anthropic_key("k")
        assert result.status == "invalid"
        assert client_cls.call_args.kwargs["timeout"] == 10
        assert client_cls.call_args.kwargs["max_retries"] == 0


class TestTable:
    def test_run_check_takes_only_the_key(self):
        with patch.object(
            provider_keys, "check_openai_key", return_value=KeyCheck("valid", "ok")
        ) as check:
            check_provider_key("openai", "k", "https://ignored.test")
        check.assert_called_once_with("k")

    def test_no_provider_needs_a_base_url(self):
        assert not [
            name
            for name, spec in provider_keys.PROVIDERS.items()
            if spec.needs_base_url
        ]

    def test_tavily_is_gone(self):
        assert "tavily" not in provider_keys.PROVIDERS
        assert not hasattr(provider_keys, "check_tavily_key")


def test_import_loads_no_ui_library():
    """The backend imports this module; it must not pull in the wizard."""
    code = (
        "import sys; import metasci_scribe.config.provider_keys; "
        "bad = [m for m in ('questionary', 'prompt_toolkit', "
        "'metasci_scribe.config.onboard') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize(
    ("check", "client", "args"),
    [
        ("check_openai_key", "openai.OpenAI", ("k",)),
        ("check_anthropic_key", "anthropic.Anthropic", ("k",)),
    ],
)
def test_every_sdk_client_gets_timeout_and_no_retries(check, client, args):
    with patch(client) as client_cls:
        getattr(provider_keys, check)(*args)
    assert client_cls.call_args.kwargs["timeout"] == 10
    assert client_cls.call_args.kwargs["max_retries"] == 0


def test_each_provider_uses_its_check():
    assert {name: spec.check for name, spec in provider_keys.PROVIDERS.items()} == {
        "openai": "check_openai_key",
        "anthropic": "check_anthropic_key",
    }
