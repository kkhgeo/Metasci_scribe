"""Tests the config-rule approval policy in ``stream.interaction``."""

from unittest.mock import MagicMock, patch

from metasci_scribe.stream import interaction as I


def _cfg(*, auto_approve=False, shell_allow_list="", dangerous_mode=False):
    cfg = MagicMock()
    cfg.auto_approve = auto_approve
    cfg.shell_allow_list = shell_allow_list
    cfg.dangerous_mode = dangerous_mode
    return cfg


class TestApproveDecisions:
    def test_length_matches_requests(self):
        assert I.approve_decisions([{"name": "a"}, {"name": "b"}]) == [
            {"type": "approve"},
            {"type": "approve"},
        ]

    def test_empty_list_still_yields_a_single_approve(self):
        # The shape ``Command(resume=...)`` expects.
        assert I.approve_decisions([]) == [{"type": "approve"}]


class TestResolveConfigDecisions:
    """``resolve_config_decisions``: approve / reject / prompt from config alone."""

    def test_empty(self):
        assert I.resolve_config_decisions([]) == []

    def test_non_execute(self):
        assert I.resolve_config_decisions([{"name": "write_file", "args": {}}]) == [
            {"type": "approve"}
        ]

    def test_execute_no_allowlist_prompts(self):
        with patch("metasci_scribe.agent._ensure_config", return_value=_cfg()):
            assert (
                I.resolve_config_decisions(
                    [{"name": "execute", "args": {"command": "rm -rf /"}}]
                )
                is None
            )

    def test_execute_allowlist_match(self):
        cfg = _cfg(shell_allow_list="ls,python")
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            assert I.resolve_config_decisions(
                [{"name": "execute", "args": {"command": "ls -la"}}]
            ) == [{"type": "approve"}]

    def test_auto_approve_config_approves(self):
        cfg = _cfg(auto_approve=True)
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            assert I.resolve_config_decisions(
                [{"name": "execute", "args": {"command": "ls"}}]
            ) == [{"type": "approve"}]

    def test_delete_is_never_cleared_by_the_allow_list(self):
        """delete has no `command` arg, so shell_allow_list must never clear it
        the way it clears execute (C1)."""
        cfg = _cfg(shell_allow_list="ls,python")
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            assert (
                I.resolve_config_decisions(
                    [{"name": "delete", "args": {"file_path": "/f.txt"}}]
                )
                is None
            )

    def test_run_in_background_not_allowlisted(self):
        cfg = _cfg(shell_allow_list="ls,cat")
        with patch("metasci_scribe.agent._ensure_config", return_value=cfg):
            assert (
                I.resolve_config_decisions(
                    [{"name": "run_in_background", "args": {"command": "rm -rf /"}}]
                )
                is None
            )

    def test_fail_closed_on_config_error(self):
        with patch(
            "metasci_scribe.agent._ensure_config", side_effect=RuntimeError("boom")
        ):
            assert (
                I.resolve_config_decisions(
                    [{"name": "execute", "args": {"command": "ls"}}]
                )
                is None
            )


class TestDecisionsAfterHumanApproval:
    def test_without_policy_rejections_everything_is_approved(self):
        reqs = [{"name": "a"}, {"name": "b"}]
        assert I.decisions_after_human_approval(reqs, {}) == [
            {"type": "approve"},
            {"type": "approve"},
        ]

    def test_policy_rejection_survives_a_human_approval(self):
        reqs = [{"name": "a"}, {"name": "b"}]
        reject = {"type": "reject", "message": "dangerous"}
        assert I.decisions_after_human_approval(reqs, {1: reject}) == [
            {"type": "approve"},
            reject,
        ]
