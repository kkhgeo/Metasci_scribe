"""Tests for metasci_scribe/prompts.py."""

import pytest

from metasci_scribe.prompts import (
    DELEGATION_STRATEGY,
    METASCI_SCRIBE_IDENTITY,
    SHELL_GUIDELINES,
    WRITING_GUIDELINES,
    WRITING_WORKFLOW,
    get_system_prompt,
)


class TestGetSystemPrompt:
    def test_returns_non_empty(self):
        result = get_system_prompt()
        assert isinstance(result, str)
        assert len(result) > 100

    def test_contains_identity(self):
        result = get_system_prompt()
        assert "Metasci Scribe" in result
        assert "paper" in result

    def test_contains_workflow(self):
        result = get_system_prompt()
        assert "Writing Workflow" in result

    def test_contains_writing_guidelines(self):
        result = get_system_prompt()
        assert "Writing Guidelines" in result

    def test_contains_shell_guidelines(self):
        result = get_system_prompt()
        assert "Shell Execution Guidelines" in result

    def test_contains_delegation(self):
        result = get_system_prompt()
        assert "Sub-Agent Delegation" in result

    def test_no_numeric_limits(self):
        result = get_system_prompt()
        assert "{max_concurrent}" not in result
        assert "{max_iterations}" not in result

    def test_workflow_constant_not_empty(self):
        assert len(WRITING_WORKFLOW) > 0

    def test_delegation_no_placeholders(self):
        assert "{max_concurrent}" not in DELEGATION_STRATEGY
        assert "{max_iterations}" not in DELEGATION_STRATEGY

    def test_section_ordering(self):
        """Identity must precede workflow; workflow must precede delegation."""
        result = get_system_prompt()
        idx_identity = result.find("# Identity")
        idx_workflow = result.find("# Writing Workflow")
        idx_delegation = result.find("# Sub-Agent Delegation")
        assert 0 <= idx_identity < idx_workflow < idx_delegation

    def test_does_not_contain_static_date(self):
        """Date is injected per-turn by runtime context, not baked into static prompt.

        Static prompt must stay byte-stable across midnight so the cache prefix
        survives. See RuntimeContextMiddleware for runtime injection.
        """
        import re

        result = get_system_prompt()
        assert not re.search(r"Current date: \d{4}-\d{2}-\d{2}", result)

    def test_no_stale_memory_path_singular(self):
        """Backend route is `/memories/`, not `/memory/`. Catch silent-bug regressions.

        Anything sent to `/memory/...` falls through to CustomSandboxBackend
        (workspace files), bypassing the persistent FilesystemBackend that
        owns persistent memory files.
        """
        result = get_system_prompt()
        # `/memory/` as a filesystem path (after a backtick or whitespace, before
        # an alpha char or another /). Excludes word-list usages like
        # "context/memory/web search".
        import re

        assert not re.search(r"[\s`]/memory/[a-zA-Z]", result), (
            "Found `/memory/<file>` in system prompt — should be `/memories/<file>`"
        )


class TestScribeIdentity:
    def test_constant_not_empty(self):
        assert len(METASCI_SCRIBE_IDENTITY) > 0

    def test_states_role(self):
        assert "You are Metasci Scribe" in METASCI_SCRIBE_IDENTITY

    def test_author_owns_the_claims(self):
        assert "The author owns the claims" in METASCI_SCRIBE_IDENTITY


class TestWritingGuidelines:
    def test_constant_not_empty(self):
        assert len(WRITING_GUIDELINES) > 0

    def test_follows_the_authors_language(self):
        assert "language" in WRITING_GUIDELINES.lower()

    def test_leaves_specialist_rules_to_skills(self):
        assert "skills" in WRITING_GUIDELINES


class TestShellGuidelines:
    def test_constant_not_empty(self):
        assert len(SHELL_GUIDELINES) > 0

    def test_mentions_timeout_limit(self):
        assert "300" in SHELL_GUIDELINES  # default timeout
        assert "3600" in SHELL_GUIDELINES  # per-command override ceiling

    def test_is_for_an_attended_conversation(self):
        assert "the author approves each command" in SHELL_GUIDELINES
        assert "only in a conversation" in SHELL_GUIDELINES

    def test_no_long_running_jobs(self):
        lowered = SHELL_GUIDELINES.lower()
        assert "do not use `run_in_background`" in lowered
        assert "gpu" not in lowered
        assert "output.log" not in lowered

    def test_not_duplicated_in_workflow(self):
        """SHELL_GUIDELINES content should live ONLY in its own constant."""
        # Sentinel phrase unique to SHELL_GUIDELINES
        assert "Sandbox limits" not in WRITING_WORKFLOW


class TestDangerousShellGuidelines:
    def test_default_uses_the_sandbox_wording(self):
        result = get_system_prompt()
        assert "Sandbox limits" in result
        assert "DANGEROUS MODE" not in result

    def test_dangerous_swaps_guidelines(self):
        result = get_system_prompt(dangerous=True, cwd="/Users/me/ws/demo")
        assert "DANGEROUS MODE" in result
        assert "/Users/me/ws/demo" in result
        # sandbox wording is gone
        assert "Sandbox limits" not in result
        assert "the author approves each command" not in result
        # privileged-command blocklist still advertised
        assert "sudo" in result
        assert "rm -rf /" in result

    def test_dangerous_without_cwd_falls_back(self):
        result = get_system_prompt(dangerous=True)
        assert "DANGEROUS MODE" in result


class TestBashOnWindowsShellGuidelines:
    def test_off_by_default(self):
        assert "Git for Windows" not in get_system_prompt()
        assert "Git for Windows" not in get_system_prompt(dangerous=True)

    def test_sandbox_names_the_shell_without_windows_paths(self):
        result = get_system_prompt(bash_on_windows=True)
        assert "The shell is bash (Git for Windows)." in result
        assert "Write Windows paths" not in result
        assert "C:/Users/you" not in result
        assert "Sandbox limits" in result

    @pytest.mark.parametrize("dangerous", [False, True])
    def test_both_modes_explain_git_bash_path_conversion(self, dangerous):
        result = get_system_prompt(dangerous=dangerous, bash_on_windows=True)
        assert "any other argument starting with `/` arrives as a path" in result
        assert "start the command with `MSYS_NO_PATHCONV=1`" in result
        assert "GNU tools (grep, sed, awk, ...) get arguments unchanged" in result
        assert "MSYS_NO_PATHCONV" not in get_system_prompt(dangerous=dangerous)

    def test_dangerous_shows_windows_paths_with_forward_slashes(self):
        result = get_system_prompt(
            dangerous=True,
            cwd=r"C:\Users\me\ws\demo",
            bash_on_windows=True,
        )
        assert "Your current working directory is `C:/Users/me/ws/demo`" in result
        assert "(e.g. `C:/Users/you/Documents/file.txt`)" in result
        assert "`/Users/you/Documents/file.txt`" not in result
        assert "The shell is bash from Git for Windows." in result

    def test_agent_prompt_follows_the_shell_decision(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from metasci_scribe import agent as evo
        from metasci_scribe import agent_shell

        cfg = SimpleNamespace(dangerous_mode=False)
        assert "Git for Windows" not in evo._configured_system_prompt(cfg, tmp_path)
        monkeypatch.setattr(agent_shell, "agent_bash", lambda: object())
        assert "Git for Windows" in evo._configured_system_prompt(cfg, tmp_path)
