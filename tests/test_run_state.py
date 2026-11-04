"""Outline reading and the run state of autonomous mode (injected fakes only)."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
from filelock import FileLock

from metasci_scribe import run as run_mod
from metasci_scribe.auth.chatgpt_plan import NotSignedIn
from metasci_scribe.knowledge import SyncReport
from metasci_scribe.llm.credentials import ModelNotConfigured
from metasci_scribe.llm.errors import PlanAction, PlanStop
from metasci_scribe.run import (
    OutlineError,
    Paragraph,
    RunExit,
    RunState,
    RunStateError,
    make_mark_paragraph_tool,
    open_markers,
    outline_digest,
    parse_paragraphs,
    run_project,
)
from metasci_scribe.sessions import is_run_thread

OUTLINE = """\
# Outline

<!-- Template note. Headings look like:
### X9 · ① ◇ hidden example inside a comment
-->

핵심 메시지: Plants grow.

## Introduction

### I1 · ① Background
- 핵심 내용: why it matters

## Results

### R1 · ① First result <!-- ◇ author note -->
- 핵심 내용: growth rises

### R2 · ② Second result
- 핵심 내용: growth saturates

## Discussion

### D1 · ① Meaning
- 핵심 내용: so what
"""

IDS = ["I1", "R1", "R2", "D1"]

# The section file the fakes write. Any one serves: the guard and the checks
# read all of Manuscript/.
DRAFT = "Manuscript/Results.md"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "program.md").write_text("# Program\n", encoding="utf-8")
    (tmp_path / "outline.md").write_text(OUTLINE, encoding="utf-8")
    (tmp_path / "Manuscript").mkdir()
    (tmp_path / DRAFT).write_text("", encoding="utf-8")
    return tmp_path


def boom(thread_id: str, prompt: str) -> None:
    raise AssertionError("the agent must not be invoked")


def todo_ids(prompt: str) -> list[str]:
    """The IDs listed as still to write (lines such as '- R1  Results: ...')."""
    head = prompt.split("Rules:")[0]
    return [line.split()[1] for line in head.splitlines() if line.startswith("- ")]


def state_file(root: Path) -> Path:
    return root / ".metasci-scribe" / "run.json"


def mark(root: Path, pid: str, status: str = "done", reason: str = "") -> str:
    tool = make_mark_paragraph_tool(root)
    return tool.invoke({"paragraph_id": pid, "status": status, "reason": reason})


class Agent:
    """A fake model: each call marks the next ``plan`` entry and writes text."""

    def __init__(
        self, root: Path, plan: list[list[tuple[str, str, str]]], text: str = "Draft.\n"
    ):
        self.root = root
        self.plan = plan
        self.text = text
        self.calls: list[tuple[str, str]] = []

    def __call__(self, thread_id: str, prompt: str) -> None:
        self.calls.append((thread_id, prompt))
        step = (
            self.plan[len(self.calls) - 1] if len(self.calls) <= len(self.plan) else []
        )
        for pid, status, reason in step:
            mark(self.root, pid, status, reason)
        if step and self.text:
            with open(self.root / DRAFT, "a", encoding="utf-8") as f:
                f.write(self.text)


def all_done(root: Path, text: str = "Draft.\n") -> Agent:
    return Agent(root, [[(i, "done", "") for i in IDS]], text)


# ------------------------------------------------------------------ parsing


def test_parse_paragraph_ids():
    paras = parse_paragraphs(OUTLINE)
    assert [p.id for p in paras] == IDS
    assert paras[0] == Paragraph("I1", "Introduction", "Background")
    assert paras[1] == Paragraph("R1", "Results", "First result")
    assert paras[3].section == "Discussion"


def test_numbering_repeats_per_section_but_id_is_the_key():
    paras = parse_paragraphs("## A\n### I1 · ① x\n## B\n### R1 · ① y\n")
    assert [p.id for p in paras] == ["I1", "R1"]


def test_headings_inside_comments_are_skipped():
    text = "<!--\n### R1 · ① hidden\n-->\n### R2 · ① shown\n"
    assert [p.id for p in parse_paragraphs(text)] == ["R2"]


def test_template_outline_parses():
    from importlib import resources

    text = (
        resources.files("metasci_scribe")
        .joinpath("templates", "outline.md")
        .read_text(encoding="utf-8")
    )
    assert [p.id for p in parse_paragraphs(text)] == ["I1", "R1", "D1"]


def test_duplicate_id_reports_line():
    text = "## A\n### R1 · ① x\n\n### R1 · ② again\n"
    with pytest.raises(OutlineError) as err:
        parse_paragraphs(text)
    assert err.value.line == 4
    assert "R1" in str(err.value)


def test_malformed_paragraph_heading_reports_line():
    with pytest.raises(OutlineError) as err:
        parse_paragraphs("## A\n### R1 - ① x\n")
    assert err.value.line == 2


def test_other_h3_headings_are_ignored():
    assert parse_paragraphs("### Notes\n### R1 · ① x\n")[0].id == "R1"


def test_unterminated_comment_is_an_error_with_its_line():
    with pytest.raises(OutlineError) as err:
        parse_paragraphs("a\n<!-- never closed\n### R1 · ① x\n")
    assert err.value.line == 2


def test_open_markers_report_lines_outside_comments_only():
    text = "a ◇ one\n<!-- ◇ hidden\n◇ hidden -->\nb\n### R1 · ① ◇ title\n"
    assert open_markers(text) == [1, 5]


def test_open_markers_none_in_clean_outline():
    assert open_markers(OUTLINE) == []


def test_digest_ignores_comments_and_blank_lines_but_not_content():
    base = outline_digest(OUTLINE)
    assert outline_digest(OUTLINE.replace("Template note", "Another note")) == base
    assert outline_digest(OUTLINE.replace("\n\n", "\n\n\n")) == base
    assert outline_digest(OUTLINE.replace("growth rises", "growth falls")) != base
    assert outline_digest(OUTLINE.replace("R2 · ②", "R3 · ②")) != base


# ------------------------------------------------------------- the run gate


def test_open_marker_blocks_start(project):
    text = OUTLINE.replace("Plants grow.", "◇ undecided")
    (project / "outline.md").write_text(text, encoding="utf-8")
    result = run_project(project, boom)
    assert result.exit == RunExit.NEEDS_DECISION
    assert "outline.md" in result.message
    assert "7" in result.message
    assert not state_file(project).exists()


def test_outline_error_is_found_before_any_model_or_knowledge_cost(
    project, monkeypatch
):
    (project / "outline.md").write_text(
        "### R1 · ① a\n### R1 · ② b\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        run_mod,
        "sync_knowledge",
        lambda *a, **k: pytest.fail("sync before outline check"),
    )
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "line 2" in result.message


def test_outline_without_paragraphs_fails(project):
    (project / "outline.md").write_text("# Outline\n\n## Intro\n", encoding="utf-8")
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "paragraph" in result.message


def test_missing_files_suggest_init(project):
    (project / "outline.md").unlink()
    (project / "program.md").unlink()
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "/init" in result.message
    assert "outline.md" in result.message
    assert "program.md" in result.message


def test_outline_that_is_not_utf8_fails_cleanly(project):
    (project / "outline.md").write_bytes("### R1 · ① 가".encode("cp949"))
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "UTF-8" in result.message


def test_network_share_without_a_drive_letter_is_refused(project, monkeypatch):
    from metasci_scribe import drive_paths
    from metasci_scribe import project as project_mod

    monkeypatch.setattr(drive_paths, "_roots", [])
    monkeypatch.setattr(project_mod, "_IS_WINDOWS", True)
    result = run_project(Path("//host/share/papers/mine"), boom)
    assert result.exit == RunExit.FAILED
    assert "drive letter" in result.message


def test_folder_holding_the_data_dir_is_refused(project, monkeypatch):
    from metasci_scribe import paths

    monkeypatch.setattr(paths, "DATA_DIR", project / "inner" / "data")
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "workspace" in result.message
    assert not state_file(project).exists()


def test_missing_folder_fails(tmp_path):
    result = run_project(tmp_path / "nope", boom)
    assert result.exit == RunExit.FAILED


# ----------------------------------------------------------------- the loop


def test_complete_when_all_done(project):
    agent = all_done(project)
    result = run_project(project, agent)
    assert result.exit == RunExit.OK
    assert (result.done, result.total) == (4, 4)
    assert len(agent.calls) == 1
    first = agent.calls[0][1]
    assert all(i in first for i in IDS)
    assert "mark_paragraph" in first
    assert "[evidence needed]" in first


def test_complete_only_when_all_done(project):
    """The model saying it is finished does not count; pending paragraphs do."""
    agent = Agent(
        project,
        [
            [("I1", "done", ""), ("R1", "done", "")],
            [("R2", "done", ""), ("D1", "done", "")],
        ],
    )
    result = run_project(project, agent)
    assert result.exit == RunExit.OK
    assert len(agent.calls) == 2
    assert agent.calls[0][0] == agent.calls[1][0]
    second = agent.calls[1][1]
    assert "R2" in second
    assert "I1" not in second


def test_gives_up_after_three_passes(project):
    agent = Agent(project, [[("I1", "done", "")], [], []])
    result = run_project(project, agent)
    assert result.exit == RunExit.FAILED
    assert len(agent.calls) == run_mod.MAX_PASSES == 3
    assert (result.done, result.total) == (1, 4)
    assert "R1" in result.message


def test_blocked_stops_with_4(project):
    agent = Agent(
        project,
        [[("I1", "done", ""), ("R1", "blocked", "No source for the growth data")]],
    )
    result = run_project(project, agent)
    assert result.exit == RunExit.NEEDS_DECISION
    assert len(agent.calls) == 1
    assert "R1" in result.message
    assert "No source for the growth data" in result.message
    assert RunState.load(project).paragraphs["R1"] == "blocked"


def test_rerun_retries_blocked_paragraphs_in_same_thread(project):
    first = Agent(project, [[("I1", "done", ""), ("R1", "blocked", "need data")]])
    run_project(project, first)
    second = Agent(project, [[(i, "done", "") for i in ("R1", "R2", "D1")]])
    result = run_project(project, second)
    assert result.exit == RunExit.OK
    assert second.calls[0][0] == first.calls[0][0]
    assert RunState.load(project).reasons == {}


def test_usage_limit_exits_3_and_keeps_state(project):
    def limited(thread_id, prompt):
        mark(project, "I1")
        raise PlanStop(
            PlanAction.STOP_USAGE, "usage limit reached", code="usage_limit_reached"
        )

    result = run_project(project, limited)
    assert result.exit == RunExit.USAGE_LIMIT
    assert "settings" in result.message.lower()
    state = RunState.load(project)
    assert state.paragraphs["I1"] == "done"
    assert (result.done, result.total) == (1, 4)


def test_other_plan_stop_fails_with_code_and_request_id(project):
    def stopped(thread_id, prompt):
        raise PlanStop(
            PlanAction.STOP_INELIGIBLE,
            "plan not eligible",
            code="not_eligible",
            request_id="req_123",
        )

    result = run_project(project, stopped)
    assert result.exit == RunExit.FAILED
    assert "not_eligible" in result.message
    assert "req_123" in result.message


def test_plan_stop_message_is_redacted(project, monkeypatch):
    monkeypatch.setenv("TEST_FAKE_TOKEN", "sk-secretsecret1234")

    def stopped(thread_id, prompt):
        raise PlanStop(PlanAction.STOP_BAD_REQUEST, "bad sk-secretsecret1234 request")

    result = run_project(project, stopped)
    assert "sk-secretsecret1234" not in result.message


@pytest.mark.parametrize(
    "exc", [NotSignedIn("Not signed in"), ModelNotConfigured("openai")]
)
def test_sign_in_problems_point_to_login(project, exc):
    def failing(thread_id, prompt):
        raise exc

    result = run_project(project, failing)
    assert result.exit == RunExit.FAILED
    assert "/login" in result.message


def test_unexpected_error_fails_and_names_the_type(project):
    def failing(thread_id, prompt):
        mark(project, "I1")
        raise ValueError("kaput")

    result = run_project(project, failing)
    assert result.exit == RunExit.FAILED
    assert "ValueError" in result.message
    assert RunState.load(project).paragraphs["I1"] == "done"


def test_keyboard_interrupt_preserves_state_and_reraises(project):
    def interrupted(thread_id, prompt):
        mark(project, "I1")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_project(project, interrupted)
    state = RunState.load(project)  # still a valid file
    assert state is not None
    assert state.paragraphs["I1"] == "done"
    assert not list(state_file(project).parent.glob("*.tmp"))
    # The lock is released: the next run resumes in the same thread.
    again = all_done(project)
    assert run_project(project, again).exit == RunExit.OK
    assert again.calls[0][0] == state.thread_id


def test_resume_uses_same_thread_when_outline_same(project):
    first = Agent(project, [[("I1", "done", "")], [], []])
    run_project(project, first)
    second = Agent(project, [[(i, "done", "") for i in ("R1", "R2", "D1")]])
    result = run_project(project, second)
    assert result.exit == RunExit.OK
    assert second.calls[0][0] == first.calls[0][0]
    assert todo_ids(second.calls[0][1]) == ["R1", "R2", "D1"]


def test_comment_only_change_keeps_thread(project):
    first = Agent(project, [[("I1", "done", "")], [], []])
    run_project(project, first)
    edited = OUTLINE.replace("Template note", "A different note").replace(
        "<!-- ◇ author note -->", "<!-- ◇ new note -->"
    )
    (project / "outline.md").write_text(edited, encoding="utf-8")
    second = Agent(project, [[(i, "done", "") for i in ("R1", "R2", "D1")]])
    run_project(project, second)
    assert second.calls[0][0] == first.calls[0][0]


def test_outline_change_new_thread_keeps_done(project):
    first = Agent(project, [[("I1", "done", ""), ("R1", "done", "")], [], []])
    run_project(project, first)
    changed = OUTLINE.replace(
        "### R2 · ② Second result", "### R3 · ② Third result"
    ).replace("### D1 · ① Meaning", "### D1 · ① Meaning changed")
    (project / "outline.md").write_text(changed, encoding="utf-8")
    second = Agent(project, [[("R3", "done", ""), ("D1", "done", "")]])
    result = run_project(project, second)
    assert result.exit == RunExit.OK
    assert second.calls[0][0] != first.calls[0][0]
    state = RunState.load(project)
    assert list(state.paragraphs) == ["I1", "R1", "R3", "D1"]
    assert "R2" not in state.paragraphs
    assert todo_ids(second.calls[0][1]) == ["R3", "D1"]  # done stays done


def test_changed_done_paragraph_stays_done(project):
    run_project(project, all_done(project))
    changed = OUTLINE.replace("growth rises", "growth rises a lot")
    (project / "outline.md").write_text(changed, encoding="utf-8")
    result = run_project(project, boom)
    assert result.exit == RunExit.OK


def test_all_done_rerun_does_not_invoke(project):
    run_project(project, all_done(project))
    assert run_project(project, boom).exit == RunExit.OK


def test_evidence_needed_counted_not_failure(project):
    agent = all_done(
        project,
        "Claim [evidence needed]. Other [Evidence Needed] and [EVIDENCE NEEDED].\n",
    )
    result = run_project(project, agent)
    assert result.exit == RunExit.OK
    assert result.evidence_needed == 3
    assert "3" in result.message


def test_all_done_but_manuscript_empty_is_a_failure(project):
    result = run_project(project, all_done(project, text=""))
    assert result.exit == RunExit.FAILED
    assert "Manuscript/" in result.message
    assert (result.done, result.total) == (4, 4)


def test_all_done_but_manuscript_missing_is_a_failure(project):
    (project / DRAFT).unlink()
    result = run_project(project, all_done(project, text=""))
    assert result.exit == RunExit.FAILED


def test_manuscript_that_is_a_folder_is_read_as_empty(project):
    (project / DRAFT).unlink()
    (project / DRAFT).mkdir()
    result = run_project(project, all_done(project, text=""))
    assert result.exit == RunExit.FAILED


def test_manuscript_with_bad_bytes_is_still_readable(project):
    agent = all_done(project, text="")
    (project / DRAFT).write_bytes("가나다 [evidence needed]".encode("cp949"))
    result = run_project(project, agent)
    assert result.exit == RunExit.OK
    assert result.evidence_needed == 1


# --------------------------------------------------------------- knowledge


def test_knowledge_busy_fails_before_invoking(project, monkeypatch):
    seen = {}

    def fake_sync(root, **kwargs):
        seen["kwargs"] = kwargs
        return SyncReport([], 0, [], [], busy=True)

    monkeypatch.setattr(run_mod, "sync_knowledge", fake_sync)
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "another" in result.message.lower()
    assert seen["kwargs"]["wait"] > 0


def test_knowledge_notes_reach_the_result_message(project, monkeypatch):
    report = SyncReport([project / "Knowledge" / "a.pdf"], 0, [], [])
    monkeypatch.setattr(run_mod, "sync_knowledge", lambda root, **k: report)
    result = run_project(project, all_done(project))
    assert result.exit == RunExit.OK
    assert "Knowledge: read the text of 1 PDF" in result.message


def test_knowledge_oserror_fails(project, monkeypatch):
    def fail(root, **k):
        raise OSError("disk full")

    monkeypatch.setattr(run_mod, "sync_knowledge", fail)
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "disk full" in result.message


def test_second_run_on_same_folder_is_refused(project):
    lock = FileLock(str(project / ".metasci-scribe" / "run.lock"))
    (project / ".metasci-scribe").mkdir()
    with lock:
        result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "already" in result.message.lower()


# -------------------------------------------------------------------- state


def test_state_roundtrip_is_plain_utf8_json(project):
    state = RunState("t1", "abc", {"R1": "done", "R2": "blocked"}, {"R2": "자료 없음"})
    state.save(project)
    raw = json.loads(state_file(project).read_text(encoding="utf-8"))
    assert raw["paragraphs"] == {"R1": "done", "R2": "blocked"}
    assert "자료 없음" in state_file(project).read_text(encoding="utf-8")
    assert RunState.load(project) == state


def test_load_without_file_is_none(project):
    assert RunState.load(project) is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        '{"thread_id": "t", "outline_digest": "d", "paragraphs": {"R1": "weird"}, "reasons": {}}',
        '{"thread_id": 3, "outline_digest": "d", "paragraphs": {}, "reasons": {}}',
    ],
)
def test_corrupt_state_is_an_error_not_a_reset(project, content):
    state_file(project).parent.mkdir()
    state_file(project).write_text(content, encoding="utf-8")
    with pytest.raises(RunStateError):
        RunState.load(project)
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "run.json" in result.message
    assert state_file(project).read_text(encoding="utf-8") == content


def test_save_is_atomic_on_crash(project, monkeypatch):
    RunState("t1", "d1", {"R1": "done"}, {}).save(project)
    before = state_file(project).read_bytes()

    def crash(src, dst):
        raise OSError("crash before replace")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError, match="crash before replace"):
        RunState("t2", "d2", {"R1": "pending"}, {}).save(project)
    monkeypatch.undo()
    assert state_file(project).read_bytes() == before
    assert RunState.load(project).thread_id == "t1"
    assert not list(state_file(project).parent.glob("*.tmp"))


# --------------------------------------------------------- mark_paragraph


@pytest.fixture
def started(project):
    RunState("t1", "d", dict.fromkeys(IDS, "pending"), {}).save(project)
    return project


def test_mark_paragraph_marks_done(started):
    reply = mark(started, "R1")
    assert RunState.load(started).paragraphs["R1"] == "done"
    assert len(reply) < 80


def test_mark_paragraph_rejects_unknown_id(started):
    reply = mark(started, "Z9")
    assert "Unknown" in reply
    assert "Z9" not in RunState.load(started).paragraphs


def test_mark_paragraph_rejects_bad_status(started):
    reply = mark(started, "R1", "finished")
    assert "Rejected" in reply
    assert RunState.load(started).paragraphs["R1"] == "pending"


def test_blocked_needs_a_reason(started):
    assert "reason" in mark(started, "R1", "blocked", "   ")
    assert RunState.load(started).paragraphs["R1"] == "pending"
    mark(started, "R1", "blocked", "no data")
    state = RunState.load(started)
    assert state.paragraphs["R1"] == "blocked"
    assert state.reasons["R1"] == "no data"


def test_done_clears_an_earlier_reason(started):
    mark(started, "R1", "blocked", "no data")
    mark(started, "R1", "done")
    assert RunState.load(started).reasons == {}


def test_mark_twice_is_safe(started):
    mark(started, "R1")
    snapshot = state_file(started).read_bytes()
    reply = mark(started, "R1")
    assert state_file(started).read_bytes() == snapshot
    assert "already" in reply


def test_mark_without_a_run_is_rejected(project):
    assert "No active run" in mark(project, "R1")
    assert not state_file(project).exists()


def test_mark_with_unreadable_state_is_rejected(started):
    state_file(started).write_text("{oops", encoding="utf-8")
    assert "Rejected" in mark(started, "R1")


def test_mark_strips_the_id(started):
    mark(started, " R1 ")
    assert RunState.load(started).paragraphs["R1"] == "done"


def test_parallel_marks_do_not_lose_updates(started):
    threads = [threading.Thread(target=mark, args=(started, i)) for i in IDS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(RunState.load(started).paragraphs.values()) == {"done"}


def test_mark_paragraph_tool_schema_lists_the_two_statuses(started):
    schema = make_mark_paragraph_tool(started).args
    assert schema["status"]["enum"] == ["done", "blocked"]


def test_a_chat_session_never_gets_the_tool():
    import metasci_scribe.agent as agent_mod

    assert "mark_paragraph" not in Path(agent_mod.__file__).read_text(encoding="utf-8")
    from metasci_scribe import tools

    assert tools.make_mark_paragraph_tool is make_mark_paragraph_tool
    assert tools.__all__.count("make_mark_paragraph_tool") == 1


# ------------------------------------------------------- fix round 1


@pytest.mark.parametrize(
    "exc",
    [
        NotSignedIn("refresh_token=rt-abcdef Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig"),
        ValueError("Authorization: Bearer eyJhbGciOiJSUzI1NiJ9.payload.sig"),
    ],
)
def test_exception_details_stay_out_of_the_result(project, exc):
    def failing(thread_id, prompt):
        raise exc

    result = run_project(project, failing)
    assert result.exit == RunExit.FAILED
    assert type(exc).__name__ in result.message
    assert "eyJ" not in result.message
    assert "rt-abcdef" not in result.message
    assert str(exc) not in result.message


def test_model_not_configured_message_is_fixed(project):
    def failing(thread_id, prompt):
        raise ModelNotConfigured("openai")

    result = run_project(project, failing)
    assert "ModelNotConfigured" in result.message
    assert "/login" in result.message


def test_ok_message_does_not_claim_the_draft_is_checked(project):
    result = run_project(project, all_done(project))
    assert "marked 4 of 4 paragraphs done" in result.message
    assert "Done:" not in result.message
    assert "written" not in result.message
    assert "check" in result.message


def test_saved_state_is_read_after_the_lock_is_held(project, monkeypatch):
    """A run that finished while we waited for the lock is the state to build on."""
    real = run_mod.FileLock

    class Racing(real):
        def acquire(self, *args, **kwargs):
            out = super().acquire(*args, **kwargs)
            digest = outline_digest(OUTLINE)
            states = dict.fromkeys(IDS, "pending") | {"I1": "done"}
            RunState("t-other", digest, states, {}).save(project)
            return out

    monkeypatch.setattr(run_mod, "FileLock", Racing)
    agent = Agent(project, [[(i, "done", "") for i in ("R1", "R2", "D1")]])
    assert run_project(project, agent).exit == RunExit.OK
    assert agent.calls[0][0] == "t-other"
    assert todo_ids(agent.calls[0][1]) == ["R1", "R2", "D1"]


def test_first_prompt_after_outline_change_says_done_paragraphs_are_in_the_manuscript(
    project,
):
    run_project(
        project, Agent(project, [[("I1", "done", ""), ("R1", "done", "")], [], []])
    )
    changed = OUTLINE.replace("R2 · ② Second result", "R3 · ② Third result")
    (project / "outline.md").write_text(changed, encoding="utf-8")
    agent = Agent(project, [[("R3", "done", ""), ("D1", "done", "")]])
    run_project(project, agent)
    prompt = agent.calls[0][1]
    assert "The section files already hold" in prompt
    assert "I1, R1" in prompt
    assert "Keep them" in prompt


def test_first_prompt_of_a_fresh_run_has_no_kept_note(project):
    agent = all_done(project)
    run_project(project, agent)
    assert "already holds" not in agent.calls[0][1]


def test_tool_rejects_text_the_file_cannot_hold(started, monkeypatch):
    def cannot_encode(self, root):
        raise UnicodeEncodeError("utf-8", "\ud800", 0, 1, "surrogates not allowed")

    monkeypatch.setattr(RunState, "save", cannot_encode)
    assert "Rejected" in mark(started, "R1")


def test_tool_with_only_a_lone_surrogate_reason_is_rejected(started):
    assert "Rejected" in mark(started, "R1", "blocked", "\ud800")
    assert RunState.load(started).paragraphs["R1"] == "pending"


def test_tool_strips_control_characters_from_the_reason(started):
    mark(started, "R1", "blocked", "\x1b[31mno data\x07")
    assert RunState.load(started).reasons["R1"] == "[31mno data"


def test_blocked_reason_is_cleaned_and_cut_in_the_result(project):
    def hostile(thread_id, prompt):
        RunState(
            thread_id,
            outline_digest(OUTLINE),
            dict.fromkeys(IDS, "pending") | {"R1": "blocked"},
            {"R1": "\x1b]0;owned\x07\x1b[31m" + "x" * 2000},
        ).save(project)

    result = run_project(project, hostile)
    assert result.exit == RunExit.NEEDS_DECISION
    assert "\x1b" not in result.message
    assert "\x07" not in result.message
    assert len(result.message) < 900


def test_stale_temp_files_are_removed_at_start(project):
    folder = project / ".metasci-scribe"
    folder.mkdir()
    stale = folder / ".run.json.abc123.tmp"
    stale.write_text("{half", encoding="utf-8")
    other = folder / "keep.tmp"
    other.write_text("x", encoding="utf-8")
    assert run_project(project, all_done(project)).exit == RunExit.OK
    assert not stale.exists()
    assert other.exists()


# ------------------------------------------------------- fix round 1: the guard


def backups(root: Path) -> list[str]:
    return sorted(p.name for p in (root / ".metasci-scribe").glob("manuscript.pass-*"))


def test_each_pass_starts_with_a_backup_and_only_three_are_kept(project):
    (project / DRAFT).write_text("Kept line.\n", encoding="utf-8")
    run_project(project, Agent(project, [], text=""))  # 3 passes, nothing marked
    assert backups(project) == [
        "manuscript.pass-1",
        "manuscript.pass-2",
        "manuscript.pass-3",
    ]
    run_project(project, Agent(project, [], text=""))
    assert backups(project) == [
        "manuscript.pass-4",
        "manuscript.pass-5",
        "manuscript.pass-6",
    ]
    copy = project / ".metasci-scribe" / "manuscript.pass-6" / "Results.md"
    assert copy.read_text(encoding="utf-8") == "Kept line.\n"


def test_a_pass_that_overwrites_finished_text_fails_naming_the_backup(project):
    (project / DRAFT).write_text("Finished one.\nFinished two.\n", encoding="utf-8")

    def overwrite(thread_id, prompt):
        (project / DRAFT).write_text("Only new text.\n", encoding="utf-8")
        for pid in IDS:
            mark(project, pid)

    result = run_project(project, overwrite)
    assert result.exit == RunExit.FAILED
    assert ".metasci-scribe/manuscript.pass-1/" in result.message
    assert "Finished one." in result.message
    kept = project / ".metasci-scribe" / "manuscript.pass-1" / "Results.md"
    assert kept.read_text(encoding="utf-8") == "Finished one.\nFinished two.\n"


def test_loss_is_found_when_the_pass_ends_in_an_error_too(project):
    (project / DRAFT).write_text("Finished one.\n", encoding="utf-8")

    def overwrite_then_fail(thread_id, prompt):
        (project / DRAFT).write_text("", encoding="utf-8")
        raise PlanStop(PlanAction.STOP_USAGE, "usage limit reached")

    result = run_project(project, overwrite_then_fail)
    assert result.exit == RunExit.FAILED  # not USAGE_LIMIT: the draft is damaged
    assert "manuscript.pass-1" in result.message


def test_adding_moving_and_blank_line_changes_are_not_a_loss(project):
    (project / DRAFT).write_text("Alpha.\n\nBeta.\n", encoding="utf-8")

    def tidy(thread_id, prompt):
        (project / DRAFT).write_text(
            "New first.\n\n\n  Beta.  \nAlpha.\nNew last.\n", encoding="utf-8"
        )
        for pid in IDS:
            mark(project, pid)

    assert run_project(project, tidy).exit == RunExit.OK


def test_no_manuscript_means_nothing_to_back_up(project):
    (project / DRAFT).unlink()

    def write(thread_id, prompt):
        (project / DRAFT).write_text("Fresh.\n", encoding="utf-8")
        for pid in IDS:
            mark(project, pid)

    assert run_project(project, write).exit == RunExit.OK
    assert backups(project) == []


def test_a_backup_that_cannot_be_written_stops_before_the_model(project, monkeypatch):
    (project / DRAFT).write_text("Finished.\n", encoding="utf-8")

    def no_space(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(run_mod, "_backup_manuscript", no_space)
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "backed up" in result.message


def test_a_mark_after_text_was_lost_is_rejected_and_the_run_fails(project):
    def lose_then_mark(thread_id, prompt):
        manuscript = project / DRAFT
        manuscript.write_text("R1 paragraph.\n", encoding="utf-8")
        assert "marked done" in mark(project, "R1")
        manuscript.write_text("A different text.\n", encoding="utf-8")  # R1 gone
        reply = mark(project, "R2")
        assert reply.startswith("Rejected")
        assert RunState.load(project).paragraphs["R2"] == "pending"

    result = run_project(project, lose_then_mark)
    assert result.exit == RunExit.FAILED
    assert "manuscript.mark" in result.message
    snapshot = project / ".metasci-scribe" / "manuscript.mark" / "Results.md"
    assert snapshot.read_text(encoding="utf-8") == "R1 paragraph.\n"


def test_loss_after_the_last_mark_is_found_at_the_end_of_the_pass(project):
    def lose_after_marking(thread_id, prompt):
        manuscript = project / DRAFT
        manuscript.write_text("R1 paragraph.\nR2 paragraph.\n", encoding="utf-8")
        for pid in IDS:
            mark(project, pid)
        manuscript.write_text("R2 paragraph.\n", encoding="utf-8")

    result = run_project(project, lose_after_marking)
    assert result.exit == RunExit.FAILED
    assert "manuscript.mark" in result.message
    assert "R1 paragraph." in result.message


def test_marks_and_snapshots_of_an_earlier_pass_do_not_leak_into_the_next(project):
    """A person may edit the section files between runs; the next run starts clean."""

    def first(thread_id, prompt):
        (project / DRAFT).write_text("R1 paragraph.\n", encoding="utf-8")
        mark(project, "R1")

    run_project(project, first)
    (project / DRAFT).write_text("Rewritten by the author.\n", encoding="utf-8")

    def second(thread_id, prompt):
        with open(project / DRAFT, "a", encoding="utf-8") as f:
            f.write("more\n")
        for pid in IDS:
            mark(project, pid)

    assert run_project(project, second).exit == RunExit.OK


# ----------------------------------------------------- section files and runs


def test_the_prompt_names_each_paragraphs_section_file(project):
    agent = all_done(project)
    run_project(project, agent)
    prompt = agent.calls[0][1]
    assert "- R1  Results: First result -> Manuscript/Results.md" in prompt
    assert "- I1  Introduction: Background -> Manuscript/Introduction.md" in prompt
    assert "Never edit manuscript.md" in prompt


def test_a_run_joins_the_section_files_into_manuscript_md(project):
    def write(thread_id, prompt):
        (project / "Manuscript" / "Introduction.md").write_text(
            "I1.\n", encoding="utf-8"
        )
        (project / DRAFT).write_text("R1.\n\nR2.\n", encoding="utf-8")
        for pid in IDS:
            mark(project, pid)

    result = run_project(project, write)
    assert result.exit == RunExit.OK
    assert (project / "manuscript.md").read_text(encoding="utf-8") == (
        "## Introduction\n\nI1.\n\n## Results\n\nR1.\n\nR2.\n"
    )
    assert "manuscript.md was updated" in result.message


def test_a_section_run_writes_only_that_section(project):
    agent = Agent(project, [[("R1", "done", ""), ("R2", "done", "")]])
    result = run_project(project, agent, section="results")
    assert result.exit == RunExit.OK
    assert todo_ids(agent.calls[0][1]) == ["R1", "R2"]
    assert "Edit only Manuscript/Results.md." in agent.calls[0][1]
    assert (result.done, result.total) == (2, 2)
    assert "2 of 2 paragraphs of Results done" in result.message
    saved = RunState.load(project).paragraphs
    assert saved == {"I1": "pending", "R1": "done", "R2": "done", "D1": "pending"}


def test_a_section_run_has_its_own_thread_and_a_whole_run_keeps_its_own(project):
    section = Agent(project, [[("R1", "done", ""), ("R2", "done", "")]])
    run_project(project, section, section="Results")
    whole = Agent(project, [[("I1", "done", ""), ("D1", "done", "")]])
    run_project(project, whole)
    state = RunState.load(project)
    assert whole.calls[0][0] == state.thread_id
    assert section.calls[0][0] != state.thread_id
    assert is_run_thread(section.calls[0][0])  # hidden from /threads like the rest
    assert todo_ids(whole.calls[0][1]) == ["I1", "D1"]


def test_an_unknown_section_is_refused_before_the_model(project):
    result = run_project(project, boom, section="Methods")
    assert result.exit == RunExit.FAILED
    assert "Introduction, Results, Discussion" in result.message


def test_open_marks_in_other_sections_do_not_stop_a_section_run(project):
    marked = OUTLINE.replace("### D1 · ① Meaning", "### D1 · ① ◇ Meaning")
    (project / "outline.md").write_text(marked, encoding="utf-8")
    assert run_project(project, boom).exit == RunExit.NEEDS_DECISION
    agent = Agent(project, [[("R1", "done", ""), ("R2", "done", "")]])
    assert run_project(project, agent, section="Results").exit == RunExit.OK
    assert (
        run_project(project, boom, section="Discussion").exit == RunExit.NEEDS_DECISION
    )


def test_open_marks_about_the_whole_paper_stop_a_section_run(project):
    marked = OUTLINE.replace("핵심 메시지: Plants grow.", "핵심 메시지: ◇")
    (project / "outline.md").write_text(marked, encoding="utf-8")
    result = run_project(project, boom, section="Results")
    assert result.exit == RunExit.NEEDS_DECISION


def test_two_sections_that_share_a_file_are_refused(project):
    clash = OUTLINE.replace("## Discussion", "## Results.")  # same file name
    (project / "outline.md").write_text(clash, encoding="utf-8")
    result = run_project(project, boom)
    assert result.exit == RunExit.FAILED
    assert "share the file" in result.message


def test_text_moved_between_section_files_is_not_a_loss(project):
    (project / DRAFT).write_text("Alpha.\n", encoding="utf-8")

    def move(thread_id, prompt):
        (project / DRAFT).write_text("", encoding="utf-8")
        (project / "Manuscript" / "Discussion.md").write_text(
            "Alpha.\n", encoding="utf-8"
        )
        for pid in IDS:
            mark(project, pid)

    assert run_project(project, move).exit == RunExit.OK
