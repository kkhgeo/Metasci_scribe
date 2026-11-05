"""The paragraph pipeline of a run: write, review, rewrite, one paragraph at a time."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from metasci_scribe import run_stages
from metasci_scribe.run import FINAL, RunExit, RunState, run_project
from metasci_scribe.run_stages import Check
from tests.test_run_state import OUTLINE, mark

ROUND = re.compile(r"(Reviews/[^\s]+?/r\d+)/")
PID = re.compile(r"ONE paragraph[^:]*?:\s*\n?(\w+\d+) ")


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "program.md").write_text("language: en\n", encoding="utf-8")
    (tmp_path / "outline.md").write_text(OUTLINE, encoding="utf-8")
    # The skills' checker scripts are tested on their own; here they pass.
    monkeypatch.setattr(run_stages, "missing_skills", lambda *a: [])
    monkeypatch.setattr(run_stages, "check_review", lambda *a: Check(True, ""))
    monkeypatch.setattr(run_stages, "check_revision", lambda *a: Check(True, ""))
    return tmp_path


class Writer:
    """A fake model that plays each turn the pipeline asks for."""

    def __init__(self, root: Path, findings: dict[str, str], anchor: bool = False):
        self.root = root
        self.findings = findings  # ID -> severity of its one finding, or ""
        self.anchor = anchor
        self.turns: list[tuple[str, str]] = []  # (kind, ID)
        self.threads: list[str] = []

    def __call__(self, thread_id: str, prompt: str) -> None:
        self.threads.append(thread_id)
        if "refused it" in prompt:  # the follow-up after a refused revision
            self.turns.append(("rewrite", self.turns[-1][1]))
            folder = self.root / ROUND.search(prompt).group(1)
            (folder / "revised.txt").write_text("Again.\n", encoding="utf-8")
            return
        pid = re.search(r"(\b[A-Z]\d+)\s{2}", prompt).group(1)
        section = (
            "Results"
            if pid.startswith("R")
            else ("Introduction" if pid.startswith("I") else "Discussion")
        )
        if prompt.startswith("You are running unattended.\n") or "Write ONE" in prompt:
            pass
        if "Write ONE paragraph" in prompt:
            self.turns.append(("write", pid))
            path = self.root / "Manuscript" / f"{section}.md"
            path.parent.mkdir(exist_ok=True)
            old = path.read_text(encoding="utf-8") if path.exists() else ""
            head = f"<!-- {pid} -->\n" if self.anchor else ""
            path.write_text(
                old + ("\n" if old else "") + f"{head}Draft of {pid} with 30%.\n",
                encoding="utf-8",
            )
            mark(self.root, pid)
        elif "Review ONE paragraph" in prompt:
            self.turns.append(("review", pid))
            folder = self.root / ROUND.search(prompt).group(1)
            severity = self.findings.get(pid, "")
            body = f"### 1. [{severity}] A problem\n" if severity else ""
            (folder / "review.md").write_text(
                f"# 메타리뷰\n## 지적 사항\n{body or '없음'}\n", encoding="utf-8"
            )
        elif "Rewrite ONE paragraph" in prompt:
            self.turns.append(("rewrite", pid))
            folder = self.root / ROUND.search(prompt).group(1)
            (folder / "revised.txt").write_text(
                f"Better {pid} with 30%.\n", encoding="utf-8"
            )


def test_each_paragraph_is_written_reviewed_and_rewritten_in_turn(project):
    agent = Writer(project, {"R1": "주요", "R2": "경미"})
    said: list[str] = []
    result = run_project(
        project, agent, section="Results", review=True, say=said.append
    )
    assert result.exit == RunExit.OK, result.message
    assert agent.turns == [
        ("write", "R1"),
        ("review", "R1"),
        ("rewrite", "R1"),
        ("write", "R2"),
        ("review", "R2"),
    ]
    text = (project / "Manuscript" / "Results.md").read_text(encoding="utf-8")
    assert text == (
        "<!-- R1 -->\nBetter R1 with 30%.\n\n<!-- R2 -->\nDraft of R2 with 30%.\n"
    )
    assert (project / "Reviews/Results/R1/r1/draft.txt").read_text(
        encoding="utf-8"
    ) == "Draft of R1 with 30%.\n"
    assert any("R1 reviewed" in s and "rewriting" in s for s in said)
    assert any("R2 reviewed" in s and "kept as written" in s for s in said)
    assert "Reviews/" in result.message
    # Every turn has its own thread, and none is the chat's.
    assert len(set(agent.threads)) == len(agent.threads)
    assert all(t.startswith("run-") for t in agent.threads)
    joined = (project / "manuscript.md").read_text(encoding="utf-8")
    assert "<!--" not in joined


def test_a_review_or_rewrite_turn_cannot_change_a_section_file(project):
    class Meddler(Writer):
        def __call__(self, thread_id, prompt):
            super().__call__(thread_id, prompt)
            if "Review ONE" in prompt:
                (self.root / "Manuscript" / "Results.md").write_text(
                    "gone\n", encoding="utf-8"
                )

    agent = Meddler(project, {}, anchor=True)
    said: list[str] = []
    result = run_project(
        project, agent, section="Results", review=True, say=said.append
    )
    assert result.exit == RunExit.OK, result.message
    text = (project / "Manuscript" / "Results.md").read_text(encoding="utf-8")
    assert "Draft of R1" in text
    assert "Draft of R2" in text
    assert any("undone" in s for s in said)


def test_a_refused_revision_keeps_the_written_paragraph(project, monkeypatch):
    monkeypatch.setattr(
        run_stages, "check_revision", lambda *a: Check(False, "ABORT numbers", 2)
    )
    agent = Writer(project, {"R1": "치명"})
    result = run_project(project, agent, section="Results", review=True)
    assert result.exit == RunExit.OK
    text = (project / "Manuscript" / "Results.md").read_text(encoding="utf-8")
    assert "Draft of R1" in text
    assert "Better R1" not in text
    assert "refused" in result.message
    # Asked once more with the check's output before giving up.
    assert agent.turns.count(("rewrite", "R1")) == 2


def test_a_stopped_run_continues_with_the_unfinished_stage(project):
    class Stops(Writer):
        def __call__(self, thread_id, prompt):
            if "Rewrite ONE" in prompt and not getattr(self, "stopped", False):
                self.stopped = True
                raise RuntimeError("network")
            super().__call__(thread_id, prompt)

    agent = Stops(project, {"R1": "주요"})
    first = run_project(project, agent, section="Results", review=True)
    assert first.exit == RunExit.FAILED
    assert RunState.load(project).stages == {"R1": "reviewed"}
    second = run_project(project, agent, section="Results", review=True)
    assert second.exit == RunExit.OK, second.message
    # The second run went straight to the rewrite of R1, then on to R2.
    assert agent.turns[-3:] == [("rewrite", "R1"), ("write", "R2"), ("review", "R2")]
    assert RunState.load(project).stages == {"R1": FINAL, "R2": FINAL}


def test_paragraphs_written_before_paragraph_review_are_left_alone(project):
    (project / "Manuscript").mkdir()
    (project / "Manuscript" / "Results.md").write_text("Old R1.\n", encoding="utf-8")
    state = RunState(
        "run-" + "0" * 32,
        "x",
        {"I1": "pending", "R1": "done", "R2": "pending", "D1": "pending"},
    )
    state.save(project)
    agent = Writer(project, {})
    result = run_project(project, agent, section="Results", review=True)
    assert result.exit == RunExit.OK, result.message
    assert ("review", "R1") not in agent.turns
    assert "before paragraph review" in result.message


def test_the_korean_manuscript_gets_the_korean_skills(project):
    (project / "program.md").write_text("language: ko\n", encoding="utf-8")
    prompts: list[str] = []

    def agent(thread_id, prompt):
        prompts.append(prompt)
        Writer(project, {"R1": "주요"})(thread_id, prompt)

    run_project(project, agent, section="Results", review=True)
    assert "/skills/scribe-writing-korean/SKILL.md" in prompts[0]
    assert "/skills/scribe-review/SKILL.md" in prompts[1]
    assert "/skills/scribe-rewriting-korean/SKILL.md" in prompts[2]
    assert "[근거 필요]" in prompts[0]


def test_without_the_skills_the_review_run_says_where_to_put_them(project, monkeypatch):
    monkeypatch.setattr(
        run_stages, "missing_skills", lambda *a: ["scribe-writing", "scribe-review"]
    )
    agent = Writer(project, {})
    result = run_project(project, agent, section="Results", review=True)
    assert result.exit == RunExit.FAILED
    assert "scribe-writing, scribe-review" in result.message
    assert "--quick" in result.message
    assert agent.turns == []
