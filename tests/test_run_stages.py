"""The helpers of the paragraph pipeline: anchors, language, reviews, checks."""

from __future__ import annotations

from collections import Counter

import pytest

from metasci_scribe import run_stages
from metasci_scribe.manuscript import joined_text
from metasci_scribe.run_stages import (
    anchor_new_paragraph,
    count_findings,
    manuscript_language,
    needs_rewrite,
    next_round,
    paragraph_text,
    replace_paragraph,
    review_dir,
    single_paragraph,
    skills_for,
)

SECTION = "<!-- R1 -->\nFirst para line one.\nLine two.\n\n<!-- R2 -->\nSecond para.\n"


def test_a_paragraph_is_found_by_its_anchor():
    assert paragraph_text(SECTION, "R1") == "First para line one.\nLine two."
    assert paragraph_text(SECTION, "R2") == "Second para."
    assert paragraph_text(SECTION, "R3") is None


def test_an_anchor_on_its_own_block_still_names_the_next_paragraph():
    text = "<!-- R1 -->\n\nFirst.\n\n<!-- R2 -->\nSecond.\n"
    assert paragraph_text(text, "R1") == "First."


def test_the_write_turn_s_new_paragraph_gets_its_anchor():
    before = "<!-- R1 -->\nFirst.\n"
    after = "<!-- R1 -->\nFirst.\n\nNew second paragraph.\n"
    anchored = anchor_new_paragraph(before, after, "R2")
    assert anchored == "<!-- R1 -->\nFirst.\n\n<!-- R2 -->\nNew second paragraph.\n"
    # An anchor the agent wrote is kept as it is.
    own = "<!-- R1 -->\nFirst.\n\n<!-- R2 -->\nMine.\n"
    assert paragraph_text(anchor_new_paragraph(before, own, "R2"), "R2") == "Mine."


def test_two_new_paragraphs_cannot_be_told_apart():
    assert anchor_new_paragraph("", "One.\n\nTwo.\n", "R1") is None
    assert anchor_new_paragraph("Same.\n", "Same.\n", "R1") is None


def test_a_revision_replaces_only_its_paragraph():
    out = replace_paragraph(SECTION, "R1", "Better first.\n")
    assert out == "<!-- R1 -->\nBetter first.\n\n<!-- R2 -->\nSecond para.\n"
    assert replace_paragraph(SECTION, "R9", "x") is None


def test_a_revision_must_be_one_paragraph():
    assert (
        single_paragraph("  One paragraph.\nStill it.\n") == "One paragraph.\nStill it."
    )
    assert single_paragraph("<!-- R1 -->\nOne.\n") == "One."
    assert single_paragraph("One.\n\nTwo.") is None
    assert single_paragraph("\n") is None


def test_the_joined_manuscript_has_no_anchors(tmp_path):
    (tmp_path / "Manuscript").mkdir()
    (tmp_path / "Manuscript" / "Results.md").write_text(SECTION, encoding="utf-8")
    text, _unused = joined_text(tmp_path, ["Results"])
    assert "<!--" not in text
    assert "First para line one.\nLine two.\n\nSecond para." in text


def test_the_language_comes_from_program_md_then_the_text():
    assert manuscript_language("language: en\n", "한국어 개요") == "en"
    assert manuscript_language("language: auto\n", "결과는 세 축으로 요약된다") == "ko"
    assert manuscript_language("", "Growth rises across sites.") == "en"
    assert skills_for("ko").writing == "scribe-writing-korean"
    assert skills_for("en").rewriting == "scribe-rewriting"
    assert skills_for("en").review == skills_for("ko").review == "scribe-review"


def test_review_rounds_and_findings(tmp_path):
    folder = review_dir(tmp_path, "Results", "R3")
    assert folder == tmp_path / "Reviews" / "Results" / "R3"
    assert next_round(folder).name == "r1"
    (folder / "r1").mkdir(parents=True)
    assert next_round(folder).name == "r2"
    report = "### 1. [주요] A\n### 2. [경미] B\n### 3. [경미] C\n"
    assert count_findings(report) == Counter({"경미": 2, "주요": 1})
    assert needs_rewrite(count_findings(report))
    assert not needs_rewrite(count_findings("### 1. [경미] only\n"))


def test_a_missing_checker_does_not_stop_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(run_stages, "skill_dirs", lambda root: [tmp_path])
    check = run_stages.check_review(tmp_path, tmp_path, tmp_path, None)
    assert check.passed
    assert "not checked" in check.output


def test_the_english_preservation_check_fails_only_on_numbers_and_citations(
    tmp_path, monkeypatch
):
    script = tmp_path / "scribe-review" / "scripts" / "check_preservation.py"
    script.parent.mkdir(parents=True)
    (script.parent.parent / "SKILL.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(run_stages, "skill_dirs", lambda root: [tmp_path])
    for output, passed in (
        ("FAIL hedges: removed=['may']", True),
        ("FAIL numbers: removed=['30%']", False),
    ):
        script.write_text(f"import sys\nprint({output!r})\nsys.exit(1)\n", "utf-8")
        assert run_stages.check_revision(tmp_path, tmp_path, "en").passed is passed


def test_the_real_korean_check_refuses_a_lost_number_and_passes_a_rewording(tmp_path):
    # The checker is the author's own and lives outside the repository.
    if run_stages.find_skill(tmp_path, "scribe-rewriting-korean") is None:
        pytest.skip("scribe-rewriting-korean is not installed on this machine")
    draft = "질산염 농도와 수온의 상관은 0.62였다(그림 3a).\n"
    (tmp_path / "draft.txt").write_text(draft, encoding="utf-8")
    (tmp_path / "revised.txt").write_text(
        "수온과 질산염 농도는 함께 변했다(그림 3a).\n",
        encoding="utf-8",
    )
    assert not run_stages.check_revision(tmp_path, tmp_path, "ko").passed
    (tmp_path / "revised.txt").write_text(
        "수온은 질산염 농도와 0.62의 상관을 보였다(그림 3a).\n",
        encoding="utf-8",
    )
    assert run_stages.check_revision(tmp_path, tmp_path, "ko").passed


def test_skills_are_found_in_the_paper_then_the_author_s_folder(tmp_path, monkeypatch):
    from metasci_scribe import paths

    home = tmp_path / "home"
    (home / "scribe-review").mkdir(parents=True)
    (home / "scribe-review" / "SKILL.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(paths, "GLOBAL_SKILLS_DIR", home)
    paper = tmp_path / "paper"
    assert run_stages.find_skill(paper, "scribe-review") == home / "scribe-review"
    own = paper / ".metasci-scribe" / "skills" / "scribe-review"
    own.mkdir(parents=True)
    (own / "SKILL.md").write_text("x", encoding="utf-8")
    assert run_stages.find_skill(paper, "scribe-review") == own
    assert "scribe-writing-korean" in run_stages.missing_skills(paper, "ko")
