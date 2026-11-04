"""Section files under Manuscript/ and the joined manuscript.md."""

from __future__ import annotations

from pathlib import Path

import pytest

from metasci_scribe.manuscript import (
    join_manuscript,
    section_file_name,
    section_names,
)

OUTLINE = """\
# Outline

<!--
## Not a section
-->

## Introduction

### I1 · ① Background

## Results

### R1 · ① First

## Discussion
"""

SECTIONS = ["Introduction", "Results", "Discussion"]


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    (tmp_path / "Manuscript").mkdir()
    return tmp_path


def write(root: Path, name: str, text: str) -> None:
    (root / "Manuscript" / name).write_text(text, encoding="utf-8")


def joined(root: Path) -> str:
    return (root / "manuscript.md").read_text(encoding="utf-8")


def test_sections_are_read_in_order_without_comments():
    assert section_names(OUTLINE) == SECTIONS


@pytest.mark.parametrize(
    ("section", "name"),
    [
        ("Results", "Results.md"),
        ("Materials and Methods", "Materials and Methods.md"),
        ("Results: part 1/2", "Results- part 1-2.md"),
        ("서론", "서론.md"),
        ("Con", "Con_.md"),
        ("...", "Section.md"),
    ],
)
def test_a_section_name_becomes_a_windows_safe_file_name(section, name):
    assert section_file_name(section) == name


def test_the_join_follows_the_outline_and_adds_the_headings(paper):
    write(paper, "Results.md", "R1 text.\n")
    write(paper, "Introduction.md", "## Introduction\n\nI1 text.\n")
    result = join_manuscript(paper, SECTIONS)
    assert result.written
    assert joined(paper) == "## Introduction\n\nI1 text.\n\n## Results\n\nR1 text.\n"


def test_an_unchanged_join_writes_nothing(paper):
    write(paper, "Results.md", "R1 text.\n")
    assert join_manuscript(paper, SECTIONS).written
    assert not join_manuscript(paper, SECTIONS).written


def test_a_join_updates_its_own_earlier_text(paper):
    write(paper, "Results.md", "R1 text.\n")
    join_manuscript(paper, SECTIONS)
    write(paper, "Results.md", "R1 text, revised.\n")
    assert join_manuscript(paper, SECTIONS).written
    assert "revised" in joined(paper)


def test_text_the_app_did_not_write_is_left_alone(paper):
    (paper / "manuscript.md").write_text("My own draft.\n", encoding="utf-8")
    write(paper, "Results.md", "R1 text.\n")
    result = join_manuscript(paper, SECTIONS)
    assert not result.written
    assert "delete manuscript.md" in result.held_back
    assert joined(paper) == "My own draft.\n"


def test_a_hand_edit_after_a_join_is_left_alone_until_deleted(paper):
    write(paper, "Results.md", "R1 text.\n")
    join_manuscript(paper, SECTIONS)
    (paper / "manuscript.md").write_text("Edited by hand.\n", encoding="utf-8")
    write(paper, "Results.md", "R1 text, revised.\n")
    assert join_manuscript(paper, SECTIONS).held_back
    (paper / "manuscript.md").unlink()
    assert join_manuscript(paper, SECTIONS).written
    assert "revised" in joined(paper)


def test_an_empty_manuscript_is_replaced(paper):
    (paper / "manuscript.md").write_text("", encoding="utf-8")
    write(paper, "Results.md", "R1 text.\n")
    assert join_manuscript(paper, SECTIONS).written


def test_nothing_to_join_makes_no_file(paper):
    assert not join_manuscript(paper, SECTIONS).written
    assert not (paper / "manuscript.md").exists()


def test_a_file_no_section_names_is_reported(paper):
    write(paper, "Results.md", "R1 text.\n")
    write(paper, "Old.md", "Left over.\n")
    result = join_manuscript(paper, SECTIONS)
    assert result.unused == ["Old.md"]
    assert "Manuscript/Old.md" in result.note()
    assert "Left over." not in joined(paper)
