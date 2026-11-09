"""The writing skills ship with the app and get editable copies in the author's folder."""

from __future__ import annotations

from pathlib import Path

from metasci_scribe import skill_seed
from metasci_scribe.skill_seed import SEEDED, seed_skills


def _shipped(tmp_path: Path, text: str = "v1") -> Path:
    source = tmp_path / "shipped"
    for name in SEEDED:
        (source / name / "references").mkdir(parents=True, exist_ok=True)
        (source / name / "SKILL.md").write_text(f"name: {name}\n{text}\n", "utf-8")
        (source / name / "references" / "a.md").write_text(text, "utf-8")
    return source


def test_a_missing_copy_is_made_and_a_second_start_says_nothing(tmp_path):
    source, target = _shipped(tmp_path), tmp_path / "home"
    notes = seed_skills(source, target)
    assert len(notes) == len(SEEDED)
    assert (target / "scribe-review" / "references" / "a.md").read_text("utf-8") == "v1"
    assert seed_skills(source, target) == []


def test_an_unedited_copy_follows_the_app(tmp_path):
    source, target = _shipped(tmp_path), tmp_path / "home"
    seed_skills(source, target)
    _shipped(tmp_path, "v2")
    notes = seed_skills(source, target)
    assert any("Updated the skill scribe-review" in n for n in notes)
    assert (target / "scribe-review" / "references" / "a.md").read_text("utf-8") == "v2"


def test_an_edited_copy_is_kept_and_the_author_is_told_once(tmp_path):
    source, target = _shipped(tmp_path), tmp_path / "home"
    seed_skills(source, target)
    mine = target / "scribe-review" / "references" / "a.md"
    mine.write_text("my rule", "utf-8")
    assert seed_skills(source, target) == []  # nothing newer shipped yet
    _shipped(tmp_path, "v2")
    notes = seed_skills(source, target)
    assert any("scribe-review" in n and "has your edits" in n for n in notes)
    assert mine.read_text("utf-8") == "my rule"
    assert not any("scribe-review" in n for n in seed_skills(source, target))


def test_a_deleted_copy_comes_back_from_the_app(tmp_path):
    import shutil

    source, target = _shipped(tmp_path), tmp_path / "home"
    seed_skills(source, target)
    (target / "scribe-review" / "references" / "a.md").write_text("mine", "utf-8")
    shutil.rmtree(target / "scribe-review")
    seed_skills(source, target)
    assert (target / "scribe-review" / "references" / "a.md").read_text("utf-8") == "v1"


def test_a_copy_that_already_matches_is_adopted(tmp_path):
    import shutil

    source, target = _shipped(tmp_path), tmp_path / "home"
    target.mkdir()
    shutil.copytree(source / "scribe-review", target / "scribe-review")
    seed_skills(source, target)
    assert (target / "scribe-review" / skill_seed.STAMP).is_file()
    _shipped(tmp_path, "v2")
    seed_skills(source, target)  # it was the shipped text, so it follows the app
    assert (target / "scribe-review" / "references" / "a.md").read_text("utf-8") == "v2"


def test_the_real_shipped_skills_are_the_ones_seeded():
    for name in SEEDED:
        assert (skill_seed.SHIPPED / name / "SKILL.md").is_file(), name
