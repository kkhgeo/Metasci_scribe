"""The outline drawn as an HTML page, the tool that opens it, and /map."""

from __future__ import annotations

import re
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.markdown import Markdown

from metasci_scribe import outline_view
from metasci_scribe.outline_view import (
    parse_outline_view,
    render_outline_html,
    show_outline,
)

OUTLINE = """\
# Outline

<!-- The template's guide.
## Not a section
-->

핵심 메시지: Plants grow.

## Results

<!-- 구성 흐름: R1 → R2 -->

### R1 · ① First result
- 역할: 핵심 결과 제시
- 핵심 내용: Growth rises.
- 전달할 내용: It rose 30%
  across sites.
- 근거: Tables/T1.csv — growth; 확인 필요: T2/units; ⚠ 자료없음: site C
- 연결: Then the limit.

### R2 · ② Second result
- 역할: 한계
- 핵심 내용: Growth saturates.
- ◇ Keep R2 or fold it into R1?

## Discussion
"""


def test_the_outline_is_read_into_sections_paragraphs_and_fields():
    view = parse_outline_view(OUTLINE)
    assert view.key_message == "Plants grow."
    assert [s.name for s in view.sections] == ["Results", "Discussion"]
    results = view.sections[0]
    assert results.notes == ["구성 흐름: R1 → R2"]
    r1, r2 = results.paragraphs
    assert (r1.id, r1.number, r1.title) == ("R1", "①", "First result")
    assert r1.value("전달할 내용") == "It rose 30% across sites."
    assert not r1.open
    assert r2.decisions == ["◇ Keep R2 or fold it into R1?"]
    assert r2.open
    assert (view.paragraph_count, view.open_count) == (2, 1)


def test_the_page_shows_the_parts_and_marks_what_is_open():
    page = render_outline_html(parse_outline_view(OUTLINE), title="Plants", saved=False)
    assert page.startswith("<!doctype html>")
    assert "제안 · 아직 저장하지 않음" in page
    assert "Plants grow." in page
    assert page.count('<li class="') == 2  # one flow item per paragraph
    assert "◇ 결정 1" in page
    assert '<div class="check">확인 필요: T2/units</div>' in page
    assert '<div class="miss">⚠ 자료없음: site C</div>' in page
    assert '<span class="path">Tables/T1.csv</span> — growth' in page
    assert "Not a section" not in page
    assert "<button" not in page
    assert "<script" not in page


def test_the_page_is_set_in_paperlogy_installed_first():
    page = render_outline_html(parse_outline_view(OUTLINE))
    assert "font-family:'Paperlogy'" in page
    assert page.index("local('Paperlogy 4 Regular')") < page.index(
        "Paperlogy-4Regular.woff2"
    )
    # The only address on the page is the font's.
    urls = set(re.findall(r"https?://[^/'\")]+", page))
    assert urls == {"https://cdn.jsdelivr.net"}


def test_text_from_the_outline_cannot_add_markup():
    hostile = (
        "## R<script>alert(1)</script>\n\n### R1 · ① <img src=x onerror=alert(1)>\n"
    )
    page = render_outline_html(parse_outline_view(hostile))
    assert "<script>" not in page
    assert "<img" not in page
    assert "&lt;script&gt;" in page


def test_the_saved_outline_is_drawn_into_the_state_folder(tmp_path):
    (tmp_path / "outline.md").write_text(OUTLINE, encoding="utf-8")
    (tmp_path / "program.md").write_text('---\ntitle: "Island plants"\n---\n', "utf-8")
    path, _view = show_outline(tmp_path, open_browser=False)
    assert path == tmp_path / ".metasci-scribe" / "view" / "outline.html"
    page = path.read_text(encoding="utf-8")
    assert "Island plants" in page
    assert "저장된 outline.md" in page
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        ".metasci-scribe",
        "outline.md",
        "program.md",
    ]


def test_no_outline_md_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        show_outline(tmp_path, open_browser=False)


def test_the_tool_opens_a_proposal_and_says_what_it_shows(tmp_path, monkeypatch):
    from metasci_scribe.tools import make_outline_map_tool

    opened: list[str] = []
    monkeypatch.setattr(outline_view.webbrowser, "open", opened.append)
    reply = make_outline_map_tool(tmp_path).invoke({"outline": OUTLINE})
    assert "proposal (not saved)" in reply
    assert "2 paragraph(s), 1 open choice(s)" in reply
    assert opened[0].startswith("file:")
    assert not (tmp_path / "outline.md").exists()


def test_the_tool_without_a_proposal_or_outline_says_so(tmp_path, monkeypatch):
    from metasci_scribe.tools import make_outline_map_tool

    monkeypatch.setattr(outline_view.webbrowser, "open", lambda url: None)
    assert "no outline.md" in make_outline_map_tool(tmp_path).invoke({})


def test_only_a_conversation_gets_the_tool(tmp_path):
    from metasci_scribe.agent import _chat_only_tools
    from metasci_scribe.paths import Workspace

    chat = _chat_only_tools(SimpleNamespace(auto_mode=False), Workspace(tmp_path))
    run = _chat_only_tools(SimpleNamespace(auto_mode=True), Workspace(tmp_path))
    assert [t.name for t in chat] == ["show_outline_map"]
    assert run == []


async def test_map_command_opens_the_saved_outline(tmp_path, monkeypatch):
    from metasci_scribe.commands import CommandContext
    from metasci_scribe.commands.manager import manager
    from metasci_scribe.paths import SessionDirs, Workspace
    from tests.fakes.common import FakeCommandUI

    (tmp_path / "outline.md").write_text(OUTLINE, encoding="utf-8")
    opened: list[str] = []
    monkeypatch.setattr(outline_view.webbrowser, "open", opened.append)
    ui = FakeCommandUI()
    ctx = CommandContext(
        agent=None, thread_id="t", ui=ui, dirs=SessionDirs(Workspace(tmp_path))
    )
    await manager.execute("/map", ctx)
    assert opened
    assert any("2 paragraph(s), 1 open choice(s)" in m for m in ui.system_messages)


def test_markdown_tables_have_a_line_between_rows():
    import metasci_scribe.stream.display  # noqa: F401  installs the grid tables

    out = StringIO()
    Console(file=out, width=60, color_system=None, legacy_windows=False).print(
        Markdown("| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |")
    )
    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert lines[0].startswith("╭")
    assert (
        sum(line.startswith("├") for line in lines) == 2
    )  # under header, between rows
