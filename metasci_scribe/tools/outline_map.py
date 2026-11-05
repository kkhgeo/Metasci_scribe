"""``show_outline_map``: draw an outline as a page in the author's browser."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool

__all__ = ["make_outline_map_tool"]


def make_outline_map_tool(root: Path) -> BaseTool:
    """The ``show_outline_map`` tool for the paper folder *root*. Chat only:
    an unattended run has nobody to look at the page."""
    root = Path(root)

    @tool(parse_docstring=True)
    def show_outline_map(outline: str = "") -> str:
        """Open the paper's outline as a visual map in the author's web browser.

        Use it after you present an outline proposal in the conversation, so the
        author can see the flow of paragraphs and the two tables laid out, and
        when the author asks to see the outline. The page is a preview: it
        changes no file of the paper.

        Args:
            outline: A proposal written exactly in the outline.md format (## sections, "### R1 · ① title" paragraphs, "- 역할:", "- 핵심 내용:", "- 전달할 내용:", "- 근거:", "- 연결:" fields, ◇ for open choices). Leave empty to show the saved outline.md.
        """
        from ..outline_view import show_outline

        try:
            path, view = show_outline(root, outline or None)
        except FileNotFoundError:
            return "There is no outline.md yet. Pass the proposal as `outline`."
        except OSError as exc:
            return f"Could not write the map ({type(exc).__name__})."
        if not view.paragraph_count:
            return (
                f"Opened {path.name}, but it shows no paragraphs: the text has no "
                "'### R1 · ① title' headings. Write the proposal in the outline.md format."
            )
        what = "the saved outline.md" if not outline else "the proposal (not saved)"
        return (
            f"Opened the map of {what} in the author's browser: "
            f"{len(view.sections)} section(s), {view.paragraph_count} paragraph(s), "
            f"{view.open_count} open choice(s). File: {path}"
        )

    return show_outline_map
