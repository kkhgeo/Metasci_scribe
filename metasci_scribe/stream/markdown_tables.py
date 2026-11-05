"""Markdown tables drawn with a line between rows.

Rich draws a Markdown table with a rule under the header only. The outline
tables of a paper hold several lines of text per row, and without row lines
the rows run into each other. ``install`` swaps Rich's table element for one
with a rounded frame and a rule between rows, for every Markdown the app
renders.
"""

from __future__ import annotations

from rich import box
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import Markdown, TableElement
from rich.table import Table

__all__ = ["GridTableElement", "install"]


class GridTableElement(TableElement):
    """A Markdown table with a frame and a line between rows."""

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        table = Table(
            box=box.ROUNDED,
            show_lines=True,
            style="markdown.table.border",
        )
        if self.header is not None and self.header.row is not None:
            for column in self.header.row.cells:
                heading = column.content.copy()
                heading.stylize("markdown.table.header")
                table.add_column(heading)
        if self.body is not None:
            for row in self.body.rows:
                table.add_row(*(element.content for element in row.cells))
        yield table


def install() -> None:
    """Use :class:`GridTableElement` for every Markdown table. Idempotent."""
    Markdown.elements["table_open"] = GridTableElement
