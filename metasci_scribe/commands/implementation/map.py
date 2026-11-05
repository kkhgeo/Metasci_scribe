"""``/map``: open the saved outline as a visual map in the browser."""

from __future__ import annotations

import asyncio

from ..base import Command, CommandContext
from ..manager import manager


class MapCommand(Command):
    """Draw ``outline.md`` as a page and open it in the browser."""

    name = "/map"
    description = "Open outline.md as a visual map in your browser"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        from ...outline_view import show_outline

        root = ctx.workspace.root
        try:
            path, view = await asyncio.to_thread(show_outline, root)
        except FileNotFoundError:
            ctx.ui.append_system(
                "✗ No outline.md here. Run /init, or draw one with /scribe-mapping.",
                style="red",
            )
            return
        except OSError as exc:
            ctx.ui.append_system(f"✗ Could not write the map: {exc}", style="red")
            return
        open_note = f", {view.open_count} open choice(s) (◇)" if view.open_count else ""
        ctx.ui.append_system(
            f"Opened the map in your browser: {len(view.sections)} section(s), "
            f"{view.paragraph_count} paragraph(s){open_note}.",
            style="green",
        )
        ctx.ui.append_system(f"  {path.resolve().as_uri()}", style="dim")


manager.register(MapCommand())
