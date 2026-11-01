"""``/init``: create the paper folder layout in the current workspace."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ... import project
from ...paths import WorkspaceRootError
from ..base import Command, CommandContext
from ..manager import manager


def _label(path: Path, root: Path) -> str:
    """A path as the person sees it: relative to the paper folder, folders with
    a trailing slash."""
    name = path.relative_to(root).as_posix()
    return name + "/" if path.is_dir() and not path.is_symlink() else name


def _show_created(ctx: CommandContext, result: project.InitResult, root: Path) -> None:
    if result.created:
        ctx.ui.append_system("Created:", style="green")
        for path in result.created:
            ctx.ui.append_system(f"  {_label(path, root)}", style="green")


class InitCommand(Command):
    """Create program.md, outline.md, manuscript.md and the folders around them."""

    name = "/init"
    description = "Create the paper files and folders here (never overwrites)"

    async def execute(self, ctx: CommandContext, args: list[str]) -> None:
        root = ctx.workspace.root
        try:
            result = await asyncio.to_thread(project.init_project, root)
        except project.InitError as exc:
            ctx.ui.append_system(f"✗ {exc}", style="red")
            _show_created(ctx, exc.result, root)
            return
        except (project.NetworkPathError, WorkspaceRootError, OSError) as exc:
            ctx.ui.append_system(f"✗ {exc}", style="red")
            return
        _show_created(ctx, result, root)
        if result.skipped:
            ctx.ui.append_system("Skipped (already exist):", style="yellow")
            for path in result.skipped:
                ctx.ui.append_system(f"  {_label(path, root)}", style="dim")
        if result.problems:
            ctx.ui.append_system("Problems (left as they are):", style="red")
            for path in result.problems:
                reason = project.problem_reason(path)
                ctx.ui.append_system(f"  {_label(path, root)} {reason}", style="red")
        if not result.created:
            ctx.ui.append_system("Nothing to create.", style="dim")


manager.register(InitCommand())
