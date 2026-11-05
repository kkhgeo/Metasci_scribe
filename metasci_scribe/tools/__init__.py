"""Tools package — re-exports all public tool symbols.

``make_mark_paragraph_tool`` belongs to autonomous runs only: the chat agent
never imports it, so a chat session has no way to mark a paragraph.
``make_outline_map_tool`` belongs to chat only: it opens a page in the author's
browser, and an unattended run has nobody to look at it.
"""

from ..run import make_mark_paragraph_tool
from .outline_map import make_outline_map_tool
from .skill_manager import make_skill_manager_tool
from .think import think_tool

__all__ = [
    "make_mark_paragraph_tool",
    "make_outline_map_tool",
    "make_skill_manager_tool",
    "think_tool",
]
