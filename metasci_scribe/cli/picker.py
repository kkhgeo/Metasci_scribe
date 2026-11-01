"""Shared helpers for the questionary pickers (``/resume``, ``/delete``, skills, MCP).

Threads are grouped into a two-level hierarchy:

  L1 header  - common ancestor path shared by 2+ workspaces, or the
               workspace path itself for standalone workspaces.
  L2 subheader - relative sub-path shown only when a group contains
                 multiple workspaces.  Run-mode dirs are marked with a loop icon.
  thread row - indented under their sub-path (or directly under L1 for
               standalone groups).
"""

from __future__ import annotations

from prompt_toolkit.styles import Style as PtStyle  # type: ignore[import-untyped]

from .. import project
from ..paths import SessionDirs

# Style for questionary pickers used by the Rich CLI ``/resume`` and
# ``/delete`` interactive selectors. Matches the slash-completion menu's
# visual language: gray (#888888) for non-selected, bold for selected,
# no background changes.
PICKER_STYLE = PtStyle.from_dict(
    {
        "questionmark": "#888888",
        "question": "",
        "pointer": "bold",
        "highlighted": "bold",
        "text": "#888888",
        "answer": "bold",
    }
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _normalize_path(path: str) -> str:
    """Strip trailing slash and replace home directory with ~."""
    import os

    if not path:
        return ""
    path = path.rstrip("/")
    # Stored folders use forward slashes on every platform; so must the home
    # prefix they are compared with.
    home = os.path.expanduser("~").replace("\\", "/")
    if path.startswith(home):
        path = "~" + path[len(home) :]
    return path


def _common_prefix_depth(p1: str, p2: str) -> int:
    """Return the number of leading path components shared by *p1* and *p2*."""
    depth = 0
    for a, b in zip(p1.split("/"), p2.split("/"), strict=False):
        if a == b:
            depth += 1
        else:
            break
    return depth


def _group_by_ancestor(norm_paths: list[str]) -> dict[str, list[str]]:
    """Group normalized paths by their deepest common ancestor.

    Two paths are placed in the same group when they share a common prefix
    of at least 2 components (e.g. ``~/Projects``).  Paths with no such
    shared prefix become standalone single-item groups keyed by their own
    full path.

    The returned dict is ordered by first appearance in *norm_paths*.

    Complexity: O(n log n) — paths are sorted lexicographically so the
    maximum common prefix depth for each path is found by comparing only
    its immediate neighbours in sorted order (not all pairs).
    """
    sorted_paths = sorted(norm_paths)
    n = len(sorted_paths)

    # Single pass over sorted list: max common prefix is always with a neighbour.
    sorted_best: dict[str, int] = {}
    for i, p in enumerate(sorted_paths):
        best = 1  # at minimum depth 1 (~)
        if i > 0:
            best = max(best, _common_prefix_depth(p, sorted_paths[i - 1]))
        if i < n - 1:
            best = max(best, _common_prefix_depth(p, sorted_paths[i + 1]))
        sorted_best[p] = best

    path_to_ancestor: dict[str, str] = {
        p: ("/".join(p.split("/")[:best]) if best >= 2 else p)
        for p, best in sorted_best.items()
    }

    groups: dict[str, list[str]] = {}
    for p in norm_paths:
        anc = path_to_ancestor[p]
        if anc not in groups:
            groups[anc] = []
        groups[anc].append(p)
    return groups


# ---------------------------------------------------------------------------
# Item builders
# ---------------------------------------------------------------------------


def _build_items(threads: list[dict]) -> list[dict]:
    """Build the flat item list rendered by the thread picker.

    Returns a list whose elements are one of::

        {"type": "header",    "label": str}
        {"type": "subheader", "label": str}
        {"type": "thread",    "thread": dict, "indented": bool}

    *indented* is True for thread rows that sit under a L2 subheader.
    """
    if not threads:
        return []

    # Map normalized work folder -> list[thread dicts], preserving first-seen
    # order. Run-mode threads sit under their run folder, next to the
    # workspace's other threads.
    raw_to_threads: dict[str, list[dict]] = {}
    seen_order: list[str] = []
    run_paths: set[str] = set()
    for t in threads:
        stored, stored_run = t.get("workspace_dir"), t.get("run_dir")
        if project.stored_network_path(stored, stored_run):
            # Listed as stored: resolving it would touch the share.
            dirs = None
            raw = str(stored_run or stored).replace("\\", "/")
            if stored_run:
                run_paths.add(_normalize_path(raw) or raw)
        else:
            dirs = SessionDirs.from_stored(stored, stored_run)
            raw = dirs.work_dir.as_posix() if dirs is not None else ""
        norm = _normalize_path(raw) or raw
        if dirs is not None and dirs.run_dir is not None:
            run_paths.add(norm)
        if norm not in raw_to_threads:
            raw_to_threads[norm] = []
            seen_order.append(norm)
        raw_to_threads[norm].append(t)

    groups = _group_by_ancestor(seen_order)

    items: list[dict] = []
    for ancestor, norm_paths in groups.items():
        multi = len(norm_paths) > 1

        # L1 header — the common ancestor (or the sole workspace path)
        items.append({"type": "header", "label": ancestor or "(no workspace)"})

        for norm_path in norm_paths:
            if multi:
                # L2 subheader — relative path from ancestor
                rel = norm_path[len(ancestor) :].lstrip("/")
                if not rel:
                    # norm_path IS the ancestor (standalone group of 1 that shares
                    # an ancestor with others); show just the last path component
                    rel = norm_path.split("/")[-1] or norm_path
                icon = "🔁" if norm_path in run_paths else "📁"
                items.append({"type": "subheader", "label": f"{icon} {rel}"})

            for t in raw_to_threads[norm_path]:
                items.append({"type": "thread", "thread": t, "indented": multi})

    return items
