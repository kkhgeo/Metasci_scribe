"""Thin wrapper over the langgraph dev built-in cron API (langgraph_sdk).

This module is the single choke-point for crons: every cron (AutoSkills')
is created with :func:`create_cron`, which tags it, and deleted with
:func:`delete_cron`, which checks it is the workspace's own.

Every cron belongs to a workspace: it carries the workspace root in
``config.configurable`` and ``metadata`` (``workspace_dir``), runs in that root,
and is listed, paused and deleted only from that workspace. Crons nobody else
can own (untagged, or tagged with a folder that no longer exists) belong to
whichever workspace the server serves.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_sdk.schema import Cron

from ..paths import SessionDirs, Workspace


def run_config(workspace: Workspace, **configurable: Any) -> dict[str, Any]:
    """Run config of a scheduled run: it works in the workspace root."""
    return {"configurable": {**configurable, **SessionDirs(workspace).metadata()}}


def workspace_key(value: Any) -> str | None:
    """The workspace a stored folder names, in the stored form, or None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return Workspace(value).key
    except (OSError, RuntimeError, ValueError):
        # A path that cannot exist on this machine names no workspace here.
        return None


def belongs_to(cron: Any, workspace: Workspace) -> bool:
    """True if *cron* is one of *workspace*'s scheduled tasks.

    Tags are compared in the stored form, so tags written before it (an
    unresolved or Windows path) still match. The server only loads the store
    of the workspace it serves, so a cron nobody else can own belongs to it:
    one created before tasks were tagged, or one tagged with a folder that no
    longer exists (the project was moved or renamed, and its store with it).
    """
    tag = ((cron or {}).get("metadata") or {}).get("workspace_dir")
    if not isinstance(tag, str) or not tag:
        return True
    key = workspace_key(tag)
    if key == workspace.key:
        return True
    return key is not None and not Path(key).is_dir()


# Crons per search page (the server's maximum). Workspaces are filtered after
# the search, so every page is read: a store full of other workspaces' crons
# cannot hide ours. Pages are newest first, so a cron created between two
# fetches shifts the rest down by one; rows are therefore collected by id.
_SEARCH_PAGE = 1000


def _extend_unseen(rows: list[Cron], seen: set[str], page: list[Cron]) -> None:
    for cron in page:
        cron_id = str(cron.get("cron_id"))
        if cron_id not in seen:
            seen.add(cron_id)
            rows.append(cron)


def search_all(client: Any, *, metadata: dict[str, Any]) -> list[Cron]:
    """Every cron whose metadata contains *metadata*, page by page."""
    rows: list[Cron] = []
    seen: set[str] = set()
    offset = 0
    while True:
        page = client.crons.search(metadata=metadata, limit=_SEARCH_PAGE, offset=offset)
        _extend_unseen(rows, seen, page)
        offset += len(page)
        if len(page) < _SEARCH_PAGE:
            return rows


async def asearch_all(client: Any, *, metadata: dict[str, Any]) -> list[Cron]:
    """Async variant of :func:`search_all`."""
    rows: list[Cron] = []
    seen: set[str] = set()
    offset = 0
    while True:
        page = await client.crons.search(
            metadata=metadata, limit=_SEARCH_PAGE, offset=offset
        )
        _extend_unseen(rows, seen, page)
        offset += len(page)
        if len(page) < _SEARCH_PAGE:
            return rows


def owned(rows: list[Cron], workspace: Workspace) -> list[Cron]:
    """The crons of *rows* that belong to *workspace* (resolves paths)."""
    return [row for row in rows if belongs_to(row, workspace)]


def create_cron(
    client: Any,
    *,
    workspace: Workspace,
    assistant_id: str,
    schedule: str,
    input: dict[str, Any],
    metadata: dict[str, str],
    timezone: str | None,
    thread_id: str | None = None,
    multitask_strategy: str | None = None,
    durability: str | None = None,
) -> Cron:
    """Create a cron of *workspace*: tagged with it, and its runs work in it.

    Without *thread_id* every run gets a fresh thread, which the runtime
    deletes after the run. With it the cron is bound to that existing thread,
    so its runs share one state; *multitask_strategy* (``"reject"`` keeps runs
    from overlapping) applies only then. *durability* sets how often a run
    checkpoints (``"exit"``: once, at the end).
    """
    kwargs: dict[str, Any] = {
        "schedule": schedule,
        "input": input,
        "metadata": {**metadata, "workspace_dir": workspace.key},
        "config": run_config(workspace),
        "timezone": timezone,
    }
    if multitask_strategy is not None:
        kwargs["multitask_strategy"] = multitask_strategy
    if durability is not None:
        kwargs["durability"] = durability
    if thread_id is not None:
        return client.crons.create_for_thread(thread_id, assistant_id, **kwargs)
    return client.crons.create(assistant_id=assistant_id, **kwargs)


def _require_own(cron: Cron, workspace: Workspace) -> str:
    if not belongs_to(cron, workspace):
        raise LookupError(f"No scheduled task {cron.get('cron_id')} in this workspace.")
    return str(cron["cron_id"])


def delete_cron(client: Any, cron: Cron, *, workspace: Workspace) -> None:
    """Delete *cron*, one of *workspace*'s crons."""
    client.crons.delete(_require_own(cron, workspace))
