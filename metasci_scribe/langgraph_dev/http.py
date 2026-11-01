"""Custom HTTP routes mounted alongside the langgraph dev server.

The langgraph-api host supports a top-level ``http`` key in
``langgraph.json`` that names an ASGI app to mount on the same
process as the graph. We use it for the one route the CLI reads across the
process boundary, ``/api/bg_process_status``, and to put the browser boundary
in front of the server.

The app's middleware, :class:`~metasci_scribe.langgraph_dev.boundary.BrowserBoundary`,
wraps every route of the server, langgraph-api's included: langgraph-api runs a
custom app's middleware outside its own.

Why Starlette: ``langgraph_api`` already depends on it, and the route has no
input models, just a query parameter and a JSON body.

Lightweight by design — nothing on this surface should pull the agent into
memory.
"""

from __future__ import annotations

import asyncio
import os

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from metasci_scribe.langgraph_dev.boundary import BOUNDARY_ENV, BrowserBoundary, Trust


async def get_bg_process_status(request: Request) -> JSONResponse:
    """Return a background process's live status for the client-side reader.

    Background processes launched by ``run_in_background`` live in THIS server
    process's registry (the served main graph runs here), so their exit is only
    observable here. The CLI's ``bg_processes`` state reader polls this route to
    detect completion across the process boundary, mirroring how the async-task
    reader polls run status. ``process_id`` is a query param; ``status`` is one of
    ``running`` / ``success`` / ``error`` / ``interrupted`` / ``unknown`` (untracked id).

    Offloaded to a thread: reading the registry takes a ``threading.Lock`` and does
    a ``Popen.poll()`` syscall, which langgraph-dev's ``blockbuster`` middleware
    refuses on the async event loop.
    """
    from metasci_scribe import background

    process_id = request.query_params.get("process_id", "")
    status = await asyncio.to_thread(background.poll_status, process_id)
    return JSONResponse({"status": status})


def _build_app() -> Starlette:
    """The custom app, then the boundary's variable removed from the environment.

    It is read by then (the guard keeps its ``Trust``); the agent's commands,
    such as a ``langgraph dev`` of the user's own project, must not inherit it.
    """
    app = Starlette(
        # Read when langgraph-api imports this app: LANGGRAPH_API_URL is set by
        # then, and a workspace ``.env`` merged later cannot change it.
        middleware=[Middleware(BrowserBoundary, trust=Trust.from_env())],
        routes=[
            Route("/api/bg_process_status", get_bg_process_status, methods=["GET"]),
        ],
    )
    os.environ.pop(BOUNDARY_ENV, None)
    return app


app = _build_app()
