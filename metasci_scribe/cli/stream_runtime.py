"""Run the Rich streaming renderer, on a worker thread when a frontend needs it."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..gateway import GraphGateway
from ..stream.display import _run_streaming

if TYPE_CHECKING:
    from ..runtime import AsyncRuntime

STREAM_CANCEL_SETTLE_TIMEOUT = 5.0


class StreamCancellationTimeout(RuntimeError):
    """A blocking renderer did not settle after its turn was cancelled."""


def _consume_late_worker_result(worker: asyncio.Task[Any]) -> None:
    """Retrieve a detached worker result so eventual failure is not unhandled."""
    try:
        worker.exception()
    except asyncio.CancelledError:
        pass


async def settle_cancelled_worker(
    worker: asyncio.Task[Any],
    *,
    on_cancel: Callable[[], Any],
) -> Any:
    """Request cooperative cancellation and wait a bounded time for settlement."""
    on_cancel()
    done, _ = await asyncio.wait(
        {worker},
        timeout=STREAM_CANCEL_SETTLE_TIMEOUT,
    )
    if not done:
        worker.add_done_callback(_consume_late_worker_result)
        raise StreamCancellationTimeout(
            "The active turn did not stop within "
            f"{STREAM_CANCEL_SETTLE_TIMEOUT:g} seconds after cancellation."
        )
    try:
        return worker.result()
    except Exception:
        return ""


def run_streaming(
    *,
    agent: Any,
    message: str,
    thread_id: str,
    show_thinking: bool,
    interactive: bool,
    on_thinking: Callable[[str], None] | None = None,
    on_todo: Callable[[list[dict]], None] | None = None,
    on_file_write: Callable[[str], None] | None = None,
    work_dir: str | Path | None = None,
    on_stream_event: Callable[[str, Any], Any] | None = None,
    status_footer_builder: Callable[[], Any] | None = None,
    metadata: dict | None = None,
    configurable_extra: dict[str, Any] | None = None,
    hitl_prompt_fn: Callable[[list], list[dict] | None] | None = None,
    hitl_outcome_fn: Callable[[list, bool], Any] | None = None,
    ask_user_prompt_fn: Callable[[dict], dict] | None = None,
    cancel_scope: str | None = None,
    gateway: GraphGateway,
    runtime: AsyncRuntime | None = None,
    rubric: str | None = None,
    approve_all_at_start: bool = False,
    goal_request: dict[str, Any] | None = None,
) -> str:
    """Run one streamed turn and return the final response text."""
    return _run_streaming(
        agent=agent,
        message=message,
        thread_id=thread_id,
        show_thinking=show_thinking,
        interactive=interactive,
        on_thinking=on_thinking,
        on_todo=on_todo,
        on_file_write=on_file_write,
        work_dir=work_dir,
        on_stream_event=on_stream_event,
        status_footer_builder=status_footer_builder,
        metadata=metadata,
        configurable_extra=configurable_extra,
        hitl_prompt_fn=hitl_prompt_fn,
        hitl_outcome_fn=hitl_outcome_fn,
        ask_user_prompt_fn=ask_user_prompt_fn,
        cancel_scope=cancel_scope,
        gateway=gateway,
        runtime=runtime,
        rubric=rubric,
        approve_all_at_start=approve_all_at_start,
        goal_request=goal_request,
    )


async def run_streaming_async(
    *,
    recover_on_cancel: bool = False,
    **kwargs: Any,
) -> str:
    """Run the synchronous Rich renderer without blocking a frontend loop.

    Cancellation requests the matching stream scope and gives the worker a
    bounded interval to unwind.  Foreground interactive turns may opt into
    recovering the frontend task after cleanup so Ctrl+C returns to the prompt.
    """
    from ..stream.display import request_stream_cancel

    worker = asyncio.create_task(asyncio.to_thread(run_streaming, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        try:
            response = await settle_cancelled_worker(
                worker,
                on_cancel=lambda: request_stream_cancel(kwargs.get("cancel_scope")),
            )
        finally:
            from ..middleware.code_interpreter import aclose_code_interpreters

            await aclose_code_interpreters()
        if recover_on_cancel:
            current = asyncio.current_task()
            if current is not None and current.uncancel() > 0:
                raise
            return response
        raise
