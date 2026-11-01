"""No human gate, no shell.

A command runs with the user's full rights, and text the agent reads can ask
for one: a prompt-injected PDF could send the saved sign-in or an unpublished
manuscript anywhere. The approval prompt is the only real stop, so a run
without it does not get ``execute`` or ``run_in_background``.

Decided per model call and per tool call, not at construction: one graph
serves runs with and without approval (see ``agent._build_hitl_interrupt_on``),
so a tool removed at build time would stay missing for a later attended run.
The tools stay registered in the graph; this hides them from the model and
refuses a call that names them anyway, as one from earlier history would.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from ..backends import no_human_gate
from ..config.settings import HITL_SHELL_TOOLS
from .utils import append_to_system_message

_NO_SHELL_NOTE = (
    "\n\nShell commands are off for this run because no person is approving "
    "them: `execute` and `run_in_background` do not exist here, whatever "
    "earlier instructions say. Work with the file tools."
)


def _run_config(request: Any) -> Any:
    return getattr(getattr(request, "runtime", None), "config", None)


class ShellGateMiddleware(AgentMiddleware):
    """Remove the shell tools from any run with no human gate.

    Args:
        closed: Keep the shell off for every run of this graph: dangerous
            mode, which is fixed at assembly and approves everything, and
            async sub-agents, which have no approval path at all.
    """

    def __init__(self, *, closed: bool = False) -> None:
        super().__init__()
        self._closed = closed

    def _gate_shut(self, request: Any) -> bool:
        return self._closed or no_human_gate(_run_config(request))

    def _without_shell(self, request: ModelRequest) -> ModelRequest:
        if not self._gate_shut(request):
            return request
        tools = [
            tool
            for tool in request.tools
            if (tool.get("name") if isinstance(tool, dict) else tool.name)
            not in HITL_SHELL_TOOLS
        ]
        return request.override(
            tools=tools,
            system_message=append_to_system_message(
                request.system_message, _NO_SHELL_NOTE
            ),
        )

    def _refusal(self, request: ToolCallRequest) -> ToolMessage | None:
        name = request.tool_call["name"]
        if name not in HITL_SHELL_TOOLS or not self._gate_shut(request):
            return None
        return ToolMessage(
            content=f"Error: {name} is not available: no person is approving "
            "commands in this run.",
            tool_call_id=request.tool_call["id"] or "",
            name=name,
            status="error",
        )

    def wrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Any]
    ) -> Any:
        return handler(self._without_shell(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        return await handler(self._without_shell(request))

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]
    ) -> Any:
        return self._refusal(request) or handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        return self._refusal(request) or await handler(request)
