"""Deployed graphs for the yaml-flagged async sub-agent and the memory workers.

One module-level binding per ``async: true`` entry in
``metasci_scribe/subagents/<name>.yaml``. Each binding is a graph compiled by
``build_async_subagent_graph`` (which reads the yaml, wires tools/skills/
backend/middleware identical to the in-process sync version, and returns a
runnable langgraph).

To add a new async sub-agent:

  1. Set ``async: true`` in ``metasci_scribe/subagents/<name>.yaml``.
  2. Add a one-line binding here::

         <snake_name> = folder_checked(
             build_async_subagent_graph("<name>", **_session), graph_id="<name>"
         )

  3. Register it in ``metasci_scribe/langgraph_dev/langgraph.json``::

         "<name>": "metasci_scribe.langgraph_dev.graphs:<snake_name>"

  4. If it works in the session's folder, add ``"<name>"`` to
     ``metasci_scribe.sessions.FOLDER_GRAPH_IDS``.

The deployed main agent (``scribe_agent``) lives in ``main_graph.py``
because it follows a different mechanism (re-exporting a lazily-constructed
attribute), not the yaml-driven factory.
"""

from metasci_scribe.langgraph_dev.folder_check import folder_checked
from metasci_scribe.memory.agents import (
    build_autoskills_graph,
    build_memory_worker_graph,
    build_observation_linker_graph,
)
from metasci_scribe.memory.types import MemorySourceType
from metasci_scribe.paths import process_session_dirs
from metasci_scribe.subagents._factory import build_async_subagent_graph

# The server serves the workspace it was started for, and works in the run
# folder of a session with a run folder (the manager sets
# ``METASCI_SCRIBE_WORKSPACE_DIR`` and ``METASCI_SCRIBE_RUN_DIR`` on the
# subprocess). Memory graphs only need the workspace.
_dirs = process_session_dirs()
_workspace = _dirs.workspace
_session = {"workspace": _workspace, "work_dir": _dirs.work_dir}

# Graphs that work in the session's folder (``FOLDER_GRAPH_IDS``) check a
# run's workspace and run folder; graphs built for the workspace root check
# the workspace only.
writing_agent = folder_checked(
    build_async_subagent_graph("writing-agent", **_session), graph_id="writing-agent"
)
evomemory_subagent_worker = folder_checked(
    build_memory_worker_graph(MemorySourceType.SUBAGENT, workspace=_workspace),
    graph_id="evomemory-subagent-worker",
)
evomemory_turn_worker = folder_checked(
    build_memory_worker_graph(MemorySourceType.TURN, workspace=_workspace),
    graph_id="evomemory-turn-worker",
)
evomemory_observation_linker = folder_checked(
    build_observation_linker_graph(workspace=_workspace),
    graph_id="evomemory-observation-linker",
)
evomemory_autoskills = folder_checked(
    build_autoskills_graph(workspace=_workspace), graph_id="evomemory-autoskills"
)
