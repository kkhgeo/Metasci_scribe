"""``langgraph dev`` deployment surface.

Holds everything needed to run the Metasci Scribe agent ecosystem on a local
``langgraph dev`` subprocess:

- ``manager`` — subprocess lifecycle (auto-start, port management, cleanup).
- ``langgraph.json`` — graph manifest consumed by ``langgraph dev --config``.
- ``main_graph`` — re-export of the lazy-loaded ``scribe_agent``.
- ``graphs`` — module-level bindings for every yaml-flagged async sub-agent
  (``async: true`` in ``metasci_scribe/subagents/<name>.yaml``).

The graphs themselves are built by ``metasci_scribe.subagents._factory.
build_async_subagent_graph`` from the canonical yaml definitions, so this
package only owns the *deployment* concern. Adding a new async sub-agent
takes three steps: flip the yaml flag, add a one-line binding in
``graphs.py``, and register it in ``langgraph.json``.
"""

# ``langgraph dev`` imports this package (to resolve the graphs listed in
# ``langgraph.json``) inside its own subprocess, before it creates the event
# loop that serves runs. Forcing the Proactor loop policy here — at import,
# ahead of loop creation — keeps MCP stdio subprocess spawning async on
# Windows so it isn't flagged as a blocking call by the dev runtime's
# ``blockbuster`` guard (see #283). This is the entrypoint where that error
# actually surfaces; the CLI entrypoint sets the same policy for its own
# process.
import os as _os

from .._winloop import ensure_proactor_event_loop_policy

ensure_proactor_event_loop_policy()

# ``langgraph dev`` also appends its working folder, the workspace, to
# ``sys.path`` before importing the graphs. The agent writes to that folder
# and this app imports lazily, so the server takes it off again here, the
# first of our modules it imports. Only in the server process: a test run
# or a plain import must keep its own path.
if _os.environ.get("METASCI_SCRIBE_SERVER_PROCESS") == "1":
    from ..policy import harden_sys_path as _harden_sys_path

    _harden_sys_path()
