"""Typer application objects — no intra-package imports to avoid circular deps."""

import typer  # type: ignore[import-untyped]

app = typer.Typer(
    no_args_is_help=False,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Config subcommand group
config_app = typer.Typer(
    help="Configuration management commands", invoke_without_command=True
)
app.add_typer(config_app, name="config")

# MCP subcommand group
_MCP_HELP = """\
Configure and manage MCP servers

Examples:
  # Add a local MCP server (stdio auto-detected):
  metasci-scribe mcp add local-server python -- /path/to/server.py

  # Add an npx-based server:
  metasci-scribe mcp add sequential-thinking npx -- -y @modelcontextprotocol/server-sequential-thinking

  # Add an HTTP server (http auto-detected from URL):
  metasci-scribe mcp add docs-langchain https://docs.langchain.com/mcp

  # Add a stdio server with env vars (hardcoded):
  metasci-scribe mcp add my-server node --env API_KEY=xxx -- server.js

  # Add a server with runtime env ref (resolved from .env at startup):
  metasci-scribe mcp add brave-search npx --env-ref BRAVE_API_KEY -- -y @modelcontextprotocol/server-brave-search

  # Expose to a specific sub-agent (e.g. research-agent):
  metasci-scribe mcp add brave-search npx --env-ref BRAVE_API_KEY -e research-agent -- -y @modelcontextprotocol/server-brave-search

  # Expose to multiple agents:
  metasci-scribe mcp add local-server python -e main,research-agent,code-agent -- /path/to/server.py

  # Explicit transport override:
  metasci-scribe mcp add my-sse https://example.com/sse --transport sse

Sub-agents (-e): planner-agent | research-agent | code-agent | debug-agent | data-analysis-agent | writing-agent
"""
mcp_app = typer.Typer(help=_MCP_HELP, invoke_without_command=True)
app.add_typer(mcp_app, name="mcp")

# Channel subcommand group
channel_app = typer.Typer(help="Channel management commands")
app.add_typer(channel_app, name="channel")

# Sessions subcommand group — diagnostic tools for the LangGraph checkpoint DB
sessions_app = typer.Typer(
    help="Inspect and manage the sessions DB (~/.metasci-scribe/sessions.db)",
    invoke_without_command=True,
)
app.add_typer(sessions_app, name="sessions")

# Background langgraph dev server management — the explicit counterpart to
# langgraph_dev_keepalive: a server that outlives its CLI needs a first-class
# way to inspect and stop it.
server_app = typer.Typer(help="Manage the background langgraph dev server")
app.add_typer(server_app, name="server")

# Configure subcommand group — re-run a single onboarding section.
configure_app = typer.Typer(
    help=(
        "Re-run one onboarding section without going through the full wizard.\n"
        "Example: metasci-scribe configure provider"
    ),
)
app.add_typer(configure_app, name="configure")
