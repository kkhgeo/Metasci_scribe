# Contributing to Metasci Scribe

Thank you for helping. Metasci Scribe is a terminal agent for writing scientific
manuscripts. Please read this before you open an issue or a pull request.

## How you can contribute

- **Report bugs and request features:** open an issue with the provided templates.
- **Propose design changes:** use an issue to describe the problem, the
  alternatives and the trade-offs before you implement anything.
- **Contribute code or docs:** submit a pull request that addresses an open
  issue, with a clear rationale and tests where they apply.

## What fits

The project stays small on purpose. A pull request should be one of:

- a bug fix or an improvement to an existing feature,
- a feature that was proposed in an issue and agreed with the maintainers,
- a documentation update,
- a meaningful addition to the test suite.

Out of scope: web or full-screen interfaces, messaging channels, web search,
scheduled or background jobs, memory that carries between papers, and
connections to Claude subscriptions. The reasons are in
`docs/superpowers/specs/2026-11-01-metasci-scribe-design.md`.

## Development setup

1. Fork and clone the repository.
2. Install the dependencies (needs [uv](https://docs.astral.sh/uv/) and Python
   3.11 to 3.13):
   ```bash
   uv sync --dev
   ```
3. Run the tests. They need no API key and no network:
   ```bash
   uv run pytest -n auto -q
   ```
4. Run the linter:
   ```bash
   uv run ruff check .
   ```

On Windows with a non-UTF-8 default locale, a few tests can fail for reasons that
have nothing to do with the code. The list is in [docs/baseline.md](docs/baseline.md).

## Submitting a pull request

1. Create a branch from `main` with a descriptive name (for example
   `fix/session-crash`).
2. Keep commits focused and well described.
3. Make sure `uv run ruff check .` and `uv run pytest -n auto -q` pass. Both also
   run in CI.
4. Open a pull request against `main` and fill in the template.

## Code style

- Ruff does the linting and formatting (`uv run ruff check .`,
  `uv run ruff format .`).
- Follow the patterns of the code around your change, and keep changes minimal.
- The command is `metasci-scribe`, the package is `metasci_scribe`, and
  environment variables start with `METASCI_SCRIBE_`.

## Project overview

| Fact | Value |
|---|---|
| Language | Python 3.11 to 3.13 |
| License | Apache 2.0 |
| Framework | [DeepAgents](https://github.com/langchain-ai/deepagents), [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Models | OpenAI (ChatGPT plan or API key) and Anthropic (API key) |
| Config file | `~/.config/metasci-scribe/config.yaml` |
| Data folder | `~/.metasci-scribe` |

### Sub-agents (`metasci_scribe/subagents/*.yaml`)

| Agent | Purpose |
|---|---|
| `planner-agent` | Organizes the outline map (sections, paragraph claims, order, gaps in the evidence); does not draft text. |
| `research-agent` | Collects what the paper folder and the extracted PDF text say about a topic, with file names and page markers; no web access. |
| `writing-agent` | Drafts one paragraph or section from a given claim and source passages; returns text and saves nothing. |

### Data flow

```text
User input (terminal)
    |
CLI (cli/interactive.py)            metasci-scribe run (run.py, run_agent.py)
    |                                    |
Main agent (agent.py) -- create_deep_agent()
    +-- System prompt (prompts.py)
    +-- Chat model (llm/: ChatGPT plan, OpenAI key, Anthropic key)
    +-- Middleware (middleware/)
    +-- Backends (backends.py): the paper folder, guarded (the agent cannot
    |     write .env, .metasci-scribe/ or .git); /skills/ is read only
    +-- MCP tools (mcp/, optional, chat only)
    |
task tool --> sub-agents
    |
Stream events --> display --> Rich terminal output
```

## License of contributions

By contributing you agree that your work is licensed under the Apache License
2.0, as the rest of the project. Parts derived from the upstream project are
listed in [NOTICE](NOTICE).
