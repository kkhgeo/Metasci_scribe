# Metasci Scribe

A terminal agent that helps you write a scientific manuscript, paragraph by
paragraph, from your own outline and your own sources. You talk to it in the
terminal, it reads and writes files in your paper folder, and it can also write
the whole draft on its own while you are away.

Status: early development (version 0.1.0.dev0). Nothing is released yet.

## What you need

- **Python 3.11, 3.12 or 3.13.** Newer versions are not supported yet.
- **[uv](https://docs.astral.sh/uv/)**, which installs the app into its own
  environment.
- **Git.** On Windows the agent's shell is Git Bash, which comes with
  [Git for Windows](https://git-scm.com/download/win).
- A way to pay for the model: a **ChatGPT plan** (see below) or your **own
  OpenAI or Anthropic API key**.

## Install

```bash
uv tool install C:\path\to\Metasci_scribe
metasci-scribe --version
```

`C:\path\to\Metasci_scribe` is your local copy of this repository; the public
URL arrives at publication. To update, run the same command again with
`--reinstall`.

## Quick start

Make a folder for the paper, **on a local disk** (for example
`C:\papers\my-paper`), and start the app in it:

```bash
mkdir my-paper
cd my-paper
metasci-scribe
```

Then, inside the app:

| Step | What you type | What happens |
|---|---|---|
| 1 | `/login` | Opens your browser. Choose **Continue with ChatGPT** and sign in. Needed once. |
| 2 | `/init` | Creates `program.md`, `outline.md`, `manuscript.md` and the folders `Knowledge/`, `Figures/` and `Tables/` here. Files that already exist are never overwritten. |
| 3 | (chat) | Put your PDFs in `Knowledge/`. Fill in `program.md` and `outline.md` yourself or ask the agent to help; it reads and writes the files in the folder. Ask it to write a paragraph from the outline. |
| 4 | `/run` | Writes the whole draft unattended, following `outline.md`, into `manuscript.md`. The same as running `metasci-scribe run` in a terminal. |

To come back to a paper later, open a terminal in the same folder and run
`metasci-scribe` again. `/threads` and `/resume` bring back an earlier
conversation (the threads an autonomous run keeps for itself are not listed),
and `/run` continues a draft where it stopped.

Other commands: `/help` lists them all. `/model` switches the model, `/logout`
signs out, `/rubric` and `/goal` set acceptance criteria that the agent's work is
checked against (see [docs/guides](docs/guides/)), and `/skills` and
`/install-skill` manage skills.

## Paying for the model

**ChatGPT plan (`/login`).** The default route. `/login` uses OpenAI's "Sign in
with ChatGPT" flow, and requests count against your plan. Plus and Pro are the
plans documented so far. **This route is a preview:** OpenAI's terms for it can
change, and the app uses only OpenAI's public API. If your account cannot use
plan usage, `/login` says so and changes nothing.

**Your own API key.** The alternative. Set `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` in your environment, or run `metasci-scribe onboard` and
choose a provider. This is billed to your API account, not to a plan. The app
never switches from a ChatGPT plan to an API key on its own, not even when the
plan runs out.

**Usage limits.** A plan has usage limits. When you reach one, a run stops with a
message and does not switch to another way of paying. Wait for the limit to
reset, then run `/run` or `metasci-scribe run` again: it continues at the first
paragraph that is not done. Exit codes of `metasci-scribe run`: 0 finished, 2
failed or could not start, 3 plan usage limit, 4 a paragraph needs your decision.

## What the agent can and cannot do

- **Autonomous runs (`/run`, `metasci-scribe run`)** have no shell, no MCP tools
  and no delete. They read and write files in the paper folder and nothing else.
- **In chat**, every shell command the agent wants to run needs your approval.
  The setting `shell_allow_list` (comma-separated command prefixes, set with
  `metasci-scribe config set shell_allow_list "..."`) approves matching commands
  without asking. A broad list weakens this protection, so keep it short and
  specific.
- A **`.env` file in the paper folder** is honored for provider API keys only
  (names such as `OPENAI_API_KEY`). Everything else in it is ignored, because a
  paper folder can come from someone else. Endpoints, proxies and
  `METASCI_SCRIBE_*` settings must come from your shell or the config file.
  The app reads `.env` itself; the agent's file tools refuse `.env` and
  `.env.*`, and those files do not appear in its listings and searches.
- With `OPENAI_BASE_URL` set in your shell, an OpenAI-compatible server works
  with no API key.
- The agent **cannot search the web.** Its evidence is the paper folder:
  `Knowledge/`, `Figures/`, `Tables/` and your own text.
- **PDFs are extracted as text.** The text of every PDF in `Knowledge/` is made
  available to the agent. A scanned PDF has no text layer and gives nothing; run
  OCR on it with another tool first.
- **Paper folders must be on a local disk.** A network drive, a mapped drive
  letter such as `Z:` or a `\\server\share` path is refused with a message.
- **Memory and scheduled features are off.** The app does not learn between
  papers and does not run anything on a schedule. With memory off the agent
  has no path to the memory folder.

## Where things live

- In the paper folder: your files, plus `.metasci-scribe/` with the app's own
  state (run progress, text extracted from PDFs, project skills). Do not edit it;
  the agent is not allowed to write there.
- In your home folder: `~/.metasci-scribe` (sign-in, sessions) and
  `~/.config/metasci-scribe/config.yaml` (settings). Environment variables use
  the `METASCI_SCRIBE_` prefix.
- Skills go to your home folder (all papers) or to the paper folder (one paper).
  Installing a skill always needs your approval.

## Development

```bash
uv sync --dev
uv run metasci-scribe --help
uv run pytest -n auto -q
uv run ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Known test failures on a Windows machine
with a non-UTF-8 default locale, and the decisions made while cutting the
project down, are in [docs/baseline.md](docs/baseline.md). The manual acceptance
checklist is [docs/acceptance.md](docs/acceptance.md).

## Origin

Metasci Scribe is a fork of [EvoScientist](https://github.com/EvoScientist/EvoScientist)
v0.3.8 (commit `e2eef14`), used under the Apache License 2.0. See
[NOTICE](NOTICE) for what was changed and [LICENSE](LICENSE) for the license
text. This project is not affiliated with or endorsed by EvoScientist.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
