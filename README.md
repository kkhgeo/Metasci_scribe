# Metasci Scribe

A terminal writing agent for scientific manuscripts.

Status: early development. The code was imported from EvoScientist v0.3.8 and
renamed; it is being cut down step by step to a terminal-only writing agent.
Most of the original features (messaging channels, WebUI, Docker image,
self-update) are still present and are going to be removed.

## Origin

Metasci Scribe is a fork of [EvoScientist](https://github.com/EvoScientist/EvoScientist)
v0.3.8 (commit `e2eef14`), used under the Apache License 2.0. See
[NOTICE](NOTICE) for what was changed and [LICENSE](LICENSE) for the license
text. This project is not affiliated with or endorsed by EvoScientist.

## Development setup

Requires Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev
uv run metasci-scribe --help
uv run pytest -n auto -q
```

The command is `metasci-scribe`; the Python package is `metasci_scribe`.
User data lives in `~/.metasci-scribe`, and environment variables use the
`METASCI_SCRIBE_` prefix.

Known test failures on a Windows machine with a non-UTF-8 default locale are
listed in [docs/baseline.md](docs/baseline.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
