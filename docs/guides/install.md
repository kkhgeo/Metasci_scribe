# Installing with the one-line command

One command installs Metasci Scribe on a machine that has nothing installed: no
uv, no Python, no Node.

**macOS and Linux**

```bash
curl -fsSL https://raw.githubusercontent.com/EvoScientist/EvoScientist/main/scripts/install.sh | bash
```

**Windows** (PowerShell)

```powershell
irm https://raw.githubusercontent.com/EvoScientist/EvoScientist/main/scripts/install.ps1 | iex
```

In mainland China, use the mirrors (jsDelivr for the script):

```bash
curl -fsSL https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist@main/scripts/install.sh | bash -s -- --cn
```

```powershell
& ([scriptblock]::Create((irm https://cdn.jsdelivr.net/gh/EvoScientist/EvoScientist@main/scripts/install.ps1))) -Cn
```

When it finishes, open a new terminal and run `metasci-scribe onboard`, then `metasci-scribe`
(terminal UI) or `metasci-scribe --ui webui`.

## Options

| bash | PowerShell | What it does |
|---|---|---|
| `--cn` | `-Cn` | Download from mainland China mirrors: uv from the Tsinghua PyPI mirror, Python from npmmirror, packages from `https://pypi.tuna.tsinghua.edu.cn/simple`, `constraints.txt` from jsDelivr, Node and the WebUI from npmmirror. `metasci-scribe setup --cn` also saves `mirror: cn`, so later WebUI updates use the mirror too. Packages that `npx` fetches (MCP servers, the `find-skills` skill) still come from `registry.npmjs.org`; add `registry=https://registry.npmmirror.com/` to `~/.npmrc` to send them to the mirror, which the agent's shell and `npx` MCP servers both read. |
| `--version <ver>` | `-Version <ver>` | Install this release instead of the newest one. 0.3.5 and older cannot be installed this way: they have no `constraints.txt`. |
| `--no-modify-path` | `-NoModifyPath` | Do not put uv's tool bin directory on PATH. |
| `--no-setup` | `-NoSetup` | Install the package and stop before `metasci-scribe setup`. |

`irm ... | iex` cannot pass options; use the `[scriptblock]::Create` form shown
above for them.

## What it does

1. Downloads uv 0.12.23 and checks it against the SHA-256 written in the
   script. It is kept in `~/.metasci-scribe/tools/uv-0.12.23/` and recorded in
   `~/.metasci-scribe/tools/uv.json`. A uv you installed yourself is not used or
   changed.
2. Picks the newest Metasci Scribe release on the package index it installs from
   (or `--version`), and downloads that release's `constraints.txt`.
3. Installs Metasci Scribe as a uv tool on uv's own Python 3.12 (a re-run keeps
   the Python version of the existing install), with exactly the
   dependency versions the release was tested with. uv downloads that Python
   once; a Python already on the machine is not used.
4. Puts uv's tool bin directory (`~/.local/bin`, or what `uv tool dir --bin`
   prints) on PATH with `uv tool update-shell`: on macOS and Linux it adds a line
   to your shell profile (`~/.bashrc` and `~/.bash_profile` or `~/.profile` for
   bash, `~/.zshenv` for zsh, the fish config) and on Windows it adds the
   directory to your user PATH. Open a new terminal afterwards.
5. Runs `metasci-scribe setup`: Node.js, Git for Windows (Windows only), a Python for
   the agent's shell when there is none, and the WebUI.

`~/.metasci-scribe` stands for the data directory: `METASCI_SCRIBE_DATA_DIR` when it
is set, else `~/.metasci-scribe`.

The scripts set `UV_SYSTEM_CERTS=1` for their own uv calls (unless you set it),
so uv trusts the certificates of your operating system, and give uv a longer
download timeout (`UV_HTTP_TIMEOUT=300`) for slow mirrors.

Supported: macOS (Apple Silicon; Intel Macs on a best-effort basis), Linux on
x86_64 or arm64 with glibc 2.28 or newer, Windows 10 and 11 on x64. The scripts
stop before downloading anything on Windows on arm64 and on Linux with musl
(such as Alpine) or an older glibc. On an Intel Mac the `stt` extra cannot be
installed (its `onnxruntime` has no Intel macOS build).

## Running it again

Running the command again repairs the install or moves it to the newest release
(or to `--version`). It keeps the extras and the packages recorded in the
tool's receipt, which the MCP installer adds with `--with` on macOS and Linux.
A package that went into the environment another way (on Windows, or with an
older release, when you added a channel or an MCP server) is not in the
receipt and is removed; add it again afterwards. Close Metasci Scribe first; on Windows the script stops and names the Metasci Scribe
processes that still run, because a running process keeps the files locked.

A dropped connection while Node.js, Git for Windows or the WebUI downloads fails
that stage: `metasci-scribe setup` finishes the other stages, and the script then ends
with an error although Metasci Scribe itself is installed. Run `metasci-scribe setup` or
the command again; that download starts from the beginning.

When `metasci-scribe setup` installed its own Node.js because none was found, the
agent's shell keeps finding that one first, also after you install a Node.js
of your own later (for example with nvm). Run `metasci-scribe setup` again from a
terminal where your Node.js 20.9 or newer is on PATH, and it switches to yours.

A script install pins the exact version, so `uv tool upgrade Metasci Scribe`
reports "Nothing to upgrade". Update with `metasci-scribe update`, which keeps the
same packages, or by running the command again. `metasci-scribe update` waits until
other Metasci Scribe sessions, servers and MCP servers started from the install
have closed, and names them; Ctrl+C cancels and changes nothing.

## Uninstalling

There is no `--uninstall` option. On macOS and Linux:

1. Remove the package with the uv the script installed, which
   `~/.metasci-scribe/tools/uv.json` names (before step 2, which deletes it), or
   with your own uv:
   ```bash
   uv=$(sed -E 's/.*"path": *"([^"]*)".*/\1/' ~/.metasci-scribe/tools/uv.json)
   "$uv" tool uninstall Metasci Scribe
   ```
   For a full removal, and only if no other uv on the machine uses them, also
   remove the Python 3.12 that uv downloaded and uv's download cache:
   ```bash
   "$uv" python uninstall 3.12
   "$uv" cache clean
   ```
2. Delete the downloaded tools and the agent's Python environment:
   ```bash
   rm -rf ~/.metasci-scribe/tools ~/.metasci-scribe/envs
   ```
3. Your settings (`~/.config/metasci-scribe/`) and your memories, skills and
   sessions (`sessions.db`) in `~/.metasci-scribe` stay, so a reinstall picks
   them up. For a full removal, delete `~/.config/metasci-scribe` and
   `~/.metasci-scribe` as well.
4. Remove the PATH entry: the `# uv` line and the `export PATH=...` line below
   it in the shell profile that the install named. Keep it if other tools live
   in that directory, which is common for `~/.local/bin`.

On Windows (PowerShell):

1. ```powershell
   $uv = (Get-Content -Raw -Encoding UTF8 -LiteralPath "$env:USERPROFILE\.metasci-scribe\tools\uv.json" | ConvertFrom-Json).path
   & $uv tool uninstall Metasci Scribe
   ```
   For a full removal, as on macOS and Linux: `& $uv python uninstall 3.12` and
   `& $uv cache clean`.
2. ```powershell
   Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -LiteralPath "$env:USERPROFILE\.metasci-scribe\tools", "$env:USERPROFILE\.metasci-scribe\envs"
   ```
3. As on macOS and Linux: `$env:USERPROFILE\.config\metasci-scribe` and
   `$env:USERPROFILE\.metasci-scribe` hold your settings and data.
4. Remove `%USERPROFILE%\.local\bin` from your user PATH (Settings > System >
   About > Advanced system settings > Environment Variables), unless other
   tools live there.

## Proxies

- **macOS and Linux**: `curl` and uv read `HTTPS_PROXY` and `HTTP_PROXY`. The
  "system proxy" switch of a proxy client (Clash, V2Ray and the like) does not
  reach them: export the variables in the terminal first, or turn on the
  client's TUN mode.
  ```bash
  export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890
  ```
- **Windows**: PowerShell uses the system proxy. uv follows a system proxy set
  by hand, but not an automatic configuration (a PAC script or WPAD) or a
  per-protocol list (`http=...;https=...`). In those cases set `HTTPS_PROXY`
  before running the command:
  ```powershell
  $env:HTTPS_PROXY = "http://proxy.example.com:8080"
  ```

## Networks that inspect TLS

A company network that inspects TLS signs traffic with its own certificate.
The scripts make uv trust the operating system's certificates; other tools need
it pointed out to them:

| Variable | Used by |
|---|---|
| `SSL_CERT_FILE` | Python, including Metasci Scribe's own downloads (`metasci-scribe setup` on macOS and Linux) |
| `UV_SYSTEM_CERTS=1` | uv calls that Metasci Scribe makes later, such as the MCP installer's |
| `PIP_CERT` | pip |
| `NODE_EXTRA_CA_CERTS` | Node programs (the WebUI, MCP servers) |
| `npm_config_cafile` | npm and `npx` in the agent's shell |

Point the file variables at a PEM file that contains your company's
certificate. MCP servers started with `npx` do not see `npm_config_cafile`:
put `cafile=` (and `registry`, `proxy` or `strict-ssl` if you need them) in
`~/.npmrc` or in the server's `env` in `mcp.yaml`, or rely on
`NODE_EXTRA_CA_CERTS`, which reaches them.
