"""``/init``, the paper folder layout and the network-path check."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import MagicMock

import pytest

from metasci_scribe import paths, project
from metasci_scribe.commands.base import CommandContext
from metasci_scribe.commands.implementation.init import InitCommand
from metasci_scribe.paths import SessionDirs, Workspace, WorkspaceRootError
from metasci_scribe.project import (
    InitResult,
    NetworkPathError,
    extract_dir,
    init_project,
    is_network_path,
)

LAYOUT = [
    "program.md",
    "outline.md",
    "manuscript.md",
    "Knowledge",
    "Figures",
    "Tables",
    ".metasci-scribe",
]


@pytest.fixture
def local_data_dir(monkeypatch, tmp_path_factory):
    """Keep the app's data folder out of the tmp_path the tests use."""
    data = tmp_path_factory.mktemp("data")
    monkeypatch.setattr(paths, "DATA_DIR", data)
    return data


def _fake_drive_type(monkeypatch, kinds: dict[str, int]):
    calls: list[str] = []

    def fake(root: str) -> int:
        calls.append(root)
        return kinds.get(root[:1].upper(), project.DRIVE_NO_ROOT_DIR)

    monkeypatch.setattr(project, "_drive_type", fake)
    monkeypatch.setattr(project, "_IS_WINDOWS", True)
    return calls


# ------------------------------------------------------------ the layout


def test_init_creates_layout(tmp_path, local_data_dir):
    r = init_project(tmp_path)
    for n in LAYOUT:
        assert (tmp_path / n).exists()
    for d in ("Knowledge", "Figures", "Tables", ".metasci-scribe"):
        assert (tmp_path / d).is_dir()
    assert [p.name for p in r.created] == LAYOUT
    assert r.skipped == []


def test_init_never_overwrites(tmp_path, local_data_dir):
    (tmp_path / "outline.md").write_text("mine", encoding="utf-8")
    r = init_project(tmp_path)
    assert (tmp_path / "outline.md").read_text(encoding="utf-8") == "mine"
    assert tmp_path / "outline.md" in r.skipped
    assert tmp_path / "outline.md" not in r.created
    assert tmp_path / "program.md" in r.created


def test_init_skips_an_existing_folder_and_keeps_its_files(tmp_path, local_data_dir):
    (tmp_path / "Knowledge").mkdir()
    (tmp_path / "Knowledge" / "a.pdf").write_bytes(b"%PDF")
    r = init_project(tmp_path)
    assert tmp_path / "Knowledge" in r.skipped
    assert (tmp_path / "Knowledge" / "a.pdf").read_bytes() == b"%PDF"


def test_init_is_idempotent(tmp_path, local_data_dir):
    first = init_project(tmp_path)
    before = {p: p.read_bytes() for p in (tmp_path / n for n in LAYOUT) if p.is_file()}
    second = init_project(tmp_path)
    assert second.created == []
    assert second.skipped == first.created
    assert {p: p.read_bytes() for p in before} == before


def test_init_does_not_follow_a_dangling_link(tmp_path, local_data_dir):
    target = tmp_path / "elsewhere.md"
    link = tmp_path / "manuscript.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks need a privilege on this machine")
    r = init_project(tmp_path)
    assert link in r.skipped
    assert not target.exists()


def test_init_creates_a_missing_root(tmp_path, local_data_dir):
    root = tmp_path / "new" / "paper"
    r = init_project(root)
    assert (root / "outline.md").is_file()
    assert len(r.created) == len(LAYOUT)


def test_init_refuses_a_file_as_root(tmp_path, local_data_dir):
    f = tmp_path / "f.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        init_project(f)


def test_extract_dir(tmp_path):
    assert extract_dir(tmp_path) == tmp_path / ".metasci-scribe" / "extracted"


# ---------------------------------------------------------- the templates


def test_templates_are_utf8_and_hold_the_parsed_markers(tmp_path, local_data_dir):
    init_project(tmp_path)
    outline = (tmp_path / "outline.md").read_text(encoding="utf-8")
    assert "### R1 · ①" in outline
    assert "◇" in outline
    assert "- 핵심 내용:" in outline
    program = (tmp_path / "program.md").read_text(encoding="utf-8")
    assert program.startswith("---\n")
    assert re.search(r"^title:", program, re.MULTILINE)
    assert re.search(r"^journal:", program, re.MULTILINE)
    assert re.search(r"^core_message:", program, re.MULTILINE)
    assert "## 저자 메모" in program


def test_outline_comment_does_not_carry_the_open_marker(tmp_path, local_data_dir):
    """A ◇ inside the explanatory comment would read as an unanswered question."""
    init_project(tmp_path)
    outline = (tmp_path / "outline.md").read_text(encoding="utf-8")
    comments = re.findall(r"<!--.*?-->", outline, re.DOTALL)
    assert comments
    assert all("◇" not in c for c in comments)
    assert all(not re.match(r"#{1,6} ", ln) for c in comments for ln in c.splitlines())


def test_manuscript_starts_empty_and_files_use_lf(tmp_path, local_data_dir):
    init_project(tmp_path)
    assert (tmp_path / "manuscript.md").read_bytes() == b""
    assert b"\r\n" not in (tmp_path / "outline.md").read_bytes()


# ---------------------------------------------------------- the guards


def test_init_refuses_a_folder_holding_the_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "home" / ".metasci-scribe")
    with pytest.raises(WorkspaceRootError):
        init_project(tmp_path)
    assert not (tmp_path / "outline.md").exists()


def test_unc_path_is_network():
    assert is_network_path(PureWindowsPath(r"\\host\share\p"))


@pytest.mark.parametrize(
    "text",
    [
        r"\\host\share\p",
        "//host/share/p",
        r"\\?\UNC\host\share\p",
    ],
)
def test_unc_spellings_are_network(text):
    assert is_network_path(PureWindowsPath(text))
    assert is_network_path(text)


def test_posix_path_is_never_network(monkeypatch):
    monkeypatch.setattr(project, "_IS_WINDOWS", False)
    assert not is_network_path(PurePosixPath("//host/share/p"))
    assert not is_network_path(Path("/mnt/z/paper"))


def test_remote_drive_is_network(monkeypatch):
    calls = _fake_drive_type(monkeypatch, {"Z": project.DRIVE_REMOTE})
    assert is_network_path(r"Z:\papers\a")
    assert is_network_path(PureWindowsPath("z:/papers/a"))
    assert calls[0] == "Z:\\"


def test_extended_prefix_on_a_remote_drive_is_network(monkeypatch):
    _fake_drive_type(monkeypatch, {"Z": project.DRIVE_REMOTE})
    assert is_network_path("\\\\?\\Z:\\papers")


def test_fixed_removable_and_missing_drives_are_not_network(monkeypatch):
    _fake_drive_type(
        monkeypatch, {"C": project.DRIVE_FIXED, "E": project.DRIVE_REMOVABLE}
    )
    assert not is_network_path(r"C:\papers")
    assert not is_network_path(r"E:\papers")
    assert not is_network_path(r"Q:\papers")  # no such drive
    assert not is_network_path("\\\\?\\C:\\papers")


def test_relative_path_is_judged_by_where_it_lands(monkeypatch):
    _fake_drive_type(monkeypatch, {"Z": project.DRIVE_REMOTE})
    monkeypatch.setattr(project.os.path, "abspath", lambda p: "Z:\\cwd\\paper")
    assert is_network_path("paper")


@pytest.mark.skipif(project.os.name != "nt", reason="Windows API")
def test_drive_type_asks_windows_for_a_local_drive():
    system_drive = project.os.environ.get("SystemDrive", "C:")
    assert project._drive_type(system_drive + "\\") == project.DRIVE_FIXED


def test_init_refuses_network_path(monkeypatch, tmp_path, local_data_dir):
    monkeypatch.setattr(project, "is_network_path", lambda p: True)
    with pytest.raises(NetworkPathError) as err:
        init_project(tmp_path)
    assert "local" in str(err.value).lower()
    assert not (tmp_path / "outline.md").exists()
    assert isinstance(err.value, RuntimeError)


def test_init_refuses_a_unc_root_without_touching_it():
    with pytest.raises(NetworkPathError):
        init_project(PureWindowsPath(r"\\host\share\paper"))  # type: ignore[arg-type]


# ----------------------------------------------------------- the command


def _ctx(root: Path) -> tuple[CommandContext, list[str]]:
    lines: list[str] = []
    ui = MagicMock()
    ui.append_system.side_effect = lambda text, style="dim": lines.append(text)
    ctx = CommandContext(
        agent=None, thread_id="t", ui=ui, dirs=SessionDirs(Workspace(root))
    )
    return ctx, lines


async def test_init_command_lists_created_then_skipped(tmp_path, local_data_dir):
    (tmp_path / "outline.md").write_text("mine", encoding="utf-8")
    ctx, lines = _ctx(tmp_path)
    await InitCommand().execute(ctx, [])
    text = "\n".join(lines)
    assert "program.md" in text
    assert re.search(r"Skipped.*\n.*outline\.md", text)
    assert text.index("Created") < text.index("outline.md")
    assert ctx.command_error is None


async def test_init_command_second_run_reports_nothing_created(
    tmp_path, local_data_dir
):
    ctx, lines = _ctx(tmp_path)
    await InitCommand().execute(ctx, [])
    lines.clear()
    await InitCommand().execute(ctx, [])
    text = "\n".join(lines)
    assert "Created" not in text
    assert "Skipped" in text


async def test_init_command_reports_a_refused_folder(
    tmp_path, monkeypatch, local_data_dir
):
    monkeypatch.setattr(project, "is_network_path", lambda p: True)
    ctx, lines = _ctx(tmp_path)
    await InitCommand().execute(ctx, [])
    assert any("local" in ln.lower() for ln in lines)
    assert not (tmp_path / "outline.md").exists()


async def test_init_command_reports_the_data_dir_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "home" / ".metasci-scribe")
    ctx, lines = _ctx(tmp_path)
    await InitCommand().execute(ctx, [])
    assert any("cannot be a workspace" in ln for ln in lines)


def test_init_command_is_registered():
    from metasci_scribe.commands import manager

    assert any(c.name == "/init" for c in manager.get_all_commands())


def test_init_result_is_a_plain_record():
    r = InitResult(created=[Path("a")], skipped=[])
    assert r.created == [Path("a")]
