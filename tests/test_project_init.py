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
    assert link in r.problems
    assert link not in r.created
    assert not target.exists()


@pytest.mark.parametrize("name", ["Knowledge", "Figures", "Tables", ".metasci-scribe"])
def test_a_file_in_place_of_a_folder_is_a_problem(tmp_path, local_data_dir, name):
    (tmp_path / name).write_text("oops", encoding="utf-8")
    r = init_project(tmp_path)
    assert tmp_path / name in r.problems
    assert tmp_path / name not in r.skipped
    assert (tmp_path / name).read_text(encoding="utf-8") == "oops"
    assert tmp_path / "outline.md" in r.created  # the rest still gets made


def test_a_folder_in_place_of_a_file_is_a_problem(tmp_path, local_data_dir):
    (tmp_path / "program.md").mkdir()
    r = init_project(tmp_path)
    assert tmp_path / "program.md" in r.problems
    assert tmp_path / "program.md" not in r.skipped


def _fail_second_write(monkeypatch):
    """Make the temp-file write of outline.md die half way, once."""
    real = project._write_temp
    state = {"armed": True}

    def flaky(tmp: Path, text: str) -> None:
        if state["armed"] and "# Outline" in text:
            tmp.write_text(text[:10], encoding="utf-8")
            raise OSError("disk full")
        real(tmp, text)

    monkeypatch.setattr(project, "_write_temp", flaky)
    return state


def test_a_failed_write_leaves_nothing_behind(tmp_path, local_data_dir, monkeypatch):
    state = _fail_second_write(monkeypatch)
    with pytest.raises(project.InitError) as err:
        init_project(tmp_path)
    assert "disk full" in str(err.value)
    assert not (tmp_path / "outline.md").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["program.md"]  # no temp files
    assert err.value.result.created == [tmp_path / "program.md"]
    # The next /init is not fooled by a half-written file.
    state["armed"] = False
    r = init_project(tmp_path)
    assert tmp_path / "outline.md" in r.created
    assert (tmp_path / "outline.md").read_text(encoding="utf-8").startswith("# Outline")


def test_a_os_error_on_the_root_is_an_init_error(tmp_path, local_data_dir, monkeypatch):
    def boom(self, *a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", boom)
    with pytest.raises(project.InitError) as err:
        init_project(tmp_path / "new")
    assert err.value.result.created == []


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


def test_outline_skeleton_covers_three_sections_with_parseable_ids(
    tmp_path, local_data_dir
):
    init_project(tmp_path)
    outline = (tmp_path / "outline.md").read_text(encoding="utf-8")
    sections = re.findall(r"^## (\w+)$", outline, re.MULTILINE)
    assert sections == ["Introduction", "Results", "Discussion"]
    heads = re.findall(r"^### ([A-Z]\d+) · (\S+) ", outline, re.MULTILINE)
    assert [h[0] for h in heads] == ["I1", "R1", "D1"]
    assert all(h[1] in "①②③④⑤⑥⑦⑧⑨" for h in heads)
    # every paragraph has the field the parser requires
    blocks = re.split(r"^### ", outline, flags=re.MULTILINE)[1:]
    assert len(blocks) == 3
    assert all(re.search(r"^- 핵심 내용:", b, re.MULTILINE) for b in blocks)


def test_outline_evidence_placeholder_matches_the_documented_format(
    tmp_path, local_data_dir
):
    init_project(tmp_path)
    outline = (tmp_path / "outline.md").read_text(encoding="utf-8")
    lines = re.findall(r"^- 근거: (.*)$", outline, re.MULTILINE)
    assert lines
    assert all(ln.startswith("확인 필요: 자료명/") for ln in lines)


def test_outline_comment_explains_the_fields_and_the_numbering(
    tmp_path, local_data_dir
):
    init_project(tmp_path)
    outline = (tmp_path / "outline.md").read_text(encoding="utf-8")
    comment = re.search(r"<!--.*?-->", outline, re.DOTALL).group(0)
    for term in ("역할", "핵심 내용", "전달할 내용", "근거", "연결", "①", "R1 · ②"):
        assert term in comment, term


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


async def test_init_command_reports_a_failed_write_and_what_was_made(
    tmp_path, local_data_dir, monkeypatch
):
    _fail_second_write(monkeypatch)
    ctx, lines = _ctx(tmp_path)
    await InitCommand().execute(ctx, [])
    text = "\n".join(lines)
    assert "disk full" in text
    assert re.search(r"Created:\s*\n\s*program\.md", text)
    assert not (tmp_path / "outline.md").exists()


async def test_init_command_reports_any_os_error(tmp_path, monkeypatch):
    def boom(root):
        raise PermissionError("denied")

    monkeypatch.setattr(project, "init_project", boom)
    ctx, lines = _ctx(tmp_path)
    await InitCommand().execute(ctx, [])
    assert any("denied" in ln for ln in lines)


async def test_init_command_lists_problems(tmp_path, local_data_dir):
    (tmp_path / "Knowledge").write_text("a file", encoding="utf-8")
    ctx, lines = _ctx(tmp_path)
    await InitCommand().execute(ctx, [])
    text = "\n".join(lines)
    assert re.search(r"Problem.*\n.*Knowledge", text)
    assert "not a folder" in text
    assert not re.search(r"Skipped.*\n(.*\n)*.*Knowledge", text)


# ------------------------------------------------- stored network folders


def test_stored_network_folder_is_refused_by_its_raw_spelling(monkeypatch):
    _fake_drive_type(monkeypatch, {"C": project.DRIVE_FIXED})
    with pytest.raises(NetworkPathError) as err:
        project.require_local_stored("//host/share/paper", None)
    assert "//host/share/paper" in str(err.value)
    with pytest.raises(NetworkPathError):
        project.require_local_stored("C:/local", "//host/share/paper/runs/x")
    project.require_local_stored(None, None)
    project.require_local_stored("", None)


def _network_stored(monkeypatch):
    """A stored UNC workspace; resolving it is an error, as a dead share would be."""
    monkeypatch.setattr(project, "_IS_WINDOWS", True)

    def never(*_a, **_k):
        raise AssertionError("the share was touched")

    monkeypatch.setattr(SessionDirs, "from_stored", never)
    monkeypatch.setattr(paths, "require_workspace_root", never)
    return {"workspace_dir": "//host/share/paper"}


async def test_resume_of_a_network_workspace_says_use_a_local_folder(monkeypatch):
    from metasci_scribe.commands.implementation.session import ResumeCommand
    from tests.fakes.common import FakeThreadStore
    from tests.test_resume_command import _ctx as resume_ctx

    stored = _network_stored(monkeypatch)
    ctx, ui = resume_ctx(
        thread_store=FakeThreadStore(resolved_thread_id="t2", metadata=stored)
    )
    before = ctx.dirs
    await ResumeCommand().execute(ctx, ["t2"])
    msgs = [c.args[0] for c in ui.append_system.call_args_list]
    assert any("local" in m.lower() and "//host/share/paper" in m for m in msgs)
    ui.handle_session_resume.assert_not_called()
    assert ctx.dirs == before
    assert ctx.thread_id == "current"


def test_cli_resume_of_a_network_workspace_stops_with_the_message(
    monkeypatch, tmp_path
):
    import typer

    from metasci_scribe.cli import commands
    from tests.test_cli_workspace_resolution import _config, _run_prompt

    stored = _network_stored(monkeypatch)
    printed: list[str] = []
    monkeypatch.setattr(commands.console, "print", lambda *a, **k: printed.append(a[0]))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(typer.Exit):
        _run_prompt(monkeypatch, _config(), thread_id="t-net", stored=stored)
    assert any("local" in line.lower() for line in printed)


def test_picker_groups_a_network_thread_without_touching_the_share(monkeypatch):
    from metasci_scribe.cli.picker import _build_items

    stored = _network_stored(monkeypatch)
    items = _build_items([{"thread_id": "t1", **stored}])
    headers = [i["label"] for i in items if i["type"] == "header"]
    assert headers == ["//host/share/paper"]
    assert [i["thread"]["thread_id"] for i in items if i["type"] == "thread"] == ["t1"]


def test_threads_list_label_for_a_network_thread_does_not_touch_the_share(monkeypatch):
    from metasci_scribe.commands.implementation.session import _workspace_label

    stored = _network_stored(monkeypatch)
    assert _workspace_label({"thread_id": "t1", **stored}) == "//host/share/paper"
    assert _workspace_label({"thread_id": "t1", **stored}, name_only=True) == "paper"
