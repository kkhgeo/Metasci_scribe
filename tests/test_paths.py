"""Tests for metasci_scribe.paths — the Workspace type, start resolution, ensure_dirs."""

import os
from pathlib import Path

import pytest

from metasci_scribe import paths
from metasci_scribe.paths import (
    SessionDirs,
    Workspace,
    normalize_path,
    process_session_dirs,
    process_workspace,
    resolve_virtual_path,
    start_workspace_path,
)


@pytest.fixture(autouse=True)
def _restore_paths():
    """Snapshot module-level path globals and restore after each test."""
    orig = {
        "DATA_DIR": paths.DATA_DIR,
        "MEMORIES_DIR": paths.MEMORIES_DIR,
        "GLOBAL_SKILLS_DIR": paths.GLOBAL_SKILLS_DIR,
        "GLOBAL_MEMORIES_DIR": paths.GLOBAL_MEMORIES_DIR,
    }
    yield
    for name, value in orig.items():
        setattr(paths, name, value)


def _fake_home(monkeypatch, home: Path) -> None:
    """Point ``~`` at *home* (Windows reads USERPROFILE, not HOME)."""
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


class TestWorkspace:
    """The Workspace value object."""

    def test_derived_dirs(self, tmp_path):
        ws = Workspace(tmp_path)
        root = tmp_path.resolve()
        assert ws.root == root
        assert ws.state_dir == root / ".metasci-scribe"
        assert ws.skills_dir == root / ".metasci-scribe" / "skills"
        assert ws.runs_dir == root / "runs"

    def test_skills_dir_lives_under_project_state_dir(self, tmp_path):
        """Project state (skills) sits in <root>/.metasci-scribe, not beside the project files."""
        ws = Workspace(tmp_path)
        assert ws.skills_dir == tmp_path.resolve() / ".metasci-scribe" / "skills"
        assert ws.skills_dir.parent == ws.state_dir
        assert not (tmp_path / "skills").exists()

    def test_accepts_string_path(self, tmp_path):
        assert Workspace(str(tmp_path)).root == tmp_path.resolve()

    def test_resolves_symlinks(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real)
        assert Workspace(link) == Workspace(real)
        assert Workspace(link).root == real.resolve()

    def test_expands_user(self, monkeypatch, tmp_path):
        _fake_home(monkeypatch, tmp_path)
        assert Workspace("~/proj").root == (tmp_path / "proj").resolve()

    def test_does_not_create_the_folder(self, tmp_path):
        missing = tmp_path / "not-there"
        ws = Workspace(missing)
        assert ws.root == missing.resolve()
        assert not missing.exists()

    def test_key_is_posix_without_trailing_slash(self, tmp_path):
        ws = Workspace(f"{tmp_path}/proj/")
        assert ws.key == (tmp_path / "proj").resolve().as_posix()
        assert not ws.key.endswith("/")

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("proj/", "proj"),
            ("proj/./sub/..", "proj"),
            ("with space", "with space"),
            ("項目 ü", "項目 ü"),
        ],
    )
    def test_key_normalises_shared_spellings(self, tmp_path, spelling, expected):
        """Spellings the WebUI normalises to the same key as the core."""
        ws = Workspace(f"{tmp_path}/{spelling}")
        assert ws.key == (tmp_path.resolve() / expected).as_posix()

    def test_is_immutable_and_hashable(self, tmp_path):
        ws = Workspace(tmp_path)
        with pytest.raises(AttributeError):
            ws.root = tmp_path / "other"  # type: ignore[misc]
        assert {ws: 1}[Workspace(tmp_path)] == 1

    def test_ignores_removed_dir_overrides(self, tmp_path, monkeypatch):
        """The per-workspace folders always derive from the root."""
        monkeypatch.setenv("METASCI_SCRIBE_SKILLS_DIR", str(tmp_path / "s"))
        monkeypatch.setenv("METASCI_SCRIBE_RUNS_DIR", str(tmp_path / "r"))
        ws = Workspace(tmp_path / "proj")
        assert ws.skills_dir == ws.root / ".metasci-scribe" / "skills"
        assert ws.runs_dir == ws.root / "runs"


class TestNormalizePath:
    def test_equal_spellings_normalise_equal(self, tmp_path):
        a = normalize_path(f"{tmp_path}/x/../y")
        b = normalize_path(tmp_path / "y")
        assert a == b

    def test_nul_byte_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="NUL byte"):
            normalize_path(f"{tmp_path}/evil\x00x")


class TestStartWorkspacePath:
    """Which folder a process starts in: workdir, then default_workdir, then cwd."""

    def test_workdir_wins(self, tmp_path):
        got = start_workspace_path(tmp_path / "a", tmp_path / "b")
        assert got == tmp_path / "a"

    def test_default_workdir_when_no_workdir(self, tmp_path):
        assert start_workspace_path(None, tmp_path / "b") == tmp_path / "b"

    def test_cwd_when_neither(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert start_workspace_path() == Path(os.getcwd())

    def test_expands_user_and_is_absolute(self, tmp_path, monkeypatch):
        _fake_home(monkeypatch, tmp_path)
        got = start_workspace_path("~/proj")
        assert got == tmp_path / "proj"
        assert got.is_absolute()

    def test_relative_workdir_is_made_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert start_workspace_path("rel") == tmp_path / "rel"

    def test_empty_strings_fall_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert start_workspace_path("", "") == Path(os.getcwd())


class TestProcessWorkspace:
    def test_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("METASCI_SCRIBE_WORKSPACE_DIR", str(tmp_path))
        assert process_workspace() == Workspace(tmp_path)

    def test_cwd_without_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("METASCI_SCRIBE_WORKSPACE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert process_workspace() == Workspace(tmp_path)

    def test_read_once_per_process(self, tmp_path, monkeypatch):
        monkeypatch.setenv("METASCI_SCRIBE_WORKSPACE_DIR", str(tmp_path / "first"))
        first = process_workspace()
        monkeypatch.setenv("METASCI_SCRIBE_WORKSPACE_DIR", str(tmp_path / "second"))
        assert process_workspace() == first


class TestSessionDirs:
    def test_daemon_session_works_in_the_root(self, tmp_path):
        dirs = SessionDirs(Workspace(tmp_path))
        assert dirs.run_dir is None
        assert dirs.work_dir == tmp_path.resolve()
        assert dirs.metadata() == {"workspace_dir": tmp_path.resolve().as_posix()}

    def test_run_session_works_in_its_run_folder(self, tmp_path):
        ws = Workspace(tmp_path)
        dirs = SessionDirs(ws, ws.runs_dir / "exp")
        assert dirs.work_dir == ws.runs_dir / "exp"
        assert dirs.metadata() == {
            "workspace_dir": ws.key,
            "run_dir": (ws.runs_dir / "exp").as_posix(),
        }

    def test_run_dir_equal_to_root_is_no_run_dir(self, tmp_path):
        dirs = SessionDirs(Workspace(tmp_path), tmp_path / "." / "")
        assert dirs.run_dir is None
        assert "run_dir" not in dirs.metadata()


class TestSessionDirsFromStored:
    def test_nothing_stored(self):
        assert SessionDirs.from_stored(None) is None
        assert SessionDirs.from_stored("") is None

    def test_a_timestamp_named_root_stays_a_root(self, tmp_path):
        """New rows carry ``run_dir`` themselves; reading never guesses."""
        folder = tmp_path / "runs" / "20260930_120000"
        assert SessionDirs.from_stored(str(folder)) == SessionDirs(Workspace(folder))

    def test_unresolved_values_are_normalised(self, tmp_path, monkeypatch):
        _fake_home(monkeypatch, tmp_path)
        assert SessionDirs.from_stored("~/proj/") == SessionDirs(
            Workspace(tmp_path / "proj")
        )

    def test_run_dir_outside_workspace_is_dropped(self, tmp_path):
        """Run folders live under ``<root>/runs``; anything else is not one."""
        dirs = SessionDirs.from_stored(
            str(tmp_path / "a"), str(tmp_path / "b" / "runs" / "20260930_120000")
        )
        assert dirs == SessionDirs(Workspace(tmp_path / "a"))

    def test_unreadable_value_reads_as_nothing_stored(self):
        assert SessionDirs.from_stored("/tmp/evil\x00x") is None

    def test_non_string_value_reads_as_nothing_stored(self, tmp_path):
        """Metadata is caller-supplied JSON, so a folder can be stored as a
        number or a list; one bad row must not break listing the rest."""
        assert SessionDirs.from_stored(123) is None
        assert SessionDirs.from_stored(str(tmp_path), ["runs", "x"]) is None


class TestSessionDirsFromLegacy:
    def test_generated_run_folder_is_split_into_workspace_and_run_dir(self, tmp_path):
        run = tmp_path / "runs" / "20260930_120000"
        assert SessionDirs.from_legacy(str(run)) == SessionDirs(
            Workspace(tmp_path), run
        )

    def test_named_run_folder_is_left_as_its_own_workspace(self, tmp_path):
        """A project can live in a folder called ``runs``; only generated
        names are known to be run folders."""
        folder = tmp_path / "runs" / "exp"
        assert SessionDirs.from_legacy(str(folder)) == SessionDirs(Workspace(folder))

    def test_names_the_cli_does_not_generate_are_not_split(self, tmp_path):
        for name in (
            "20260930_120000_1",
            "2026093_120000",
            "x20260930_120000",
            "99999999_999999",
        ):
            folder = tmp_path / "runs" / name
            assert SessionDirs.from_legacy(str(folder)).run_dir is None, name

    def test_timestamp_folder_outside_runs_is_not_split(self, tmp_path):
        folder = tmp_path / "other" / "20260930_120000"
        assert SessionDirs.from_legacy(str(folder)) == SessionDirs(Workspace(folder))

    def test_generated_names_round_trip(self):
        from datetime import datetime

        name = datetime(2026, 9, 30, 12, 0, 0).strftime(paths.RUN_NAME_FORMAT)
        assert paths.is_generated_run_name(name)


class TestProcessSessionDirs:
    def test_with_run_dir(self, tmp_path, monkeypatch):
        run = tmp_path / "runs" / "exp"
        monkeypatch.setenv("METASCI_SCRIBE_WORKSPACE_DIR", str(tmp_path))
        monkeypatch.setenv("METASCI_SCRIBE_RUN_DIR", str(run))
        assert process_session_dirs() == SessionDirs(Workspace(tmp_path), run)


class TestReloadEnvDirs:
    """An override merged into the environment after import (project .env)."""

    def test_picks_up_a_late_memories_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("METASCI_SCRIBE_MEMORY_DIR", raising=False)
        monkeypatch.setenv("METASCI_SCRIBE_MEMORIES_DIR", str(tmp_path / "mem"))
        paths.reload_env_dirs()
        assert paths.MEMORIES_DIR == tmp_path / "mem"

    def test_falls_back_to_global_memories(self, monkeypatch):
        monkeypatch.delenv("METASCI_SCRIBE_MEMORIES_DIR", raising=False)
        monkeypatch.delenv("METASCI_SCRIBE_MEMORY_DIR", raising=False)
        paths.reload_env_dirs()
        assert paths.MEMORIES_DIR == paths.GLOBAL_MEMORIES_DIR


class TestResolveVirtualPath:
    def test_absolute_virtual_path(self, tmp_path):
        assert (
            resolve_virtual_path(tmp_path, "/a/b.png")
            == (tmp_path / "a/b.png").resolve()
        )

    def test_relative_virtual_path(self, tmp_path):
        assert resolve_virtual_path(tmp_path, "a.png") == (tmp_path / "a.png").resolve()

    def test_root(self, tmp_path):
        assert resolve_virtual_path(tmp_path, "/") == tmp_path.resolve()

    def test_depends_only_on_the_given_work_dir(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        assert resolve_virtual_path(a, "/f") != resolve_virtual_path(b, "/f")


class TestEnsureDirs:
    """ensure_dirs creates the global data and memories folders only."""

    def test_creates_global_dirs_not_workspace_dirs(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        memories = tmp_path / "memories"
        ws_root = tmp_path / "workspace"
        ws_root.mkdir()
        monkeypatch.setattr(paths, "DATA_DIR", data)
        monkeypatch.setattr(paths, "MEMORIES_DIR", memories)
        monkeypatch.chdir(ws_root)

        paths.ensure_dirs()

        assert data.is_dir()
        assert memories.is_dir()
        assert not (ws_root / "memories").exists()
        assert not (ws_root / "skills").exists()  # created on demand by install_skill()


class TestDataDir:
    """Tests for DATA_DIR and global data-dir helpers."""

    def test_global_skills_dir_under_data_dir(self):
        """GLOBAL_SKILLS_DIR must live under DATA_DIR."""
        assert paths.GLOBAL_SKILLS_DIR == paths.DATA_DIR / "skills"

    def test_global_memories_dir_under_data_dir(self):
        """GLOBAL_MEMORIES_DIR must live under DATA_DIR."""
        assert paths.GLOBAL_MEMORIES_DIR == paths.DATA_DIR / "memories"


class TestSharedDirFromEnv:
    """The data and config folders, which other programs locate too, follow one
    path rule."""

    @pytest.mark.parametrize("value", ["", "   "])
    def test_shared_dir_from_env_blank_is_unset(self, value, monkeypatch):
        monkeypatch.setenv("METASCI_SCRIBE_DATA_DIR", value)
        assert paths._shared_dir_from_env("METASCI_SCRIBE_DATA_DIR") is None

    @pytest.mark.parametrize(("value", "suffix"), [("~", ""), ("~/evo", "evo")])
    def test_shared_dir_from_env_expands_a_leading_tilde(
        self, value, suffix, tmp_path, monkeypatch
    ):
        _fake_home(monkeypatch, tmp_path)
        monkeypatch.setenv("METASCI_SCRIBE_DATA_DIR", value)
        assert (
            paths._shared_dir_from_env("METASCI_SCRIBE_DATA_DIR") == tmp_path / suffix
        )

    @pytest.mark.parametrize("value", ["~nobody/evo", "$HOME/evo"])
    def test_shared_dir_from_env_expands_nothing_else(
        self, value, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("METASCI_SCRIBE_DATA_DIR", value)
        assert paths._shared_dir_from_env("METASCI_SCRIBE_DATA_DIR") == tmp_path / value

    def test_shared_dir_from_env_resolves_a_relative_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("METASCI_SCRIBE_DATA_DIR", "data")
        assert (
            paths._shared_dir_from_env("METASCI_SCRIBE_DATA_DIR") == tmp_path / "data"
        )

    @pytest.mark.parametrize("value", [None, "data"])
    def test_data_dir_follows_scribe_data_dir(self, value, tmp_path, monkeypatch):
        _fake_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        if value is None:
            monkeypatch.delenv("METASCI_SCRIBE_DATA_DIR", raising=False)
            expected = tmp_path / ".metasci-scribe"
        else:
            monkeypatch.setenv("METASCI_SCRIBE_DATA_DIR", value)
            expected = tmp_path / value
        assert paths._data_dir() == expected

    @pytest.mark.parametrize("xdg", [None, "cfg"])
    def test_config_dir_follows_xdg_config_home(self, xdg, tmp_path, monkeypatch):
        _fake_home(monkeypatch, tmp_path)
        monkeypatch.chdir(tmp_path)
        if xdg is None:
            monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
            expected = tmp_path / ".config" / "metasci-scribe"
        else:
            monkeypatch.setenv("XDG_CONFIG_HOME", xdg)
            expected = tmp_path / xdg / "metasci-scribe"
        assert paths._config_dir() == expected

    @pytest.mark.parametrize("xdg_set", [True, False])
    def test_dir_env_gives_children_this_process_folders(
        self, xdg_set, tmp_path, monkeypatch
    ):
        """Children run in other folders and load their own ``.env``;
        ``XDG_CONFIG_HOME`` is passed only when this process started with it."""
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path / "cfg" / "metasci-scribe")
        monkeypatch.setattr(paths, "_XDG_CONFIG_HOME_SET", xdg_set)
        env = paths.dir_env(
            {"METASCI_SCRIBE_DATA_DIR": "rel", "XDG_CONFIG_HOME": "rel", "OTHER": "x"}
        )
        expected = {"METASCI_SCRIBE_DATA_DIR": str(tmp_path / "data"), "OTHER": "x"}
        if xdg_set:
            expected["XDG_CONFIG_HOME"] = str(tmp_path / "cfg")
        assert env == expected
