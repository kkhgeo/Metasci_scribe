"""Tests for /install-skill and /uninstall-skill commands."""

from unittest.mock import MagicMock, patch

from metasci_scribe.paths import SessionDirs


def _ctx(workspace):
    from metasci_scribe.commands.base import CommandContext

    ui = MagicMock()
    ui.supports_interactive = True
    return CommandContext(
        agent=None, thread_id="tid", ui=ui, dirs=SessionDirs(workspace)
    ), ui


class TestInstallSkill:
    async def test_usage_message_when_no_args(self, workspace):
        from metasci_scribe.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx(workspace)
        await InstallSkill().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Usage:" in m for m in msgs)

    async def test_happy_path(self, workspace):
        from metasci_scribe.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "metasci_scribe.tools.skills_manager.install_skill",
            return_value={
                "success": True,
                "name": "demo-skill",
                "description": "demo",
                "path": "/tmp/demo",
            },
        ) as install_mock:
            await InstallSkill().execute(ctx, ["./some-path"])
        assert install_mock.call_args.kwargs["workspace"] is workspace
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Installed: demo-skill" in m for m in msgs)

    async def test_missing_git_prints_the_git_message(
        self, tmp_path, workspace, no_git
    ):
        from metasci_scribe.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx(workspace)
        with patch("metasci_scribe.paths.GLOBAL_SKILLS_DIR", tmp_path / "skills"):
            await InstallSkill().execute(ctx, ["owner/repo@skill"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any(m.startswith("Failed: git was not found on PATH.") for m in msgs)


class TestUninstallSkill:
    async def test_usage_message_when_no_args(self, workspace):
        from metasci_scribe.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx(workspace)
        await UninstallSkill().execute(ctx, [])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Usage:" in m for m in msgs)

    async def test_uninstall_success(self, workspace):
        from metasci_scribe.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "metasci_scribe.tools.skills_manager.uninstall_skill",
            return_value={"success": True},
        ) as uninstall_mock:
            await UninstallSkill().execute(ctx, ["demo-skill"])
        uninstall_mock.assert_called_once_with("demo-skill", workspace=workspace)
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Uninstalled: demo-skill" in m for m in msgs)

    async def test_uninstall_failure(self, workspace):
        from metasci_scribe.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "metasci_scribe.tools.skills_manager.uninstall_skill",
            return_value={"success": False, "error": "not found"},
        ):
            await UninstallSkill().execute(ctx, ["missing"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Failed: not found" in m for m in msgs)


class TestNoReloadHint:
    async def test_install_prints_no_reload_hint(self, workspace):
        from metasci_scribe.commands.implementation.skills import InstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "metasci_scribe.tools.skills_manager.install_skill",
            return_value={
                "success": True,
                "name": "demo-skill",
                "description": "demo",
                "path": "/tmp/demo",
            },
        ):
            await InstallSkill().execute(ctx, ["./some-path"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert any("Installed: demo-skill" in m for m in msgs)
        assert not any("/new" in m for m in msgs)

    async def test_uninstall_prints_no_reload_hint(self, workspace):
        from metasci_scribe.commands.implementation.skills import UninstallSkill

        ctx, ui = _ctx(workspace)
        with patch(
            "metasci_scribe.tools.skills_manager.uninstall_skill",
            return_value={"success": True},
        ):
            await UninstallSkill().execute(ctx, ["demo-skill"])
        msgs = [c.args[0] for c in ui.append_system.call_args_list]
        assert msgs == ["Uninstalled: demo-skill"]


def _make_skill(parent, name, *, expert=False):
    d = parent / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: demo\n---\n\n# {name}\n"
    )
    if expert:
        (d / "EXPERT.md").write_text("You are a demo expert.\n")
    return d
