"""Tests for CommandManager subcommand metadata."""

from metasci_scribe.commands.base import SubCommand
from metasci_scribe.commands.manager import CommandManager


class TestSubCommand:
    def test_creation_with_defaults(self):
        sc = SubCommand("list", "List servers")
        assert sc.name == "list"
        assert sc.description == "List servers"
        assert sc.arguments == []

    def test_creation_with_arguments(self):
        from metasci_scribe.commands.base import Argument

        sc = SubCommand(
            "add",
            "Add a server",
            arguments=[Argument("name", str, "Server name", required=True)],
        )
        assert len(sc.arguments) == 1
        assert sc.arguments[0].name == "name"


class TestCommandManagerSubcommands:
    def test_mcp_has_six_subcommands(self):
        """``/mcp`` must expose all 6 subcommands via the manager."""
        manager = CommandManager()
        from metasci_scribe.commands.implementation.mcp import MCPCommand

        manager.register(MCPCommand())
        scs = manager.list_subcommands("/mcp")
        names = {name for name, _desc in scs}
        assert names == {"list", "config", "add", "edit", "remove", "install"}

    def test_command_without_subcommands_returns_empty(self):
        """A command with no subcommands must return an empty list."""
        manager = CommandManager()
        from metasci_scribe.commands.implementation.general import HelpCommand

        manager.register(HelpCommand())
        assert manager.list_subcommands("/help") == []
        assert manager.get_subcommands("/help") == []

    def test_unknown_command_returns_empty(self):
        """get_subcommands for a nonexistent command must return empty list."""
        manager = CommandManager()
        assert manager.list_subcommands("/nonexistent") == []
        assert manager.get_subcommands("/nonexistent") == []

    def test_subcommand_via_alias(self):
        """Registry by alias should still expose subcommands."""
        from typing import ClassVar

        from metasci_scribe.commands.base import Command, SubCommand

        class _Aliased(Command):
            name = "/aliased"
            alias: ClassVar[list[str]] = ["/al"]
            description = "test"
            subcommands: ClassVar[list[SubCommand]] = [
                SubCommand("list", "List things"),
                SubCommand("add", "Add a thing"),
            ]

            async def execute(self, ctx, args):
                return None

        manager = CommandManager()
        manager.register(_Aliased())
        scs = manager.list_subcommands("/al")
        names = {name for name, _desc in scs}
        assert {"list", "add"} <= names
