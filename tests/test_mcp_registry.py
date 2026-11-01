"""Tests for metasci_scribe.mcp.registry."""

import pytest

from metasci_scribe.git_cli import GitNotFoundError
from metasci_scribe.mcp.registry import _MARKETPLACE_CACHE, fetch_marketplace_index


class TestFetchMarketplaceIndex:
    def test_missing_git_raises_git_not_found(self, no_git):
        _MARKETPLACE_CACHE.clear()
        with pytest.raises(GitNotFoundError):
            fetch_marketplace_index(repo="test/repo")
