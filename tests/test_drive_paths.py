"""A mapped network drive keeps its letter when a path is resolved."""

from __future__ import annotations

import os

import pytest

from metasci_scribe import drive_paths


@pytest.fixture
def mapped(monkeypatch):
    monkeypatch.setattr(
        drive_paths,
        "_roots",
        [(r"\\RaiDrive-KEI\Synology", "Z:"), (r"\\host\share", "Y:")],
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"\\RaiDrive-KEI\Synology\KKH\paper", r"Z:\KKH\paper"),
        (r"\\raidrive-kei\synology\KKH", r"Z:\KKH"),
        (r"\\RaiDrive-KEI\Synology", "Z:\\"),
        ("\\\\?\\UNC\\host\\share\\paper", r"Y:\paper"),
        (r"\\host\shared\paper", r"\\host\shared\paper"),  # another share
        (r"\\other\share\paper", r"\\other\share\paper"),
        (r"C:\dev\paper", r"C:\dev\paper"),
        ("\\\\?\\C:\\dev", "\\\\?\\C:\\dev"),
    ],
)
def test_a_unc_name_under_a_mapped_drive_gets_the_letter(mapped, text, expected):
    assert drive_paths.drive_letter_path(text) == expected


def test_the_wrapper_maps_what_realpath_returns(mapped, monkeypatch):
    monkeypatch.setattr(
        drive_paths,
        "_original_realpath",
        lambda path, **kwargs: r"\\RaiDrive-KEI\Synology\KKH\paper",
    )
    assert drive_paths._realpath_with_drive_letters(r"Z:\KKH\paper") == r"Z:\KKH\paper"


@pytest.mark.skipif(os.name != "nt", reason="Windows only")
def test_the_package_installs_the_wrapper():
    import metasci_scribe  # noqa: F401

    assert os.path.realpath is drive_paths._realpath_with_drive_letters
    assert os.path.realpath(r"C:\Windows") == r"C:\Windows"
