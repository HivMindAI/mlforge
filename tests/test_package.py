"""Tests for the MLForge package metadata."""

from mlforge import __version__


def test_version() -> None:
    """The package should expose its current version."""
    assert __version__ == "0.1.0"
