"""Tests for the MLForge package metadata."""

import subprocess
import sys
from importlib import resources
from importlib.metadata import entry_points, files, metadata, requires, version
from pathlib import Path

from mlforge import __version__

DISTRIBUTION_NAME = "hivmind-mlforge"


def test_version_matches_installed_distribution() -> None:
    """Runtime and distribution metadata should use the same version."""
    assert __version__ == version(DISTRIBUTION_NAME)


def test_console_script_is_registered() -> None:
    """The installed distribution should expose the documented CLI."""
    scripts = {
        entry_point.name: entry_point.value for entry_point in entry_points(group="console_scripts")
    }

    assert scripts["mlforge"] == "mlforge.cli:main"


def test_runtime_dependency_metadata() -> None:
    """The distribution should declare its implemented tabular and pipeline boundaries."""
    requirements = requires(DISTRIBUTION_NAME) or []

    assert any(requirement.startswith("pandas<4,>=3.0") for requirement in requirements)
    assert any(requirement.startswith("scikit-learn<2,>=1.9") for requirement in requirements)


def test_distribution_declares_and_contains_apache_license() -> None:
    """Installed metadata and files should carry the owner-selected license."""
    package_metadata = metadata(DISTRIBUTION_NAME)
    distribution_files = files(DISTRIBUTION_NAME) or ()

    assert package_metadata["Name"] == DISTRIBUTION_NAME
    assert package_metadata["License-Expression"] == "Apache-2.0"
    assert any(
        str(path).replace("\\", "/").endswith(".dist-info/licenses/LICENSE")
        for path in distribution_files
    )


def test_distribution_is_marked_as_typed() -> None:
    """Type checkers should be able to consume MLForge's inline annotations."""
    marker = resources.files("mlforge").joinpath("py.typed")

    assert marker.is_file()


def test_import_has_no_application_side_effects(tmp_path: Path) -> None:
    """Importing foundation modules should not configure logging or write files."""
    code = """
import logging

logger = logging.getLogger("mlforge")
before = (tuple(logger.handlers), logger.level, logger.propagate)

import mlforge
import mlforge.artifacts
import mlforge.config
import mlforge.errors
import mlforge.inference
import mlforge.logging_config

after = (tuple(logger.handlers), logger.level, logger.propagate)
assert after == before
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_release_tag_validation_matches_runtime_version() -> None:
    """The publishing workflow must reject a tag that differs from package metadata."""
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "check_release_tag.py"

    accepted = subprocess.run(
        [sys.executable, script, f"v{__version__}"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    rejected = subprocess.run(
        [sys.executable, script, "v999.0.0"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert accepted.returncode == 0, accepted.stderr
    assert rejected.returncode == 1
    assert "does not match package version" in rejected.stderr
