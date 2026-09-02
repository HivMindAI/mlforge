"""Validate that an MLForge source distribution contains the complete product source."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

REQUIRED_SUFFIXES = (
    "compose.private.yaml",
    "deployment/backend.Dockerfile",
    "deployment/frontend.Dockerfile",
    "frontend/package.json",
    "frontend/package-lock.json",
    "frontend/playwright.config.ts",
    "frontend/tests/e2e/golden-path.spec.ts",
    "tests/web/test_schema_version.py",
)
FORBIDDEN_PARTS = ("node_modules", ".next", "playwright-report", "test-results")


def _source_archive(location: Path) -> Path:
    if location.is_file():
        if not location.name.endswith(".tar.gz"):
            raise ValueError(f"Source archive must end in .tar.gz: {location}")
        return location

    archives = tuple(sorted(location.glob("*.tar.gz")))
    if len(archives) != 1:
        raise ValueError(
            f"Expected exactly one .tar.gz source archive in {location}, found {len(archives)}."
        )
    return archives[0]


def validate_source_archive(archive: Path) -> None:
    """Raise ``ValueError`` when the archive is incomplete or contains build output."""
    try:
        with tarfile.open(archive, mode="r:gz") as source:
            members = tuple(member.name for member in source.getmembers())
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"Could not read source archive {archive}.") from error

    missing = tuple(
        suffix
        for suffix in REQUIRED_SUFFIXES
        if not any(name == suffix or name.endswith(f"/{suffix}") for name in members)
    )
    if missing:
        raise ValueError(f"Source archive is missing required files: {', '.join(missing)}")

    forbidden = tuple(
        name for name in members if any(part in FORBIDDEN_PARTS for part in Path(name).parts)
    )
    if forbidden:
        raise ValueError(f"Source archive contains generated files: {forbidden[0]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "distribution",
        nargs="?",
        type=Path,
        default=Path("dist"),
        help="A .tar.gz archive or a directory containing exactly one (default: dist).",
    )
    arguments = parser.parse_args()
    archive = _source_archive(arguments.distribution)
    try:
        validate_source_archive(archive)
    except ValueError as error:
        parser.error(str(error))
    print(f"Validated complete source distribution: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
