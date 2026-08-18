"""Fail a release build when its tag does not match the package version."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from mlforge import __version__  # noqa: E402


def main(tag: str) -> int:
    """Validate one exact ``v<package-version>`` release tag."""
    expected = f"v{__version__}"
    if tag != expected:
        print(f"release tag {tag!r} does not match package version {expected!r}", file=sys.stderr)
        return 1
    print(f"release tag {tag!r} matches package version {__version__}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_release_tag.py TAG", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
