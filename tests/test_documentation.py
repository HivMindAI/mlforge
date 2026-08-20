"""Repository documentation integrity tests."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "ROADMAP.md",
    REPOSITORY_ROOT / "CONTRIBUTING.md",
    REPOSITORY_ROOT / "SECURITY.md",
    *(REPOSITORY_ROOT / "docs").glob("*.md"),
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
PYTHON_BLOCK = re.compile(r"```python[^\n]*\n(.*?)```", re.DOTALL)


def test_required_documentation_exists() -> None:
    """Release-facing documentation should be present at predictable paths."""
    expected = {
        "README.md",
        "ROADMAP.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/api.md",
        "docs/architecture.md",
        "docs/compatibility.md",
        "docs/release-validation.md",
        "docs/releasing.md",
        "docs/security.md",
        "docs/tutorial.md",
    }
    actual = {
        str(path.relative_to(REPOSITORY_ROOT)).replace("\\", "/")
        for path in DOCUMENTS
        if path.is_file()
    }

    assert expected <= actual


def test_local_documentation_links_resolve_inside_repository() -> None:
    """Relative Markdown links should not drift or escape the repository."""
    for document in DOCUMENTS:
        content = document.read_text(encoding="utf-8")
        local_targets = (*MARKDOWN_LINK.findall(content), *MARKDOWN_IMAGE.findall(content))
        for raw_target in local_targets:
            target = raw_target.strip().strip("<>").split("#", maxsplit=1)[0]
            if not target or urlparse(target).scheme:
                continue
            resolved = (document.parent / unquote(target)).resolve()

            assert resolved.is_relative_to(REPOSITORY_ROOT), (document, raw_target)
            assert resolved.exists(), (document, raw_target)


def test_documented_python_blocks_are_valid_syntax() -> None:
    """Python examples in Markdown should at least remain parseable."""
    for document in DOCUMENTS:
        content = document.read_text(encoding="utf-8")
        for index, block in enumerate(PYTHON_BLOCK.findall(content), start=1):
            compile(
                textwrap.dedent(block),
                f"{document.relative_to(REPOSITORY_ROOT)}:python-block-{index}",
                "exec",
            )


def test_terminal_demo_is_a_valid_accessible_svg() -> None:
    """The repository-owned benchmark visual should remain valid and self-describing."""
    asset = REPOSITORY_ROOT / "docs" / "assets" / "benchmark-terminal.svg"
    root = ElementTree.parse(asset).getroot()
    namespace = "{http://www.w3.org/2000/svg}"

    assert root.tag == f"{namespace}svg"
    assert root.attrib["role"] == "img"
    assert root.find(f"{namespace}title") is not None
    assert root.find(f"{namespace}desc") is not None
