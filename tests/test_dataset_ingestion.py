"""Tests for safe CSV dataset ingestion."""

import hashlib
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from mlforge.datasets import CsvLoadOptions, load_csv
from mlforge.errors import (
    ConfigurationError,
    DatasetFormatError,
    DatasetPathError,
    DatasetValidationError,
)


def _write_csv(path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    path.write_text(content, encoding=encoding, newline="")
    return path


def test_load_csv_returns_data_and_stable_metadata(tmp_path: Path) -> None:
    """A valid CSV should retain values and record its source identity."""
    content = "feature,label\n1,yes\n2,no\n"
    path = _write_csv(tmp_path / "training.csv", content)

    dataset = load_csv(path, target="label")

    assert dataset.frame.to_dict(orient="list") == {
        "feature": [1, 2],
        "label": ["yes", "no"],
    }
    assert dataset.metadata.source_path == path.resolve()
    assert dataset.metadata.file_size_bytes == len(content.encode())
    assert dataset.metadata.sha256 == hashlib.sha256(content.encode()).hexdigest()
    assert dataset.metadata.row_count == 2
    assert dataset.metadata.column_count == 2
    assert tuple(column.name for column in dataset.metadata.columns) == ("feature", "label")
    assert dataset.metadata.target == "label"
    assert dataset.metadata.to_dict()["source_path"] == str(path.resolve())


def test_custom_delimiter_and_encoding(tmp_path: Path) -> None:
    """Explicit parser options should support non-default CSV representations."""
    path = _write_csv(tmp_path / "latin.csv", "city;target\nCafé;1\n", encoding="latin-1")

    dataset = load_csv(
        path,
        target="target",
        options=CsvLoadOptions(encoding="latin-1", delimiter=";"),
    )

    assert dataset.frame.loc[0, "city"] == "Café"
    assert dataset.metadata.encoding == "latin-1"
    assert dataset.metadata.delimiter == ";"


def test_default_encoding_accepts_a_utf8_byte_order_mark(tmp_path: Path) -> None:
    """The UTF-8 default should not leak a BOM into the first column name."""
    path = tmp_path / "bom.csv"
    path.write_bytes("feature,target\n1,yes\n".encode("utf-8-sig"))

    dataset = load_csv(path, target="target")

    assert tuple(dataset.frame.columns) == ("feature", "target")


def test_csv_load_options_type_is_constructible() -> None:
    """The default options should remain a usable public API."""
    assert CsvLoadOptions().encoding == "utf-8-sig"


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: CsvLoadOptions(encoding=" "), "encoding must not be blank"),
        (lambda: CsvLoadOptions(encoding=1), "encoding must not be blank"),  # type: ignore[arg-type]
        (lambda: CsvLoadOptions(delimiter="::"), "delimiter must be one"),
        (lambda: CsvLoadOptions(delimiter="\n"), "delimiter must be one"),
        (lambda: CsvLoadOptions(delimiter=1), "delimiter must be one"),  # type: ignore[arg-type]
        (lambda: CsvLoadOptions(max_file_size_bytes=0), "must be greater than zero"),
        (lambda: CsvLoadOptions(max_file_size_bytes=True), "must be greater than zero"),
        (
            lambda: CsvLoadOptions(max_file_size_bytes=1.5),  # type: ignore[arg-type]
            "must be greater than zero",
        ),
    ],
)
def test_invalid_load_options(factory: Callable[[], CsvLoadOptions], message: str) -> None:
    """Invalid resource/parser options should fail before filesystem access."""
    with pytest.raises(ConfigurationError, match=message):
        factory()


def test_missing_path(tmp_path: Path) -> None:
    """Missing sources should raise a path-domain error."""
    with pytest.raises(DatasetPathError, match="does not exist"):
        load_csv(tmp_path / "missing.csv", target="target")


def test_directory_path(tmp_path: Path) -> None:
    """Directories should never be passed to the CSV parser."""
    directory = tmp_path / "dataset.csv"
    directory.mkdir()

    with pytest.raises(DatasetPathError, match="not a file"):
        load_csv(directory, target="target")


def test_non_csv_extension(tmp_path: Path) -> None:
    """The first reader should have an explicit supported file boundary."""
    path = _write_csv(tmp_path / "dataset.txt", "value,target\n1,0\n")

    with pytest.raises(DatasetPathError, match=r"\.csv extension"):
        load_csv(path, target="target")


def test_empty_file(tmp_path: Path) -> None:
    """A zero-byte file should produce an actionable validation error."""
    path = _write_csv(tmp_path / "empty.csv", "")

    with pytest.raises(DatasetValidationError, match="file is empty"):
        load_csv(path, target="target")


def test_file_size_limit(tmp_path: Path) -> None:
    """Resource limits should be checked before parsing."""
    path = _write_csv(tmp_path / "large.csv", "value,target\n1,0\n")

    with pytest.raises(DatasetPathError, match="exceeding the configured"):
        load_csv(
            path,
            target="target",
            options=CsvLoadOptions(max_file_size_bytes=5),
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("value,target\n", "header but no data"),
        (",target\n1,0\n", "blank column names"),
        ("value,value\n1,0\n", "duplicate column names"),
        ("value,tar\0get\n1,0\n", "header contains a null byte"),
        ("value,target\n1\n", "has 1 fields; expected 2"),
        ("value,target\n1,0,extra\n", "has 3 fields; expected 2"),
        ('value,target\n"unterminated,0\n', "Malformed CSV"),
    ],
)
def test_invalid_csv_structure(tmp_path: Path, content: str, message: str) -> None:
    """Structural corruption should fail rather than be skipped or filled silently."""
    path = _write_csv(tmp_path / "invalid.csv", content)

    with pytest.raises((DatasetFormatError, DatasetValidationError), match=message):
        load_csv(path, target="target")


def test_invalid_text_encoding(tmp_path: Path) -> None:
    """Undecodable bytes should identify the configured text boundary."""
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"feature,target\n\xff,1\n")

    with pytest.raises(DatasetFormatError, match="not valid utf-8-sig text"):
        load_csv(path, target="target")


def test_unknown_encoding(tmp_path: Path) -> None:
    """Unknown codecs should be configuration errors, not parser tracebacks."""
    path = _write_csv(tmp_path / "valid.csv", "value,target\n1,0\n")

    with pytest.raises(ConfigurationError, match="Unknown CSV encoding"):
        load_csv(path, target="target", options=CsvLoadOptions(encoding="not-a-codec"))


@pytest.mark.parametrize("target", ["", "   ", "missing"])
def test_invalid_target(tmp_path: Path, target: str) -> None:
    """The configured target must be nonblank and present exactly."""
    path = _write_csv(tmp_path / "valid.csv", "value,target\n1,0\n")

    with pytest.raises(DatasetValidationError, match="Target column"):
        load_csv(path, target=target)


def test_invalid_runtime_argument_types_raise_domain_errors(tmp_path: Path) -> None:
    """Public ingestion boundaries should not leak incidental attribute or type errors."""
    path = _write_csv(tmp_path / "valid.csv", "value,target\n1,0\n")

    with pytest.raises(DatasetValidationError, match="Target column"):
        load_csv(path, target=1)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="CsvLoadOptions"):
        load_csv(path, target="target", options=object())  # type: ignore[arg-type]
    with pytest.raises(DatasetPathError, match="path-like"):
        load_csv(None, target="target")  # type: ignore[arg-type]


def test_missing_values_are_retained_for_profiling(tmp_path: Path) -> None:
    """Ingestion should preserve missing cells instead of silently imputing them."""
    path = _write_csv(tmp_path / "missing.csv", "value,target\n,yes\n2,\n")

    dataset = load_csv(path, target="target")

    assert bool(pd.isna(dataset.frame.loc[0, "value"]))
    assert bool(pd.isna(dataset.frame.loc[1, "target"]))
