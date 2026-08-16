"""Safe, explicit CSV dataset ingestion."""

import csv
import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike, stat_result
from pathlib import Path

import pandas as pd

from mlforge.datasets.types import ColumnMetadata, DatasetMetadata, LoadedDataset
from mlforge.errors import (
    ConfigurationError,
    DatasetFormatError,
    DatasetPathError,
    DatasetValidationError,
)

DEFAULT_MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class CsvLoadOptions:
    """Explicit parsing and resource limits for one CSV load."""

    encoding: str = "utf-8-sig"
    delimiter: str = ","
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES

    def __post_init__(self) -> None:
        """Reject invalid parser and resource settings early."""
        if not isinstance(self.encoding, str) or not self.encoding.strip():
            raise ConfigurationError("CSV encoding must not be blank.")
        if (
            not isinstance(self.delimiter, str)
            or len(self.delimiter) != 1
            or self.delimiter in {"\r", "\n", "\0"}
        ):
            raise ConfigurationError("CSV delimiter must be one non-newline character.")
        if (
            isinstance(self.max_file_size_bytes, bool)
            or not isinstance(self.max_file_size_bytes, int)
            or self.max_file_size_bytes <= 0
        ):
            raise ConfigurationError("Maximum CSV file size must be greater than zero bytes.")


def _resolve_csv_path(
    path: str | PathLike[str],
    max_file_size_bytes: int,
) -> tuple[Path, stat_result]:
    try:
        candidate = Path(path).expanduser()
    except TypeError as error:
        raise DatasetPathError("Dataset path must be a string or path-like value.") from error
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise DatasetPathError(
            f"Dataset path does not exist or cannot be resolved: {candidate}"
        ) from error

    if not resolved.is_file():
        raise DatasetPathError(f"Dataset path is not a file: {resolved}")
    if resolved.suffix.lower() != ".csv":
        raise DatasetPathError(f"Dataset must use the .csv extension: {resolved}")

    try:
        file_status = resolved.stat()
    except OSError as error:
        raise DatasetPathError(f"Could not inspect dataset file: {resolved}") from error

    file_size = file_status.st_size
    if file_size == 0:
        raise DatasetValidationError(f"Dataset file is empty: {resolved}")
    if file_size > max_file_size_bytes:
        raise DatasetPathError(
            f"Dataset file is {file_size} bytes, exceeding the configured "
            f"{max_file_size_bytes}-byte limit: {resolved}"
        )
    return resolved, file_status


def _nonblank_rows(reader: Iterable[list[str]]) -> Iterator[list[str]]:
    for row in reader:
        if row:
            yield row


def _validate_header(header: list[str]) -> tuple[str, ...]:
    if not header:
        raise DatasetValidationError("CSV header must contain at least one column.")

    blank_positions = [str(index + 1) for index, name in enumerate(header) if not name.strip()]
    if blank_positions:
        positions = ", ".join(blank_positions)
        raise DatasetValidationError(
            f"CSV header contains blank column names at positions: {positions}."
        )
    if any("\0" in name for name in header):
        raise DatasetFormatError("CSV header contains a null byte.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for name in header:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        names = ", ".join(repr(name) for name in duplicates)
        raise DatasetValidationError(f"CSV header contains duplicate column names: {names}.")
    return tuple(header)


def _validate_csv_structure(path: Path, options: CsvLoadOptions) -> tuple[tuple[str, ...], str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as binary_file:
            for chunk in iter(lambda: binary_file.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)

        with path.open("r", encoding=options.encoding, errors="strict", newline="") as text_file:
            reader = csv.reader(text_file, delimiter=options.delimiter, strict=True)
            rows = _nonblank_rows(reader)
            try:
                header = _validate_header(next(rows))
            except StopIteration as error:
                raise DatasetValidationError("CSV file contains no header row.") from error

            data_row_count = 0
            for row in rows:
                data_row_count += 1
                if any("\0" in value for value in row):
                    raise DatasetFormatError(
                        f"CSV contains a null byte in the row ending at line {reader.line_num}."
                    )
                if len(row) != len(header):
                    raise DatasetFormatError(
                        f"CSV row ending at line {reader.line_num} has {len(row)} fields; "
                        f"expected {len(header)}."
                    )
    except LookupError as error:
        raise ConfigurationError(f"Unknown CSV encoding {options.encoding!r}.") from error
    except UnicodeDecodeError as error:
        raise DatasetFormatError(
            f"Dataset is not valid {options.encoding} text near byte {error.start}: {path}"
        ) from error
    except csv.Error as error:
        raise DatasetFormatError(f"Malformed CSV near line {reader.line_num}: {error}") from error
    except OSError as error:
        raise DatasetPathError(f"Could not read dataset file: {path}") from error

    if data_row_count == 0:
        raise DatasetValidationError("CSV file contains a header but no data rows.")
    return header, digest.hexdigest()


def _load_csv_frame(
    path: str | PathLike[str],
    options: CsvLoadOptions,
) -> tuple[pd.DataFrame, Path, stat_result, str]:
    resolved_path, before = _resolve_csv_path(path, options.max_file_size_bytes)
    header, fingerprint = _validate_csv_structure(resolved_path, options)

    try:
        frame = pd.read_csv(
            resolved_path,
            sep=options.delimiter,
            encoding=options.encoding,
            encoding_errors="strict",
            on_bad_lines="error",
            low_memory=False,
        )
    except pd.errors.EmptyDataError as error:
        raise DatasetValidationError("CSV file contains no tabular data.") from error
    except pd.errors.ParserError as error:
        raise DatasetFormatError(f"Could not parse CSV dataset: {error}") from error
    except UnicodeDecodeError as error:
        raise DatasetFormatError(
            f"Dataset is not valid {options.encoding} text near byte {error.start}: {resolved_path}"
        ) from error
    except OSError as error:
        raise DatasetPathError(f"Could not read dataset file: {resolved_path}") from error

    try:
        after = resolved_path.stat()
    except OSError as error:
        raise DatasetPathError(f"Could not re-inspect dataset file: {resolved_path}") from error
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise DatasetPathError(f"Dataset file changed while it was being loaded: {resolved_path}")

    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != header:
        raise DatasetFormatError(
            "CSV parser produced columns that do not match the validated header."
        )
    if frame.empty:
        raise DatasetValidationError("CSV file contains no data rows after parsing.")
    return frame, resolved_path, before, fingerprint


def load_feature_csv(
    path: str | PathLike[str],
    *,
    options: CsvLoadOptions | None = None,
) -> pd.DataFrame:
    """Load a validated target-free CSV frame for schema-checked batch inference."""
    if options is not None and not isinstance(options, CsvLoadOptions):
        raise ConfigurationError("CSV load options must be a CsvLoadOptions value.")
    load_options = options if options is not None else CsvLoadOptions()
    frame, _, _, _ = _load_csv_frame(path, load_options)
    return frame


def load_csv(
    path: str | PathLike[str],
    *,
    target: str,
    options: CsvLoadOptions | None = None,
) -> LoadedDataset:
    """Load and validate one local CSV dataset without mutating its source."""
    if not isinstance(target, str) or not target.strip():
        raise DatasetValidationError("Target column name must not be blank.")

    if options is not None and not isinstance(options, CsvLoadOptions):
        raise ConfigurationError("CSV load options must be a CsvLoadOptions value.")
    load_options = options if options is not None else CsvLoadOptions()
    frame, resolved_path, before, fingerprint = _load_csv_frame(path, load_options)
    actual_columns = tuple(str(column) for column in frame.columns)
    if target not in frame.columns:
        available = ", ".join(repr(column) for column in actual_columns)
        raise DatasetValidationError(
            f"Target column {target!r} was not found. Available columns: {available}."
        )

    column_metadata = tuple(
        ColumnMetadata(name=str(name), pandas_dtype=str(dtype))
        for name, dtype in frame.dtypes.items()
    )
    metadata = DatasetMetadata(
        source_path=resolved_path,
        file_size_bytes=before.st_size,
        sha256=fingerprint,
        row_count=len(frame),
        column_count=len(frame.columns),
        columns=column_metadata,
        target=target,
        encoding=load_options.encoding,
        delimiter=load_options.delimiter,
    )
    return LoadedDataset(frame=frame, metadata=metadata)
