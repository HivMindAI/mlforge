"""Typed dataset results shared by ingestion and profiling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ColumnMetadata:
    """Stable metadata for one loaded dataframe column."""

    name: str
    pandas_dtype: str

    def to_dict(self) -> JsonObject:
        """Return JSON-serializable column metadata."""
        return {"name": self.name, "pandas_dtype": self.pandas_dtype}


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """Stable identity and shape metadata for a loaded CSV dataset."""

    source_path: Path
    file_size_bytes: int
    sha256: str
    row_count: int
    column_count: int
    columns: tuple[ColumnMetadata, ...]
    target: str
    encoding: str
    delimiter: str

    def to_dict(self) -> JsonObject:
        """Return JSON-serializable dataset metadata."""
        return {
            "source_path": str(self.source_path),
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [column.to_dict() for column in self.columns],
            "target": self.target,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
        }


@dataclass(slots=True)
class LoadedDataset:
    """A validated dataframe and the metadata describing its source."""

    frame: pd.DataFrame
    metadata: DatasetMetadata
