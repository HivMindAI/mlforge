"""Integrity checks for loaded datasets crossing module boundaries."""

from __future__ import annotations

from mlforge.datasets.types import LoadedDataset
from mlforge.errors import DatasetValidationError


def validate_loaded_dataset(dataset: LoadedDataset, *, operation: str) -> None:
    """Ensure a loaded dataframe still matches the metadata produced at ingestion."""
    frame = dataset.frame
    metadata = dataset.metadata
    current_columns = tuple(str(column) for column in frame.columns)
    expected_columns = tuple(column.name for column in metadata.columns)
    current_dtypes = tuple(str(dtype) for dtype in frame.dtypes)
    expected_dtypes = tuple(column.pandas_dtype for column in metadata.columns)

    if (
        frame.shape != (metadata.row_count, metadata.column_count)
        or current_columns != expected_columns
        or current_dtypes != expected_dtypes
        or not frame.columns.is_unique
        or current_columns.count(metadata.target) != 1
    ):
        raise DatasetValidationError(
            "Loaded dataset dataframe no longer matches its validated metadata. Reload the CSV "
            f"before {operation}."
        )
