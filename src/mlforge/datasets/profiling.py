"""Deterministic, JSON-safe dataset quality profiling."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd

from mlforge.datasets.types import DatasetMetadata, JsonObject, LoadedDataset
from mlforge.datasets.validation import validate_loaded_dataset

_HIGH_CARDINALITY_MINIMUM = 50
_HIGH_CARDINALITY_RATIO = 0.5
_MAX_CLASS_DISTRIBUTION_VALUES = 20
_IMBALANCE_RATIO_THRESHOLD = 0.2


class ColumnKind(StrEnum):
    """Physical value kind inferred by pandas during CSV loading."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    STRING = "string"
    OTHER = "other"


class TaskHint(StrEnum):
    """Conservative task suggestion inferred from the target values."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Finite-value summary for a numeric column."""

    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float | None

    def to_dict(self) -> JsonObject:
        """Return a JSON-safe numeric summary."""
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "standard_deviation": self.standard_deviation,
        }


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Quality and cardinality information for one dataset column."""

    name: str
    kind: ColumnKind
    pandas_dtype: str
    non_missing_count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    infinite_count: int
    is_constant: bool
    is_high_cardinality: bool
    is_likely_identifier: bool
    numeric_summary: NumericSummary | None

    def to_dict(self) -> JsonObject:
        """Return a JSON-safe column profile."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "pandas_dtype": self.pandas_dtype,
            "non_missing_count": self.non_missing_count,
            "missing_count": self.missing_count,
            "missing_ratio": self.missing_ratio,
            "unique_count": self.unique_count,
            "unique_ratio": self.unique_ratio,
            "infinite_count": self.infinite_count,
            "is_constant": self.is_constant,
            "is_high_cardinality": self.is_high_cardinality,
            "is_likely_identifier": self.is_likely_identifier,
            "numeric_summary": (
                self.numeric_summary.to_dict() if self.numeric_summary is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ValueFrequency:
    """One display-safe class value and its frequency."""

    value: str
    count: int
    ratio: float

    def to_dict(self) -> JsonObject:
        """Return a JSON-safe class frequency."""
        return {"value": self.value, "count": self.count, "ratio": self.ratio}


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """Task and balance hints for the configured target column."""

    name: str
    task_hint: TaskHint
    non_missing_count: int
    missing_count: int
    unique_count: int
    class_distribution: tuple[ValueFrequency, ...]
    distribution_truncated: bool
    imbalance_warning: bool

    def to_dict(self) -> JsonObject:
        """Return a JSON-safe target profile."""
        return {
            "name": self.name,
            "task_hint": self.task_hint.value,
            "non_missing_count": self.non_missing_count,
            "missing_count": self.missing_count,
            "unique_count": self.unique_count,
            "class_distribution": [item.to_dict() for item in self.class_distribution],
            "distribution_truncated": self.distribution_truncated,
            "imbalance_warning": self.imbalance_warning,
        }


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Serializable profile of one validated loaded dataset."""

    metadata: DatasetMetadata
    missing_cell_count: int
    missing_cell_ratio: float
    duplicate_row_count: int
    columns: tuple[ColumnProfile, ...]
    target: TargetProfile
    warnings: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        """Return the complete profile as JSON-safe primitives."""
        return {
            "metadata": self.metadata.to_dict(),
            "missing_cell_count": self.missing_cell_count,
            "missing_cell_ratio": self.missing_cell_ratio,
            "duplicate_row_count": self.duplicate_row_count,
            "columns": [column.to_dict() for column in self.columns],
            "target": self.target.to_dict(),
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the profile as standards-compliant deterministic JSON."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )


def _column_kind(series: pd.Series[Any]) -> ColumnKind:
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return ColumnKind.BOOLEAN
    if pd.api.types.is_integer_dtype(dtype):
        return ColumnKind.INTEGER
    if pd.api.types.is_float_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
        return ColumnKind.FLOAT
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return ColumnKind.DATETIME
    if isinstance(dtype, pd.CategoricalDtype):
        return ColumnKind.CATEGORICAL
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return ColumnKind.STRING
    return ColumnKind.OTHER


def _numeric_summary(series: pd.Series[Any]) -> tuple[NumericSummary | None, int]:
    values: list[float] = []
    infinite_count = 0
    for raw_value in series.dropna().tolist():
        value = float(raw_value)
        if math.isfinite(value):
            values.append(value)
        else:
            infinite_count += 1

    if not values:
        return None, infinite_count
    standard_deviation = statistics.stdev(values) if len(values) > 1 else None
    return (
        NumericSummary(
            minimum=min(values),
            maximum=max(values),
            mean=statistics.fmean(values),
            median=statistics.median(values),
            standard_deviation=standard_deviation,
        ),
        infinite_count,
    )


def _identifier_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return normalized in {"id", "uuid", "guid", "key", "identifier"} or normalized.endswith(
        ("_id", "_uuid", "_guid", "_key")
    )


def _profile_column(name: str, series: pd.Series[Any], row_count: int) -> ColumnProfile:
    kind = _column_kind(series)
    missing_count = int(series.isna().sum())
    non_missing_count = row_count - missing_count
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = unique_count / non_missing_count if non_missing_count else 0.0
    summary: NumericSummary | None = None
    infinite_count = 0
    if kind in {ColumnKind.INTEGER, ColumnKind.FLOAT}:
        summary, infinite_count = _numeric_summary(series)

    is_high_cardinality = (
        kind in {ColumnKind.STRING, ColumnKind.CATEGORICAL}
        and unique_count >= _HIGH_CARDINALITY_MINIMUM
        and unique_ratio >= _HIGH_CARDINALITY_RATIO
    )
    is_likely_identifier = (
        row_count > 0
        and non_missing_count == row_count
        and unique_count == row_count
        and _identifier_name(name)
    )
    return ColumnProfile(
        name=name,
        kind=kind,
        pandas_dtype=str(series.dtype),
        non_missing_count=non_missing_count,
        missing_count=missing_count,
        missing_ratio=missing_count / row_count,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        infinite_count=infinite_count,
        is_constant=unique_count <= 1,
        is_high_cardinality=is_high_cardinality,
        is_likely_identifier=is_likely_identifier,
        numeric_summary=summary,
    )


def _infer_task_hint(kind: ColumnKind, unique_count: int, non_missing_count: int) -> TaskHint:
    if unique_count < 2 or non_missing_count == 0:
        return TaskHint.UNDETERMINED
    if kind in {ColumnKind.BOOLEAN, ColumnKind.STRING, ColumnKind.CATEGORICAL}:
        return TaskHint.CLASSIFICATION
    if kind in {ColumnKind.INTEGER, ColumnKind.FLOAT}:
        classification_limit = min(20, max(2, math.isqrt(non_missing_count)))
        if unique_count <= classification_limit:
            return TaskHint.CLASSIFICATION
        return TaskHint.REGRESSION
    return TaskHint.UNDETERMINED


def _target_profile(
    series: pd.Series[Any],
    column_profile: ColumnProfile,
) -> TargetProfile:
    task_hint = _infer_task_hint(
        column_profile.kind,
        column_profile.unique_count,
        column_profile.non_missing_count,
    )
    frequencies: tuple[ValueFrequency, ...] = ()
    imbalance_warning = False

    if task_hint is TaskHint.CLASSIFICATION:
        counts = sorted(
            ((str(value), int(count)) for value, count in series.value_counts(dropna=True).items()),
            key=lambda item: (-item[1], item[0]),
        )
        frequencies = tuple(
            ValueFrequency(
                value=value,
                count=count,
                ratio=count / column_profile.non_missing_count,
            )
            for value, count in counts[:_MAX_CLASS_DISTRIBUTION_VALUES]
        )
        if len(counts) >= 2:
            largest_count = counts[0][1]
            smallest_count = counts[-1][1]
            imbalance_warning = smallest_count / largest_count < _IMBALANCE_RATIO_THRESHOLD

    return TargetProfile(
        name=column_profile.name,
        task_hint=task_hint,
        non_missing_count=column_profile.non_missing_count,
        missing_count=column_profile.missing_count,
        unique_count=column_profile.unique_count,
        class_distribution=frequencies,
        distribution_truncated=(
            task_hint is TaskHint.CLASSIFICATION and column_profile.unique_count > len(frequencies)
        ),
        imbalance_warning=imbalance_warning,
    )


def _profile_warnings(
    columns: tuple[ColumnProfile, ...],
    target: TargetProfile,
    missing_cell_count: int,
    duplicate_row_count: int,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if missing_cell_count:
        warnings.append(f"Dataset contains {missing_cell_count} missing cells.")
    if duplicate_row_count:
        warnings.append(f"Dataset contains {duplicate_row_count} duplicate rows.")

    constant_columns = [column.name for column in columns if column.is_constant]
    if constant_columns:
        warnings.append(f"Constant columns: {', '.join(constant_columns)}.")
    high_cardinality_columns = [column.name for column in columns if column.is_high_cardinality]
    if high_cardinality_columns:
        warnings.append(f"High-cardinality columns: {', '.join(high_cardinality_columns)}.")
    identifier_columns = [column.name for column in columns if column.is_likely_identifier]
    if identifier_columns:
        warnings.append(f"Likely identifier columns: {', '.join(identifier_columns)}.")
    non_finite_columns = [column.name for column in columns if column.infinite_count]
    if non_finite_columns:
        warnings.append(f"Columns containing infinite values: {', '.join(non_finite_columns)}.")
    if target.missing_count:
        warnings.append(f"Target column {target.name!r} contains missing values.")
    if target.imbalance_warning:
        warnings.append(f"Target column {target.name!r} appears imbalanced.")
    if target.task_hint is TaskHint.UNDETERMINED:
        warnings.append(f"Task type could not be inferred from target column {target.name!r}.")
    return tuple(warnings)


def profile_dataset(dataset: LoadedDataset) -> DatasetProfile:
    """Profile a validated loaded dataset without mutating its dataframe."""
    validate_loaded_dataset(dataset, operation="profiling")
    frame = dataset.frame
    metadata = dataset.metadata
    current_columns = tuple(str(column) for column in frame.columns)

    columns = tuple(
        _profile_column(name, frame[name], metadata.row_count) for name in current_columns
    )
    profiles_by_name = {column.name: column for column in columns}
    target = _target_profile(frame[metadata.target], profiles_by_name[metadata.target])
    missing_cell_count = int(frame.isna().sum().sum())
    cell_count = metadata.row_count * metadata.column_count
    duplicate_row_count = int(frame.duplicated(keep="first").sum())
    warnings = _profile_warnings(
        columns,
        target,
        missing_cell_count,
        duplicate_row_count,
    )
    return DatasetProfile(
        metadata=metadata,
        missing_cell_count=missing_cell_count,
        missing_cell_ratio=missing_cell_count / cell_count,
        duplicate_row_count=duplicate_row_count,
        columns=columns,
        target=target,
        warnings=warnings,
    )
