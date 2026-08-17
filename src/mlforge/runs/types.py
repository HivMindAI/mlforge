"""Versioned, immutable data structures for local training run records."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, TypeAlias, cast
from uuid import UUID

from mlforge.errors import RunStoreError

RUN_MANIFEST_SCHEMA_VERSION = 1
JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, Any]


class RunStatus(StrEnum):
    """Terminal state of a recorded training attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RunStoreError(f"Run manifest {label} must be a JSON object.")
    return cast(dict[str, object], value)


def _keys(value: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if extra:
            details.append(f"unexpected {extra!r}")
        raise RunStoreError(f"Run manifest {label} has invalid fields: {', '.join(details)}.")


def _string(value: object, label: str, *, allow_blank: bool = False) -> str:
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        raise RunStoreError(f"Run manifest {label} must be a non-blank string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunStoreError(f"Run manifest {label} must be an integer.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunStoreError(f"Run manifest {label} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise RunStoreError(f"Run manifest {label} must be finite.")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise RunStoreError(f"Run manifest {label} must be true or false.")
    return value


def _optional_boolean(value: object, label: str) -> bool | None:
    if value is None:
        return None
    return _boolean(value, label)


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise RunStoreError(f"Run manifest {label} must be a JSON array.")
    return cast(list[object], value)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{label} entry") for item in _array(value, label))


def _primitive(value: object, label: str) -> JsonPrimitive:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise RunStoreError(f"Run manifest {label} must be a finite JSON primitive.")


@dataclass(frozen=True, slots=True)
class RunParameter:
    """One explicitly recorded estimator parameter."""

    name: str
    value: JsonPrimitive

    def __post_init__(self) -> None:
        _string(self.name, "estimator parameter name")
        _primitive(self.value, f"estimator parameter {self.name!r}")

    def to_dict(self) -> JsonObject:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_object(cls, value: object) -> RunParameter:
        data = _object(value, "estimator parameter")
        _keys(data, {"name", "value"}, "estimator parameter")
        name = _string(data["name"], "estimator parameter name")
        return cls(name=name, value=_primitive(data["value"], f"parameter {name!r}"))


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    """Effective training configuration captured independently of changing defaults."""

    task: str
    estimator: str
    validation_fraction: float
    random_seed: int
    stratify_requested: bool | None
    numeric_imputation: str
    scale_numeric: bool
    categorical_fill_value: str
    numeric_overrides: tuple[str, ...]
    categorical_overrides: tuple[str, ...]
    estimator_parameters: tuple[RunParameter, ...]

    def __post_init__(self) -> None:
        _string(self.task, "configuration task")
        if self.task not in {"classification", "regression"}:
            raise RunStoreError(f"Recorded training task is unsupported: {self.task!r}.")
        _string(self.estimator, "configuration estimator")
        _number(self.validation_fraction, "configuration validation_fraction")
        if not 0 < self.validation_fraction < 1:
            raise RunStoreError("Recorded validation fraction must be between 0 and 1.")
        _integer(self.random_seed, "configuration random_seed")
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise RunStoreError("Recorded random seed is outside the supported range.")
        _optional_boolean(self.stratify_requested, "configuration stratify_requested")
        _string(self.numeric_imputation, "configuration numeric_imputation")
        if self.numeric_imputation not in {"mean", "median"}:
            raise RunStoreError(
                f"Recorded numeric imputation is unsupported: {self.numeric_imputation!r}."
            )
        _boolean(self.scale_numeric, "configuration scale_numeric")
        _string(self.categorical_fill_value, "configuration categorical_fill_value")
        if not isinstance(self.numeric_overrides, tuple) or not isinstance(
            self.categorical_overrides, tuple
        ):
            raise RunStoreError("Recorded feature overrides must be immutable tuples.")
        for role, names in (
            ("numeric", self.numeric_overrides),
            ("categorical", self.categorical_overrides),
        ):
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise RunStoreError(f"Recorded {role} overrides contain an invalid name.")
            if len(set(names)) != len(names):
                raise RunStoreError(f"Recorded {role} overrides must be unique.")
        if set(self.numeric_overrides).intersection(self.categorical_overrides):
            raise RunStoreError("Recorded numeric and categorical overrides must not overlap.")
        if not isinstance(self.estimator_parameters, tuple) or any(
            not isinstance(item, RunParameter) for item in self.estimator_parameters
        ):
            raise RunStoreError("Recorded estimator parameters must be RunParameter tuples.")
        parameter_names = tuple(item.name for item in self.estimator_parameters)
        if parameter_names != tuple(sorted(parameter_names)) or len(set(parameter_names)) != len(
            parameter_names
        ):
            raise RunStoreError("Recorded estimator parameters must have unique sorted names.")

    def to_dict(self) -> JsonObject:
        return {
            "task": self.task,
            "estimator": self.estimator,
            "validation_fraction": self.validation_fraction,
            "random_seed": self.random_seed,
            "stratify_requested": self.stratify_requested,
            "numeric_imputation": self.numeric_imputation,
            "scale_numeric": self.scale_numeric,
            "categorical_fill_value": self.categorical_fill_value,
            "numeric_overrides": list(self.numeric_overrides),
            "categorical_overrides": list(self.categorical_overrides),
            "estimator_parameters": [item.to_dict() for item in self.estimator_parameters],
        }

    @classmethod
    def from_object(cls, value: object) -> RunConfiguration:
        data = _object(value, "configuration")
        expected = {
            "task",
            "estimator",
            "validation_fraction",
            "random_seed",
            "stratify_requested",
            "numeric_imputation",
            "scale_numeric",
            "categorical_fill_value",
            "numeric_overrides",
            "categorical_overrides",
            "estimator_parameters",
        }
        _keys(data, expected, "configuration")
        parameters = tuple(
            RunParameter.from_object(item)
            for item in _array(data["estimator_parameters"], "estimator_parameters")
        )
        return cls(
            task=_string(data["task"], "configuration task"),
            estimator=_string(data["estimator"], "configuration estimator"),
            validation_fraction=_number(
                data["validation_fraction"], "configuration validation_fraction"
            ),
            random_seed=_integer(data["random_seed"], "configuration random_seed"),
            stratify_requested=_optional_boolean(
                data["stratify_requested"], "configuration stratify_requested"
            ),
            numeric_imputation=_string(
                data["numeric_imputation"], "configuration numeric_imputation"
            ),
            scale_numeric=_boolean(data["scale_numeric"], "configuration scale_numeric"),
            categorical_fill_value=_string(
                data["categorical_fill_value"], "configuration categorical_fill_value"
            ),
            numeric_overrides=_string_tuple(
                data["numeric_overrides"], "configuration numeric_overrides"
            ),
            categorical_overrides=_string_tuple(
                data["categorical_overrides"], "configuration categorical_overrides"
            ),
            estimator_parameters=parameters,
        )


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    """Stable source identity referenced by a run."""

    source_path: str
    sha256: str
    file_size_bytes: int
    row_count: int
    column_count: int
    target: str
    encoding: str
    delimiter: str

    def __post_init__(self) -> None:
        _string(self.source_path, "dataset source_path")
        _string(self.sha256, "dataset sha256")
        _string(self.target, "dataset target")
        _string(self.encoding, "dataset encoding")
        if (
            not isinstance(self.delimiter, str)
            or len(self.delimiter) != 1
            or self.delimiter in {"\r", "\n", "\0"}
        ):
            raise RunStoreError("Recorded dataset delimiter must be one non-newline character.")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise RunStoreError("Recorded dataset SHA-256 is invalid.")
        _integer(self.file_size_bytes, "dataset file_size_bytes")
        _integer(self.row_count, "dataset row_count")
        _integer(self.column_count, "dataset column_count")
        if self.file_size_bytes <= 0 or self.row_count <= 0 or self.column_count <= 0:
            raise RunStoreError("Recorded dataset dimensions and file size must be positive.")

    def to_dict(self) -> JsonObject:
        return {
            "source_path": self.source_path,
            "sha256": self.sha256,
            "file_size_bytes": self.file_size_bytes,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "target": self.target,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
        }

    @classmethod
    def from_object(cls, value: object) -> DatasetSnapshot:
        data = _object(value, "dataset")
        expected = {
            "source_path",
            "sha256",
            "file_size_bytes",
            "row_count",
            "column_count",
            "target",
            "encoding",
            "delimiter",
        }
        _keys(data, expected, "dataset")
        return cls(
            source_path=_string(data["source_path"], "dataset source_path"),
            sha256=_string(data["sha256"], "dataset sha256"),
            file_size_bytes=_integer(data["file_size_bytes"], "dataset file_size_bytes"),
            row_count=_integer(data["row_count"], "dataset row_count"),
            column_count=_integer(data["column_count"], "dataset column_count"),
            target=_string(data["target"], "dataset target"),
            encoding=_string(data["encoding"], "dataset encoding"),
            delimiter=_string(data["delimiter"], "dataset delimiter"),
        )


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    """Runtime versions needed to interpret or reproduce a run."""

    python: str
    mlforge: str
    pandas: str
    numpy: str
    scipy: str
    scikit_learn: str

    def __post_init__(self) -> None:
        for name, value in (
            ("python", self.python),
            ("mlforge", self.mlforge),
            ("pandas", self.pandas),
            ("numpy", self.numpy),
            ("scipy", self.scipy),
            ("scikit_learn", self.scikit_learn),
        ):
            _string(value, f"environment {name}")

    def to_dict(self) -> JsonObject:
        return {
            "python": self.python,
            "mlforge": self.mlforge,
            "pandas": self.pandas,
            "numpy": self.numpy,
            "scipy": self.scipy,
            "scikit_learn": self.scikit_learn,
        }

    @classmethod
    def from_object(cls, value: object) -> EnvironmentSnapshot:
        data = _object(value, "environment")
        expected = {"python", "mlforge", "pandas", "numpy", "scipy", "scikit_learn"}
        _keys(data, expected, "environment")
        return cls(
            python=_string(data["python"], "environment python"),
            mlforge=_string(data["mlforge"], "environment mlforge"),
            pandas=_string(data["pandas"], "environment pandas"),
            numpy=_string(data["numpy"], "environment numpy"),
            scipy=_string(data["scipy"], "environment scipy"),
            scikit_learn=_string(data["scikit_learn"], "environment scikit_learn"),
        )


@dataclass(frozen=True, slots=True)
class SplitSnapshot:
    """Actual partition sizes and policy used by a training attempt."""

    train_rows: int
    validation_rows: int
    feature_count: int
    stratified: bool
    partition_sha256: str

    def __post_init__(self) -> None:
        _integer(self.train_rows, "split train_rows")
        _integer(self.validation_rows, "split validation_rows")
        _integer(self.feature_count, "split feature_count")
        _boolean(self.stratified, "split stratified")
        _string(self.partition_sha256, "split partition_sha256")
        if len(self.partition_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.partition_sha256
        ):
            raise RunStoreError("Recorded split partition SHA-256 is invalid.")
        if self.train_rows <= 0 or self.validation_rows <= 0 or self.feature_count <= 0:
            raise RunStoreError("Recorded split dimensions must be positive.")

    def to_dict(self) -> JsonObject:
        return {
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "feature_count": self.feature_count,
            "stratified": self.stratified,
            "partition_sha256": self.partition_sha256,
        }

    @classmethod
    def from_object(cls, value: object) -> SplitSnapshot:
        data = _object(value, "split")
        expected = {
            "train_rows",
            "validation_rows",
            "feature_count",
            "stratified",
            "partition_sha256",
        }
        _keys(data, expected, "split")
        return cls(
            train_rows=_integer(data["train_rows"], "split train_rows"),
            validation_rows=_integer(data["validation_rows"], "split validation_rows"),
            feature_count=_integer(data["feature_count"], "split feature_count"),
            stratified=_boolean(data["stratified"], "split stratified"),
            partition_sha256=_string(data["partition_sha256"], "split partition_sha256"),
        )


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One finite evaluation metric and its comparison direction."""

    name: str
    value: float
    higher_is_better: bool

    def __post_init__(self) -> None:
        _string(self.name, "metric name")
        _boolean(self.higher_is_better, "metric direction")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise RunStoreError("Metric values must be numbers.")
        if not math.isfinite(self.value):
            raise RunStoreError("Metric names must be non-blank and values must be finite.")

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
        }

    @classmethod
    def from_object(cls, value: object) -> MetricValue:
        data = _object(value, "metric")
        _keys(data, {"name", "value", "higher_is_better"}, "metric")
        return cls(
            name=_string(data["name"], "metric name"),
            value=_number(data["value"], "metric value"),
            higher_is_better=_boolean(data["higher_is_better"], "metric direction"),
        )


@dataclass(frozen=True, slots=True)
class RunFailure:
    """Safe terminal error information without a traceback or data values."""

    error_type: str
    message: str

    def __post_init__(self) -> None:
        _string(self.error_type, "failure error_type")
        _string(self.message, "failure message")

    def to_dict(self) -> JsonObject:
        return {"error_type": self.error_type, "message": self.message}

    @classmethod
    def from_object(cls, value: object) -> RunFailure:
        data = _object(value, "failure")
        _keys(data, {"error_type", "message"}, "failure")
        return cls(
            error_type=_string(data["error_type"], "failure error_type"),
            message=_string(data["message"], "failure message"),
        )


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Versioned terminal record for one local training attempt."""

    schema_version: int
    run_id: str
    status: RunStatus
    started_at: str
    completed_at: str
    configuration: RunConfiguration
    dataset: DatasetSnapshot
    environment: EnvironmentSnapshot
    split: SplitSnapshot | None
    metrics: tuple[MetricValue, ...]
    warnings: tuple[str, ...]
    failure: RunFailure | None

    def __post_init__(self) -> None:
        _integer(self.schema_version, "schema_version")
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise RunStoreError(f"Unsupported run manifest schema version: {self.schema_version}.")
        _string(self.run_id, "run_id")
        if not isinstance(self.status, RunStatus):
            raise RunStoreError("Run manifest status must be a RunStatus value.")
        _string(self.started_at, "started_at")
        _string(self.completed_at, "completed_at")
        if not isinstance(self.configuration, RunConfiguration):
            raise RunStoreError("Run manifest configuration is invalid.")
        if not isinstance(self.dataset, DatasetSnapshot):
            raise RunStoreError("Run manifest dataset is invalid.")
        if not isinstance(self.environment, EnvironmentSnapshot):
            raise RunStoreError("Run manifest environment is invalid.")
        if self.split is not None and not isinstance(self.split, SplitSnapshot):
            raise RunStoreError("Run manifest split is invalid.")
        if not isinstance(self.metrics, tuple) or any(
            not isinstance(metric, MetricValue) for metric in self.metrics
        ):
            raise RunStoreError("Run manifest metrics must be a MetricValue tuple.")
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(warning, str) for warning in self.warnings
        ):
            raise RunStoreError("Run manifest warnings must be a string tuple.")
        if self.failure is not None and not isinstance(self.failure, RunFailure):
            raise RunStoreError("Run manifest failure is invalid.")
        try:
            parsed_id = UUID(self.run_id)
        except (ValueError, AttributeError, TypeError) as error:
            raise RunStoreError("Run manifest run_id must be a UUID.") from error
        if str(parsed_id) != self.run_id:
            raise RunStoreError("Run manifest run_id must use canonical lowercase UUID form.")
        try:
            started = datetime.fromisoformat(self.started_at)
            completed = datetime.fromisoformat(self.completed_at)
        except ValueError as error:
            raise RunStoreError("Run manifest timestamps must use ISO 8601 format.") from error
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            raise RunStoreError("Run manifest timestamps must be timezone-aware and ordered.")
        metric_names = tuple(metric.name for metric in self.metrics)
        if metric_names != tuple(sorted(metric_names)) or len(set(metric_names)) != len(
            metric_names
        ):
            raise RunStoreError("Run metrics must have unique sorted names.")
        if any(not warning.strip() for warning in self.warnings):
            raise RunStoreError("Run warnings must not contain blank entries.")
        if len(set(self.warnings)) != len(self.warnings):
            raise RunStoreError("Run warnings must not contain duplicate entries.")
        if self.status is RunStatus.SUCCEEDED:
            if self.split is None or not self.metrics or self.failure is not None:
                raise RunStoreError(
                    "A successful run requires a split and metrics without failure."
                )
        elif self.failure is None or self.metrics:
            raise RunStoreError("A failed run requires failure details and no metrics.")

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "configuration": self.configuration.to_dict(),
            "dataset": self.dataset.to_dict(),
            "environment": self.environment.to_dict(),
            "split": self.split.to_dict() if self.split is not None else None,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "warnings": list(self.warnings),
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministic standards-compliant manifest JSON."""
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, content: str) -> RunManifest:
        """Parse and fully validate one manifest from untrusted JSON text."""
        try:
            raw: object = json.loads(content)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise RunStoreError(f"Run manifest is not valid JSON: {error}") from error
        data = _object(raw, "root")
        expected = {
            "schema_version",
            "run_id",
            "status",
            "started_at",
            "completed_at",
            "configuration",
            "dataset",
            "environment",
            "split",
            "metrics",
            "warnings",
            "failure",
        }
        _keys(data, expected, "root")
        try:
            status = RunStatus(_string(data["status"], "status"))
        except ValueError as error:
            raise RunStoreError(f"Unsupported run status: {data['status']!r}.") from error
        split = None if data["split"] is None else SplitSnapshot.from_object(data["split"])
        failure = None if data["failure"] is None else RunFailure.from_object(data["failure"])
        return cls(
            schema_version=_integer(data["schema_version"], "schema_version"),
            run_id=_string(data["run_id"], "run_id"),
            status=status,
            started_at=_string(data["started_at"], "started_at"),
            completed_at=_string(data["completed_at"], "completed_at"),
            configuration=RunConfiguration.from_object(data["configuration"]),
            dataset=DatasetSnapshot.from_object(data["dataset"]),
            environment=EnvironmentSnapshot.from_object(data["environment"]),
            split=split,
            metrics=tuple(
                MetricValue.from_object(item) for item in _array(data["metrics"], "metrics")
            ),
            warnings=_string_tuple(data["warnings"], "warnings"),
            failure=failure,
        )
