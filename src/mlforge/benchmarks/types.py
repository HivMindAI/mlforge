"""Typed configuration, manifests, and results for local classification benchmarks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast
from uuid import UUID

from mlforge.errors import BenchmarkError, BenchmarkStoreError, ConfigurationError, RunStoreError
from mlforge.pipelines import (
    FeatureOverrides,
    PreprocessingConfig,
    SplitConfig,
    TaskType,
)
from mlforge.runs import DatasetSnapshot, RunFailure, RunManifest, RunStatus, SplitSnapshot
from mlforge.training import (
    CLASSIFICATION_ESTIMATORS,
    CLASSIFICATION_METRICS,
    DUMMY_CLASSIFIER,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_CLASSIFIER,
    TrainingResult,
)

BENCHMARK_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CLASSIFICATION_BENCHMARK_ESTIMATORS = (
    DUMMY_CLASSIFIER,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_CLASSIFIER,
)

JsonObject: TypeAlias = dict[str, Any]


class BenchmarkStatus(StrEnum):
    """Terminal aggregate status of a local benchmark."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BenchmarkStoreError(f"Benchmark manifest {label} must be a JSON object.")
    return cast(dict[str, object], value)


def _keys(value: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing!r}")
    if extra:
        details.append(f"unexpected {extra!r}")
    raise BenchmarkStoreError(
        f"Benchmark manifest {label} has invalid fields: {', '.join(details)}."
    )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkStoreError(f"Benchmark manifest {label} must be a non-blank string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkStoreError(f"Benchmark manifest {label} must be an integer.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkStoreError(f"Benchmark manifest {label} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise BenchmarkStoreError(f"Benchmark manifest {label} must be finite.")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise BenchmarkStoreError(f"Benchmark manifest {label} must be true or false.")
    return value


def _optional_boolean(value: object, label: str) -> bool | None:
    return None if value is None else _boolean(value, label)


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise BenchmarkStoreError(f"Benchmark manifest {label} must be a JSON array.")
    return cast(list[object], value)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{label} entry") for item in _array(value, label))


def _validate_uuid(value: str, label: str) -> None:
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError, TypeError) as error:
        raise BenchmarkStoreError(f"Benchmark manifest {label} must be a UUID.") from error
    if str(parsed) != value:
        raise BenchmarkStoreError(
            f"Benchmark manifest {label} must use canonical lowercase UUID form."
        )


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise BenchmarkStoreError(f"Benchmark manifest {label} is not a valid SHA-256 value.")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Explicit configuration shared by every run in one local benchmark."""

    estimators: tuple[str, ...] = DEFAULT_CLASSIFICATION_BENCHMARK_ESTIMATORS
    primary_metric: str = "balanced_accuracy"
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_overrides: FeatureOverrides = field(default_factory=FeatureOverrides)

    def __post_init__(self) -> None:
        if not isinstance(self.estimators, tuple):
            raise ConfigurationError("Benchmark estimators must be provided as a tuple.")
        if len(self.estimators) < 2:
            raise ConfigurationError("A benchmark requires at least two classification estimators.")
        if any(
            not isinstance(estimator, str) or estimator not in CLASSIFICATION_ESTIMATORS
            for estimator in self.estimators
        ):
            choices = ", ".join(sorted(CLASSIFICATION_ESTIMATORS))
            raise ConfigurationError(
                "Benchmark estimators must be supported classification estimators. "
                f"Choose from: {choices}."
            )
        if len(set(self.estimators)) != len(self.estimators):
            raise ConfigurationError("Benchmark estimators must be unique.")
        if not isinstance(self.primary_metric, str) or self.primary_metric not in (
            CLASSIFICATION_METRICS
        ):
            choices = ", ".join(CLASSIFICATION_METRICS)
            raise ConfigurationError(
                f"Unsupported benchmark metric {self.primary_metric!r}. Choose one of: {choices}."
            )
        if not isinstance(self.split, SplitConfig):
            raise ConfigurationError("Benchmark split must be a SplitConfig value.")
        if not isinstance(self.preprocessing, PreprocessingConfig):
            raise ConfigurationError("Benchmark preprocessing must be a PreprocessingConfig value.")
        if not isinstance(self.feature_overrides, FeatureOverrides):
            raise ConfigurationError(
                "Benchmark feature_overrides must be a FeatureOverrides value."
            )


@dataclass(frozen=True, slots=True)
class BenchmarkConfiguration:
    """Serializable effective configuration for an immutable benchmark manifest."""

    task: str
    estimators: tuple[str, ...]
    primary_metric: str
    validation_fraction: float
    random_seed: int
    stratify_requested: bool | None
    numeric_imputation: str
    scale_numeric: bool
    categorical_fill_value: str
    numeric_overrides: tuple[str, ...]
    categorical_overrides: tuple[str, ...]

    def __post_init__(self) -> None:
        _string(self.task, "configuration task")
        if self.task != TaskType.CLASSIFICATION.value:
            raise BenchmarkStoreError("Benchmark manifests currently support classification only.")
        if not isinstance(self.estimators, tuple) or len(self.estimators) < 2:
            raise BenchmarkStoreError(
                "Benchmark manifest estimators must contain at least two entries."
            )
        if any(estimator not in CLASSIFICATION_ESTIMATORS for estimator in self.estimators):
            raise BenchmarkStoreError(
                "Benchmark manifest contains an unsupported classification estimator."
            )
        if len(set(self.estimators)) != len(self.estimators):
            raise BenchmarkStoreError("Benchmark manifest estimators must be unique.")
        _string(self.primary_metric, "configuration primary_metric")
        if self.primary_metric not in CLASSIFICATION_METRICS:
            raise BenchmarkStoreError(
                f"Benchmark manifest metric is unsupported: {self.primary_metric!r}."
            )
        _number(self.validation_fraction, "configuration validation_fraction")
        if not 0 < self.validation_fraction < 1:
            raise BenchmarkStoreError(
                "Benchmark manifest validation fraction must be between 0 and 1."
            )
        _integer(self.random_seed, "configuration random_seed")
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise BenchmarkStoreError("Benchmark manifest random seed is outside the valid range.")
        _optional_boolean(self.stratify_requested, "configuration stratify_requested")
        _string(self.numeric_imputation, "configuration numeric_imputation")
        if self.numeric_imputation not in {"mean", "median"}:
            raise BenchmarkStoreError("Benchmark manifest numeric imputation is unsupported.")
        _boolean(self.scale_numeric, "configuration scale_numeric")
        _string(self.categorical_fill_value, "configuration categorical_fill_value")
        for role, names in (
            ("numeric", self.numeric_overrides),
            ("categorical", self.categorical_overrides),
        ):
            if not isinstance(names, tuple):
                raise BenchmarkStoreError(f"Benchmark manifest {role} overrides must be a tuple.")
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise BenchmarkStoreError(
                    f"Benchmark manifest {role} overrides contain an invalid name."
                )
            if len(set(names)) != len(names):
                raise BenchmarkStoreError(f"Benchmark manifest {role} overrides must be unique.")
        if set(self.numeric_overrides).intersection(self.categorical_overrides):
            raise BenchmarkStoreError("Benchmark manifest feature overrides must not overlap.")

    @classmethod
    def from_config(cls, config: BenchmarkConfig) -> BenchmarkConfiguration:
        """Capture explicit effective values without depending on future defaults."""
        return cls(
            task=TaskType.CLASSIFICATION.value,
            estimators=config.estimators,
            primary_metric=config.primary_metric,
            validation_fraction=float(config.split.validation_fraction),
            random_seed=config.split.random_seed,
            stratify_requested=config.split.stratify,
            numeric_imputation=config.preprocessing.numeric_imputation.value,
            scale_numeric=config.preprocessing.scale_numeric,
            categorical_fill_value=config.preprocessing.categorical_fill_value,
            numeric_overrides=config.feature_overrides.numeric,
            categorical_overrides=config.feature_overrides.categorical,
        )

    def to_dict(self) -> JsonObject:
        return {
            "task": self.task,
            "estimators": list(self.estimators),
            "primary_metric": self.primary_metric,
            "validation_fraction": self.validation_fraction,
            "random_seed": self.random_seed,
            "stratify_requested": self.stratify_requested,
            "numeric_imputation": self.numeric_imputation,
            "scale_numeric": self.scale_numeric,
            "categorical_fill_value": self.categorical_fill_value,
            "numeric_overrides": list(self.numeric_overrides),
            "categorical_overrides": list(self.categorical_overrides),
        }

    @classmethod
    def from_object(cls, value: object) -> BenchmarkConfiguration:
        data = _object(value, "configuration")
        _keys(
            data,
            {
                "task",
                "estimators",
                "primary_metric",
                "validation_fraction",
                "random_seed",
                "stratify_requested",
                "numeric_imputation",
                "scale_numeric",
                "categorical_fill_value",
                "numeric_overrides",
                "categorical_overrides",
            },
            "configuration",
        )
        return cls(
            task=_string(data["task"], "configuration task"),
            estimators=_string_tuple(data["estimators"], "configuration estimators"),
            primary_metric=_string(data["primary_metric"], "configuration primary_metric"),
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
                data["categorical_overrides"], "configuration categorical overrides"
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkEntry:
    """One estimator outcome referenced by a benchmark manifest."""

    estimator: str
    run_id: str
    status: RunStatus
    duration_seconds: float
    rank: int | None
    primary_metric_value: float | None
    failure: RunFailure | None

    def __post_init__(self) -> None:
        _string(self.estimator, "entry estimator")
        if self.estimator not in CLASSIFICATION_ESTIMATORS:
            raise BenchmarkStoreError("Benchmark entry estimator is unsupported.")
        _string(self.run_id, "entry run_id")
        _validate_uuid(self.run_id, "entry run_id")
        if not isinstance(self.status, RunStatus):
            raise BenchmarkStoreError("Benchmark entry status must be a RunStatus value.")
        _number(self.duration_seconds, "entry duration_seconds")
        if self.duration_seconds < 0:
            raise BenchmarkStoreError("Benchmark entry duration must not be negative.")
        if self.status is RunStatus.SUCCEEDED:
            if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
                raise BenchmarkStoreError("Successful benchmark entries require a positive rank.")
            if self.primary_metric_value is None:
                raise BenchmarkStoreError(
                    "Successful benchmark entries require a primary metric value."
                )
            _number(self.primary_metric_value, "entry primary_metric_value")
            if self.failure is not None:
                raise BenchmarkStoreError("Successful benchmark entries cannot contain failures.")
        elif self.rank is not None or self.primary_metric_value is not None or self.failure is None:
            raise BenchmarkStoreError(
                "Failed benchmark entries require failure details and cannot be ranked."
            )
        if self.failure is not None and not isinstance(self.failure, RunFailure):
            raise BenchmarkStoreError("Benchmark entry failure is invalid.")

    def to_dict(self) -> JsonObject:
        return {
            "estimator": self.estimator,
            "run_id": self.run_id,
            "status": self.status.value,
            "duration_seconds": self.duration_seconds,
            "rank": self.rank,
            "primary_metric_value": self.primary_metric_value,
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }

    @classmethod
    def from_object(cls, value: object) -> BenchmarkEntry:
        data = _object(value, "entry")
        _keys(
            data,
            {
                "estimator",
                "run_id",
                "status",
                "duration_seconds",
                "rank",
                "primary_metric_value",
                "failure",
            },
            "entry",
        )
        try:
            status = RunStatus(_string(data["status"], "entry status"))
        except ValueError as error:
            raise BenchmarkStoreError(
                f"Benchmark entry status is unsupported: {data['status']!r}."
            ) from error
        raw_rank = data["rank"]
        raw_metric = data["primary_metric_value"]
        try:
            failure = None if data["failure"] is None else RunFailure.from_object(data["failure"])
        except RunStoreError as error:
            raise BenchmarkStoreError(f"Benchmark entry failure is invalid: {error}") from error
        return cls(
            estimator=_string(data["estimator"], "entry estimator"),
            run_id=_string(data["run_id"], "entry run_id"),
            status=status,
            duration_seconds=_number(data["duration_seconds"], "entry duration_seconds"),
            rank=None if raw_rank is None else _integer(raw_rank, "entry rank"),
            primary_metric_value=(
                None if raw_metric is None else _number(raw_metric, "entry primary_metric_value")
            ),
            failure=failure,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """Versioned terminal record that aggregates immutable training runs."""

    schema_version: int
    benchmark_id: str
    status: BenchmarkStatus
    started_at: str
    completed_at: str
    configuration: BenchmarkConfiguration
    dataset: DatasetSnapshot
    split: SplitSnapshot | None
    higher_is_better: bool
    entries: tuple[BenchmarkEntry, ...]

    def __post_init__(self) -> None:
        _integer(self.schema_version, "schema_version")
        if self.schema_version != BENCHMARK_MANIFEST_SCHEMA_VERSION:
            raise BenchmarkStoreError(
                f"Unsupported benchmark manifest schema version: {self.schema_version}."
            )
        _string(self.benchmark_id, "benchmark_id")
        _validate_uuid(self.benchmark_id, "benchmark_id")
        if not isinstance(self.status, BenchmarkStatus):
            raise BenchmarkStoreError("Benchmark manifest status is invalid.")
        _string(self.started_at, "started_at")
        _string(self.completed_at, "completed_at")
        try:
            started = datetime.fromisoformat(self.started_at)
            completed = datetime.fromisoformat(self.completed_at)
        except ValueError as error:
            raise BenchmarkStoreError(
                "Benchmark manifest timestamps must use ISO 8601 format."
            ) from error
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            raise BenchmarkStoreError(
                "Benchmark manifest timestamps must be timezone-aware and ordered."
            )
        if not isinstance(self.configuration, BenchmarkConfiguration):
            raise BenchmarkStoreError("Benchmark manifest configuration is invalid.")
        if not isinstance(self.dataset, DatasetSnapshot):
            raise BenchmarkStoreError("Benchmark manifest dataset is invalid.")
        if self.split is not None and not isinstance(self.split, SplitSnapshot):
            raise BenchmarkStoreError("Benchmark manifest split is invalid.")
        _boolean(self.higher_is_better, "higher_is_better")
        if not self.higher_is_better:
            raise BenchmarkStoreError(
                "Benchmark classification metrics must use higher-is-better ranking."
            )
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, BenchmarkEntry) for entry in self.entries
        ):
            raise BenchmarkStoreError("Benchmark manifest entries must be a tuple.")
        if tuple(entry.estimator for entry in self.entries) != self.configuration.estimators:
            raise BenchmarkStoreError(
                "Benchmark entries must follow the configured estimator order."
            )
        run_ids = tuple(entry.run_id for entry in self.entries)
        if len(set(run_ids)) != len(run_ids):
            raise BenchmarkStoreError("Benchmark entry run IDs must be unique.")

        successful = tuple(entry for entry in self.entries if entry.status is RunStatus.SUCCEEDED)
        expected_status = (
            BenchmarkStatus.FAILED
            if not successful
            else BenchmarkStatus.SUCCEEDED
            if len(successful) == len(self.entries)
            else BenchmarkStatus.PARTIAL
        )
        if self.status is not expected_status:
            raise BenchmarkStoreError(
                "Benchmark aggregate status does not match its estimator outcomes."
            )
        if successful and self.split is None:
            raise BenchmarkStoreError("A benchmark with successful runs requires a split snapshot.")
        if not successful and self.split is not None:
            raise BenchmarkStoreError("A failed benchmark cannot contain a successful split.")
        if self.split is not None and (
            self.split.train_rows + self.split.validation_rows != self.dataset.row_count
            or self.split.feature_count != self.dataset.column_count - 1
        ):
            raise BenchmarkStoreError(
                "Benchmark split dimensions do not match the recorded dataset."
            )
        ranks = tuple(entry.rank for entry in successful)
        if set(ranks) != set(range(1, len(successful) + 1)):
            raise BenchmarkStoreError(
                "Successful benchmark entry ranks must form a complete one-based sequence."
            )
        ordered = sorted(
            successful,
            key=lambda entry: (
                -cast(float, entry.primary_metric_value),
                entry.estimator,
                entry.run_id,
            ),
        )
        if any(entry.rank != rank for rank, entry in enumerate(ordered, start=1)):
            raise BenchmarkStoreError(
                "Benchmark ranks do not match the recorded primary metric values."
            )

    @property
    def winner(self) -> BenchmarkEntry | None:
        """Return the rank-one observed result, if any estimator succeeded."""
        return next((entry for entry in self.entries if entry.rank == 1), None)

    def to_dict(self) -> JsonObject:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "configuration": self.configuration.to_dict(),
            "dataset": self.dataset.to_dict(),
            "split": self.split.to_dict() if self.split is not None else None,
            "higher_is_better": self.higher_is_better,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministic standards-compliant benchmark JSON."""
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, content: str) -> BenchmarkManifest:
        """Parse and fully validate one untrusted benchmark manifest."""
        try:
            raw: object = json.loads(content)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise BenchmarkStoreError(f"Benchmark manifest is not valid JSON: {error}") from error
        data = _object(raw, "root")
        _keys(
            data,
            {
                "schema_version",
                "benchmark_id",
                "status",
                "started_at",
                "completed_at",
                "configuration",
                "dataset",
                "split",
                "higher_is_better",
                "entries",
            },
            "root",
        )
        try:
            status = BenchmarkStatus(_string(data["status"], "status"))
        except ValueError as error:
            raise BenchmarkStoreError(
                f"Unsupported benchmark status: {data['status']!r}."
            ) from error
        try:
            dataset = DatasetSnapshot.from_object(data["dataset"])
            split = None if data["split"] is None else SplitSnapshot.from_object(data["split"])
        except RunStoreError as error:
            raise BenchmarkStoreError(
                f"Benchmark manifest contains an invalid run snapshot: {error}"
            ) from error
        return cls(
            schema_version=_integer(data["schema_version"], "schema_version"),
            benchmark_id=_string(data["benchmark_id"], "benchmark_id"),
            status=status,
            started_at=_string(data["started_at"], "started_at"),
            completed_at=_string(data["completed_at"], "completed_at"),
            configuration=BenchmarkConfiguration.from_object(data["configuration"]),
            dataset=dataset,
            split=split,
            higher_is_better=_boolean(data["higher_is_better"], "higher_is_better"),
            entries=tuple(
                BenchmarkEntry.from_object(item) for item in _array(data["entries"], "entries")
            ),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Persisted benchmark plus fitted successful pipelines and all run records."""

    manifest: BenchmarkManifest
    manifest_path: Path
    training_results: tuple[TrainingResult, ...]
    run_manifests: tuple[RunManifest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, BenchmarkManifest):
            raise BenchmarkError("Benchmark result manifest is invalid.")
        if not isinstance(self.manifest_path, Path):
            raise BenchmarkError("Benchmark result manifest_path must be a pathlib.Path.")
        if not isinstance(self.training_results, tuple) or any(
            not isinstance(result, TrainingResult) for result in self.training_results
        ):
            raise BenchmarkError("Benchmark training results must be a tuple.")
        if not isinstance(self.run_manifests, tuple) or any(
            not isinstance(manifest, RunManifest) for manifest in self.run_manifests
        ):
            raise BenchmarkError("Benchmark run manifests must be a tuple.")
        entry_run_ids = tuple(entry.run_id for entry in self.manifest.entries)
        if tuple(manifest.run_id for manifest in self.run_manifests) != entry_run_ids:
            raise BenchmarkError("Benchmark result run manifests do not match its entries.")
        successful_ids = tuple(
            entry.run_id for entry in self.manifest.entries if entry.status is RunStatus.SUCCEEDED
        )
        if tuple(result.manifest.run_id for result in self.training_results) != successful_ids:
            raise BenchmarkError("Benchmark fitted results do not match its successful entries.")

    @property
    def winner(self) -> TrainingResult:
        """Return the fitted pipeline belonging to the rank-one benchmark entry."""
        winning_entry = self.manifest.winner
        if winning_entry is None:
            raise BenchmarkError("A failed benchmark does not have a winning training result.")
        return next(
            result
            for result in self.training_results
            if result.manifest.run_id == winning_entry.run_id
        )
