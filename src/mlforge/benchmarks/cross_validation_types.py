"""Typed configuration and immutable evidence for stratified K-fold benchmarks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev

from mlforge.benchmarks.types import (
    DEFAULT_CLASSIFICATION_BENCHMARK_ESTIMATORS,
    BenchmarkConfig,
    BenchmarkStatus,
    JsonObject,
    _array,
    _boolean,
    _integer,
    _keys,
    _number,
    _object,
    _string,
    _string_tuple,
    _validate_sha256,
    _validate_uuid,
)
from mlforge.errors import BenchmarkError, BenchmarkStoreError, ConfigurationError, RunStoreError
from mlforge.pipelines import (
    CrossValidationSplitConfig,
    FeatureOverrides,
    NumericImputationStrategy,
    PreprocessingConfig,
    TaskType,
)
from mlforge.runs import (
    DatasetSnapshot,
    EnvironmentSnapshot,
    MetricValue,
    RunFailure,
    RunParameter,
    RunStatus,
)
from mlforge.training import CLASSIFICATION_ESTIMATORS, CLASSIFICATION_METRICS

CROSS_VALIDATION_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CrossValidationConfig:
    """Shared estimator, metric, fold, and preprocessing configuration."""

    estimators: tuple[str, ...] = DEFAULT_CLASSIFICATION_BENCHMARK_ESTIMATORS
    primary_metric: str = "balanced_accuracy"
    split: CrossValidationSplitConfig = field(default_factory=CrossValidationSplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_overrides: FeatureOverrides = field(default_factory=FeatureOverrides)

    def __post_init__(self) -> None:
        BenchmarkConfig(
            estimators=self.estimators,
            primary_metric=self.primary_metric,
            preprocessing=self.preprocessing,
            feature_overrides=self.feature_overrides,
        )
        if not isinstance(self.split, CrossValidationSplitConfig):
            raise ConfigurationError(
                "Cross-validation split must be a CrossValidationSplitConfig value."
            )


@dataclass(frozen=True, slots=True)
class CrossValidationConfiguration:
    """Serializable effective configuration independent of future defaults."""

    task: str
    estimators: tuple[str, ...]
    primary_metric: str
    fold_count: int
    random_seed: int
    numeric_imputation: str
    scale_numeric: bool
    categorical_fill_value: str
    numeric_overrides: tuple[str, ...]
    categorical_overrides: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            config = CrossValidationConfig(
                estimators=self.estimators,
                primary_metric=self.primary_metric,
                split=CrossValidationSplitConfig(
                    fold_count=self.fold_count,
                    random_seed=self.random_seed,
                ),
                preprocessing=PreprocessingConfig(
                    numeric_imputation=NumericImputationStrategy(self.numeric_imputation),
                    scale_numeric=self.scale_numeric,
                    categorical_fill_value=self.categorical_fill_value,
                ),
                feature_overrides=FeatureOverrides(
                    numeric=self.numeric_overrides,
                    categorical=self.categorical_overrides,
                ),
            )
        except (ConfigurationError, ValueError) as error:
            raise BenchmarkStoreError(
                f"Cross-validation manifest configuration is invalid: {error}"
            ) from error
        if self.task != TaskType.CLASSIFICATION.value:
            raise BenchmarkStoreError(
                "Cross-validation manifests currently support classification only."
            )
        if config.primary_metric != self.primary_metric:
            raise BenchmarkStoreError("Cross-validation primary metric is invalid.")

    @classmethod
    def from_config(cls, config: CrossValidationConfig) -> CrossValidationConfiguration:
        return cls(
            task=TaskType.CLASSIFICATION.value,
            estimators=config.estimators,
            primary_metric=config.primary_metric,
            fold_count=config.split.fold_count,
            random_seed=config.split.random_seed,
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
            "fold_count": self.fold_count,
            "random_seed": self.random_seed,
            "numeric_imputation": self.numeric_imputation,
            "scale_numeric": self.scale_numeric,
            "categorical_fill_value": self.categorical_fill_value,
            "numeric_overrides": list(self.numeric_overrides),
            "categorical_overrides": list(self.categorical_overrides),
        }

    @classmethod
    def from_object(cls, value: object) -> CrossValidationConfiguration:
        data = _object(value, "cross-validation configuration")
        _keys(
            data,
            {
                "task",
                "estimators",
                "primary_metric",
                "fold_count",
                "random_seed",
                "numeric_imputation",
                "scale_numeric",
                "categorical_fill_value",
                "numeric_overrides",
                "categorical_overrides",
            },
            "cross-validation configuration",
        )
        return cls(
            task=_string(data["task"], "cross-validation configuration task"),
            estimators=_string_tuple(
                data["estimators"], "cross-validation configuration estimators"
            ),
            primary_metric=_string(
                data["primary_metric"], "cross-validation configuration primary_metric"
            ),
            fold_count=_integer(data["fold_count"], "cross-validation configuration fold_count"),
            random_seed=_integer(data["random_seed"], "cross-validation configuration random_seed"),
            numeric_imputation=_string(
                data["numeric_imputation"],
                "cross-validation configuration numeric_imputation",
            ),
            scale_numeric=_boolean(
                data["scale_numeric"], "cross-validation configuration scale_numeric"
            ),
            categorical_fill_value=_string(
                data["categorical_fill_value"],
                "cross-validation configuration categorical_fill_value",
            ),
            numeric_overrides=_string_tuple(
                data["numeric_overrides"],
                "cross-validation configuration numeric_overrides",
            ),
            categorical_overrides=_string_tuple(
                data["categorical_overrides"],
                "cross-validation configuration categorical_overrides",
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossValidationFoldSnapshot:
    """One shared fold's row counts and exact partition fingerprint."""

    fold_number: int
    train_rows: int
    validation_rows: int
    partition_sha256: str

    def __post_init__(self) -> None:
        for label, value in (
            ("fold_number", self.fold_number),
            ("train_rows", self.train_rows),
            ("validation_rows", self.validation_rows),
        ):
            _integer(value, f"cross-validation fold {label}")
            if value <= 0:
                raise BenchmarkStoreError(f"Cross-validation fold {label} must be positive.")
        _string(self.partition_sha256, "cross-validation fold partition_sha256")
        _validate_sha256(self.partition_sha256, "cross-validation fold partition_sha256")

    def to_dict(self) -> JsonObject:
        return {
            "fold_number": self.fold_number,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "partition_sha256": self.partition_sha256,
        }

    @classmethod
    def from_object(cls, value: object) -> CrossValidationFoldSnapshot:
        data = _object(value, "cross-validation fold")
        _keys(
            data,
            {"fold_number", "train_rows", "validation_rows", "partition_sha256"},
            "cross-validation fold",
        )
        return cls(
            fold_number=_integer(data["fold_number"], "cross-validation fold_number"),
            train_rows=_integer(data["train_rows"], "cross-validation fold train_rows"),
            validation_rows=_integer(
                data["validation_rows"], "cross-validation fold validation_rows"
            ),
            partition_sha256=_string(
                data["partition_sha256"], "cross-validation fold partition_sha256"
            ),
        )


def fold_plan_sha256(folds: tuple[CrossValidationFoldSnapshot, ...]) -> str:
    """Fingerprint the ordered complete fold plan."""
    content = json.dumps(
        [fold.to_dict() for fold in folds],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


@dataclass(frozen=True, slots=True)
class CrossValidationFoldResult:
    """Metrics, warnings, and observed duration for one successful estimator fold."""

    fold_number: int
    metrics: tuple[MetricValue, ...]
    duration_seconds: float
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        _integer(self.fold_number, "cross-validation result fold_number")
        if self.fold_number <= 0:
            raise BenchmarkStoreError("Cross-validation result fold number must be positive.")
        if (
            not isinstance(self.metrics, tuple)
            or not self.metrics
            or any(not isinstance(metric, MetricValue) for metric in self.metrics)
        ):
            raise BenchmarkStoreError(
                "Cross-validation fold metrics must be a non-empty MetricValue tuple."
            )
        names = tuple(metric.name for metric in self.metrics)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise BenchmarkStoreError(
                "Cross-validation fold metric names must be unique and sorted."
            )
        if names != tuple(sorted(CLASSIFICATION_METRICS)):
            raise BenchmarkStoreError(
                "Cross-validation folds must contain the complete classification metric set."
            )
        if any(not metric.higher_is_better for metric in self.metrics):
            raise BenchmarkStoreError(
                "Cross-validation classification metric directions are invalid."
            )
        _number(self.duration_seconds, "cross-validation fold duration_seconds")
        if self.duration_seconds < 0:
            raise BenchmarkStoreError("Cross-validation fold duration must not be negative.")
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(message, str) or not message.strip() for message in self.warnings
        ):
            raise BenchmarkStoreError("Cross-validation fold warnings must be non-blank strings.")
        if len(set(self.warnings)) != len(self.warnings):
            raise BenchmarkStoreError("Cross-validation fold warnings must be unique.")

    def to_dict(self) -> JsonObject:
        return {
            "fold_number": self.fold_number,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "duration_seconds": self.duration_seconds,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_object(cls, value: object) -> CrossValidationFoldResult:
        data = _object(value, "cross-validation fold result")
        _keys(
            data,
            {"fold_number", "metrics", "duration_seconds", "warnings"},
            "cross-validation fold result",
        )
        try:
            metrics = tuple(
                MetricValue.from_object(item)
                for item in _array(data["metrics"], "cross-validation fold metrics")
            )
        except RunStoreError as error:
            raise BenchmarkStoreError(
                f"Cross-validation fold metric is invalid: {error}"
            ) from error
        return cls(
            fold_number=_integer(data["fold_number"], "cross-validation result fold_number"),
            metrics=metrics,
            duration_seconds=_number(
                data["duration_seconds"], "cross-validation fold duration_seconds"
            ),
            warnings=_string_tuple(data["warnings"], "cross-validation fold warnings"),
        )


@dataclass(frozen=True, slots=True)
class CrossValidationMetricSummary:
    """Per-fold values plus mean and population standard deviation."""

    name: str
    fold_values: tuple[float, ...]
    mean: float
    standard_deviation: float
    higher_is_better: bool

    def __post_init__(self) -> None:
        _string(self.name, "cross-validation metric name")
        if self.name not in CLASSIFICATION_METRICS:
            raise BenchmarkStoreError(f"Cross-validation metric is unsupported: {self.name!r}.")
        if not isinstance(self.fold_values, tuple) or len(self.fold_values) < 2:
            raise BenchmarkStoreError("Cross-validation metric requires at least two fold values.")
        values = tuple(
            _number(value, f"cross-validation metric {self.name!r} fold value")
            for value in self.fold_values
        )
        _number(self.mean, f"cross-validation metric {self.name!r} mean")
        _number(
            self.standard_deviation,
            f"cross-validation metric {self.name!r} standard_deviation",
        )
        if self.standard_deviation < 0:
            raise BenchmarkStoreError(
                "Cross-validation metric standard deviation must not be negative."
            )
        _boolean(
            self.higher_is_better,
            f"cross-validation metric {self.name!r} higher_is_better",
        )
        if not self.higher_is_better:
            raise BenchmarkStoreError(
                "Cross-validation classification metric directions are invalid."
            )
        if not math.isclose(self.mean, fmean(values), rel_tol=1e-12, abs_tol=1e-12):
            raise BenchmarkStoreError(
                "Cross-validation metric mean does not match its fold values."
            )
        if not math.isclose(
            self.standard_deviation,
            pstdev(values),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise BenchmarkStoreError(
                "Cross-validation metric standard deviation does not match its fold values."
            )

    def to_dict(self) -> JsonObject:
        return {
            "name": self.name,
            "fold_values": list(self.fold_values),
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "higher_is_better": self.higher_is_better,
        }

    @classmethod
    def from_object(cls, value: object) -> CrossValidationMetricSummary:
        data = _object(value, "cross-validation metric summary")
        _keys(
            data,
            {
                "name",
                "fold_values",
                "mean",
                "standard_deviation",
                "higher_is_better",
            },
            "cross-validation metric summary",
        )
        name = _string(data["name"], "cross-validation metric summary name")
        return cls(
            name=name,
            fold_values=tuple(
                _number(item, f"cross-validation metric {name!r} fold value")
                for item in _array(data["fold_values"], "cross-validation metric fold_values")
            ),
            mean=_number(data["mean"], f"cross-validation metric {name!r} mean"),
            standard_deviation=_number(
                data["standard_deviation"],
                f"cross-validation metric {name!r} standard_deviation",
            ),
            higher_is_better=_boolean(
                data["higher_is_better"],
                f"cross-validation metric {name!r} higher_is_better",
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossValidationEntry:
    """One estimator's complete shared-fold outcome."""

    estimator: str
    parameters: tuple[RunParameter, ...]
    status: RunStatus
    rank: int | None
    primary_metric_mean: float | None
    primary_metric_standard_deviation: float | None
    metrics: tuple[CrossValidationMetricSummary, ...]
    folds: tuple[CrossValidationFoldResult, ...]
    duration_seconds: float
    failure_fold: int | None
    failure_partition_sha256: str | None
    failure: RunFailure | None

    def __post_init__(self) -> None:
        if self.estimator not in CLASSIFICATION_ESTIMATORS:
            raise BenchmarkStoreError(
                f"Cross-validation estimator is unsupported: {self.estimator!r}."
            )
        if not isinstance(self.parameters, tuple) or any(
            not isinstance(parameter, RunParameter) for parameter in self.parameters
        ):
            raise BenchmarkStoreError("Cross-validation estimator parameters must be a tuple.")
        parameter_names = tuple(parameter.name for parameter in self.parameters)
        if parameter_names != tuple(sorted(parameter_names)) or len(set(parameter_names)) != len(
            parameter_names
        ):
            raise BenchmarkStoreError(
                "Cross-validation estimator parameters must have unique sorted names."
            )
        if not isinstance(self.status, RunStatus):
            raise BenchmarkStoreError("Cross-validation entry status is invalid.")
        if not isinstance(self.folds, tuple) or any(
            not isinstance(fold, CrossValidationFoldResult) for fold in self.folds
        ):
            raise BenchmarkStoreError("Cross-validation fold results must be a tuple.")
        fold_numbers = tuple(fold.fold_number for fold in self.folds)
        if fold_numbers != tuple(range(1, len(self.folds) + 1)):
            raise BenchmarkStoreError(
                "Cross-validation completed fold results must be an ordered prefix."
            )
        _number(self.duration_seconds, "cross-validation entry duration_seconds")
        if self.duration_seconds < sum(fold.duration_seconds for fold in self.folds):
            raise BenchmarkStoreError(
                "Cross-validation entry duration cannot be shorter than completed folds."
            )
        if self.status is RunStatus.SUCCEEDED:
            if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
                raise BenchmarkStoreError(
                    "Successful cross-validation entries require a positive rank."
                )
            if self.primary_metric_mean is None or self.primary_metric_standard_deviation is None:
                raise BenchmarkStoreError(
                    "Successful cross-validation entries require primary metric aggregates."
                )
            _number(
                self.primary_metric_mean,
                "cross-validation entry primary_metric_mean",
            )
            _number(
                self.primary_metric_standard_deviation,
                "cross-validation entry primary_metric_standard_deviation",
            )
            if not self.metrics or any(
                not isinstance(metric, CrossValidationMetricSummary) for metric in self.metrics
            ):
                raise BenchmarkStoreError(
                    "Successful cross-validation entries require metric summaries."
                )
            names = tuple(metric.name for metric in self.metrics)
            if names != tuple(sorted(names)) or len(set(names)) != len(names):
                raise BenchmarkStoreError(
                    "Cross-validation metric summaries must have unique sorted names."
                )
            if (
                self.failure_fold is not None
                or self.failure_partition_sha256 is not None
                or self.failure is not None
            ):
                raise BenchmarkStoreError(
                    "Successful cross-validation entries cannot contain failure details."
                )
        else:
            if (
                self.rank is not None
                or self.primary_metric_mean is not None
                or self.primary_metric_standard_deviation is not None
                or self.metrics
            ):
                raise BenchmarkStoreError(
                    "Failed cross-validation entries cannot be ranked or summarized."
                )
            if (
                isinstance(self.failure_fold, bool)
                or not isinstance(self.failure_fold, int)
                or self.failure_fold <= 0
                or self.failure_partition_sha256 is None
                or self.failure is None
            ):
                raise BenchmarkStoreError(
                    "Failed cross-validation entries require fold, partition, and failure details."
                )
            _validate_sha256(
                self.failure_partition_sha256,
                "cross-validation failure partition_sha256",
            )
        if self.failure is not None and not isinstance(self.failure, RunFailure):
            raise BenchmarkStoreError("Cross-validation entry failure is invalid.")

    def to_dict(self) -> JsonObject:
        return {
            "estimator": self.estimator,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "status": self.status.value,
            "rank": self.rank,
            "primary_metric_mean": self.primary_metric_mean,
            "primary_metric_standard_deviation": self.primary_metric_standard_deviation,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "folds": [fold.to_dict() for fold in self.folds],
            "duration_seconds": self.duration_seconds,
            "failure_fold": self.failure_fold,
            "failure_partition_sha256": self.failure_partition_sha256,
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }

    @classmethod
    def from_object(cls, value: object) -> CrossValidationEntry:
        data = _object(value, "cross-validation entry")
        _keys(
            data,
            {
                "estimator",
                "parameters",
                "status",
                "rank",
                "primary_metric_mean",
                "primary_metric_standard_deviation",
                "metrics",
                "folds",
                "duration_seconds",
                "failure_fold",
                "failure_partition_sha256",
                "failure",
            },
            "cross-validation entry",
        )
        try:
            status = RunStatus(_string(data["status"], "cross-validation entry status"))
            parameters = tuple(
                RunParameter.from_object(item)
                for item in _array(data["parameters"], "cross-validation estimator parameters")
            )
            failure = None if data["failure"] is None else RunFailure.from_object(data["failure"])
        except (ValueError, RunStoreError) as error:
            raise BenchmarkStoreError(
                f"Cross-validation entry contains invalid run evidence: {error}"
            ) from error
        raw_rank = data["rank"]
        raw_mean = data["primary_metric_mean"]
        raw_deviation = data["primary_metric_standard_deviation"]
        raw_failure_fold = data["failure_fold"]
        raw_failure_partition = data["failure_partition_sha256"]
        return cls(
            estimator=_string(data["estimator"], "cross-validation entry estimator"),
            parameters=parameters,
            status=status,
            rank=None if raw_rank is None else _integer(raw_rank, "cross-validation entry rank"),
            primary_metric_mean=(
                None
                if raw_mean is None
                else _number(raw_mean, "cross-validation entry primary_metric_mean")
            ),
            primary_metric_standard_deviation=(
                None
                if raw_deviation is None
                else _number(
                    raw_deviation,
                    "cross-validation entry primary_metric_standard_deviation",
                )
            ),
            metrics=tuple(
                CrossValidationMetricSummary.from_object(item)
                for item in _array(data["metrics"], "cross-validation entry metrics")
            ),
            folds=tuple(
                CrossValidationFoldResult.from_object(item)
                for item in _array(data["folds"], "cross-validation entry folds")
            ),
            duration_seconds=_number(
                data["duration_seconds"], "cross-validation entry duration_seconds"
            ),
            failure_fold=(
                None
                if raw_failure_fold is None
                else _integer(raw_failure_fold, "cross-validation failure_fold")
            ),
            failure_partition_sha256=(
                None
                if raw_failure_partition is None
                else _string(
                    raw_failure_partition,
                    "cross-validation failure partition_sha256",
                )
            ),
            failure=failure,
        )


@dataclass(frozen=True, slots=True)
class CrossValidationManifest:
    """Versioned terminal aggregate for one shared stratified K-fold benchmark."""

    schema_version: int
    benchmark_id: str
    status: BenchmarkStatus
    started_at: str
    completed_at: str
    configuration: CrossValidationConfiguration
    dataset: DatasetSnapshot
    environment: EnvironmentSnapshot
    folds: tuple[CrossValidationFoldSnapshot, ...]
    fold_plan_sha256: str
    warnings: tuple[str, ...]
    entries: tuple[CrossValidationEntry, ...]

    def __post_init__(self) -> None:
        _integer(self.schema_version, "cross-validation schema_version")
        if self.schema_version != CROSS_VALIDATION_MANIFEST_SCHEMA_VERSION:
            raise BenchmarkStoreError(
                f"Unsupported cross-validation manifest schema version: {self.schema_version}."
            )
        _string(self.benchmark_id, "cross-validation benchmark_id")
        _validate_uuid(self.benchmark_id, "cross-validation benchmark_id")
        if not isinstance(self.status, BenchmarkStatus):
            raise BenchmarkStoreError("Cross-validation manifest status is invalid.")
        _string(self.started_at, "cross-validation started_at")
        _string(self.completed_at, "cross-validation completed_at")
        try:
            started = datetime.fromisoformat(self.started_at)
            completed = datetime.fromisoformat(self.completed_at)
        except ValueError as error:
            raise BenchmarkStoreError(
                "Cross-validation timestamps must use ISO 8601 format."
            ) from error
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            raise BenchmarkStoreError(
                "Cross-validation timestamps must be timezone-aware and ordered."
            )
        if not isinstance(self.configuration, CrossValidationConfiguration):
            raise BenchmarkStoreError("Cross-validation configuration is invalid.")
        if not isinstance(self.dataset, DatasetSnapshot):
            raise BenchmarkStoreError("Cross-validation dataset snapshot is invalid.")
        if not isinstance(self.environment, EnvironmentSnapshot):
            raise BenchmarkStoreError("Cross-validation environment snapshot is invalid.")
        if not isinstance(self.folds, tuple) or any(
            not isinstance(fold, CrossValidationFoldSnapshot) for fold in self.folds
        ):
            raise BenchmarkStoreError("Cross-validation fold plan must be a tuple.")
        if len(self.folds) != self.configuration.fold_count:
            raise BenchmarkStoreError(
                "Cross-validation fold plan does not match configured fold count."
            )
        if tuple(fold.fold_number for fold in self.folds) != tuple(range(1, len(self.folds) + 1)):
            raise BenchmarkStoreError(
                "Cross-validation fold plan must be a complete one-based sequence."
            )
        if len({fold.partition_sha256 for fold in self.folds}) != len(self.folds):
            raise BenchmarkStoreError(
                "Cross-validation fold partition fingerprints must be unique."
            )
        if (
            any(
                fold.train_rows + fold.validation_rows != self.dataset.row_count
                for fold in self.folds
            )
            or sum(fold.validation_rows for fold in self.folds) != self.dataset.row_count
        ):
            raise BenchmarkStoreError(
                "Cross-validation fold row counts do not cover the dataset exactly once."
            )
        _validate_sha256(self.fold_plan_sha256, "cross-validation fold_plan_sha256")
        if self.fold_plan_sha256 != fold_plan_sha256(self.folds):
            raise BenchmarkStoreError(
                "Cross-validation fold plan fingerprint does not match its folds."
            )
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(message, str) or not message.strip() for message in self.warnings
        ):
            raise BenchmarkStoreError("Cross-validation warnings must be non-blank strings.")
        if len(set(self.warnings)) != len(self.warnings):
            raise BenchmarkStoreError("Cross-validation warnings must be unique.")
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, CrossValidationEntry) for entry in self.entries
        ):
            raise BenchmarkStoreError("Cross-validation entries must be a tuple.")
        if tuple(entry.estimator for entry in self.entries) != self.configuration.estimators:
            raise BenchmarkStoreError(
                "Cross-validation entries must follow configured estimator order."
            )

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
                "Cross-validation aggregate status does not match estimator outcomes."
            )
        ranks = tuple(entry.rank for entry in successful)
        if set(ranks) != set(range(1, len(successful) + 1)):
            raise BenchmarkStoreError(
                "Cross-validation ranks must form a complete one-based sequence."
            )
        for entry in self.entries:
            if entry.status is RunStatus.SUCCEEDED:
                if len(entry.folds) != self.configuration.fold_count:
                    raise BenchmarkStoreError(
                        "Successful cross-validation entries must complete every fold."
                    )
                metric_by_name = {metric.name: metric for metric in entry.metrics}
                if tuple(metric_by_name) != tuple(sorted(CLASSIFICATION_METRICS)):
                    raise BenchmarkStoreError(
                        "Cross-validation entry must summarize every classification metric."
                    )
                primary = metric_by_name.get(self.configuration.primary_metric)
                if primary is None:
                    raise BenchmarkStoreError(
                        "Cross-validation entry is missing the primary metric summary."
                    )
                if (
                    entry.primary_metric_mean != primary.mean
                    or entry.primary_metric_standard_deviation != primary.standard_deviation
                ):
                    raise BenchmarkStoreError(
                        "Cross-validation primary metric aggregate is inconsistent."
                    )
                for summary in entry.metrics:
                    observed_values: list[float] = []
                    for fold in entry.folds:
                        matching = tuple(
                            metric.value for metric in fold.metrics if metric.name == summary.name
                        )
                        if len(matching) != 1:
                            raise BenchmarkStoreError(
                                "Cross-validation fold is missing a summarized metric."
                            )
                        observed_values.append(matching[0])
                    observed = tuple(observed_values)
                    if observed != summary.fold_values:
                        raise BenchmarkStoreError(
                            "Cross-validation metric summary does not match fold evidence."
                        )
            else:
                if entry.failure_fold is None or entry.failure_fold > len(self.folds):
                    raise BenchmarkStoreError(
                        "Cross-validation failure fold is outside the shared plan."
                    )
                expected_completed = tuple(range(1, entry.failure_fold))
                if tuple(fold.fold_number for fold in entry.folds) != expected_completed:
                    raise BenchmarkStoreError(
                        "Failed cross-validation entry has inconsistent completed folds."
                    )
                expected_partition = self.folds[entry.failure_fold - 1].partition_sha256
                if entry.failure_partition_sha256 != expected_partition:
                    raise BenchmarkStoreError(
                        "Cross-validation failure partition does not match the shared plan."
                    )
        primary_by_estimator = {
            entry.estimator: next(
                metric
                for metric in entry.metrics
                if metric.name == self.configuration.primary_metric
            )
            for entry in successful
        }
        ordered = sorted(
            successful,
            key=lambda entry: (
                -primary_by_estimator[entry.estimator].mean
                if primary_by_estimator[entry.estimator].higher_is_better
                else primary_by_estimator[entry.estimator].mean,
                primary_by_estimator[entry.estimator].standard_deviation,
                entry.estimator,
            ),
        )
        if any(entry.rank != rank for rank, entry in enumerate(ordered, start=1)):
            raise BenchmarkStoreError(
                "Cross-validation ranks do not match the declared metric aggregates."
            )

    @property
    def winner(self) -> CrossValidationEntry | None:
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
            "environment": self.environment.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "fold_plan_sha256": self.fold_plan_sha256,
            "warnings": list(self.warnings),
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, content: str) -> CrossValidationManifest:
        try:
            raw: object = json.loads(content)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise BenchmarkStoreError(
                f"Cross-validation manifest is not valid JSON: {error}"
            ) from error
        data = _object(raw, "cross-validation root")
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
                "environment",
                "folds",
                "fold_plan_sha256",
                "warnings",
                "entries",
            },
            "cross-validation root",
        )
        try:
            status = BenchmarkStatus(_string(data["status"], "cross-validation status"))
            dataset = DatasetSnapshot.from_object(data["dataset"])
            environment = EnvironmentSnapshot.from_object(data["environment"])
        except (ValueError, RunStoreError) as error:
            raise BenchmarkStoreError(
                f"Cross-validation manifest contains invalid shared evidence: {error}"
            ) from error
        return cls(
            schema_version=_integer(data["schema_version"], "cross-validation schema_version"),
            benchmark_id=_string(data["benchmark_id"], "cross-validation benchmark_id"),
            status=status,
            started_at=_string(data["started_at"], "cross-validation started_at"),
            completed_at=_string(data["completed_at"], "cross-validation completed_at"),
            configuration=CrossValidationConfiguration.from_object(data["configuration"]),
            dataset=dataset,
            environment=environment,
            folds=tuple(
                CrossValidationFoldSnapshot.from_object(item)
                for item in _array(data["folds"], "cross-validation folds")
            ),
            fold_plan_sha256=_string(data["fold_plan_sha256"], "cross-validation fold_plan_sha256"),
            warnings=_string_tuple(data["warnings"], "cross-validation warnings"),
            entries=tuple(
                CrossValidationEntry.from_object(item)
                for item in _array(data["entries"], "cross-validation entries")
            ),
        )


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Persisted cross-validation benchmark selection evidence."""

    manifest: CrossValidationManifest
    manifest_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, CrossValidationManifest):
            raise BenchmarkError("Cross-validation result manifest is invalid.")
        if not isinstance(self.manifest_path, Path):
            raise BenchmarkError("Cross-validation result manifest_path must be a pathlib.Path.")
