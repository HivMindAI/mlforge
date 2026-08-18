"""Leakage-safe shared-fold classification benchmark application service."""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from sklearn.base import BaseEstimator

from mlforge.benchmarks.cross_validation_store import LocalCrossValidationStore
from mlforge.benchmarks.cross_validation_types import (
    CROSS_VALIDATION_MANIFEST_SCHEMA_VERSION,
    CrossValidationConfig,
    CrossValidationConfiguration,
    CrossValidationEntry,
    CrossValidationFoldResult,
    CrossValidationFoldSnapshot,
    CrossValidationManifest,
    CrossValidationMetricSummary,
    CrossValidationResult,
    fold_plan_sha256,
)
from mlforge.benchmarks.types import BenchmarkStatus
from mlforge.datasets import LoadedDataset, profile_dataset
from mlforge.errors import BenchmarkError, BenchmarkFailedError, MLForgeError
from mlforge.pipelines import (
    DatasetSplit,
    SplitConfig,
    TaskType,
    build_model_pipeline,
    split_classification_folds,
    split_partition_sha256,
)
from mlforge.runs import (
    DatasetSnapshot,
    EnvironmentSnapshot,
    RunFailure,
    RunParameter,
    RunStatus,
)
from mlforge.runs.types import JsonPrimitive
from mlforge.training import TrainingConfig, evaluate_predictions
from mlforge.training.estimators import create_estimator


@dataclass(frozen=True, slots=True)
class _EstimatorAttempt:
    estimator: str
    parameters: tuple[RunParameter, ...]
    folds: tuple[CrossValidationFoldResult, ...]
    metrics: tuple[CrossValidationMetricSummary, ...]
    duration_seconds: float
    failure_fold: int | None
    failure_partition_sha256: str | None
    failure: RunFailure | None

    @property
    def succeeded(self) -> bool:
        return self.failure is None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _dataset_snapshot(dataset: LoadedDataset) -> DatasetSnapshot:
    metadata = dataset.metadata
    return DatasetSnapshot(
        source_path=str(metadata.source_path),
        sha256=metadata.sha256,
        file_size_bytes=metadata.file_size_bytes,
        row_count=metadata.row_count,
        column_count=metadata.column_count,
        target=metadata.target,
        encoding=metadata.encoding,
        delimiter=metadata.delimiter,
    )


def _environment_snapshot() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        python=sys.version.split()[0],
        mlforge=version("hivmind-mlforge"),
        pandas=version("pandas"),
        numpy=version("numpy"),
        scipy=version("scipy"),
        scikit_learn=version("scikit-learn"),
    )


def _parameter_value(value: object, *, name: str) -> JsonPrimitive:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise BenchmarkError(
        f"Estimator parameter {name!r} is not a reproducible JSON primitive: "
        f"{type(value).__name__}."
    )


def _parameters(estimator: BaseEstimator) -> tuple[RunParameter, ...]:
    return tuple(
        RunParameter(name=name, value=_parameter_value(value, name=name))
        for name, value in sorted(estimator.get_params(deep=False).items())
    )


def _failure_message(error: Exception) -> str:
    compact = " ".join(str(error).split())
    return compact[:2_000] if compact else "Cross-validation fold failed without an error message."


def _warning_messages(captured: list[warnings.WarningMessage]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            f"{item.category.__name__}: {' '.join(str(item.message).split())}"
            for item in captured
            if str(item.message).strip()
        )
    )


def _summaries(
    folds: tuple[CrossValidationFoldResult, ...],
) -> tuple[CrossValidationMetricSummary, ...]:
    if not folds:
        return ()
    first_metrics = folds[0].metrics
    summaries: list[CrossValidationMetricSummary] = []
    for reference in first_metrics:
        values = tuple(
            next(metric.value for metric in fold.metrics if metric.name == reference.name)
            for fold in folds
        )
        directions = {
            metric.higher_is_better
            for fold in folds
            for metric in fold.metrics
            if metric.name == reference.name
        }
        if directions != {reference.higher_is_better}:
            raise BenchmarkError(f"Metric direction changed across folds for {reference.name!r}.")
        summaries.append(
            CrossValidationMetricSummary(
                name=reference.name,
                fold_values=values,
                mean=fmean(values),
                standard_deviation=pstdev(values),
                higher_is_better=reference.higher_is_better,
            )
        )
    return tuple(summaries)


def _primary(
    attempt: _EstimatorAttempt,
    metric: str,
) -> CrossValidationMetricSummary:
    for summary in attempt.metrics:
        if summary.name == metric:
            return summary
    raise BenchmarkError(
        f"Estimator {attempt.estimator!r} did not produce primary metric {metric!r}."
    )


def _run_estimator(
    config: CrossValidationConfig,
    estimator_name: str,
    folds: tuple[DatasetSplit, ...],
    fold_snapshots: tuple[CrossValidationFoldSnapshot, ...],
) -> _EstimatorAttempt:
    attempt_started = perf_counter()
    completed: list[CrossValidationFoldResult] = []
    parameters: tuple[RunParameter, ...] = ()
    try:
        estimator = create_estimator(
            TrainingConfig(
                task=TaskType.CLASSIFICATION,
                estimator=estimator_name,
                split=SplitConfig(random_seed=config.split.random_seed),
                preprocessing=config.preprocessing,
                feature_overrides=config.feature_overrides,
            )
        )
        parameters = _parameters(estimator)
    except (MLForgeError, ValueError, TypeError, OverflowError) as error:
        return _EstimatorAttempt(
            estimator=estimator_name,
            parameters=parameters,
            folds=(),
            metrics=(),
            duration_seconds=max(0.0, perf_counter() - attempt_started),
            failure_fold=1,
            failure_partition_sha256=fold_snapshots[0].partition_sha256,
            failure=RunFailure(
                error_type=type(error).__name__,
                message=_failure_message(error),
            ),
        )

    for fold_number, split in enumerate(folds, start=1):
        fold_started = perf_counter()
        captured_warnings: list[warnings.WarningMessage] = []
        try:
            pipeline = build_model_pipeline(
                split,
                estimator,
                config=config.preprocessing,
                overrides=config.feature_overrides,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    pipeline.fit(split.train_features, split.train_target)
                    predictions = cast(Sequence[Any], pipeline.predict(split.validation_features))
                    metrics = evaluate_predictions(
                        task=TaskType.CLASSIFICATION,
                        actual=split.validation_target,
                        predicted=predictions,
                    )
                finally:
                    captured_warnings = list(caught)
        except (MLForgeError, ValueError, TypeError, OverflowError) as error:
            return _EstimatorAttempt(
                estimator=estimator_name,
                parameters=parameters,
                folds=tuple(completed),
                metrics=(),
                duration_seconds=max(0.0, perf_counter() - attempt_started),
                failure_fold=fold_number,
                failure_partition_sha256=fold_snapshots[fold_number - 1].partition_sha256,
                failure=RunFailure(
                    error_type=type(error).__name__,
                    message=_failure_message(error),
                ),
            )
        completed.append(
            CrossValidationFoldResult(
                fold_number=fold_number,
                metrics=metrics,
                duration_seconds=max(0.0, perf_counter() - fold_started),
                warnings=_warning_messages(captured_warnings),
            )
        )

    completed_folds = tuple(completed)
    return _EstimatorAttempt(
        estimator=estimator_name,
        parameters=parameters,
        folds=completed_folds,
        metrics=_summaries(completed_folds),
        duration_seconds=max(0.0, perf_counter() - attempt_started),
        failure_fold=None,
        failure_partition_sha256=None,
        failure=None,
    )


def cross_validate_benchmark(
    dataset: LoadedDataset,
    config: CrossValidationConfig,
    *,
    store: LocalCrossValidationStore | None = None,
) -> CrossValidationResult:
    """Run and persist one shared stratified K-fold classification benchmark."""
    if not isinstance(dataset, LoadedDataset):
        raise BenchmarkError("dataset must be a LoadedDataset value.")
    if not isinstance(config, CrossValidationConfig):
        raise BenchmarkError("config must be a CrossValidationConfig value.")
    destination = store or LocalCrossValidationStore(Path("mlbenchmarks") / "cross-validation")
    if not isinstance(destination, LocalCrossValidationStore):
        raise BenchmarkError("store must be a LocalCrossValidationStore value.")

    splits = split_classification_folds(dataset, config=config.split)
    fold_snapshots = tuple(
        CrossValidationFoldSnapshot(
            fold_number=index,
            train_rows=len(split.train_features),
            validation_rows=len(split.validation_features),
            partition_sha256=split_partition_sha256(split),
        )
        for index, split in enumerate(splits, start=1)
    )
    profile = profile_dataset(dataset)
    benchmark_id = str(uuid4())
    started_at = _now()
    attempts = tuple(
        _run_estimator(
            config,
            estimator,
            splits,
            fold_snapshots,
        )
        for estimator in config.estimators
    )
    successful = tuple(attempt for attempt in attempts if attempt.succeeded)
    primary_by_estimator = {
        attempt.estimator: _primary(attempt, config.primary_metric) for attempt in successful
    }
    ordered = sorted(
        successful,
        key=lambda attempt: (
            -primary_by_estimator[attempt.estimator].mean
            if primary_by_estimator[attempt.estimator].higher_is_better
            else primary_by_estimator[attempt.estimator].mean,
            primary_by_estimator[attempt.estimator].standard_deviation,
            attempt.estimator,
        ),
    )
    ranks = {attempt.estimator: rank for rank, attempt in enumerate(ordered, start=1)}
    entries = tuple(
        CrossValidationEntry(
            estimator=attempt.estimator,
            parameters=attempt.parameters,
            status=(RunStatus.SUCCEEDED if attempt.succeeded else RunStatus.FAILED),
            rank=ranks.get(attempt.estimator),
            primary_metric_mean=(
                primary_by_estimator[attempt.estimator].mean if attempt.succeeded else None
            ),
            primary_metric_standard_deviation=(
                primary_by_estimator[attempt.estimator].standard_deviation
                if attempt.succeeded
                else None
            ),
            metrics=attempt.metrics,
            folds=attempt.folds,
            duration_seconds=attempt.duration_seconds,
            failure_fold=attempt.failure_fold,
            failure_partition_sha256=attempt.failure_partition_sha256,
            failure=attempt.failure,
        )
        for attempt in attempts
    )
    status = (
        BenchmarkStatus.FAILED
        if not successful
        else BenchmarkStatus.SUCCEEDED
        if len(successful) == len(attempts)
        else BenchmarkStatus.PARTIAL
    )
    manifest = CrossValidationManifest(
        schema_version=CROSS_VALIDATION_MANIFEST_SCHEMA_VERSION,
        benchmark_id=benchmark_id,
        status=status,
        started_at=started_at,
        completed_at=_now(),
        configuration=CrossValidationConfiguration.from_config(config),
        dataset=_dataset_snapshot(dataset),
        environment=_environment_snapshot(),
        folds=fold_snapshots,
        fold_plan_sha256=fold_plan_sha256(fold_snapshots),
        warnings=profile.warnings,
        entries=entries,
    )
    manifest_path = destination.write(manifest)
    if status is BenchmarkStatus.FAILED:
        raise BenchmarkFailedError(
            f"Cross-validation benchmark {benchmark_id} failed because every estimator failed.",
            benchmark_id=benchmark_id,
            manifest_path=str(manifest_path),
        )
    return CrossValidationResult(manifest=manifest, manifest_path=manifest_path)
