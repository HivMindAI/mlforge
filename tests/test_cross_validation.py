"""Tests for leakage-safe classification cross-validation and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

from mlforge.benchmarks import (
    BenchmarkStatus,
    CrossValidationConfig,
    CrossValidationManifest,
    LocalCrossValidationStore,
    cross_validate_benchmark,
)
from mlforge.datasets import LoadedDataset, load_csv
from mlforge.errors import BenchmarkFailedError, BenchmarkStoreError
from mlforge.pipelines import CrossValidationSplitConfig, TaskType
from mlforge.runs import MetricValue, RunStatus
from mlforge.training import (
    DUMMY_CLASSIFIER,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_CLASSIFIER,
    RANDOM_FOREST_REGRESSOR,
    RIDGE_REGRESSION,
    TrainingConfig,
)
from mlforge.training.estimators import create_estimator


def _classification_dataset(tmp_path: Path) -> LoadedDataset:
    rows = ["value,segment,target"]
    for index in range(60):
        segment = ("north", "south", "east")[index % 3]
        target = "yes" if index % 4 in {0, 1} else "no"
        rows.append(f"{index},{segment},{target}")
    path = tmp_path / "cross-validation.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_csv(path, target="target")


def _regression_dataset(tmp_path: Path) -> LoadedDataset:
    rows = ["value,segment,target"]
    for index in range(60):
        segment = ("north", "south", "east")[index % 3]
        segment_effect = {"north": 2.0, "south": -3.0, "east": 5.0}[segment]
        rows.append(f"{index},{segment},{index * 4.5 + segment_effect}")
    path = tmp_path / "regression-cross-validation.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_csv(path, target="target")


class _FailingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, features: object, target: object) -> _FailingClassifier:
        raise ValueError("intentional fold failure")


def test_cross_validation_records_shared_folds_ranked_aggregates_and_round_trip(
    tmp_path: Path,
) -> None:
    """Every estimator should use one fold plan with strict aggregate evidence."""
    store = LocalCrossValidationStore(tmp_path / "benchmarks")

    result = cross_validate_benchmark(
        _classification_dataset(tmp_path),
        CrossValidationConfig(
            split=CrossValidationSplitConfig(fold_count=3, random_seed=9),
        ),
        store=store,
    )

    manifest = result.manifest
    assert manifest.status is BenchmarkStatus.SUCCEEDED
    assert manifest.configuration.fold_count == 3
    assert len(manifest.folds) == 3
    assert len(manifest.fold_plan_sha256) == 64
    assert {entry.rank for entry in manifest.entries} == {1, 2, 3}
    assert manifest.winner is not None
    for entry in manifest.entries:
        assert entry.status is RunStatus.SUCCEEDED
        assert tuple(fold.fold_number for fold in entry.folds) == (1, 2, 3)
        assert all(len(metric.fold_values) == 3 for metric in entry.metrics)
        primary = next(
            metric
            for metric in entry.metrics
            if metric.name == manifest.configuration.primary_metric
        )
        assert entry.primary_metric_mean == primary.mean
        assert entry.primary_metric_standard_deviation == primary.standard_deviation
    assert store.read(manifest.benchmark_id) == manifest
    assert CrossValidationManifest.from_json(manifest.to_json()) == manifest


def test_cross_validation_is_deterministic_except_for_identity_and_timing(tmp_path: Path) -> None:
    """A fixed seed should reproduce fold identity and model scores."""
    dataset = _classification_dataset(tmp_path)
    config = CrossValidationConfig(
        estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION),
        split=CrossValidationSplitConfig(fold_count=3, random_seed=23),
    )

    first = cross_validate_benchmark(
        dataset,
        config,
        store=LocalCrossValidationStore(tmp_path / "first"),
    ).manifest
    second = cross_validate_benchmark(
        dataset,
        config,
        store=LocalCrossValidationStore(tmp_path / "second"),
    ).manifest

    assert first.fold_plan_sha256 == second.fold_plan_sha256
    assert first.folds == second.folds
    assert tuple((entry.estimator, entry.metrics, entry.rank) for entry in first.entries) == tuple(
        (entry.estimator, entry.metrics, entry.rank) for entry in second.entries
    )


def test_regression_cross_validation_ranks_lower_rmse_and_round_trips(tmp_path: Path) -> None:
    """Regression comparison should preserve task metrics and lower-is-better ranking evidence."""
    result = cross_validate_benchmark(
        _regression_dataset(tmp_path),
        CrossValidationConfig(
            task=TaskType.REGRESSION,
            estimators=(RIDGE_REGRESSION, RANDOM_FOREST_REGRESSOR),
            primary_metric="root_mean_squared_error",
            split=CrossValidationSplitConfig(fold_count=3, random_seed=31),
        ),
        store=LocalCrossValidationStore(tmp_path / "regression-benchmarks"),
    )

    manifest = result.manifest
    assert manifest.configuration.task == "regression"
    assert manifest.configuration.primary_metric == "root_mean_squared_error"
    assert manifest.winner is not None
    assert manifest.winner.primary_metric_mean == min(
        entry.primary_metric_mean
        for entry in manifest.entries
        if entry.primary_metric_mean is not None
    )
    for entry in manifest.entries:
        directions = {metric.name: metric.higher_is_better for metric in entry.metrics}
        assert directions == {
            "mean_absolute_error": False,
            "r2": True,
            "root_mean_squared_error": False,
        }
        assert all(fold.metrics[0].name == "mean_absolute_error" for fold in entry.folds)
    assert CrossValidationManifest.from_json(manifest.to_json()) == manifest


def test_legacy_classification_cross_validation_manifest_remains_readable(
    tmp_path: Path,
) -> None:
    """Schema v2 readers should preserve immutable v1 classification evidence."""
    result = cross_validate_benchmark(
        _classification_dataset(tmp_path),
        CrossValidationConfig(
            estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION),
            split=CrossValidationSplitConfig(fold_count=3),
        ),
        store=LocalCrossValidationStore(tmp_path / "benchmarks"),
    )
    legacy = result.manifest.to_dict()
    legacy["schema_version"] = 1

    restored = CrossValidationManifest.from_json(json.dumps(legacy))

    assert restored.schema_version == 1
    assert restored.configuration.task == "classification"


def test_ranking_uses_mean_then_stability_then_estimator_identifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact mean ties should prefer stability before the deterministic name fallback."""
    scheduled_values = iter(
        (
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.4,
            0.5,
            0.6,
        )
    )

    def scheduled_metrics(**kwargs: object) -> tuple[MetricValue, ...]:
        value = next(scheduled_values)
        return tuple(
            MetricValue(name=name, value=value, higher_is_better=True)
            for name in (
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
                "precision_macro",
                "recall_macro",
            )
        )

    monkeypatch.setattr(
        "mlforge.benchmarks.cross_validation_service.evaluate_predictions",
        scheduled_metrics,
    )
    manifest = cross_validate_benchmark(
        _classification_dataset(tmp_path),
        CrossValidationConfig(
            estimators=(
                LOGISTIC_REGRESSION,
                DUMMY_CLASSIFIER,
                RANDOM_FOREST_CLASSIFIER,
            ),
            split=CrossValidationSplitConfig(fold_count=3),
        ),
        store=LocalCrossValidationStore(tmp_path / "benchmarks"),
    ).manifest

    ranks = {entry.estimator: entry.rank for entry in manifest.entries}
    assert ranks == {
        DUMMY_CLASSIFIER: 1,
        LOGISTIC_REGRESSION: 2,
        RANDOM_FOREST_CLASSIFIER: 3,
    }


def test_preprocessing_is_fit_only_on_each_training_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numeric preprocessing must never learn from the full dataset or validation rows."""
    observed_fit_rows: list[int] = []
    original_fit = StandardScaler.fit

    def recording_fit(
        scaler: StandardScaler,
        features: np.ndarray,
        target: Any = None,
        sample_weight: Any = None,
    ) -> StandardScaler:
        observed_fit_rows.append(features.shape[0])
        return original_fit(scaler, features, target, sample_weight=sample_weight)

    monkeypatch.setattr(StandardScaler, "fit", recording_fit)
    cross_validate_benchmark(
        _classification_dataset(tmp_path),
        CrossValidationConfig(
            estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION),
            split=CrossValidationSplitConfig(fold_count=3),
        ),
        store=LocalCrossValidationStore(tmp_path / "benchmarks"),
    )

    assert observed_fit_rows == [40] * 6
    assert 60 not in observed_fit_rows


def test_partial_cross_validation_preserves_failure_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken estimator should not discard successful fold evidence from its peers."""

    def estimator_factory(config: TrainingConfig) -> BaseEstimator:
        if config.estimator == RANDOM_FOREST_CLASSIFIER:
            return _FailingClassifier()
        return create_estimator(config)

    monkeypatch.setattr(
        "mlforge.benchmarks.cross_validation_service.create_estimator",
        estimator_factory,
    )
    store = LocalCrossValidationStore(tmp_path / "benchmarks")
    result = cross_validate_benchmark(
        _classification_dataset(tmp_path),
        CrossValidationConfig(split=CrossValidationSplitConfig(fold_count=3)),
        store=store,
    )

    assert result.manifest.status is BenchmarkStatus.PARTIAL
    failed = next(entry for entry in result.manifest.entries if entry.status is RunStatus.FAILED)
    assert failed.estimator == RANDOM_FOREST_CLASSIFIER
    assert failed.failure_fold == 1
    assert failed.failure_partition_sha256 == result.manifest.folds[0].partition_sha256
    assert failed.failure is not None
    assert failed.failure.message == "intentional fold failure"
    assert failed.rank is None
    assert len(tuple(entry for entry in result.manifest.entries if entry.rank is not None)) == 2


def test_complete_cross_validation_failure_is_persisted_before_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An all-failed comparison should remain inspectable as terminal evidence."""

    def estimator_factory(config: TrainingConfig) -> BaseEstimator:
        return _FailingClassifier()

    monkeypatch.setattr(
        "mlforge.benchmarks.cross_validation_service.create_estimator",
        estimator_factory,
    )
    store = LocalCrossValidationStore(tmp_path / "benchmarks")

    with pytest.raises(BenchmarkFailedError) as captured:
        cross_validate_benchmark(
            _classification_dataset(tmp_path),
            CrossValidationConfig(
                estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION),
                split=CrossValidationSplitConfig(fold_count=3),
            ),
            store=store,
        )

    manifest = store.read(captured.value.benchmark_id)
    assert Path(captured.value.manifest_path).is_file()
    assert manifest.status is BenchmarkStatus.FAILED
    assert manifest.winner is None
    assert all(entry.failure_fold == 1 for entry in manifest.entries)


def test_cross_validation_store_is_immutable_path_safe_and_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-validation evidence should be create-only and fail closed on unsafe input."""
    store = LocalCrossValidationStore(tmp_path / "benchmarks")
    result = cross_validate_benchmark(
        _classification_dataset(tmp_path),
        CrossValidationConfig(
            estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION),
            split=CrossValidationSplitConfig(fold_count=3),
        ),
        store=store,
    )

    with pytest.raises(BenchmarkStoreError, match="immutable"):
        store.write(result.manifest)
    with pytest.raises(BenchmarkStoreError, match="canonical UUID"):
        store.read("../escape")

    corrupt_id = str(uuid4())
    (store.root / f"{corrupt_id}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkStoreError, match="invalid fields"):
        store.read(corrupt_id)
    assert not list(store.root.glob("*.tmp"))

    interrupted_store = LocalCrossValidationStore(tmp_path / "interrupted")

    def fail_link(source: object, destination: object) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr("mlforge.benchmarks.cross_validation_store.os.link", fail_link)
    with pytest.raises(BenchmarkStoreError, match="atomically write"):
        interrupted_store.write(result.manifest)
    assert not list(interrupted_store.root.glob("*"))
