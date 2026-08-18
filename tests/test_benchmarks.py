"""Tests for local classification benchmark orchestration and persistence."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from mlforge.benchmarks import (
    BenchmarkConfig,
    BenchmarkManifest,
    BenchmarkStatus,
    LocalBenchmarkStore,
    benchmark,
)
from mlforge.datasets import LoadedDataset, load_csv
from mlforge.errors import (
    BenchmarkFailedError,
    BenchmarkStoreError,
    ConfigurationError,
)
from mlforge.runs import LocalRunStore, RunStatus
from mlforge.training import (
    DUMMY_CLASSIFIER,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_CLASSIFIER,
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
    path = tmp_path / "benchmark.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_csv(path, target="target")


class _FailingClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, features: object, target: object) -> _FailingClassifier:
        raise ValueError("intentional estimator failure")


def test_default_benchmark_records_fair_ranked_runs(tmp_path: Path) -> None:
    """Default baselines should share a partition and produce one immutable leaderboard."""
    dataset = _classification_dataset(tmp_path)
    run_store = LocalRunStore(tmp_path / "runs")
    benchmark_store = LocalBenchmarkStore(tmp_path / "benchmarks")

    result = benchmark(
        dataset,
        BenchmarkConfig(primary_metric="f1_macro"),
        run_store=run_store,
        benchmark_store=benchmark_store,
    )

    assert result.manifest.status is BenchmarkStatus.SUCCEEDED
    assert result.manifest.configuration.estimators == (
        DUMMY_CLASSIFIER,
        LOGISTIC_REGRESSION,
        RANDOM_FOREST_CLASSIFIER,
    )
    assert {entry.rank for entry in result.manifest.entries} == {1, 2, 3}
    assert all(entry.primary_metric_value is not None for entry in result.manifest.entries)
    assert result.manifest.winner is not None
    assert result.winner.manifest.run_id == result.manifest.winner.run_id
    assert benchmark_store.read(result.manifest.benchmark_id) == result.manifest
    assert BenchmarkManifest.from_json(result.manifest.to_json()) == result.manifest
    assert len(run_store.list_manifests()) == 3

    split_hashes = {
        manifest.split.partition_sha256
        for manifest in result.run_manifests
        if manifest.split is not None
    }
    assert len(split_hashes) == 1
    assert result.manifest.split is not None
    assert result.manifest.split.partition_sha256 in split_hashes


def test_benchmark_configuration_rejects_ambiguous_comparisons() -> None:
    """A benchmark should declare a unique multi-model classification comparison."""
    with pytest.raises(ConfigurationError, match="at least two"):
        BenchmarkConfig(estimators=(LOGISTIC_REGRESSION,))
    with pytest.raises(ConfigurationError, match="unique"):
        BenchmarkConfig(estimators=(LOGISTIC_REGRESSION, LOGISTIC_REGRESSION))
    with pytest.raises(ConfigurationError, match="classification"):
        BenchmarkConfig(estimators=(LOGISTIC_REGRESSION, RIDGE_REGRESSION))
    with pytest.raises(ConfigurationError, match="metric"):
        BenchmarkConfig(primary_metric="unknown")


def test_partial_benchmark_preserves_failed_estimator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One estimator failure should remain visible without discarding successful runs."""

    def estimator_factory(config: TrainingConfig) -> BaseEstimator:
        if config.estimator == RANDOM_FOREST_CLASSIFIER:
            return _FailingClassifier()
        return create_estimator(config)

    monkeypatch.setattr("mlforge.training.service.create_estimator", estimator_factory)
    run_store = LocalRunStore(tmp_path / "runs")
    result = benchmark(
        _classification_dataset(tmp_path),
        BenchmarkConfig(),
        run_store=run_store,
        benchmark_store=LocalBenchmarkStore(tmp_path / "benchmarks"),
    )

    assert result.manifest.status is BenchmarkStatus.PARTIAL
    assert len(result.training_results) == 2
    failed = next(entry for entry in result.manifest.entries if entry.status is RunStatus.FAILED)
    assert failed.estimator == RANDOM_FOREST_CLASSIFIER
    assert failed.rank is None
    assert failed.primary_metric_value is None
    assert failed.failure is not None
    assert failed.failure.message == "intentional estimator failure"
    assert run_store.read(failed.run_id).status is RunStatus.FAILED


def test_complete_benchmark_failure_is_recorded_before_raising(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If no model succeeds, the aggregate and every failed run should still be inspectable."""

    def estimator_factory(config: TrainingConfig) -> BaseEstimator:
        return _FailingClassifier()

    monkeypatch.setattr("mlforge.training.service.create_estimator", estimator_factory)
    run_store = LocalRunStore(tmp_path / "runs")
    benchmark_store = LocalBenchmarkStore(tmp_path / "benchmarks")

    with pytest.raises(BenchmarkFailedError) as captured:
        benchmark(
            _classification_dataset(tmp_path),
            BenchmarkConfig(),
            run_store=run_store,
            benchmark_store=benchmark_store,
        )

    error = captured.value
    manifest = benchmark_store.read(error.benchmark_id)
    assert Path(error.manifest_path).is_file()
    assert manifest.status is BenchmarkStatus.FAILED
    assert manifest.winner is None
    assert manifest.split is None
    assert all(entry.failure is not None for entry in manifest.entries)
    assert {item.status for item in run_store.list_manifests()} == {RunStatus.FAILED}


def test_benchmark_store_is_immutable_and_path_safe(tmp_path: Path) -> None:
    """Aggregate records should be create-only and reject unsafe identifiers."""
    benchmark_store = LocalBenchmarkStore(tmp_path / "benchmarks")
    result = benchmark(
        _classification_dataset(tmp_path),
        BenchmarkConfig(estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION)),
        run_store=LocalRunStore(tmp_path / "runs"),
        benchmark_store=benchmark_store,
    )

    with pytest.raises(BenchmarkStoreError, match="immutable"):
        benchmark_store.write(result.manifest)
    with pytest.raises(BenchmarkStoreError, match="canonical UUID"):
        benchmark_store.read("../escape")
    assert not list((tmp_path / "benchmarks").glob("*.tmp"))


def test_benchmark_store_rejects_corruption_and_cleans_atomic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted aggregate JSON and interrupted publication must fail closed."""
    result = benchmark(
        _classification_dataset(tmp_path),
        BenchmarkConfig(estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION)),
        run_store=LocalRunStore(tmp_path / "runs"),
        benchmark_store=LocalBenchmarkStore(tmp_path / "benchmarks"),
    )
    corrupt_store = LocalBenchmarkStore(tmp_path / "corrupt")
    corrupt_store.root.mkdir()
    corrupt_id = str(uuid4())
    (corrupt_store.root / f"{corrupt_id}.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BenchmarkStoreError, match="invalid fields"):
        corrupt_store.read(corrupt_id)

    interrupted_store = LocalBenchmarkStore(tmp_path / "interrupted")

    def fail_link(source: object, destination: object) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr("mlforge.benchmarks.store.os.link", fail_link)
    with pytest.raises(BenchmarkStoreError, match="atomically write"):
        interrupted_store.write(result.manifest)
    assert not list(interrupted_store.root.glob("*"))


def test_benchmark_manifest_rejects_semantically_false_ranking_and_split(
    tmp_path: Path,
) -> None:
    """Strict aggregate reads must reject plausible but internally false evidence."""
    result = benchmark(
        _classification_dataset(tmp_path),
        BenchmarkConfig(estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION)),
        run_store=LocalRunStore(tmp_path / "runs"),
        benchmark_store=LocalBenchmarkStore(tmp_path / "benchmarks"),
    )
    wrong_ranking = result.manifest.to_dict()
    entries = wrong_ranking["entries"]
    assert isinstance(entries, list)
    first_entry = entries[0]
    second_entry = entries[1]
    assert isinstance(first_entry, dict)
    assert isinstance(second_entry, dict)
    first_entry["rank"], second_entry["rank"] = second_entry["rank"], first_entry["rank"]
    with pytest.raises(BenchmarkStoreError, match="ranks do not match"):
        BenchmarkManifest.from_json(json.dumps(wrong_ranking))

    wrong_split = result.manifest.to_dict()
    split = wrong_split["split"]
    assert isinstance(split, dict)
    train_rows = split["train_rows"]
    assert isinstance(train_rows, int)
    split["train_rows"] = train_rows + 1
    with pytest.raises(BenchmarkStoreError, match="dimensions do not match"):
        BenchmarkManifest.from_json(json.dumps(wrong_split))
