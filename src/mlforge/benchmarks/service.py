"""Application service for running and recording a local classification benchmark."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from mlforge.benchmarks.store import LocalBenchmarkStore
from mlforge.benchmarks.types import (
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    BenchmarkConfig,
    BenchmarkConfiguration,
    BenchmarkEntry,
    BenchmarkManifest,
    BenchmarkResult,
    BenchmarkStatus,
)
from mlforge.datasets import LoadedDataset
from mlforge.errors import BenchmarkError, BenchmarkFailedError, TrainingFailedError
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore, MetricValue, RunManifest, RunStatus, compare_runs
from mlforge.training import TrainingConfig, TrainingResult, train


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _primary_metric(manifest: RunManifest, name: str) -> MetricValue:
    for metric in manifest.metrics:
        if metric.name == name:
            return metric
    raise BenchmarkError(
        f"Successful run {manifest.run_id} did not record benchmark metric {name!r}."
    )


def _rank_successful_runs(
    manifests: tuple[RunManifest, ...],
    *,
    metric: str,
) -> tuple[bool, dict[str, tuple[int, float]]]:
    if not manifests:
        return True, {}
    if len(manifests) == 1:
        value = _primary_metric(manifests[0], metric)
        return value.higher_is_better, {manifests[0].run_id: (1, value.value)}

    comparison = compare_runs(manifests, metric=metric)
    entries = sorted(
        comparison.entries,
        key=lambda entry: (
            -entry.value if comparison.higher_is_better else entry.value,
            entry.estimator,
            entry.run_id,
        ),
    )
    return comparison.higher_is_better, {
        entry.run_id: (rank, entry.value) for rank, entry in enumerate(entries, start=1)
    }


def benchmark(
    dataset: LoadedDataset,
    config: BenchmarkConfig,
    *,
    run_store: LocalRunStore | None = None,
    benchmark_store: LocalBenchmarkStore | None = None,
) -> BenchmarkResult:
    """Train, fairly rank, and record multiple local classification baselines."""
    if not isinstance(dataset, LoadedDataset):
        raise BenchmarkError("dataset must be a LoadedDataset value.")
    if not isinstance(config, BenchmarkConfig):
        raise BenchmarkError("config must be a BenchmarkConfig value.")
    runs = run_store or LocalRunStore(Path("mlruns"))
    benchmarks = benchmark_store or LocalBenchmarkStore(Path("mlbenchmarks"))
    if not isinstance(runs, LocalRunStore):
        raise BenchmarkError("run_store must be a LocalRunStore value.")
    if not isinstance(benchmarks, LocalBenchmarkStore):
        raise BenchmarkError("benchmark_store must be a LocalBenchmarkStore value.")

    benchmark_id = str(uuid4())
    started_at = _now()
    training_results: list[TrainingResult] = []
    run_manifests: list[RunManifest] = []
    durations: dict[str, float] = {}

    for estimator in config.estimators:
        attempt_started = perf_counter()
        try:
            result = train(
                dataset,
                TrainingConfig(
                    task=TaskType.CLASSIFICATION,
                    estimator=estimator,
                    split=config.split,
                    preprocessing=config.preprocessing,
                    feature_overrides=config.feature_overrides,
                ),
                run_store=runs,
            )
        except TrainingFailedError as error:
            run_manifest = runs.read(error.run_id)
        else:
            training_results.append(result)
            run_manifest = result.manifest
        duration = max(0.0, perf_counter() - attempt_started)
        durations[run_manifest.run_id] = round(duration, 6)
        run_manifests.append(run_manifest)

    successful = tuple(
        manifest for manifest in run_manifests if manifest.status is RunStatus.SUCCEEDED
    )
    higher_is_better, rankings = _rank_successful_runs(
        successful,
        metric=config.primary_metric,
    )
    reference = successful[0] if successful else run_manifests[0]
    entries = tuple(
        BenchmarkEntry(
            estimator=manifest.configuration.estimator,
            run_id=manifest.run_id,
            status=manifest.status,
            duration_seconds=durations[manifest.run_id],
            rank=rankings.get(manifest.run_id, (None, None))[0],
            primary_metric_value=rankings.get(manifest.run_id, (None, None))[1],
            failure=manifest.failure,
        )
        for manifest in run_manifests
    )
    status = (
        BenchmarkStatus.FAILED
        if not successful
        else BenchmarkStatus.SUCCEEDED
        if len(successful) == len(run_manifests)
        else BenchmarkStatus.PARTIAL
    )
    benchmark_manifest = BenchmarkManifest(
        schema_version=BENCHMARK_MANIFEST_SCHEMA_VERSION,
        benchmark_id=benchmark_id,
        status=status,
        started_at=started_at,
        completed_at=_now(),
        configuration=BenchmarkConfiguration.from_config(config),
        dataset=reference.dataset,
        split=successful[0].split if successful else None,
        higher_is_better=higher_is_better,
        entries=entries,
    )
    manifest_path = benchmarks.write(benchmark_manifest)
    if status is BenchmarkStatus.FAILED:
        raise BenchmarkFailedError(
            f"Benchmark {benchmark_id} failed because every requested estimator failed.",
            benchmark_id=benchmark_id,
            manifest_path=str(manifest_path),
        )
    return BenchmarkResult(
        manifest=benchmark_manifest,
        manifest_path=manifest_path,
        training_results=tuple(training_results),
        run_manifests=tuple(run_manifests),
    )
