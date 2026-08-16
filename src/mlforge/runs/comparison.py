"""Fair comparison of successful runs over the same validation contract."""

from __future__ import annotations

from dataclasses import dataclass

from mlforge.errors import RunComparisonError
from mlforge.runs.types import JsonObject, MetricValue, RunManifest, RunStatus


@dataclass(frozen=True, slots=True)
class RunComparisonEntry:
    """One ranked run in a metric comparison."""

    rank: int
    run_id: str
    estimator: str
    value: float

    def to_dict(self) -> JsonObject:
        return {
            "rank": self.rank,
            "run_id": self.run_id,
            "estimator": self.estimator,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class RunComparison:
    """Ranked same-data, same-split results for one named metric."""

    metric: str
    higher_is_better: bool
    entries: tuple[RunComparisonEntry, ...]

    def to_dict(self) -> JsonObject:
        return {
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _metric(manifest: RunManifest, name: str) -> MetricValue:
    for metric in manifest.metrics:
        if metric.name == name:
            return metric
    raise RunComparisonError(f"Metric {name!r} is not present in run {manifest.run_id}.")


def compare_runs(manifests: tuple[RunManifest, ...], *, metric: str) -> RunComparison:
    """Rank at least two compatible successful run manifests by one metric."""
    if len(manifests) < 2:
        raise RunComparisonError("At least two runs are required for comparison.")
    if not metric.strip():
        raise RunComparisonError("Comparison metric must not be blank.")
    if len({manifest.run_id for manifest in manifests}) != len(manifests):
        raise RunComparisonError("Run comparison inputs must be unique.")
    if any(manifest.status is not RunStatus.SUCCEEDED for manifest in manifests):
        raise RunComparisonError("Only successful runs can be compared.")

    reference = manifests[0]
    reference_contract = (
        reference.configuration.task,
        reference.dataset.sha256,
        reference.dataset.target,
        reference.configuration.validation_fraction,
        reference.configuration.random_seed,
        reference.split.stratified if reference.split is not None else None,
        reference.split.partition_sha256 if reference.split is not None else None,
    )
    for manifest in manifests[1:]:
        contract = (
            manifest.configuration.task,
            manifest.dataset.sha256,
            manifest.dataset.target,
            manifest.configuration.validation_fraction,
            manifest.configuration.random_seed,
            manifest.split.stratified if manifest.split is not None else None,
            manifest.split.partition_sha256 if manifest.split is not None else None,
        )
        if contract != reference_contract:
            raise RunComparisonError(
                "Runs must use the same task, dataset fingerprint, target, validation fraction, "
                "random seed, stratification policy, and exact row partition."
            )

    metric_values = tuple((manifest, _metric(manifest, metric)) for manifest in manifests)
    directions = {item.higher_is_better for _, item in metric_values}
    if len(directions) != 1:
        raise RunComparisonError("Metric comparison direction is inconsistent across runs.")
    higher_is_better = directions.pop()
    ordered = sorted(
        metric_values,
        key=lambda pair: (
            -pair[1].value if higher_is_better else pair[1].value,
            pair[0].run_id,
        ),
    )
    entries = tuple(
        RunComparisonEntry(
            rank=index,
            run_id=manifest.run_id,
            estimator=manifest.configuration.estimator,
            value=value.value,
        )
        for index, (manifest, value) in enumerate(ordered, start=1)
    )
    return RunComparison(metric=metric, higher_is_better=higher_is_better, entries=entries)
