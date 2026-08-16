"""Tests for immutable run manifests, local storage, and comparison."""

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from mlforge.errors import RunComparisonError, RunStoreError
from mlforge.runs import (
    RUN_MANIFEST_SCHEMA_VERSION,
    DatasetSnapshot,
    EnvironmentSnapshot,
    LocalRunStore,
    MetricValue,
    RunConfiguration,
    RunFailure,
    RunManifest,
    RunParameter,
    RunStatus,
    SplitSnapshot,
    compare_runs,
)


def _manifest(
    *,
    run_id: str | None = None,
    estimator: str = "logistic-regression",
    metric_name: str = "accuracy",
    metric_value: float = 0.75,
    higher_is_better: bool = True,
    sha256: str = "a" * 64,
    random_seed: int = 42,
    partition_sha256: str = "c" * 64,
    status: RunStatus = RunStatus.SUCCEEDED,
) -> RunManifest:
    failure = RunFailure(error_type="TrainingError", message="failed")
    return RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=run_id or str(uuid4()),
        status=status,
        started_at="2026-08-12T10:00:00+00:00",
        completed_at="2026-08-12T10:00:01+00:00",
        configuration=RunConfiguration(
            task="classification",
            estimator=estimator,
            validation_fraction=0.2,
            random_seed=random_seed,
            stratify_requested=None,
            numeric_imputation="median",
            scale_numeric=True,
            categorical_fill_value="__mlforge_missing__",
            numeric_overrides=(),
            categorical_overrides=(),
            estimator_parameters=(RunParameter(name="random_state", value=random_seed),),
        ),
        dataset=DatasetSnapshot(
            source_path="C:/data/training.csv",
            sha256=sha256,
            file_size_bytes=100,
            row_count=20,
            column_count=3,
            target="label",
            encoding="utf-8-sig",
            delimiter=",",
        ),
        environment=EnvironmentSnapshot(
            python="3.12.0",
            mlforge="0.1.0",
            pandas="3.0.1",
            numpy="2.3.5",
            scipy="1.18.0",
            scikit_learn="1.9.0",
        ),
        split=SplitSnapshot(
            train_rows=16,
            validation_rows=4,
            feature_count=2,
            stratified=True,
            partition_sha256=partition_sha256,
        ),
        metrics=(
            ()
            if status is RunStatus.FAILED
            else (
                MetricValue(
                    name=metric_name,
                    value=metric_value,
                    higher_is_better=higher_is_better,
                ),
            )
        ),
        warnings=("A warning.",),
        failure=failure if status is RunStatus.FAILED else None,
    )


def test_manifest_json_round_trip_is_deterministic_and_validated() -> None:
    """A written manifest should reconstruct the same immutable value object."""
    manifest = _manifest()

    content = manifest.to_json()
    restored = RunManifest.from_json(content)

    assert restored == manifest
    assert content == restored.to_json()
    assert json.loads(content)["schema_version"] == 1


def test_local_store_atomically_writes_reads_and_lists(tmp_path: Path) -> None:
    """A store should create a complete record without leaving temporary files."""
    store = LocalRunStore(tmp_path / "runs")
    later = _manifest()
    earlier = RunManifest.from_json(
        _manifest()
        .to_json()
        .replace(
            "2026-08-12T10:00:00+00:00",
            "2026-08-12T09:00:00+00:00",
        )
        .replace(
            "2026-08-12T10:00:01+00:00",
            "2026-08-12T09:00:01+00:00",
        )
    )

    first_path = store.write(later)
    second_path = store.write(earlier)

    assert first_path.read_text(encoding="utf-8").endswith("\n")
    assert store.read(later.run_id) == later
    assert store.list_manifests() == (earlier, later)
    assert not list((tmp_path / "runs").glob("*.tmp"))
    assert second_path.is_file()


def test_local_store_never_overwrites_an_existing_run(tmp_path: Path) -> None:
    """Run IDs are immutable after their first successful manifest write."""
    store = LocalRunStore(tmp_path / "runs")
    manifest = _manifest()
    path = store.write(manifest)
    original = path.read_bytes()

    with pytest.raises(RunStoreError, match="immutable"):
        store.write(manifest)

    assert path.read_bytes() == original


def test_atomic_write_failure_leaves_no_partial_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure before the atomic link must leave neither final nor temporary JSON."""
    store = LocalRunStore(tmp_path / "runs")
    manifest = _manifest()

    def fail_link(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError("simulated link failure")

    monkeypatch.setattr("mlforge.runs.store.os.link", fail_link)

    with pytest.raises(RunStoreError, match="atomically"):
        store.write(manifest)

    assert not list((tmp_path / "runs").iterdir())


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        json.dumps({"schema_version": 999}),
        _manifest().to_json().replace('"schema_version": 1', '"schema_version": 2'),
    ],
)
def test_corrupt_or_unsupported_manifest_fails_closed(tmp_path: Path, content: str) -> None:
    """Inspection must validate JSON and schema instead of returning partial records."""
    store = LocalRunStore(tmp_path / "runs")
    run_id = str(uuid4())
    directory = tmp_path / "runs"
    directory.mkdir()
    (directory / f"{run_id}.json").write_text(content, encoding="utf-8")

    with pytest.raises(RunStoreError):
        store.read(run_id)


@pytest.mark.parametrize("run_id", ["../escape", "NOT-A-UUID", str(uuid4()).upper()])
def test_run_ids_cannot_escape_the_store(tmp_path: Path, run_id: str) -> None:
    """Run inspection paths should accept canonical UUIDs only."""
    store = LocalRunStore(tmp_path / "runs")
    (tmp_path / "runs").mkdir()

    with pytest.raises(RunStoreError, match="canonical"):
        store.read(run_id)


def test_compare_runs_orders_higher_and_lower_metrics() -> None:
    """Comparison direction must be metric metadata, not a hardcoded name guess."""
    weak = _manifest(estimator="logistic-regression", metric_value=0.6)
    strong = _manifest(estimator="random-forest-classifier", metric_value=0.8)

    accuracy = compare_runs((weak, strong), metric="accuracy")
    loss = compare_runs(
        (
            _manifest(metric_name="loss", metric_value=0.4, higher_is_better=False),
            _manifest(metric_name="loss", metric_value=0.2, higher_is_better=False),
        ),
        metric="loss",
    )

    assert [entry.value for entry in accuracy.entries] == [0.8, 0.6]
    assert [entry.value for entry in loss.entries] == [0.2, 0.4]
    assert [entry.rank for entry in accuracy.entries] == [1, 2]


def test_compare_runs_rejects_unfair_or_invalid_inputs() -> None:
    """Different data, split seeds, failed runs, and missing metrics are not comparable."""
    reference = _manifest()

    with pytest.raises(RunComparisonError, match="At least two"):
        compare_runs((reference,), metric="accuracy")
    with pytest.raises(RunComparisonError, match="unique"):
        compare_runs((reference, reference), metric="accuracy")
    with pytest.raises(RunComparisonError, match="successful"):
        compare_runs((reference, _manifest(status=RunStatus.FAILED)), metric="accuracy")
    with pytest.raises(RunComparisonError, match="same task, dataset"):
        compare_runs((reference, _manifest(sha256="b" * 64)), metric="accuracy")
    with pytest.raises(RunComparisonError, match="exact row partition"):
        compare_runs((reference, _manifest(partition_sha256="d" * 64)), metric="accuracy")
    with pytest.raises(RunComparisonError, match="not present"):
        compare_runs((reference, _manifest()), metric="f1_weighted")
