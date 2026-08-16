"""Security, integrity, and persistence tests for local model artifacts."""

import json
import os
import stat
import zipfile
from pathlib import Path
from typing import Any, cast

import pytest

from mlforge.artifacts import (
    ArtifactEnvironment,
    LocalArtifactStore,
    inspect_artifact,
    load_artifact,
    verify_run_manifest,
)
from mlforge.datasets import LoadedDataset, load_csv
from mlforge.errors import (
    ArtifactCompatibilityError,
    ArtifactFormatError,
    ArtifactIntegrityError,
    ArtifactPathError,
    ArtifactTrustError,
)
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig, TrainingResult, train


def _training_result(tmp_path: Path) -> tuple[TrainingResult, LoadedDataset]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "training.csv"
    rows = ["amount,region,target"]
    for index in range(30):
        rows.append(
            f"{index + 1},{'north' if index % 2 else 'south'},{'yes' if index % 3 else 'no'}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset = load_csv(path, target="target")
    result = train(
        dataset,
        TrainingConfig(task=TaskType.CLASSIFICATION, estimator=LOGISTIC_REGRESSION),
        run_store=LocalRunStore(tmp_path / "runs"),
    )
    return result, dataset


def _rewrite_archive(
    path: Path, *, mutate_manifest: bool = False, mutate_pipeline: bool = False
) -> None:
    with zipfile.ZipFile(path, mode="r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        pipeline = archive.read("pipeline.pkl")
    if mutate_manifest:
        manifest["schema_version"] = 999
    if mutate_pipeline:
        pipeline = pipeline[:-1] + bytes([pipeline[-1] ^ 0xFF])
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        archive.writestr("pipeline.pkl", pipeline)


def test_save_inspect_load_and_prediction_parity(tmp_path: Path) -> None:
    """A saved pipeline should survive strict inspection and trusted loading unchanged."""
    result, dataset = _training_result(tmp_path)
    store = LocalArtifactStore(tmp_path / "artifacts")

    saved = store.save(result)
    inspected = inspect_artifact(saved.path)
    loaded = load_artifact(saved.path, trusted=True)
    features = dataset.frame.drop(columns=[dataset.metadata.target])

    assert saved.path.name == f"{result.manifest.run_id}.mlforge"
    assert inspected == saved.manifest == loaded.manifest
    assert list(loaded.pipeline.predict(features)) == list(result.pipeline.predict(features))
    assert tuple(feature.name for feature in inspected.features) == tuple(features.columns)
    assert inspected.pipeline_size_bytes > 0
    verify_run_manifest(inspected, result.manifest)


def test_loading_requires_explicit_trust_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default loading path must never execute pickle payloads."""
    result, _ = _training_result(tmp_path)
    saved = LocalArtifactStore(tmp_path / "artifacts").save(result)
    called = False

    def unexpected_loads(payload: bytes) -> object:
        nonlocal called
        called = True
        raise AssertionError("pickle.loads must not run")

    monkeypatch.setattr("mlforge.artifacts.store.pickle.loads", unexpected_loads)

    with pytest.raises(ArtifactTrustError, match="trusted=True"):
        load_artifact(saved.path)

    assert called is False


def test_dependency_mismatch_fails_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported cross-version loading should fail closed before Python objects execute."""
    result, _ = _training_result(tmp_path)
    saved = LocalArtifactStore(tmp_path / "artifacts").save(result)
    current = saved.manifest.environment
    monkeypatch.setattr(
        "mlforge.artifacts.store._current_environment",
        lambda: ArtifactEnvironment(
            python=current.python,
            mlforge=current.mlforge,
            pandas=current.pandas,
            numpy=current.numpy,
            scipy=current.scipy,
            scikit_learn="999.0",
        ),
    )
    called = False

    def unexpected_loads(payload: bytes) -> object:
        nonlocal called
        called = True
        raise AssertionError("pickle.loads must not run")

    monkeypatch.setattr("mlforge.artifacts.store.pickle.loads", unexpected_loads)

    with pytest.raises(ArtifactCompatibilityError, match="scikit_learn"):
        load_artifact(saved.path, trusted=True)

    assert called is False


def test_checksum_and_manifest_version_tampering_fail_closed(tmp_path: Path) -> None:
    """Safe inspection must reject modified payloads and unsupported schemas."""
    first, _ = _training_result(tmp_path / "first")
    first_path = LocalArtifactStore(tmp_path / "first-artifacts").save(first).path
    _rewrite_archive(first_path, mutate_pipeline=True)

    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        inspect_artifact(first_path)

    second, _ = _training_result(tmp_path / "second")
    second_path = LocalArtifactStore(tmp_path / "second-artifacts").save(second).path
    _rewrite_archive(second_path, mutate_manifest=True)

    with pytest.raises(ArtifactFormatError, match="schema version"):
        inspect_artifact(second_path)


@pytest.mark.parametrize("failure", ["compressed", "extra-member", "symbolic-link"])
def test_unsupported_archive_layouts_fail_closed(tmp_path: Path, failure: str) -> None:
    """Only the documented two stored regular members should pass safe inspection."""
    result, _ = _training_result(tmp_path)
    path = LocalArtifactStore(tmp_path / "artifacts").save(result).path
    with zipfile.ZipFile(path, mode="r") as archive:
        manifest = archive.read("manifest.json")
        pipeline = archive.read("pipeline.pkl")

    compression = zipfile.ZIP_DEFLATED if failure == "compressed" else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, mode="w", compression=compression) as archive:
        archive.writestr("manifest.json", manifest)
        if failure == "symbolic-link":
            pipeline_info = zipfile.ZipInfo("pipeline.pkl")
            pipeline_info.create_system = 3
            pipeline_info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(pipeline_info, pipeline)
        else:
            archive.writestr("pipeline.pkl", pipeline)
        if failure == "extra-member":
            archive.writestr("unexpected.txt", b"unexpected")

    message = "stored ZIP format" if failure == "compressed" else "must contain exactly"
    if failure == "symbolic-link":
        message = "regular files"
    with pytest.raises(ArtifactFormatError, match=message):
        inspect_artifact(path)


def test_artifacts_are_immutable_and_atomic(tmp_path: Path) -> None:
    """A run may publish one complete artifact and may never overwrite it."""
    result, _ = _training_result(tmp_path)
    store = LocalArtifactStore(tmp_path / "artifacts")
    path = store.save(result).path
    original = path.read_bytes()

    with pytest.raises(ArtifactPathError, match="immutable"):
        store.save(result)

    assert path.read_bytes() == original
    assert not list((tmp_path / "artifacts").glob("*.tmp"))


def test_atomic_write_failure_leaves_no_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A publication failure should leave neither a final archive nor temporary bytes."""
    result, _ = _training_result(tmp_path)
    store = LocalArtifactStore(tmp_path / "artifacts")

    def fail_link(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr("mlforge.artifacts.store.os.link", fail_link)

    with pytest.raises(ArtifactPathError, match="atomically"):
        store.save(result)

    assert not list((tmp_path / "artifacts").iterdir())


@pytest.mark.parametrize("run_id", ["../escape", "NOT-A-UUID"])
def test_store_run_ids_cannot_escape_the_artifact_root(tmp_path: Path, run_id: str) -> None:
    """Store lookup accepts canonical UUIDs only."""
    store = LocalArtifactStore(tmp_path / "artifacts")
    (tmp_path / "artifacts").mkdir()

    with pytest.raises(ArtifactPathError, match="canonical"):
        store.inspect(run_id)


def test_artifact_requires_the_exact_persisted_run_manifest(tmp_path: Path) -> None:
    """Artifact lineage cannot be built from an in-memory run detached from storage."""
    result, _ = _training_result(tmp_path)
    result.manifest_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="persisted training run"):
        LocalArtifactStore(tmp_path / "artifacts").save(result)


def test_artifact_rejects_pipeline_schema_drift_before_publication(tmp_path: Path) -> None:
    """A mutated fitted pipeline must not produce an artifact that cannot be loaded."""
    result, _ = _training_result(tmp_path)
    preprocessor = cast(Any, result.pipeline.named_steps["preprocessor"])
    preprocessor.feature_names_in_ = preprocessor.feature_names_in_[::-1]

    with pytest.raises(ArtifactIntegrityError, match="feature names"):
        LocalArtifactStore(tmp_path / "artifacts").save(result)

    assert not (tmp_path / "artifacts").exists()
