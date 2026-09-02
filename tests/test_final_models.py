"""Behavioral tests for explicit cross-validation-to-final-model fitting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mlforge.artifacts import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactLineageKind,
    LocalArtifactStore,
    inspect_artifact,
    load_artifact,
    verify_final_model_manifest,
)
from mlforge.benchmarks import (
    CrossValidationConfig,
    CrossValidationResult,
    LocalCrossValidationStore,
    cross_validate_benchmark,
)
from mlforge.cli import main
from mlforge.config import LOG_LEVEL_ENVIRONMENT_VARIABLE
from mlforge.datasets import LoadedDataset, load_csv
from mlforge.errors import (
    ArtifactIntegrityError,
    BenchmarkFailedError,
    FinalModelFailedError,
    FinalModelLineageError,
    FinalModelStoreError,
    PreprocessingError,
)
from mlforge.final_models import (
    FINAL_MODEL_FIT_SCOPE,
    FinalModelManifest,
    LocalFinalModelStore,
    fit_selected_model,
)
from mlforge.inference import predict_frame
from mlforge.pipelines import (
    CrossValidationSplitConfig,
    FeatureOverrides,
    NumericImputationStrategy,
    PreprocessingConfig,
    TaskType,
)
from mlforge.runs import RunStatus
from mlforge.training import (
    DUMMY_CLASSIFIER,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_REGRESSOR,
    RIDGE_REGRESSION,
)


def _classification_dataset(tmp_path: Path, *, name: str = "selected.csv") -> LoadedDataset:
    rows = ["amount,region,target"]
    for index in range(60):
        target = "yes" if index >= 30 else "no"
        rows.append(f"{index},{'north' if index % 3 else 'south'},{target}")
    path = tmp_path / name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_csv(path, target="target")


def _regression_dataset(tmp_path: Path) -> LoadedDataset:
    rows = ["amount,region,target"]
    for index in range(60):
        region = "north" if index % 2 else "south"
        rows.append(f"{index},{region},{index * 3.25 + (4 if region == 'north' else -2)}")
    path = tmp_path / "regression-selected.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_csv(path, target="target")


def _selection(
    dataset: LoadedDataset,
    root: Path,
    *,
    config: CrossValidationConfig | None = None,
) -> CrossValidationResult:
    return cross_validate_benchmark(
        dataset,
        config
        or CrossValidationConfig(
            estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION),
            split=CrossValidationSplitConfig(fold_count=3, random_seed=19),
        ),
        store=LocalCrossValidationStore(root),
    )


def test_final_model_refits_every_row_and_persists_exact_selection_lineage(
    tmp_path: Path,
) -> None:
    """Final fitting must use all rows without converting CV evidence into training metrics."""
    dataset = _classification_dataset(tmp_path)
    selection = _selection(dataset, tmp_path / "benchmarks")
    store = LocalFinalModelStore(tmp_path / "final-models")

    result = fit_selected_model(dataset, selection, final_model_store=store)

    assert result.manifest.status is RunStatus.SUCCEEDED
    assert result.manifest.fit_scope == FINAL_MODEL_FIT_SCOPE
    assert result.manifest.training_rows == len(dataset.frame) == 60
    assert result.manifest.configuration.estimator == LOGISTIC_REGRESSION
    assert selection.manifest.winner is not None
    assert (
        result.manifest.configuration.estimator_parameters == selection.manifest.winner.parameters
    )
    assert result.manifest.selection.benchmark_id == selection.manifest.benchmark_id
    expected_digest = hashlib.sha256(
        selection.manifest.to_json(indent=None).encode("utf-8")
    ).hexdigest()
    assert result.manifest.selection.manifest_sha256 == expected_digest
    assert result.manifest.failure is None
    assert result.manifest.artifact is not None
    assert result.manifest.artifact.artifact_id == result.manifest.final_model_id
    assert result.artifact_path is not None
    assert result.artifact_path.is_file()
    assert store.read(result.manifest.final_model_id) == result.manifest
    assert FinalModelManifest.from_json(result.manifest.to_json()) == result.manifest
    legacy_payload = result.manifest.to_dict()
    legacy_payload["schema_version"] = 1
    assert FinalModelManifest.from_json(json.dumps(legacy_payload)).schema_version == 1

    preprocessor = result.pipeline.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.n_samples_seen_ == 60


def test_regression_winner_refits_and_produces_numeric_predictions(tmp_path: Path) -> None:
    """A persisted regression winner should use the same final-model and artifact boundary."""
    dataset = _regression_dataset(tmp_path)
    selection = _selection(
        dataset,
        tmp_path / "regression-benchmarks",
        config=CrossValidationConfig(
            task=TaskType.REGRESSION,
            estimators=(RIDGE_REGRESSION, RANDOM_FOREST_REGRESSOR),
            primary_metric="root_mean_squared_error",
            split=CrossValidationSplitConfig(fold_count=3, random_seed=19),
        ),
    )

    result = fit_selected_model(
        dataset,
        selection,
        final_model_store=LocalFinalModelStore(tmp_path / "regression-final-models"),
        artifact_store=LocalArtifactStore(tmp_path / "regression-artifacts"),
    )
    assert result.artifact_path is not None
    loaded = load_artifact(result.artifact_path, trusted=True)
    predictions = predict_frame(loaded, dataset.frame.drop(columns=["target"]))

    assert result.manifest.configuration.task == "regression"
    assert result.manifest.selection.primary_metric == "root_mean_squared_error"
    assert predictions.row_count == len(dataset.frame)
    assert all(isinstance(record.prediction, float) for record in predictions.predictions)


def test_final_model_reconstructs_preprocessing_exactly(tmp_path: Path) -> None:
    """Final fitting must use the preprocessing contract persisted during selection."""
    dataset = _classification_dataset(tmp_path)
    config = CrossValidationConfig(
        estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION),
        split=CrossValidationSplitConfig(fold_count=3, random_seed=73),
        preprocessing=PreprocessingConfig(
            numeric_imputation=NumericImputationStrategy.MEAN,
            scale_numeric=False,
            categorical_fill_value="__selected_missing__",
        ),
        feature_overrides=FeatureOverrides(
            numeric=("amount",),
            categorical=("region",),
        ),
    )
    selection = _selection(dataset, tmp_path / "benchmarks", config=config)

    result = fit_selected_model(
        dataset,
        selection,
        final_model_store=LocalFinalModelStore(tmp_path / "final-models"),
    )

    recorded = result.manifest.configuration
    assert recorded.random_seed == 73
    assert recorded.numeric_imputation == "mean"
    assert recorded.scale_numeric is False
    assert recorded.categorical_fill_value == "__selected_missing__"
    assert recorded.numeric_overrides == ("amount",)
    assert recorded.categorical_overrides == ("region",)
    preprocessor = result.pipeline.named_steps["preprocessor"]
    numeric = preprocessor.named_transformers_["numeric"]
    categorical = preprocessor.named_transformers_["categorical"]
    assert numeric.named_steps["imputer"].strategy == "mean"
    assert "scaler" not in numeric.named_steps
    assert categorical.named_steps["imputer"].fill_value == "__selected_missing__"


def test_final_model_uses_fresh_deterministic_estimator_state(tmp_path: Path) -> None:
    """Repeated final fits must create distinct state with deterministic predictions."""
    dataset = _classification_dataset(tmp_path)
    selection = _selection(dataset, tmp_path / "benchmarks")

    first = fit_selected_model(
        dataset,
        selection,
        final_model_store=LocalFinalModelStore(tmp_path / "first" / "final-models"),
    )
    second = fit_selected_model(
        dataset,
        selection,
        final_model_store=LocalFinalModelStore(tmp_path / "second" / "final-models"),
    )

    first_estimator = first.pipeline.named_steps["estimator"]
    second_estimator = second.pipeline.named_steps["estimator"]
    features = dataset.frame.drop(columns=["target"])
    assert first_estimator is not second_estimator
    assert list(first.pipeline.predict(features)) == list(second.pipeline.predict(features))
    assert first.manifest.final_model_id != second.manifest.final_model_id
    assert first.manifest.configuration == second.manifest.configuration
    assert first.manifest.artifact is not None
    assert second.manifest.artifact is not None
    assert first.manifest.artifact.pipeline_sha256 == second.manifest.artifact.pipeline_sha256


def test_final_model_rejects_estimator_parameter_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Current factory defaults cannot silently replace persisted winner parameters."""
    dataset = _classification_dataset(tmp_path)
    selection = _selection(dataset, tmp_path / "benchmarks")
    destination = tmp_path / "final-models"

    monkeypatch.setattr(
        "mlforge.final_models.service.create_estimator",
        lambda config: LogisticRegression(max_iter=999, random_state=config.split.random_seed),
    )

    with pytest.raises(FinalModelLineageError, match="parameters"):
        fit_selected_model(
            dataset,
            selection,
            final_model_store=LocalFinalModelStore(destination),
        )

    assert not destination.exists()


def test_final_model_artifact_is_inspectable_loadable_and_predictive(tmp_path: Path) -> None:
    """A final-model artifact should preserve lineage and support the existing trust boundary."""
    dataset = _classification_dataset(tmp_path)
    artifact_store = LocalArtifactStore(tmp_path / "artifacts")
    final_result = fit_selected_model(
        dataset,
        _selection(dataset, tmp_path / "benchmarks"),
        final_model_store=LocalFinalModelStore(tmp_path / "final-models"),
        artifact_store=artifact_store,
    )
    assert final_result.artifact_path is not None
    inspected = inspect_artifact(final_result.artifact_path)
    verify_final_model_manifest(inspected, final_result.manifest)
    loaded = load_artifact(final_result.artifact_path, trusted=True)
    predictions = predict_frame(loaded, dataset.frame.drop(columns=["target"]))

    assert inspected.schema_version == ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert inspected.lineage_kind is ArtifactLineageKind.FINAL_MODEL
    assert inspected.model_id == final_result.manifest.final_model_id
    assert inspected.to_dict()["model_id"] == final_result.manifest.final_model_id
    assert "run_id" not in inspected.to_dict()
    assert predictions.run_id == final_result.manifest.final_model_id
    assert predictions.row_count == 60
    assert list(loaded.pipeline.predict(dataset.frame.drop(columns=["target"]))) == [
        record.prediction for record in predictions.predictions
    ]


def test_final_artifact_rejects_manifest_copied_under_wrong_filename(tmp_path: Path) -> None:
    """Artifact lineage must include the canonical persisted manifest filename."""
    dataset = _classification_dataset(tmp_path)
    final_result = fit_selected_model(
        dataset,
        _selection(dataset, tmp_path / "benchmarks"),
        final_model_store=LocalFinalModelStore(tmp_path / "final-models"),
    )
    copied_path = tmp_path / "copied-final-model.json"
    copied_path.write_text(final_result.manifest.to_json(), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="persisted filename"):
        LocalArtifactStore(tmp_path / "copied-artifacts").save_final(
            replace(final_result, manifest_path=copied_path)
        )


def test_final_model_rejects_any_dataset_identity_change_before_writing(tmp_path: Path) -> None:
    """Selection lineage must not be reused with a similar but different dataset."""
    selected = _classification_dataset(tmp_path, name="selected.csv")
    selection = _selection(selected, tmp_path / "benchmarks")
    changed_path = tmp_path / "changed.csv"
    changed = selected.frame.copy()
    changed.loc[0, "amount"] = 999
    changed.to_csv(changed_path, index=False)
    changed_dataset = load_csv(changed_path, target="target")
    destination = tmp_path / "final-models"

    with pytest.raises(FinalModelLineageError, match="exact dataset"):
        fit_selected_model(
            changed_dataset,
            selection,
            final_model_store=LocalFinalModelStore(destination),
        )

    assert not destination.exists()


@pytest.mark.parametrize("mutation", ["frame", "source"])
def test_final_model_rejects_dataset_mutation_after_ingestion(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Final fitting must revalidate both the loaded frame and its source file."""
    dataset = _classification_dataset(tmp_path)
    selection = _selection(dataset, tmp_path / "benchmarks")
    if mutation == "frame":
        dataset.frame.loc[0, "amount"] = 999
    else:
        source = dataset.metadata.source_path
        content = source.read_text(encoding="utf-8")
        source.write_text(content.replace("0,south,no", "9,south,no", 1), encoding="utf-8")
    destination = tmp_path / "final-models"

    with pytest.raises(FinalModelLineageError, match="changed after ingestion"):
        fit_selected_model(
            dataset,
            selection,
            final_model_store=LocalFinalModelStore(destination),
        )

    assert not destination.exists()


def test_final_model_rejects_tampered_or_unpersisted_selection(tmp_path: Path) -> None:
    """A caller cannot bypass the immutable cross-validation manifest lineage check."""
    dataset = _classification_dataset(tmp_path)
    selection = _selection(dataset, tmp_path / "benchmarks")
    selection.manifest_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(FinalModelLineageError, match="persisted cross-validation"):
        fit_selected_model(
            dataset,
            selection,
            final_model_store=LocalFinalModelStore(tmp_path / "final-models"),
        )


def test_final_model_rejects_complete_but_failed_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal selection without any successful winner cannot be finalized."""
    dataset = _classification_dataset(tmp_path)
    store = LocalCrossValidationStore(tmp_path / "benchmarks")

    def fail_fold(*args: object, **kwargs: object) -> object:
        raise PreprocessingError("intentional selection failure")

    monkeypatch.setattr(
        "mlforge.benchmarks.cross_validation_service.build_model_pipeline",
        fail_fold,
    )
    with pytest.raises(BenchmarkFailedError) as captured:
        cross_validate_benchmark(
            dataset,
            CrossValidationConfig(
                estimators=(DUMMY_CLASSIFIER, LOGISTIC_REGRESSION),
                split=CrossValidationSplitConfig(fold_count=3, random_seed=19),
            ),
            store=store,
        )
    failed = CrossValidationResult(
        manifest=store.read(captured.value.benchmark_id),
        manifest_path=Path(captured.value.manifest_path),
    )

    with pytest.raises(FinalModelLineageError, match="successful rank-one"):
        fit_selected_model(
            dataset,
            failed,
            final_model_store=LocalFinalModelStore(tmp_path / "final-models"),
        )


def test_expected_final_fit_failure_is_recorded_without_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expected estimator/preprocessing failures must leave terminal evidence."""
    dataset = _classification_dataset(tmp_path)
    selection = _selection(dataset, tmp_path / "benchmarks")
    store = LocalFinalModelStore(tmp_path / "final-models")

    def fail_build(*args: object, **kwargs: object) -> object:
        raise PreprocessingError("intentional final-fit failure")

    monkeypatch.setattr("mlforge.final_models.service.build_final_model_pipeline", fail_build)

    with pytest.raises(FinalModelFailedError, match="intentional final-fit failure") as captured:
        fit_selected_model(dataset, selection, final_model_store=store)

    manifest = store.read(captured.value.final_model_id)
    assert manifest.status is RunStatus.FAILED
    assert manifest.failure is not None
    assert manifest.artifact is None
    assert manifest.failure.error_type == "PreprocessingError"
    assert manifest.selection.benchmark_id == selection.manifest.benchmark_id


def test_final_model_store_is_create_only_and_strict(tmp_path: Path) -> None:
    """Final-model history must reject overwrites and unexpected JSON fields."""
    dataset = _classification_dataset(tmp_path)
    store = LocalFinalModelStore(tmp_path / "final-models")
    result = fit_selected_model(
        dataset,
        _selection(dataset, tmp_path / "benchmarks"),
        final_model_store=store,
    )

    with pytest.raises(FinalModelStoreError, match="immutable"):
        store.write(result.manifest)

    payload = result.manifest.to_dict()
    payload["unexpected"] = True
    with pytest.raises(FinalModelStoreError, match="unexpected"):
        FinalModelManifest.from_json(json.dumps(payload))


def test_finalize_cli_completes_selection_artifact_inspection_and_prediction(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should expose one explicit CV-selection-to-prediction workflow."""
    dataset = _classification_dataset(tmp_path)
    benchmarks_root = tmp_path / "benchmarks-root"
    selection = _selection(dataset, benchmarks_root / "cross-validation")
    final_models = tmp_path / "final-models"
    artifacts = tmp_path / "artifacts"
    monkeypatch.delenv(LOG_LEVEL_ENVIRONMENT_VARIABLE, raising=False)
    monkeypatch.setattr("mlforge.cli.configure_logging", lambda level: None)

    assert (
        main(
            [
                "finalize",
                str(dataset.metadata.source_path),
                "--target",
                "target",
                "--benchmark-id",
                selection.manifest.benchmark_id,
                "--benchmarks-dir",
                str(benchmarks_root),
                "--final-models-dir",
                str(final_models),
                "--artifacts-dir",
                str(artifacts),
                "--json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    final_model_id = output["final_model"]["final_model_id"]
    assert output["final_model"]["final_fit"]["fit_scope"] == "all_rows"
    assert output["final_model"]["artifact"]["artifact_id"] == final_model_id
    artifact_path = Path(output["artifact"]["path"])
    assert output["artifact"]["manifest"]["lineage_kind"] == "final-model"
    assert output["artifact"]["manifest"]["model_id"] == final_model_id
    assert (final_models / f"{final_model_id}.json").is_file()
    assert artifact_path.is_file()

    assert main(["artifacts", "inspect", str(artifact_path), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["model_id"] == final_model_id
    assert inspected["lineage_kind"] == "final-model"

    prediction_path = tmp_path / "prediction.csv"
    pd.DataFrame({"region": ["north", "new"], "amount": [10, 50]}).to_csv(
        prediction_path,
        index=False,
    )
    assert (
        main(
            [
                "predict",
                str(artifact_path),
                str(prediction_path),
                "--trust-artifact",
                "--json",
            ]
        )
        == 0
    )
    predictions = json.loads(capsys.readouterr().out)
    assert predictions["run_id"] == final_model_id
    assert predictions["row_count"] == 2
