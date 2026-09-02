"""Explicit full-dataset refitting from persisted cross-validation selection evidence."""

from __future__ import annotations

import hashlib
import math
import pickle
import sys
import warnings
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

from sklearn.base import BaseEstimator

from mlforge.artifacts.store import LocalArtifactStore
from mlforge.artifacts.types import ARTIFACT_SERIALIZATION_FORMAT
from mlforge.benchmarks import CrossValidationManifest, CrossValidationResult
from mlforge.datasets import CsvLoadOptions, LoadedDataset, load_csv, profile_dataset
from mlforge.errors import (
    FinalModelError,
    FinalModelFailedError,
    FinalModelLineageError,
    FinalModelStoreError,
    MLForgeError,
)
from mlforge.final_models.store import LocalFinalModelStore
from mlforge.final_models.types import (
    FINAL_MODEL_FIT_SCOPE,
    FINAL_MODEL_MANIFEST_SCHEMA_VERSION,
    FinalModelArtifact,
    FinalModelConfiguration,
    FinalModelManifest,
    FinalModelResult,
    FinalModelSelection,
)
from mlforge.pipelines import (
    FeatureOverrides,
    NumericImputationStrategy,
    PreprocessingConfig,
    SplitConfig,
    TaskType,
    build_final_model_pipeline,
    infer_feature_schema,
)
from mlforge.runs import (
    DatasetSnapshot,
    EnvironmentSnapshot,
    RunFailure,
    RunParameter,
    RunStatus,
)
from mlforge.runs.types import JsonPrimitive
from mlforge.training import TrainingConfig
from mlforge.training.estimators import create_estimator


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
    raise FinalModelError(
        f"Estimator parameter {name!r} is not a reproducible JSON primitive: "
        f"{type(value).__name__}."
    )


def _parameters(estimator: BaseEstimator) -> tuple[RunParameter, ...]:
    return tuple(
        RunParameter(name=name, value=_parameter_value(value, name=name))
        for name, value in sorted(estimator.get_params(deep=False).items())
    )


def _selection_digest(manifest: CrossValidationManifest) -> str:
    return hashlib.sha256(manifest.to_json(indent=None).encode("utf-8")).hexdigest()


def _persisted_selection(result: CrossValidationResult) -> CrossValidationManifest:
    path = result.manifest_path
    if path.is_symlink() or not path.is_file():
        raise FinalModelLineageError(
            "Cross-validation selection must reference its persisted regular-file manifest."
        )
    try:
        persisted = CrossValidationManifest.from_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, MLForgeError) as error:
        raise FinalModelLineageError(
            "Could not validate the persisted cross-validation selection manifest."
        ) from error
    if persisted != result.manifest:
        raise FinalModelLineageError(
            "Cross-validation selection does not match its persisted immutable manifest."
        )
    if path.stem != persisted.benchmark_id:
        raise FinalModelLineageError(
            "Cross-validation selection manifest ID does not match its filename."
        )
    return persisted


def _verify_dataset(dataset: LoadedDataset, selection: CrossValidationManifest) -> None:
    if not isinstance(dataset, LoadedDataset):
        raise FinalModelError("dataset must be a LoadedDataset value.")
    metadata = dataset.metadata
    expected = selection.dataset
    mismatches = [
        label
        for label, actual, selected in (
            ("SHA-256", metadata.sha256, expected.sha256),
            ("file size", metadata.file_size_bytes, expected.file_size_bytes),
            ("row count", metadata.row_count, expected.row_count),
            ("column count", metadata.column_count, expected.column_count),
            ("target", metadata.target, expected.target),
            ("encoding", metadata.encoding, expected.encoding),
            ("delimiter", metadata.delimiter, expected.delimiter),
        )
        if actual != selected
    ]
    if mismatches:
        raise FinalModelLineageError(
            "Final fitting requires the exact dataset selected by cross-validation; mismatched "
            + ", ".join(mismatches)
            + "."
        )
    actual_columns = tuple(str(name) for name in dataset.frame.columns)
    metadata_columns = tuple(column.name for column in metadata.columns)
    actual_dtypes = tuple(str(dtype) for dtype in dataset.frame.dtypes)
    metadata_dtypes = tuple(column.pandas_dtype for column in metadata.columns)
    if (
        actual_columns != metadata_columns
        or actual_dtypes != metadata_dtypes
        or len(dataset.frame) != metadata.row_count
    ):
        raise FinalModelLineageError(
            "Loaded dataset frame no longer matches the metadata captured during ingestion."
        )
    if metadata.target not in dataset.frame.columns:
        raise FinalModelLineageError("Selected target is missing from the loaded dataset frame.")
    try:
        reloaded = load_csv(
            metadata.source_path,
            target=metadata.target,
            options=CsvLoadOptions(
                encoding=metadata.encoding,
                delimiter=metadata.delimiter,
                max_file_size_bytes=metadata.file_size_bytes,
            ),
        )
    except MLForgeError as error:
        raise FinalModelLineageError(
            "Could not revalidate the selected dataset source before final fitting."
        ) from error
    if reloaded.metadata != metadata or not reloaded.frame.equals(dataset.frame):
        raise FinalModelLineageError(
            "Loaded dataset contents or source changed after ingestion; reload and rerun selection."
        )


def _failure_message(error: Exception) -> str:
    compact = " ".join(str(error).split())
    return compact[:2_000] if compact else "Final-model fitting failed without an error message."


def _warning_messages(
    profile_warnings: tuple[str, ...],
    captured: list[warnings.WarningMessage],
) -> tuple[str, ...]:
    messages = list(profile_warnings)
    messages.extend(
        f"{item.category.__name__}: {' '.join(str(item.message).split())}" for item in captured
    )
    return tuple(dict.fromkeys(message for message in messages if message.strip()))


def fit_selected_model(
    dataset: LoadedDataset,
    selection: CrossValidationResult,
    *,
    final_model_store: LocalFinalModelStore | None = None,
    artifact_store: LocalArtifactStore | None = None,
) -> FinalModelResult:
    """Refit persisted selection evidence on all rows and save its prediction-ready artifact."""
    if not isinstance(selection, CrossValidationResult):
        raise FinalModelError("selection must be a CrossValidationResult value.")
    manifest_destination = final_model_store or LocalFinalModelStore(Path("mlfinalmodels"))
    if not isinstance(manifest_destination, LocalFinalModelStore):
        raise FinalModelError("final_model_store must be a LocalFinalModelStore value.")
    default_artifact_root = (
        manifest_destination.root.parent / "artifacts"
        if final_model_store is not None
        else Path("artifacts")
    )
    artifact_destination = artifact_store or LocalArtifactStore(default_artifact_root)
    if not isinstance(artifact_destination, LocalArtifactStore):
        raise FinalModelError("artifact_store must be a LocalArtifactStore value.")

    persisted_selection = _persisted_selection(selection)
    _verify_dataset(dataset, persisted_selection)
    winner = persisted_selection.winner
    if winner is None or winner.primary_metric_mean is None:
        raise FinalModelLineageError(
            "Cross-validation selection must contain a successful rank-one estimator."
        )
    if winner.primary_metric_standard_deviation is None:
        raise FinalModelLineageError(
            "Cross-validation winner is missing primary-metric variability."
        )

    selected_configuration = persisted_selection.configuration
    selected_task = TaskType(selected_configuration.task)
    preprocessing = PreprocessingConfig(
        numeric_imputation=NumericImputationStrategy(selected_configuration.numeric_imputation),
        scale_numeric=selected_configuration.scale_numeric,
        categorical_fill_value=selected_configuration.categorical_fill_value,
    )
    overrides = FeatureOverrides(
        numeric=selected_configuration.numeric_overrides,
        categorical=selected_configuration.categorical_overrides,
    )
    training_config = TrainingConfig(
        task=selected_task,
        estimator=winner.estimator,
        split=SplitConfig(random_seed=selected_configuration.random_seed),
        preprocessing=preprocessing,
        feature_overrides=overrides,
    )
    estimator = create_estimator(training_config)
    estimator_parameters = _parameters(estimator)
    if estimator_parameters != winner.parameters:
        raise FinalModelLineageError(
            "Current estimator parameters do not match the persisted cross-validation winner."
        )

    selection_snapshot = FinalModelSelection(
        benchmark_id=persisted_selection.benchmark_id,
        manifest_sha256=_selection_digest(persisted_selection),
        estimator=winner.estimator,
        primary_metric=selected_configuration.primary_metric,
        primary_metric_mean=winner.primary_metric_mean,
        primary_metric_standard_deviation=winner.primary_metric_standard_deviation,
        fold_count=selected_configuration.fold_count,
        fold_plan_sha256=persisted_selection.fold_plan_sha256,
    )
    final_configuration = FinalModelConfiguration(
        task=selected_task.value,
        estimator=winner.estimator,
        random_seed=selected_configuration.random_seed,
        numeric_imputation=selected_configuration.numeric_imputation,
        scale_numeric=selected_configuration.scale_numeric,
        categorical_fill_value=selected_configuration.categorical_fill_value,
        numeric_overrides=selected_configuration.numeric_overrides,
        categorical_overrides=selected_configuration.categorical_overrides,
        estimator_parameters=estimator_parameters,
    )
    dataset_snapshot = _dataset_snapshot(dataset)
    environment = _environment_snapshot()
    final_model_id = str(uuid4())
    started_at = _now()
    features = dataset.frame.drop(columns=[dataset.metadata.target]).copy()
    target = dataset.frame[dataset.metadata.target].copy()
    feature_count = features.shape[1]
    profile_warnings: tuple[str, ...] = ()
    captured_warnings: list[warnings.WarningMessage] = []

    try:
        profile_warnings = profile_dataset(dataset).warnings
        feature_schema = infer_feature_schema(features, overrides=overrides)
        pipeline = build_final_model_pipeline(
            features,
            estimator,
            config=preprocessing,
            overrides=overrides,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                pipeline.fit(features, target)
            finally:
                captured_warnings = list(caught)
        pipeline_bytes = pickle.dumps(pipeline, protocol=5)
    except FinalModelStoreError:
        raise
    except (MLForgeError, ValueError, TypeError, OverflowError, pickle.PickleError) as error:
        failed_manifest = FinalModelManifest(
            schema_version=FINAL_MODEL_MANIFEST_SCHEMA_VERSION,
            final_model_id=final_model_id,
            status=RunStatus.FAILED,
            started_at=started_at,
            completed_at=_now(),
            selection_evidence=selection_snapshot,
            configuration=final_configuration,
            dataset=dataset_snapshot,
            environment=environment,
            fit_scope=FINAL_MODEL_FIT_SCOPE,
            training_rows=dataset_snapshot.row_count,
            feature_count=feature_count,
            artifact=None,
            warnings=_warning_messages(profile_warnings, captured_warnings),
            failure=RunFailure(
                error_type=type(error).__name__,
                message=_failure_message(error),
            ),
        )
        failed_path = manifest_destination.write(failed_manifest)
        raise FinalModelFailedError(
            f"Final-model fit {final_model_id} failed: {_failure_message(error)}",
            final_model_id=final_model_id,
            manifest_path=str(failed_path),
        ) from error

    artifact_contract = FinalModelArtifact(
        artifact_id=final_model_id,
        serialization_format=ARTIFACT_SERIALIZATION_FORMAT,
        pipeline_sha256=hashlib.sha256(pipeline_bytes).hexdigest(),
        pipeline_size_bytes=len(pipeline_bytes),
    )
    succeeded_manifest = FinalModelManifest(
        schema_version=FINAL_MODEL_MANIFEST_SCHEMA_VERSION,
        final_model_id=final_model_id,
        status=RunStatus.SUCCEEDED,
        started_at=started_at,
        completed_at=_now(),
        selection_evidence=selection_snapshot,
        configuration=final_configuration,
        dataset=dataset_snapshot,
        environment=environment,
        fit_scope=FINAL_MODEL_FIT_SCOPE,
        training_rows=dataset_snapshot.row_count,
        feature_count=feature_count,
        artifact=artifact_contract,
        warnings=_warning_messages(profile_warnings, captured_warnings),
        failure=None,
    )
    manifest_path = manifest_destination.write(succeeded_manifest)
    fitted = FinalModelResult(
        pipeline=pipeline,
        manifest=succeeded_manifest,
        manifest_path=manifest_path,
        feature_schema=feature_schema,
        feature_dtypes=tuple(
            (name, str(features[name].dtype)) for name in feature_schema.all_features
        ),
    )
    saved = artifact_destination.save_final(fitted)
    return replace(fitted, artifact_path=saved.path)
