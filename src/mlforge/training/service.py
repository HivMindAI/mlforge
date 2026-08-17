"""Application service for fitting, evaluating, and recording one local run."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import warnings
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from sklearn.base import BaseEstimator

from mlforge.datasets import LoadedDataset, profile_dataset
from mlforge.errors import MLForgeError, RunStoreError, TrainingError, TrainingFailedError
from mlforge.pipelines import (
    DatasetSplit,
    FeatureSchema,
    build_model_pipeline,
    infer_feature_schema,
    split_dataset,
)
from mlforge.runs import (
    RUN_MANIFEST_SCHEMA_VERSION,
    DatasetSnapshot,
    EnvironmentSnapshot,
    LocalRunStore,
    RunConfiguration,
    RunFailure,
    RunManifest,
    RunParameter,
    RunStatus,
    SplitSnapshot,
)
from mlforge.training.estimators import create_estimator
from mlforge.training.evaluation import evaluate_predictions
from mlforge.training.types import TrainingConfig, TrainingResult


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


def _parameter_value(value: object, *, name: str) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TrainingError(
        f"Estimator parameter {name!r} is not a reproducible JSON primitive: "
        f"{type(value).__name__}."
    )


def _configuration_snapshot(
    config: TrainingConfig,
    estimator: BaseEstimator,
) -> RunConfiguration:
    raw_parameters = estimator.get_params(deep=False)
    parameters = tuple(
        RunParameter(name=name, value=_parameter_value(value, name=name))
        for name, value in sorted(raw_parameters.items())
    )
    return RunConfiguration(
        task=config.task.value,
        estimator=config.estimator,
        validation_fraction=float(config.split.validation_fraction),
        random_seed=config.split.random_seed,
        stratify_requested=config.split.stratify,
        numeric_imputation=config.preprocessing.numeric_imputation.value,
        scale_numeric=config.preprocessing.scale_numeric,
        categorical_fill_value=config.preprocessing.categorical_fill_value,
        numeric_overrides=config.feature_overrides.numeric,
        categorical_overrides=config.feature_overrides.categorical,
        estimator_parameters=parameters,
    )


def _split_snapshot(split: DatasetSplit) -> SplitSnapshot:
    partitions: dict[str, list[int]] = {}
    for name, index in (
        ("train", split.train_features.index),
        ("validation", split.validation_features.index),
    ):
        values = index.tolist()
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TrainingError(
                "Training requires the integer source-row index created during CSV ingestion."
            )
        partitions[name] = values
    partition_content = json.dumps(
        partitions,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return SplitSnapshot(
        train_rows=len(split.train_features),
        validation_rows=len(split.validation_features),
        feature_count=split.train_features.shape[1],
        stratified=split.stratified,
        partition_sha256=hashlib.sha256(partition_content).hexdigest(),
    )


def _warning_messages(
    profile_warnings: tuple[str, ...],
    captured: list[warnings.WarningMessage],
) -> tuple[str, ...]:
    messages = list(profile_warnings)
    messages.extend(
        f"{item.category.__name__}: {' '.join(str(item.message).split())}" for item in captured
    )
    return tuple(dict.fromkeys(message for message in messages if message.strip()))


def _failure_message(error: Exception) -> str:
    compact = " ".join(str(error).split())
    return compact[:2_000] if compact else "Training failed without an error message."


def train(
    dataset: LoadedDataset,
    config: TrainingConfig,
    *,
    run_store: LocalRunStore | None = None,
) -> TrainingResult:
    """Fit and evaluate one baseline pipeline, always recording expected terminal outcomes."""
    if not isinstance(config, TrainingConfig):
        raise TrainingError("config must be a TrainingConfig value.")
    store = run_store or LocalRunStore(Path("mlruns"))
    if not isinstance(store, LocalRunStore):
        raise TrainingError("run_store must be a LocalRunStore value.")

    run_id = str(uuid4())
    started_at = _now()
    estimator = create_estimator(config)
    configuration = _configuration_snapshot(config, estimator)
    dataset_snapshot = _dataset_snapshot(dataset)
    environment = _environment_snapshot()
    split_snapshot: SplitSnapshot | None = None
    feature_schema: FeatureSchema | None = None
    profile_warnings: tuple[str, ...] = ()
    captured_warnings: list[warnings.WarningMessage] = []

    try:
        profile = profile_dataset(dataset)
        profile_warnings = profile.warnings
        split = split_dataset(dataset, task=config.task, config=config.split)
        split_snapshot = _split_snapshot(split)
        feature_schema = infer_feature_schema(
            split.train_features,
            overrides=config.feature_overrides,
        )
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
                    task=config.task,
                    actual=split.validation_target,
                    predicted=predictions,
                )
            finally:
                captured_warnings = list(caught)
    except RunStoreError:
        raise
    except (MLForgeError, ValueError, TypeError, OverflowError) as error:
        failed_manifest = RunManifest(
            schema_version=RUN_MANIFEST_SCHEMA_VERSION,
            run_id=run_id,
            status=RunStatus.FAILED,
            started_at=started_at,
            completed_at=_now(),
            configuration=configuration,
            dataset=dataset_snapshot,
            environment=environment,
            split=split_snapshot,
            metrics=(),
            warnings=_warning_messages(profile_warnings, captured_warnings),
            failure=RunFailure(
                error_type=type(error).__name__,
                message=_failure_message(error),
            ),
        )
        failed_path = store.write(failed_manifest)
        raise TrainingFailedError(
            f"Training run {run_id} failed: {_failure_message(error)}",
            run_id=run_id,
            manifest_path=str(failed_path),
        ) from error

    succeeded_manifest = RunManifest(
        schema_version=RUN_MANIFEST_SCHEMA_VERSION,
        run_id=run_id,
        status=RunStatus.SUCCEEDED,
        started_at=started_at,
        completed_at=_now(),
        configuration=configuration,
        dataset=dataset_snapshot,
        environment=environment,
        split=split_snapshot,
        metrics=metrics,
        warnings=_warning_messages(profile_warnings, captured_warnings),
        failure=None,
    )
    succeeded_path = store.write(succeeded_manifest)
    if feature_schema is None:
        raise TrainingError("Training completed without capturing its feature schema.")
    return TrainingResult(
        pipeline=pipeline,
        manifest=succeeded_manifest,
        manifest_path=succeeded_path,
        feature_schema=feature_schema,
        feature_dtypes=tuple(
            (name, str(split.train_features[name].dtype)) for name in feature_schema.all_features
        ),
    )
