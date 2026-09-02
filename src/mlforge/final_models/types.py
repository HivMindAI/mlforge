"""Strict configuration, lineage, and results for explicit final-model fitting."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sklearn.pipeline import Pipeline

from mlforge.errors import (
    ConfigurationError,
    FinalModelError,
    FinalModelStoreError,
    RunStoreError,
)
from mlforge.pipelines import (
    FeatureOverrides,
    FeatureSchema,
    NumericImputationStrategy,
    PreprocessingConfig,
    SplitConfig,
    TaskType,
)
from mlforge.runs import (
    DatasetSnapshot,
    EnvironmentSnapshot,
    RunFailure,
    RunParameter,
    RunStatus,
)
from mlforge.training import (
    ALL_ESTIMATORS,
    CLASSIFICATION_ESTIMATORS,
    CLASSIFICATION_METRICS,
    REGRESSION_ESTIMATORS,
    REGRESSION_METRICS,
    TrainingConfig,
)

FINAL_MODEL_MANIFEST_SCHEMA_VERSION = 2
_SUPPORTED_FINAL_MODEL_MANIFEST_SCHEMA_VERSIONS = frozenset({1, 2})
FINAL_MODEL_FIT_SCOPE = "all_rows"


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise FinalModelStoreError(f"Final-model manifest {label} must be a JSON object.")
    return cast(dict[str, object], value)


def _keys(value: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing!r}")
    if extra:
        details.append(f"unexpected {extra!r}")
    raise FinalModelStoreError(
        f"Final-model manifest {label} has invalid fields: {', '.join(details)}."
    )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalModelStoreError(f"Final-model manifest {label} must be a non-blank string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FinalModelStoreError(f"Final-model manifest {label} must be an integer.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalModelStoreError(f"Final-model manifest {label} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise FinalModelStoreError(f"Final-model manifest {label} must be finite.")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise FinalModelStoreError(f"Final-model manifest {label} must be true or false.")
    return value


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise FinalModelStoreError(f"Final-model manifest {label} must be a JSON array.")
    return cast(list[object], value)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{label} entry") for item in _array(value, label))


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FinalModelStoreError(f"Final-model manifest {label} must be a lowercase SHA-256.")
    return digest


def _uuid(value: object, label: str) -> str:
    identifier = _string(value, label)
    try:
        parsed = UUID(identifier)
    except (ValueError, AttributeError, TypeError) as error:
        raise FinalModelStoreError(f"Final-model manifest {label} must be a UUID.") from error
    if str(parsed) != identifier:
        raise FinalModelStoreError(
            f"Final-model manifest {label} must use canonical lowercase UUID form."
        )
    return identifier


@dataclass(frozen=True, slots=True)
class FinalModelSelection:
    """Exact successful cross-validation decision used for final fitting."""

    benchmark_id: str
    manifest_sha256: str
    estimator: str
    primary_metric: str
    primary_metric_mean: float
    primary_metric_standard_deviation: float
    fold_count: int
    fold_plan_sha256: str

    def __post_init__(self) -> None:
        _uuid(self.benchmark_id, "selection benchmark_id")
        _sha256(self.manifest_sha256, "selection manifest_sha256")
        if self.estimator not in ALL_ESTIMATORS:
            raise FinalModelStoreError(
                f"Final-model selection estimator is unsupported: {self.estimator!r}."
            )
        if self.primary_metric not in CLASSIFICATION_METRICS + REGRESSION_METRICS:
            raise FinalModelStoreError(
                f"Final-model selection metric is unsupported: {self.primary_metric!r}."
            )
        _number(self.primary_metric_mean, "selection primary_metric_mean")
        deviation = _number(
            self.primary_metric_standard_deviation,
            "selection primary_metric_standard_deviation",
        )
        if deviation < 0:
            raise FinalModelStoreError(
                "Final-model selection metric standard deviation must not be negative."
            )
        folds = _integer(self.fold_count, "selection fold_count")
        if not 2 <= folds <= 10:
            raise FinalModelStoreError("Final-model selection fold_count must be between 2 and 10.")
        _sha256(self.fold_plan_sha256, "selection fold_plan_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "manifest_sha256": self.manifest_sha256,
            "estimator": self.estimator,
            "primary_metric": self.primary_metric,
            "primary_metric_mean": self.primary_metric_mean,
            "primary_metric_standard_deviation": self.primary_metric_standard_deviation,
            "fold_count": self.fold_count,
            "fold_plan_sha256": self.fold_plan_sha256,
        }

    @classmethod
    def from_object(cls, value: object) -> FinalModelSelection:
        data = _object(value, "selection")
        _keys(
            data,
            {
                "benchmark_id",
                "manifest_sha256",
                "estimator",
                "primary_metric",
                "primary_metric_mean",
                "primary_metric_standard_deviation",
                "fold_count",
                "fold_plan_sha256",
            },
            "selection",
        )
        return cls(
            benchmark_id=_uuid(data["benchmark_id"], "selection benchmark_id"),
            manifest_sha256=_sha256(data["manifest_sha256"], "selection manifest_sha256"),
            estimator=_string(data["estimator"], "selection estimator"),
            primary_metric=_string(data["primary_metric"], "selection primary_metric"),
            primary_metric_mean=_number(
                data["primary_metric_mean"], "selection primary_metric_mean"
            ),
            primary_metric_standard_deviation=_number(
                data["primary_metric_standard_deviation"],
                "selection primary_metric_standard_deviation",
            ),
            fold_count=_integer(data["fold_count"], "selection fold_count"),
            fold_plan_sha256=_sha256(data["fold_plan_sha256"], "selection fold_plan_sha256"),
        )


@dataclass(frozen=True, slots=True)
class FinalModelConfiguration:
    """Complete full-data fitting configuration captured independently of defaults."""

    task: str
    estimator: str
    random_seed: int
    numeric_imputation: str
    scale_numeric: bool
    categorical_fill_value: str
    numeric_overrides: tuple[str, ...]
    categorical_overrides: tuple[str, ...]
    estimator_parameters: tuple[RunParameter, ...]

    def __post_init__(self) -> None:
        try:
            TrainingConfig(
                task=TaskType(self.task),
                estimator=self.estimator,
                split=SplitConfig(random_seed=self.random_seed),
                preprocessing=PreprocessingConfig(
                    numeric_imputation=NumericImputationStrategy(self.numeric_imputation),
                    scale_numeric=self.scale_numeric,
                    categorical_fill_value=self.categorical_fill_value,
                ),
                feature_overrides=FeatureOverrides(
                    numeric=self.numeric_overrides,
                    categorical=self.categorical_overrides,
                ),
            )
        except (ConfigurationError, ValueError) as error:
            raise FinalModelStoreError(
                f"Final-model manifest configuration is invalid: {error}"
            ) from error
        if not isinstance(self.estimator_parameters, tuple) or any(
            not isinstance(item, RunParameter) for item in self.estimator_parameters
        ):
            raise FinalModelStoreError(
                "Final-model estimator parameters must be a RunParameter tuple."
            )
        names = tuple(item.name for item in self.estimator_parameters)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise FinalModelStoreError(
                "Final-model estimator parameters must have unique sorted names."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "estimator": self.estimator,
            "random_seed": self.random_seed,
            "numeric_imputation": self.numeric_imputation,
            "scale_numeric": self.scale_numeric,
            "categorical_fill_value": self.categorical_fill_value,
            "numeric_overrides": list(self.numeric_overrides),
            "categorical_overrides": list(self.categorical_overrides),
            "estimator_parameters": [item.to_dict() for item in self.estimator_parameters],
        }

    @classmethod
    def from_object(cls, value: object) -> FinalModelConfiguration:
        data = _object(value, "configuration")
        _keys(
            data,
            {
                "task",
                "estimator",
                "random_seed",
                "numeric_imputation",
                "scale_numeric",
                "categorical_fill_value",
                "numeric_overrides",
                "categorical_overrides",
                "estimator_parameters",
            },
            "configuration",
        )
        try:
            parameters = tuple(
                RunParameter.from_object(item)
                for item in _array(data["estimator_parameters"], "estimator_parameters")
            )
        except RunStoreError as error:
            raise FinalModelStoreError(
                f"Final-model estimator parameters are invalid: {error}"
            ) from error
        return cls(
            task=_string(data["task"], "configuration task"),
            estimator=_string(data["estimator"], "configuration estimator"),
            random_seed=_integer(data["random_seed"], "configuration random_seed"),
            numeric_imputation=_string(
                data["numeric_imputation"], "configuration numeric_imputation"
            ),
            scale_numeric=_boolean(data["scale_numeric"], "configuration scale_numeric"),
            categorical_fill_value=_string(
                data["categorical_fill_value"], "configuration categorical_fill_value"
            ),
            numeric_overrides=_string_tuple(
                data["numeric_overrides"], "configuration numeric_overrides"
            ),
            categorical_overrides=_string_tuple(
                data["categorical_overrides"], "configuration categorical_overrides"
            ),
            estimator_parameters=parameters,
        )


@dataclass(frozen=True, slots=True)
class FinalModelArtifact:
    """Prediction-ready artifact identity and executable-payload contract."""

    artifact_id: str
    serialization_format: str
    pipeline_sha256: str
    pipeline_size_bytes: int

    def __post_init__(self) -> None:
        _uuid(self.artifact_id, "artifact artifact_id")
        _string(self.serialization_format, "artifact serialization_format")
        _sha256(self.pipeline_sha256, "artifact pipeline_sha256")
        if _integer(self.pipeline_size_bytes, "artifact pipeline_size_bytes") <= 0:
            raise FinalModelStoreError("Final-model artifact pipeline_size_bytes must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "serialization_format": self.serialization_format,
            "pipeline_sha256": self.pipeline_sha256,
            "pipeline_size_bytes": self.pipeline_size_bytes,
        }

    @classmethod
    def from_object(cls, value: object) -> FinalModelArtifact:
        data = _object(value, "artifact")
        _keys(
            data,
            {
                "artifact_id",
                "serialization_format",
                "pipeline_sha256",
                "pipeline_size_bytes",
            },
            "artifact",
        )
        return cls(
            artifact_id=_uuid(data["artifact_id"], "artifact artifact_id"),
            serialization_format=_string(
                data["serialization_format"], "artifact serialization_format"
            ),
            pipeline_sha256=_sha256(data["pipeline_sha256"], "artifact pipeline_sha256"),
            pipeline_size_bytes=_integer(
                data["pipeline_size_bytes"], "artifact pipeline_size_bytes"
            ),
        )


@dataclass(frozen=True, slots=True)
class FinalModelManifest:
    """Immutable terminal record for one explicit full-dataset fitting attempt."""

    schema_version: int
    final_model_id: str
    status: RunStatus
    started_at: str
    completed_at: str
    selection_evidence: FinalModelSelection
    configuration: FinalModelConfiguration
    dataset: DatasetSnapshot
    environment: EnvironmentSnapshot
    fit_scope: str
    training_rows: int
    feature_count: int
    artifact: FinalModelArtifact | None
    warnings: tuple[str, ...]
    failure: RunFailure | None

    def __post_init__(self) -> None:
        if (
            _integer(self.schema_version, "schema_version")
            not in _SUPPORTED_FINAL_MODEL_MANIFEST_SCHEMA_VERSIONS
        ):
            raise FinalModelStoreError(
                f"Unsupported final-model manifest schema version: {self.schema_version}."
            )
        _uuid(self.final_model_id, "final_model_id")
        if not isinstance(self.status, RunStatus):
            raise FinalModelStoreError("Final-model status must be a RunStatus value.")
        for label, value in (("started_at", self.started_at), ("completed_at", self.completed_at)):
            _string(value, label)
        try:
            started = datetime.fromisoformat(self.started_at)
            completed = datetime.fromisoformat(self.completed_at)
        except ValueError as error:
            raise FinalModelStoreError(
                "Final-model timestamps must use ISO 8601 format."
            ) from error
        if started.tzinfo is None or completed.tzinfo is None or completed < started:
            raise FinalModelStoreError("Final-model timestamps must be timezone-aware and ordered.")
        if not isinstance(self.selection_evidence, FinalModelSelection):
            raise FinalModelStoreError("Final-model selection is invalid.")
        if not isinstance(self.configuration, FinalModelConfiguration):
            raise FinalModelStoreError("Final-model configuration is invalid.")
        task = TaskType(self.configuration.task)
        if self.schema_version == 1 and task is not TaskType.CLASSIFICATION:
            raise FinalModelStoreError(
                "Final-model manifest schema version 1 supports classification only."
            )
        expected_estimators = (
            CLASSIFICATION_ESTIMATORS if task is TaskType.CLASSIFICATION else REGRESSION_ESTIMATORS
        )
        expected_metrics = (
            CLASSIFICATION_METRICS if task is TaskType.CLASSIFICATION else REGRESSION_METRICS
        )
        if self.selection_evidence.estimator not in expected_estimators:
            raise FinalModelStoreError(
                "Final-model selection estimator does not match the configured task."
            )
        if self.selection_evidence.primary_metric not in expected_metrics:
            raise FinalModelStoreError(
                "Final-model selection metric does not match the configured task."
            )
        if self.selection_evidence.estimator != self.configuration.estimator:
            raise FinalModelStoreError(
                "Final-model configuration estimator does not match its selection."
            )
        if not isinstance(self.dataset, DatasetSnapshot):
            raise FinalModelStoreError("Final-model dataset snapshot is invalid.")
        if not isinstance(self.environment, EnvironmentSnapshot):
            raise FinalModelStoreError("Final-model environment snapshot is invalid.")
        if self.fit_scope != FINAL_MODEL_FIT_SCOPE:
            raise FinalModelStoreError(f"Final-model fit_scope must be {FINAL_MODEL_FIT_SCOPE!r}.")
        if _integer(self.training_rows, "training_rows") != self.dataset.row_count:
            raise FinalModelStoreError(
                "Final-model training row count must equal the selected dataset row count."
            )
        if _integer(self.feature_count, "feature_count") <= 0:
            raise FinalModelStoreError("Final-model feature_count must be positive.")
        if self.feature_count + 1 != self.dataset.column_count:
            raise FinalModelStoreError(
                "Final-model feature count must exclude exactly the target column."
            )
        if self.artifact is not None and not isinstance(self.artifact, FinalModelArtifact):
            raise FinalModelStoreError("Final-model artifact contract is invalid.")
        if self.artifact is not None and self.artifact.artifact_id != self.final_model_id:
            raise FinalModelStoreError(
                "Final-model artifact identity must equal the final model identity."
            )
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(message, str) or not message.strip() for message in self.warnings
        ):
            raise FinalModelStoreError("Final-model warnings must be non-blank strings.")
        if len(set(self.warnings)) != len(self.warnings):
            raise FinalModelStoreError("Final-model warnings must be unique.")
        if self.status is RunStatus.SUCCEEDED:
            if self.failure is not None:
                raise FinalModelStoreError("A successful final model cannot contain a failure.")
            if self.artifact is None:
                raise FinalModelStoreError(
                    "A successful final model requires an artifact contract."
                )
        elif self.failure is None:
            raise FinalModelStoreError("A failed final model requires failure details.")
        elif self.artifact is not None:
            raise FinalModelStoreError("A failed final model cannot claim an artifact contract.")
        if self.failure is not None and not isinstance(self.failure, RunFailure):
            raise FinalModelStoreError("Final-model failure is invalid.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "final_model_id": self.final_model_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "selection_evidence": self.selection_evidence.to_dict(),
            "configuration": self.configuration.to_dict(),
            "dataset": self.dataset.to_dict(),
            "environment": self.environment.to_dict(),
            "final_fit": {
                "fit_scope": self.fit_scope,
                "training_rows": self.training_rows,
                "feature_count": self.feature_count,
            },
            "artifact": self.artifact.to_dict() if self.artifact is not None else None,
            "warnings": list(self.warnings),
            "failure": self.failure.to_dict() if self.failure is not None else None,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, content: str) -> FinalModelManifest:
        try:
            raw: object = json.loads(content)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise FinalModelStoreError(
                f"Final-model manifest is not valid JSON: {error}"
            ) from error
        data = _object(raw, "root")
        _keys(
            data,
            {
                "schema_version",
                "final_model_id",
                "status",
                "started_at",
                "completed_at",
                "selection_evidence",
                "configuration",
                "dataset",
                "environment",
                "final_fit",
                "artifact",
                "warnings",
                "failure",
            },
            "root",
        )
        final_fit = _object(data["final_fit"], "final_fit")
        _keys(final_fit, {"fit_scope", "training_rows", "feature_count"}, "final_fit")
        try:
            status = RunStatus(_string(data["status"], "status"))
            dataset = DatasetSnapshot.from_object(data["dataset"])
            environment = EnvironmentSnapshot.from_object(data["environment"])
            failure = None if data["failure"] is None else RunFailure.from_object(data["failure"])
        except (ValueError, RunStoreError) as error:
            raise FinalModelStoreError(
                f"Final-model manifest contains invalid shared evidence: {error}"
            ) from error
        return cls(
            schema_version=_integer(data["schema_version"], "schema_version"),
            final_model_id=_uuid(data["final_model_id"], "final_model_id"),
            status=status,
            started_at=_string(data["started_at"], "started_at"),
            completed_at=_string(data["completed_at"], "completed_at"),
            selection_evidence=FinalModelSelection.from_object(data["selection_evidence"]),
            configuration=FinalModelConfiguration.from_object(data["configuration"]),
            dataset=dataset,
            environment=environment,
            fit_scope=_string(final_fit["fit_scope"], "final_fit fit_scope"),
            training_rows=_integer(final_fit["training_rows"], "final_fit training_rows"),
            feature_count=_integer(final_fit["feature_count"], "final_fit feature_count"),
            artifact=(
                None
                if data["artifact"] is None
                else FinalModelArtifact.from_object(data["artifact"])
            ),
            warnings=_string_tuple(data["warnings"], "warnings"),
            failure=failure,
        )

    @property
    def selection(self) -> FinalModelSelection:
        """Return the source evaluation evidence without treating it as final-fit metrics."""
        return self.selection_evidence


@dataclass(frozen=True, slots=True)
class FinalModelResult:
    """Fitted full-dataset pipeline and its persisted immutable lineage record."""

    pipeline: Pipeline
    manifest: FinalModelManifest
    manifest_path: Path
    feature_schema: FeatureSchema
    feature_dtypes: tuple[tuple[str, str], ...]
    artifact_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, Pipeline):
            raise FinalModelError("Final-model pipeline must be a scikit-learn Pipeline.")
        if not isinstance(self.manifest, FinalModelManifest):
            raise FinalModelError("Final-model result manifest is invalid.")
        if self.manifest.status is not RunStatus.SUCCEEDED:
            raise FinalModelError("Final-model result requires a successful manifest.")
        if not isinstance(self.manifest_path, Path):
            raise FinalModelError("Final-model manifest_path must be a pathlib.Path.")
        if self.artifact_path is not None and not isinstance(self.artifact_path, Path):
            raise FinalModelError("Final-model artifact_path must be a pathlib.Path or None.")
        if not isinstance(self.feature_schema, FeatureSchema):
            raise FinalModelError("Final-model feature_schema is invalid.")
        if not isinstance(self.feature_dtypes, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or any(not isinstance(value, str) or not value.strip() for value in item)
            for item in self.feature_dtypes
        ):
            raise FinalModelError("Final-model feature dtypes must be named string pairs.")
        if tuple(name for name, _ in self.feature_dtypes) != self.feature_schema.all_features:
            raise FinalModelError(
                "Final-model feature dtypes must follow the complete feature schema order."
            )
