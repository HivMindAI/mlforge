"""Typed configuration and results for local supervised training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sklearn.pipeline import Pipeline

from mlforge.errors import ConfigurationError, TrainingError
from mlforge.pipelines import (
    FeatureOverrides,
    FeatureSchema,
    PreprocessingConfig,
    SplitConfig,
    TaskType,
)
from mlforge.runs import RunManifest

LOGISTIC_REGRESSION = "logistic-regression"
DUMMY_CLASSIFIER = "dummy-classifier"
RANDOM_FOREST_CLASSIFIER = "random-forest-classifier"
RIDGE_REGRESSION = "ridge-regression"
RANDOM_FOREST_REGRESSOR = "random-forest-regressor"

CLASSIFICATION_ESTIMATORS = frozenset(
    {DUMMY_CLASSIFIER, LOGISTIC_REGRESSION, RANDOM_FOREST_CLASSIFIER}
)
REGRESSION_ESTIMATORS = frozenset({RIDGE_REGRESSION, RANDOM_FOREST_REGRESSOR})
ALL_ESTIMATORS = tuple(sorted(CLASSIFICATION_ESTIMATORS | REGRESSION_ESTIMATORS))


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Complete explicit configuration for one local training attempt."""

    task: TaskType
    estimator: str
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_overrides: FeatureOverrides = field(default_factory=FeatureOverrides)

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskType):
            raise ConfigurationError("Training task must be a TaskType value.")
        if not isinstance(self.estimator, str) or self.estimator not in ALL_ESTIMATORS:
            supported = ", ".join(ALL_ESTIMATORS)
            raise ConfigurationError(
                f"Unsupported estimator {self.estimator!r}. Choose one of: {supported}."
            )
        supported_for_task = (
            CLASSIFICATION_ESTIMATORS
            if self.task is TaskType.CLASSIFICATION
            else REGRESSION_ESTIMATORS
        )
        if self.estimator not in supported_for_task:
            choices = ", ".join(sorted(supported_for_task))
            raise ConfigurationError(
                f"Estimator {self.estimator!r} does not support task {self.task.value!r}. "
                f"Choose one of: {choices}."
            )
        if not isinstance(self.split, SplitConfig):
            raise ConfigurationError("Training split must be a SplitConfig value.")
        if not isinstance(self.preprocessing, PreprocessingConfig):
            raise ConfigurationError("Training preprocessing must be a PreprocessingConfig value.")
        if not isinstance(self.feature_overrides, FeatureOverrides):
            raise ConfigurationError("Training feature_overrides must be a FeatureOverrides value.")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Fitted in-memory pipeline and its persisted successful run record."""

    pipeline: Pipeline
    manifest: RunManifest
    manifest_path: Path
    feature_schema: FeatureSchema
    feature_dtypes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pipeline, Pipeline):
            raise TrainingError("Training result pipeline must be a scikit-learn Pipeline.")
        if not isinstance(self.manifest, RunManifest):
            raise TrainingError("Training result manifest must be a RunManifest.")
        if not isinstance(self.manifest_path, Path):
            raise TrainingError("Training result manifest_path must be a pathlib.Path.")
        if not isinstance(self.feature_schema, FeatureSchema):
            raise TrainingError("Training result feature_schema must be a FeatureSchema.")
        if not isinstance(self.feature_dtypes, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or any(not isinstance(value, str) or not value.strip() for value in item)
            for item in self.feature_dtypes
        ):
            raise TrainingError("Training result feature dtypes must be named string pairs.")
        dtype_names = tuple(name for name, _ in self.feature_dtypes)
        if dtype_names != self.feature_schema.all_features:
            raise TrainingError(
                "Training result feature dtypes must follow the complete feature schema order."
            )
