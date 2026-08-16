"""Compatibility tests for MLForge's deliberately explicit public API."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any, cast

PUBLIC_EXPORTS = {
    "mlforge": {"__version__"},
    "mlforge.artifacts": {
        "ARTIFACT_MANIFEST_SCHEMA_VERSION",
        "ARTIFACT_SERIALIZATION_FORMAT",
        "ARTIFACT_SUFFIX",
        "ArtifactEnvironment",
        "ArtifactFeature",
        "ArtifactManifest",
        "FeatureRole",
        "LoadedArtifact",
        "LocalArtifactStore",
        "SavedArtifact",
        "inspect_artifact",
        "load_artifact",
        "verify_run_manifest",
    },
    "mlforge.config": {
        "ApplicationConfig",
        "LOG_LEVEL_ENVIRONMENT_VARIABLE",
        "LogLevel",
    },
    "mlforge.datasets": {
        "ColumnKind",
        "ColumnMetadata",
        "ColumnProfile",
        "CsvLoadOptions",
        "DatasetMetadata",
        "DatasetProfile",
        "LoadedDataset",
        "NumericSummary",
        "TargetProfile",
        "TaskHint",
        "ValueFrequency",
        "load_csv",
        "load_feature_csv",
        "profile_dataset",
    },
    "mlforge.errors": {
        "ArtifactCompatibilityError",
        "ArtifactError",
        "ArtifactFormatError",
        "ArtifactIntegrityError",
        "ArtifactPathError",
        "ArtifactTrustError",
        "ConfigurationError",
        "DatasetError",
        "DatasetFormatError",
        "DatasetPathError",
        "DatasetSplitError",
        "DatasetValidationError",
        "InferenceError",
        "MLForgeError",
        "PipelineError",
        "PredictionSchemaError",
        "PreprocessingError",
        "RunComparisonError",
        "RunError",
        "RunStoreError",
        "TrainingError",
        "TrainingFailedError",
    },
    "mlforge.inference": {
        "PredictionRecord",
        "PredictionResult",
        "PredictionValue",
        "predict_csv",
        "predict_frame",
    },
    "mlforge.logging_config": {"LOGGER_NAME", "configure_logging"},
    "mlforge.pipelines": {
        "DatasetSplit",
        "FeatureOverrides",
        "FeatureSchema",
        "NumericImputationStrategy",
        "PreprocessingConfig",
        "SplitConfig",
        "TaskType",
        "build_model_pipeline",
        "build_preprocessor",
        "infer_feature_schema",
        "split_dataset",
    },
    "mlforge.runs": {
        "RUN_MANIFEST_SCHEMA_VERSION",
        "DatasetSnapshot",
        "EnvironmentSnapshot",
        "LocalRunStore",
        "MetricValue",
        "RunComparison",
        "RunComparisonEntry",
        "RunConfiguration",
        "RunFailure",
        "RunManifest",
        "RunParameter",
        "RunStatus",
        "SplitSnapshot",
        "compare_runs",
    },
    "mlforge.training": {
        "ALL_ESTIMATORS",
        "CLASSIFICATION_ESTIMATORS",
        "LOGISTIC_REGRESSION",
        "RANDOM_FOREST_CLASSIFIER",
        "RANDOM_FOREST_REGRESSOR",
        "REGRESSION_ESTIMATORS",
        "RIDGE_REGRESSION",
        "TrainingConfig",
        "TrainingResult",
        "evaluate_predictions",
        "train",
    },
}

FUNCTION_PARAMETERS = {
    "mlforge.artifacts.inspect_artifact": ("path",),
    "mlforge.artifacts.load_artifact": ("path", "trusted"),
    "mlforge.artifacts.verify_run_manifest": ("artifact", "run"),
    "mlforge.datasets.load_csv": ("path", "target", "options"),
    "mlforge.datasets.load_feature_csv": ("path", "options"),
    "mlforge.datasets.profile_dataset": ("dataset",),
    "mlforge.inference.predict_csv": ("artifact", "path", "options"),
    "mlforge.inference.predict_frame": ("artifact", "frame"),
    "mlforge.pipelines.build_model_pipeline": ("split", "estimator", "config", "overrides"),
    "mlforge.pipelines.build_preprocessor": ("split", "config", "overrides"),
    "mlforge.pipelines.infer_feature_schema": ("features", "overrides"),
    "mlforge.pipelines.split_dataset": ("dataset", "task", "config"),
    "mlforge.runs.compare_runs": ("manifests", "metric"),
    "mlforge.training.evaluate_predictions": ("task", "actual", "predicted"),
    "mlforge.training.train": ("dataset", "config", "run_store"),
}


def _resolve(qualified_name: str) -> Callable[..., Any]:
    module_name, attribute = qualified_name.rsplit(".", maxsplit=1)
    value = getattr(importlib.import_module(module_name), attribute)
    assert callable(value)
    return cast(Callable[..., Any], value)


def test_domain_modules_have_exact_explicit_exports() -> None:
    """New or removed public names should require an intentional contract change."""
    for module_name, expected in PUBLIC_EXPORTS.items():
        module = importlib.import_module(module_name)
        actual = set(module.__all__)

        assert actual == expected, module_name
        assert all(hasattr(module, name) for name in actual), module_name


def test_primary_function_parameter_names_are_stable() -> None:
    """Keyword-capable parameter names are part of the documented Python API."""
    for qualified_name, expected in FUNCTION_PARAMETERS.items():
        actual = tuple(inspect.signature(_resolve(qualified_name)).parameters)

        assert actual == expected, qualified_name
