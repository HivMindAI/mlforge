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
        "ArtifactLineageKind",
        "ArtifactManifest",
        "FeatureRole",
        "LoadedArtifact",
        "LocalArtifactStore",
        "SavedArtifact",
        "inspect_artifact",
        "load_artifact",
        "verify_final_model_manifest",
        "verify_run_manifest",
    },
    "mlforge.benchmarks": {
        "BENCHMARK_MANIFEST_SCHEMA_VERSION",
        "CROSS_VALIDATION_MANIFEST_SCHEMA_VERSION",
        "DEFAULT_CLASSIFICATION_BENCHMARK_ESTIMATORS",
        "BenchmarkConfig",
        "BenchmarkConfiguration",
        "BenchmarkEntry",
        "BenchmarkManifest",
        "BenchmarkResult",
        "BenchmarkStatus",
        "CrossValidationConfig",
        "CrossValidationConfiguration",
        "CrossValidationEntry",
        "CrossValidationFoldResult",
        "CrossValidationFoldSnapshot",
        "CrossValidationManifest",
        "CrossValidationMetricSummary",
        "CrossValidationResult",
        "LocalBenchmarkStore",
        "LocalCrossValidationStore",
        "benchmark",
        "cross_validate_benchmark",
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
        "BenchmarkError",
        "BenchmarkFailedError",
        "BenchmarkStoreError",
        "ConfigurationError",
        "DatasetError",
        "DatasetFormatError",
        "DatasetPathError",
        "DatasetSplitError",
        "DatasetValidationError",
        "FinalModelError",
        "FinalModelFailedError",
        "FinalModelLineageError",
        "FinalModelStoreError",
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
    "mlforge.final_models": {
        "FINAL_MODEL_FIT_SCOPE",
        "FINAL_MODEL_MANIFEST_SCHEMA_VERSION",
        "FinalModelArtifact",
        "FinalModelConfiguration",
        "FinalModelManifest",
        "FinalModelResult",
        "FinalModelSelection",
        "LocalFinalModelStore",
        "fit_selected_model",
    },
    "mlforge.inference": {
        "PredictionRecord",
        "PredictionResult",
        "PredictionValue",
        "predict_csv",
        "predict_frame",
        "write_predictions_csv",
    },
    "mlforge.logging_config": {"LOGGER_NAME", "configure_logging"},
    "mlforge.pipelines": {
        "CrossValidationSplitConfig",
        "DatasetSplit",
        "FeatureOverrides",
        "FeatureSchema",
        "NumericImputationStrategy",
        "PreprocessingConfig",
        "SplitConfig",
        "TaskType",
        "build_final_model_pipeline",
        "build_final_preprocessor",
        "build_model_pipeline",
        "build_preprocessor",
        "infer_feature_schema",
        "split_dataset",
        "split_classification_folds",
        "split_partition_sha256",
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
        "CLASSIFICATION_METRICS",
        "DUMMY_CLASSIFIER",
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
    "mlforge.artifacts.verify_final_model_manifest": ("artifact", "final_model"),
    "mlforge.artifacts.verify_run_manifest": ("artifact", "run"),
    "mlforge.benchmarks.benchmark": (
        "dataset",
        "config",
        "run_store",
        "benchmark_store",
    ),
    "mlforge.benchmarks.cross_validate_benchmark": ("dataset", "config", "store"),
    "mlforge.datasets.load_csv": ("path", "target", "options"),
    "mlforge.datasets.load_feature_csv": ("path", "options"),
    "mlforge.datasets.profile_dataset": ("dataset",),
    "mlforge.final_models.fit_selected_model": (
        "dataset",
        "selection",
        "final_model_store",
        "artifact_store",
    ),
    "mlforge.inference.predict_csv": ("artifact", "path", "options"),
    "mlforge.inference.predict_frame": ("artifact", "frame"),
    "mlforge.inference.write_predictions_csv": ("result", "path"),
    "mlforge.pipelines.build_model_pipeline": ("split", "estimator", "config", "overrides"),
    "mlforge.pipelines.build_final_model_pipeline": (
        "features",
        "estimator",
        "config",
        "overrides",
    ),
    "mlforge.pipelines.build_final_preprocessor": ("features", "config", "overrides"),
    "mlforge.pipelines.build_preprocessor": ("split", "config", "overrides"),
    "mlforge.pipelines.infer_feature_schema": ("features", "overrides"),
    "mlforge.pipelines.split_dataset": ("dataset", "task", "config"),
    "mlforge.pipelines.split_classification_folds": ("dataset", "config"),
    "mlforge.pipelines.split_partition_sha256": ("split",),
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
