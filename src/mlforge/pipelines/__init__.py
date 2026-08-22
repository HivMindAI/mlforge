"""Leakage-safe splitting and preprocessing public API."""

from mlforge.pipelines.preprocessing import (
    build_final_model_pipeline,
    build_final_preprocessor,
    build_model_pipeline,
    build_preprocessor,
    infer_feature_schema,
)
from mlforge.pipelines.splitting import (
    split_classification_folds,
    split_dataset,
    split_partition_sha256,
)
from mlforge.pipelines.types import (
    CrossValidationSplitConfig,
    DatasetSplit,
    FeatureOverrides,
    FeatureSchema,
    NumericImputationStrategy,
    PreprocessingConfig,
    SplitConfig,
    TaskType,
)

__all__ = [
    "DatasetSplit",
    "CrossValidationSplitConfig",
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
]
