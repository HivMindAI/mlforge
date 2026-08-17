"""Leakage-safe splitting and preprocessing public API."""

from mlforge.pipelines.preprocessing import (
    build_model_pipeline,
    build_preprocessor,
    infer_feature_schema,
)
from mlforge.pipelines.splitting import split_dataset
from mlforge.pipelines.types import (
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
]
