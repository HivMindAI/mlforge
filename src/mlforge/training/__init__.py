"""Local baseline training and evaluation public API."""

from mlforge.training.evaluation import CLASSIFICATION_METRICS, evaluate_predictions
from mlforge.training.service import train
from mlforge.training.types import (
    ALL_ESTIMATORS,
    CLASSIFICATION_ESTIMATORS,
    DUMMY_CLASSIFIER,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_CLASSIFIER,
    RANDOM_FOREST_REGRESSOR,
    REGRESSION_ESTIMATORS,
    RIDGE_REGRESSION,
    TrainingConfig,
    TrainingResult,
)

__all__ = [
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
]
