"""Local baseline training and evaluation public API."""

from mlforge.training.evaluation import evaluate_predictions
from mlforge.training.service import train
from mlforge.training.types import (
    ALL_ESTIMATORS,
    CLASSIFICATION_ESTIMATORS,
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
