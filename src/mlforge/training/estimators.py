"""Small explicit registry of reproducible baseline estimators."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from mlforge.errors import TrainingError
from mlforge.training.types import (
    DUMMY_CLASSIFIER,
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_CLASSIFIER,
    RANDOM_FOREST_REGRESSOR,
    RIDGE_REGRESSION,
    TrainingConfig,
)


def create_estimator(config: TrainingConfig) -> BaseEstimator:
    """Construct one supported estimator with deterministic resource-bounded defaults."""
    seed = config.split.random_seed
    if config.estimator == DUMMY_CLASSIFIER:
        return DummyClassifier(strategy="prior", random_state=seed)
    if config.estimator == LOGISTIC_REGRESSION:
        return LogisticRegression(max_iter=1_000, random_state=seed)
    if config.estimator == RANDOM_FOREST_CLASSIFIER:
        return RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=seed)
    if config.estimator == RIDGE_REGRESSION:
        return Ridge(alpha=1.0)
    if config.estimator == RANDOM_FOREST_REGRESSOR:
        return RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=seed)
    raise TrainingError(f"No estimator factory exists for {config.estimator!r}.")
