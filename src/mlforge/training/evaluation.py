"""Task-appropriate finite metrics for held-out validation predictions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from mlforge.errors import TrainingError
from mlforge.pipelines import TaskType
from mlforge.runs import MetricValue


def _metric(name: str, value: float, *, higher_is_better: bool) -> MetricValue:
    if not math.isfinite(value):
        raise TrainingError(f"Evaluation metric {name!r} produced a non-finite value.")
    return MetricValue(name=name, value=value, higher_is_better=higher_is_better)


def evaluate_predictions(
    *,
    task: TaskType,
    actual: pd.Series[Any],
    predicted: Sequence[Any],
) -> tuple[MetricValue, ...]:
    """Evaluate held-out predictions with a stable metric set for the selected task."""
    if len(actual) != len(predicted):
        raise TrainingError("Prediction and validation target row counts do not match.")
    if len(actual) == 0:
        raise TrainingError("At least one validation prediction is required for evaluation.")

    if task is TaskType.CLASSIFICATION:
        metrics = (
            _metric(
                name="accuracy",
                value=float(accuracy_score(actual, predicted)),
                higher_is_better=True,
            ),
            _metric(
                name="balanced_accuracy",
                value=float(balanced_accuracy_score(actual, predicted)),
                higher_is_better=True,
            ),
            _metric(
                name="f1_weighted",
                value=float(f1_score(actual, predicted, average="weighted", zero_division=0)),
                higher_is_better=True,
            ),
        )
    elif task is TaskType.REGRESSION:
        if len(actual) < 2:
            raise TrainingError(
                "Regression evaluation requires at least two validation rows to calculate "
                "R-squared."
            )
        metrics = (
            _metric(
                name="mean_absolute_error",
                value=float(mean_absolute_error(actual, predicted)),
                higher_is_better=False,
            ),
            _metric(
                name="r2",
                value=float(r2_score(actual, predicted)),
                higher_is_better=True,
            ),
            _metric(
                name="root_mean_squared_error",
                value=float(root_mean_squared_error(actual, predicted)),
                higher_is_better=False,
            ),
        )
    else:
        raise TrainingError(f"Unsupported evaluation task: {task!r}.")
    return tuple(sorted(metrics, key=lambda metric: metric.name))
