"""Deterministic, task-aware dataset splitting before any fitting occurs."""

from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd
from sklearn.model_selection import train_test_split

from mlforge.datasets import LoadedDataset
from mlforge.datasets.validation import validate_loaded_dataset
from mlforge.errors import DatasetSplitError
from mlforge.pipelines.types import DatasetSplit, SplitConfig, TaskType


def _validate_target(target: pd.Series[Any], task: TaskType) -> None:
    if bool(target.isna().any()):
        raise DatasetSplitError(
            f"Target column {target.name!r} contains missing values; remove or repair them before "
            "splitting."
        )

    unique_count = int(target.nunique(dropna=False))
    if pd.api.types.is_complex_dtype(target.dtype):
        raise DatasetSplitError("Target values must not be complex numbers.")
    if (
        pd.api.types.is_numeric_dtype(target.dtype)
        and not pd.api.types.is_bool_dtype(target.dtype)
        and any(not math.isfinite(float(value)) for value in target.tolist())
    ):
        raise DatasetSplitError("Numeric target values must all be finite.")

    if task is TaskType.CLASSIFICATION:
        if unique_count < 2:
            raise DatasetSplitError("Classification requires at least two target classes.")
        return

    if pd.api.types.is_bool_dtype(target.dtype) or not pd.api.types.is_numeric_dtype(target.dtype):
        raise DatasetSplitError("Regression requires a numeric, non-boolean target column.")


def _validate_stratification(
    target: pd.Series[Any],
    *,
    validation_rows: int,
    training_rows: int,
) -> None:
    class_counts = target.value_counts(dropna=False)
    class_count = len(class_counts)
    if int(class_counts.min()) < 2:
        raise DatasetSplitError(
            "Stratification requires at least two rows in every target class. Add data or set "
            "stratify=False explicitly."
        )
    if validation_rows < class_count or training_rows < class_count:
        raise DatasetSplitError(
            "The requested split is too small to place every target class in both partitions. "
            "Adjust validation_fraction, add data, or set stratify=False explicitly."
        )


def split_dataset(
    dataset: LoadedDataset,
    *,
    task: TaskType,
    config: SplitConfig | None = None,
) -> DatasetSplit:
    """Split a validated dataset into features and target without fitting anything."""
    if not isinstance(task, TaskType):
        raise DatasetSplitError("Task must be a TaskType value.")

    validate_loaded_dataset(dataset, operation="splitting")
    effective_config = config or SplitConfig()
    frame = dataset.frame
    target_name = dataset.metadata.target
    features = frame.drop(columns=[target_name])
    if features.shape[1] == 0:
        raise DatasetSplitError("Supervised splitting requires at least one feature column.")
    if len(frame) < 2:
        raise DatasetSplitError("At least two rows are required for a train/validation split.")

    target = frame[target_name]
    if not isinstance(target, pd.Series):
        raise DatasetSplitError("The configured target must resolve to exactly one column.")
    _validate_target(target, task)

    if task is TaskType.REGRESSION and effective_config.stratify is True:
        raise DatasetSplitError("Target stratification is supported only for classification.")
    stratified = (
        task is TaskType.CLASSIFICATION
        if effective_config.stratify is None
        else effective_config.stratify
    )

    validation_rows = math.ceil(len(frame) * float(effective_config.validation_fraction))
    training_rows = len(frame) - validation_rows
    if validation_rows < 1 or training_rows < 1:
        raise DatasetSplitError(
            "The requested validation fraction must leave at least one row in each partition."
        )
    if stratified:
        _validate_stratification(
            target,
            validation_rows=validation_rows,
            training_rows=training_rows,
        )

    try:
        raw_split = train_test_split(
            features,
            target,
            test_size=float(effective_config.validation_fraction),
            random_state=effective_config.random_seed,
            shuffle=True,
            stratify=target if stratified else None,
        )
    except ValueError as error:
        raise DatasetSplitError(f"Could not create the requested dataset split: {error}") from error

    train_features = cast(pd.DataFrame, raw_split[0]).copy()
    validation_features = cast(pd.DataFrame, raw_split[1]).copy()
    train_target = cast("pd.Series[Any]", raw_split[2]).copy()
    validation_target = cast("pd.Series[Any]", raw_split[3]).copy()

    if task is TaskType.CLASSIFICATION and int(train_target.nunique(dropna=False)) < 2:
        raise DatasetSplitError(
            "The training partition contains fewer than two classes. Use stratification, add "
            "data, or choose a different random seed."
        )

    return DatasetSplit(
        train_features=train_features,
        validation_features=validation_features,
        train_target=train_target,
        validation_target=validation_target,
        target_name=target_name,
        task=task,
        config=effective_config,
        stratified=stratified,
    )
