"""Typed configuration and results for supervised tabular pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import pandas as pd

from mlforge.errors import ConfigurationError, PreprocessingError


class TaskType(StrEnum):
    """Supported supervised learning task types."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class NumericImputationStrategy(StrEnum):
    """Supported training-derived statistics for missing numeric values."""

    MEAN = "mean"
    MEDIAN = "median"


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Deterministic train/validation split settings."""

    validation_fraction: float = 0.2
    random_seed: int = 42
    stratify: bool | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous or invalid splitting behavior."""
        if isinstance(self.validation_fraction, bool) or not isinstance(
            self.validation_fraction, (int, float)
        ):
            raise ConfigurationError("Validation fraction must be a number between 0 and 1.")
        if not 0 < float(self.validation_fraction) < 1:
            raise ConfigurationError("Validation fraction must be greater than 0 and less than 1.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ConfigurationError("Random seed must be an integer.")
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise ConfigurationError("Random seed must be between 0 and 4294967295.")
        if self.stratify is not None and not isinstance(self.stratify, bool):
            raise ConfigurationError("Stratify must be true, false, or unset.")


@dataclass(frozen=True, slots=True)
class CrossValidationSplitConfig:
    """Resource-bounded deterministic stratified K-fold settings."""

    fold_count: int = 5
    random_seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.fold_count, bool) or not isinstance(self.fold_count, int):
            raise ConfigurationError("Cross-validation fold count must be an integer.")
        if not 2 <= self.fold_count <= 10:
            raise ConfigurationError("Cross-validation fold count must be between 2 and 10.")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ConfigurationError("Cross-validation random seed must be an integer.")
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise ConfigurationError(
                "Cross-validation random seed must be between 0 and 4294967295."
            )


@dataclass(frozen=True, slots=True)
class FeatureOverrides:
    """Explicit roles for columns whose pandas dtype is ambiguous."""

    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Require unique, non-overlapping, named feature overrides."""
        for role, names in (("numeric", self.numeric), ("categorical", self.categorical)):
            if not isinstance(names, tuple):
                raise ConfigurationError(
                    f"{role.capitalize()} feature overrides must be provided as a tuple."
                )
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ConfigurationError(f"{role.capitalize()} feature names must not be blank.")
            if len(set(names)) != len(names):
                raise ConfigurationError(f"{role.capitalize()} feature overrides must be unique.")

        overlap = sorted(set(self.numeric).intersection(self.categorical))
        if overlap:
            overlap_names = ", ".join(repr(name) for name in overlap)
            raise ConfigurationError(
                f"Feature overrides cannot be both numeric and categorical: {overlap_names}."
            )


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configuration for an unfitted tabular feature transformer."""

    numeric_imputation: NumericImputationStrategy = NumericImputationStrategy.MEDIAN
    scale_numeric: bool = True
    categorical_fill_value: str = "__mlforge_missing__"

    def __post_init__(self) -> None:
        """Reject preprocessing options that cannot be represented safely."""
        if not isinstance(self.numeric_imputation, NumericImputationStrategy):
            raise ConfigurationError(
                "Numeric imputation must be a NumericImputationStrategy value."
            )
        if not isinstance(self.scale_numeric, bool):
            raise ConfigurationError("Scale numeric must be true or false.")
        if (
            not isinstance(self.categorical_fill_value, str)
            or not self.categorical_fill_value.strip()
        ):
            raise ConfigurationError("Categorical fill value must not be blank.")


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    """Ordered assignment of every input feature to one transformer family."""

    all_features: tuple[str, ...]
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]

    def __post_init__(self) -> None:
        """Keep feature-role metadata complete and non-overlapping."""
        for role, names in (
            ("all", self.all_features),
            ("numeric", self.numeric_features),
            ("categorical", self.categorical_features),
        ):
            if not isinstance(names, tuple):
                raise PreprocessingError(
                    f"{role.capitalize()} feature schema entries must be provided as a tuple."
                )
        if not self.all_features:
            raise PreprocessingError("At least one feature column is required.")
        if len(set(self.all_features)) != len(self.all_features):
            raise PreprocessingError("Feature column names must be unique.")
        assigned = self.numeric_features + self.categorical_features
        if len(set(assigned)) != len(assigned) or set(assigned) != set(self.all_features):
            raise PreprocessingError(
                "Numeric and categorical feature roles must partition all feature columns."
            )


@dataclass(slots=True)
class DatasetSplit:
    """Index-preserving train/validation data for one supervised task."""

    train_features: pd.DataFrame
    validation_features: pd.DataFrame
    train_target: pd.Series[Any]
    validation_target: pd.Series[Any]
    target_name: str
    task: TaskType
    config: SplitConfig
    stratified: bool
