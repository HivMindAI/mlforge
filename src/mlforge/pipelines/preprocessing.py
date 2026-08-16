"""Construction of unfitted, leakage-safe scikit-learn pipelines."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mlforge.errors import PreprocessingError
from mlforge.pipelines.types import (
    DatasetSplit,
    FeatureOverrides,
    FeatureSchema,
    PreprocessingConfig,
)


def _categorical_object_frame(values: Any) -> Any:
    """Permit a string missing-value marker even for all-missing numeric-looking columns."""
    if isinstance(values, pd.DataFrame):
        return values.astype("object")
    return values.astype(object)


def _validate_feature_frame(features: pd.DataFrame) -> tuple[str, ...]:
    if not isinstance(features, pd.DataFrame):
        raise PreprocessingError("Features must be provided as a pandas DataFrame.")
    if features.empty:
        raise PreprocessingError("Training features must contain at least one row and one column.")
    if not features.columns.is_unique:
        raise PreprocessingError("Feature column names must be unique.")
    if any(not isinstance(name, str) or not name.strip() for name in features.columns):
        raise PreprocessingError("Feature column names must be non-blank strings.")
    return tuple(str(name) for name in features.columns)


def _validate_split(split: DatasetSplit) -> None:
    training_columns = _validate_feature_frame(split.train_features)
    validation_columns = tuple(str(name) for name in split.validation_features.columns)
    if validation_columns != training_columns:
        raise PreprocessingError(
            "Training and validation feature columns no longer match; recreate the dataset split."
        )
    if len(split.train_features) != len(split.train_target):
        raise PreprocessingError("Training feature and target row counts no longer match.")
    if len(split.validation_features) != len(split.validation_target):
        raise PreprocessingError("Validation feature and target row counts no longer match.")
    if not split.train_features.index.equals(split.train_target.index):
        raise PreprocessingError("Training feature and target row indices no longer align.")
    if not split.validation_features.index.equals(split.validation_target.index):
        raise PreprocessingError("Validation feature and target row indices no longer align.")
    if set(split.train_features.index).intersection(split.validation_features.index):
        raise PreprocessingError("Training and validation row indices must be disjoint.")


def _validate_finite_numeric_features(
    features: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    partition: str,
) -> None:
    non_finite = [
        name
        for name in columns
        if any(not math.isfinite(float(value)) for value in features[name].dropna().tolist())
    ]
    if non_finite:
        names = ", ".join(repr(name) for name in non_finite)
        raise PreprocessingError(
            f"{partition.capitalize()} numeric features contain infinite values: {names}. "
            "Replace them with finite values or missing values before preprocessing."
        )


def infer_feature_schema(
    features: pd.DataFrame,
    *,
    overrides: FeatureOverrides | None = None,
) -> FeatureSchema:
    """Assign every training feature to numeric or categorical preprocessing."""
    all_features = _validate_feature_frame(features)
    effective_overrides = overrides or FeatureOverrides()
    unknown = sorted(
        (set(effective_overrides.numeric) | set(effective_overrides.categorical))
        - set(all_features)
    )
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise PreprocessingError(f"Feature overrides reference unknown columns: {names}.")

    forced_numeric = set(effective_overrides.numeric)
    forced_categorical = set(effective_overrides.categorical)
    numeric: list[str] = []
    categorical: list[str] = []
    unsupported: list[str] = []

    for name in all_features:
        dtype = features[name].dtype
        if name in forced_numeric:
            if pd.api.types.is_bool_dtype(dtype) or not pd.api.types.is_numeric_dtype(dtype):
                raise PreprocessingError(
                    f"Feature {name!r} was forced numeric but has pandas dtype {dtype!s}."
                )
            numeric.append(name)
        elif name in forced_categorical or pd.api.types.is_bool_dtype(dtype):
            categorical.append(name)
        elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
            numeric.append(name)
        elif (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
        ):
            categorical.append(name)
        else:
            unsupported.append(f"{name!r} ({dtype!s})")

    if unsupported:
        details = ", ".join(unsupported)
        raise PreprocessingError(
            "Unsupported feature dtypes require explicit feature engineering before MLForge "
            f"preprocessing: {details}."
        )
    return FeatureSchema(
        all_features=all_features,
        numeric_features=tuple(numeric),
        categorical_features=tuple(categorical),
    )


def build_preprocessor(
    split: DatasetSplit,
    *,
    config: PreprocessingConfig | None = None,
    overrides: FeatureOverrides | None = None,
) -> ColumnTransformer:
    """Build a transformer whose learned state must come only from the training partition."""
    _validate_split(split)
    effective_config = config or PreprocessingConfig()
    schema = infer_feature_schema(split.train_features, overrides=overrides)
    _validate_finite_numeric_features(
        split.train_features,
        schema.numeric_features,
        partition="training",
    )
    _validate_finite_numeric_features(
        split.validation_features,
        schema.numeric_features,
        partition="validation",
    )
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if schema.numeric_features:
        numeric_steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(
                    strategy=effective_config.numeric_imputation.value,
                    keep_empty_features=True,
                ),
            )
        ]
        if effective_config.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(
            ("numeric", Pipeline(steps=numeric_steps), list(schema.numeric_features))
        )

    if schema.categorical_features:
        fill_value = effective_config.categorical_fill_value
        collisions = sorted(
            {
                name
                for features in (split.train_features, split.validation_features)
                for name in schema.categorical_features
                if bool(features[name].eq(fill_value).fillna(False).any())
            }
        )
        if collisions:
            names = ", ".join(repr(name) for name in collisions)
            raise PreprocessingError(
                f"Categorical fill value {fill_value!r} already occurs in split columns: "
                f"{names}. Configure a distinct fill value."
            )
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "to_object",
                    FunctionTransformer(
                        _categorical_object_frame,
                        validate=False,
                        feature_names_out="one-to-one",
                    ),
                ),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=fill_value,
                        keep_empty_features=True,
                    ),
                ),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                list(schema.categorical_features),
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model_pipeline(
    split: DatasetSplit,
    estimator: BaseEstimator,
    *,
    config: PreprocessingConfig | None = None,
    overrides: FeatureOverrides | None = None,
) -> Pipeline:
    """Return an unfitted preprocessing-and-estimator pipeline cloned from the input estimator."""
    if not isinstance(estimator, BaseEstimator):
        raise PreprocessingError("Estimator must follow scikit-learn's BaseEstimator convention.")
    try:
        estimator_copy = clone(estimator)
    except (TypeError, ValueError) as error:
        raise PreprocessingError(f"Estimator could not be cloned safely: {error}") from error

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(split, config=config, overrides=overrides),
            ),
            ("estimator", estimator_copy),
        ]
    )
