"""Behavioral tests for leakage-safe preprocessing construction."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from mlforge.datasets import load_csv
from mlforge.errors import ConfigurationError, PreprocessingError
from mlforge.pipelines import (
    DatasetSplit,
    FeatureOverrides,
    NumericImputationStrategy,
    PreprocessingConfig,
    SplitConfig,
    TaskType,
    build_model_pipeline,
    build_preprocessor,
    infer_feature_schema,
    split_dataset,
)


def _classification_split(tmp_path: Path, *, include_empty: bool = False) -> DatasetSplit:
    header = "number,city,empty,target" if include_empty else "number,city,target"
    rows = []
    for index in range(20):
        target = "yes" if index % 2 else "no"
        if include_empty:
            rows.append(f"{index},{'north' if index % 3 else 'south'},,{target}")
        else:
            rows.append(f"{index},{'north' if index % 3 else 'south'},{target}")
    path = tmp_path / "classification.csv"
    path.write_text("\n".join([header, *rows, ""]), encoding="utf-8")
    dataset = load_csv(path, target="target")
    return split_dataset(
        dataset,
        task=TaskType.CLASSIFICATION,
        config=SplitConfig(validation_fraction=0.25, random_seed=3),
    )


def test_schema_inference_assigns_numeric_boolean_and_string_features() -> None:
    """Physical dtypes should map to the two documented transformer families."""
    features = pd.DataFrame({"age": [20, 30], "active": [True, False], "city": ["Kabul", "Herat"]})

    schema = infer_feature_schema(features)

    assert schema.all_features == ("age", "active", "city")
    assert schema.numeric_features == ("age",)
    assert schema.categorical_features == ("active", "city")


def test_explicit_override_resolves_an_all_missing_categorical_column() -> None:
    """Users should be able to classify a dtype-ambiguous empty column explicitly."""
    features = pd.DataFrame({"empty": [math.nan, math.nan], "value": [1.0, 2.0]})

    schema = infer_feature_schema(
        features,
        overrides=FeatureOverrides(categorical=("empty",)),
    )

    assert schema.numeric_features == ("value",)
    assert schema.categorical_features == ("empty",)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: FeatureOverrides(numeric=("age", "age")),
        lambda: FeatureOverrides(numeric=("age",), categorical=("age",)),
        lambda: FeatureOverrides(numeric=["age"]),  # type: ignore[arg-type]
        lambda: PreprocessingConfig(categorical_fill_value=" "),
        lambda: PreprocessingConfig(scale_numeric=1),  # type: ignore[arg-type]
        lambda: PreprocessingConfig(numeric_imputation="median"),  # type: ignore[arg-type]
    ],
)
def test_invalid_preprocessing_configuration_is_rejected(factory: Callable[[], object]) -> None:
    """Invalid roles and options should fail when configuration is constructed."""
    with pytest.raises(ConfigurationError):
        factory()


def test_unknown_and_incompatible_overrides_are_actionable() -> None:
    """Overrides must refer to existing columns and respect forced numeric types."""
    features = pd.DataFrame({"city": ["Kabul", "Herat"]})

    with pytest.raises(PreprocessingError, match="unknown columns"):
        infer_feature_schema(features, overrides=FeatureOverrides(categorical=("country",)))
    with pytest.raises(PreprocessingError, match="forced numeric"):
        infer_feature_schema(features, overrides=FeatureOverrides(numeric=("city",)))


def test_unsupported_datetime_features_require_explicit_engineering() -> None:
    """Datetimes should not be silently coerced to arbitrary model numbers."""
    features = pd.DataFrame({"created_at": pd.to_datetime(["2026-01-01", "2026-01-02"])})

    with pytest.raises(PreprocessingError, match="Unsupported feature dtypes"):
        infer_feature_schema(features)


def test_built_pipeline_is_unfitted_cloned_and_estimator_compatible(tmp_path: Path) -> None:
    """The public builder should return a fresh pipeline that can fit and predict end to end."""
    split = _classification_split(tmp_path)
    original = LogisticRegression(max_iter=200)
    original.fit([[0], [1]], ["no", "yes"])

    pipeline = build_model_pipeline(split, original)

    with pytest.raises(NotFittedError):
        check_is_fitted(pipeline)
    assert pipeline.named_steps["estimator"] is not original

    pipeline.fit(split.train_features, split.train_target)
    predictions = pipeline.predict(split.validation_features)

    assert len(predictions) == len(split.validation_target)
    check_is_fitted(pipeline)


def test_validation_values_do_not_influence_fitted_imputation_state(tmp_path: Path) -> None:
    """Only training rows may determine learned numeric preprocessing statistics."""
    path = tmp_path / "regression.csv"
    rows = [f"{'' if index in {2, 7} else index},{index * 1.5}" for index in range(12)]
    path.write_text("\n".join(["value,target", *rows, ""]), encoding="utf-8")
    dataset = load_csv(path, target="target")
    split = split_dataset(
        dataset,
        task=TaskType.REGRESSION,
        config=SplitConfig(validation_fraction=0.25, random_seed=11),
    )
    split.validation_features.loc[:, "value"] = 1_000_000.0
    expected_training_median = float(split.train_features["value"].median())
    pipeline = build_model_pipeline(
        split,
        DummyRegressor(strategy="mean"),
        config=PreprocessingConfig(
            numeric_imputation=NumericImputationStrategy.MEDIAN,
            scale_numeric=False,
        ),
    )

    pipeline.fit(split.train_features, split.train_target)

    preprocessor = cast(Any, pipeline.named_steps["preprocessor"])
    numeric_pipeline = cast(Any, preprocessor.named_transformers_["numeric"])
    imputer = cast(Any, numeric_pipeline.named_steps["imputer"])
    assert float(imputer.statistics_[0]) == expected_training_median


def test_unseen_validation_categories_are_encoded_without_failure(tmp_path: Path) -> None:
    """Categories absent from training should transform to an all-zero category block."""
    split = _classification_split(tmp_path)
    split.validation_features.loc[:, "city"] = "never-seen"
    pipeline = build_model_pipeline(split, DummyClassifier(strategy="most_frequent"))

    pipeline.fit(split.train_features, split.train_target)
    predictions = pipeline.predict(split.validation_features)

    assert len(predictions) == len(split.validation_features)


def test_all_missing_numeric_and_forced_categorical_columns_are_retained(tmp_path: Path) -> None:
    """Empty training columns should retain stable output positions instead of disappearing."""
    split = _classification_split(tmp_path, include_empty=True)

    numeric_preprocessor = build_preprocessor(
        split,
        config=PreprocessingConfig(scale_numeric=False),
    )
    numeric_preprocessor.fit(split.train_features, split.train_target)
    assert "numeric__empty" in numeric_preprocessor.get_feature_names_out()

    categorical_preprocessor = build_preprocessor(
        split,
        overrides=FeatureOverrides(categorical=("empty",)),
    )
    categorical_preprocessor.fit(split.train_features, split.train_target)
    names = categorical_preprocessor.get_feature_names_out()
    assert any("empty___mlforge_missing__" in name for name in names)


@pytest.mark.parametrize("partition", ["training", "validation"])
def test_categorical_fill_marker_must_not_collide_with_split_data(
    tmp_path: Path,
    partition: str,
) -> None:
    """A real category must not be merged silently with the configured missing marker."""
    split = _classification_split(tmp_path)
    features = split.train_features if partition == "training" else split.validation_features
    features.loc[features.index[0], "city"] = "missing"

    with pytest.raises(PreprocessingError, match="already occurs"):
        build_preprocessor(
            split,
            config=PreprocessingConfig(categorical_fill_value="missing"),
        )


@pytest.mark.parametrize("partition", ["training", "validation"])
def test_infinite_numeric_features_are_rejected_before_fitting(
    tmp_path: Path,
    partition: str,
) -> None:
    """The standard numeric transformer must receive finite or missing feature values."""
    split = _classification_split(tmp_path)
    features = split.train_features if partition == "training" else split.validation_features
    features["number"] = features["number"].astype(float)
    features.loc[features.index[0], "number"] = math.inf

    with pytest.raises(PreprocessingError, match=partition.capitalize()):
        build_preprocessor(split)


def test_mutated_split_schema_is_rejected(tmp_path: Path) -> None:
    """Preprocessing should not continue after train/validation schema drift."""
    split = _classification_split(tmp_path)
    split.validation_features = split.validation_features.drop(columns=["city"])

    with pytest.raises(PreprocessingError, match="no longer match"):
        build_preprocessor(split)


@pytest.mark.parametrize("partition", ["training", "validation"])
def test_mutated_split_target_alignment_is_rejected(tmp_path: Path, partition: str) -> None:
    """Equal row counts must not hide feature/label index misalignment."""
    split = _classification_split(tmp_path)
    if partition == "training":
        split.train_target.index = split.train_target.index[::-1]
    else:
        split.validation_target.index = split.validation_target.index[::-1]

    with pytest.raises(PreprocessingError, match=f"{partition.capitalize()}.*indices"):
        build_preprocessor(split)


def test_non_sklearn_estimator_is_rejected(tmp_path: Path) -> None:
    """The extension point is the existing estimator convention, not any object with fit."""
    split = _classification_split(tmp_path)

    with pytest.raises(PreprocessingError, match="BaseEstimator"):
        build_model_pipeline(split, cast(Any, object()))
