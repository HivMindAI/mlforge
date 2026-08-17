"""Offline end-to-end validation with real datasets bundled by scikit-learn."""

from pathlib import Path
from typing import cast

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from mlforge.artifacts import LocalArtifactStore
from mlforge.datasets import CsvLoadOptions, load_csv
from mlforge.inference import predict_csv, write_predictions_csv
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import (
    LOGISTIC_REGRESSION,
    RIDGE_REGRESSION,
    TrainingConfig,
    train,
)


def test_real_breast_cancer_classification_workflow(tmp_path: Path) -> None:
    """A real classification dataset should support mixed, missing, semicolon CSV data."""
    features, target = cast(
        "tuple[pd.DataFrame, pd.Series[int]]",
        load_breast_cancer(return_X_y=True, as_frame=True),
    )
    source = features.copy()
    source["diagnosis"] = target.map({0: "malignant", 1: "benign"})
    source["radius band"] = pd.cut(
        source["mean radius"],
        bins=3,
        labels=("small", "medium", "large"),
    ).astype("string")
    source.loc[source.index[::37], "mean radius"] = float("nan")
    source.loc[source.index[::53], "radius band"] = pd.NA
    options = CsvLoadOptions(encoding="utf-8", delimiter=";")
    training_path = tmp_path / "wisconsin_diagnostic.csv"
    source.to_csv(training_path, index=False, sep=";", encoding="utf-8")

    dataset = load_csv(training_path, target="diagnosis", options=options)
    trained = train(
        dataset,
        TrainingConfig(task=TaskType.CLASSIFICATION, estimator=LOGISTIC_REGRESSION),
        run_store=LocalRunStore(tmp_path / "classification-runs"),
    )
    accuracy = next(
        metric.value for metric in trained.manifest.metrics if metric.name == "accuracy"
    )

    artifacts = LocalArtifactStore(tmp_path / "classification-artifacts")
    saved = artifacts.save(trained)
    prediction_path = tmp_path / "wisconsin_prediction.csv"
    source.drop(columns=["diagnosis"]).tail(80).to_csv(
        prediction_path,
        index=False,
        sep=";",
        encoding="utf-8",
    )
    predictions = predict_csv(
        artifacts.load(saved.manifest.run_id, trusted=True),
        prediction_path,
        options=options,
    )
    output = write_predictions_csv(predictions, tmp_path / "classification_predictions.csv")

    assert dataset.metadata.row_count == 569
    assert dataset.metadata.delimiter == ";"
    assert dataset.frame["mean radius"].isna().any()
    assert dataset.frame["radius band"].isna().any()
    assert accuracy >= 0.9
    assert predictions.row_count == 80
    assert len(pd.read_csv(output)) == 80


def test_real_diabetes_regression_workflow(tmp_path: Path) -> None:
    """A real regression dataset should support mixed, missing, Latin-1 pipe CSV data."""
    features, target = cast(
        "tuple[pd.DataFrame, pd.Series[float]]",
        load_diabetes(return_X_y=True, as_frame=True),
    )
    source = features.rename(columns={"s1": "serum cholest\u00e9rol"}).copy()
    source["progression"] = target
    source["bmi band"] = pd.cut(
        source["bmi"],
        bins=3,
        labels=("lower", "middle", "higher"),
    ).astype("string")
    source.loc[source.index[::31], "bmi"] = float("nan")
    source.loc[source.index[::47], "bmi band"] = pd.NA
    options = CsvLoadOptions(encoding="latin-1", delimiter="|")
    training_path = tmp_path / "diabetes_progression.csv"
    source.to_csv(training_path, index=False, sep="|", encoding="latin-1")

    dataset = load_csv(training_path, target="progression", options=options)
    trained = train(
        dataset,
        TrainingConfig(task=TaskType.REGRESSION, estimator=RIDGE_REGRESSION),
        run_store=LocalRunStore(tmp_path / "regression-runs"),
    )
    r_squared = next(metric.value for metric in trained.manifest.metrics if metric.name == "r2")

    artifacts = LocalArtifactStore(tmp_path / "regression-artifacts")
    saved = artifacts.save(trained)
    prediction_path = tmp_path / "diabetes_prediction.csv"
    source.drop(columns=["progression"]).tail(60).to_csv(
        prediction_path,
        index=False,
        sep="|",
        encoding="latin-1",
    )
    predictions = predict_csv(
        artifacts.load(saved.manifest.run_id, trusted=True),
        prediction_path,
        options=options,
    )
    output = write_predictions_csv(predictions, tmp_path / "regression_predictions.csv")

    assert dataset.metadata.row_count == 442
    assert dataset.metadata.encoding == "latin-1"
    assert dataset.frame["bmi"].isna().any()
    assert dataset.frame["bmi band"].isna().any()
    assert r_squared > 0
    assert predictions.row_count == 60
    assert len(pd.read_csv(output)) == 60
