"""Behavioral tests for schema-validated dataframe and CSV batch inference."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mlforge.artifacts import LoadedArtifact, LocalArtifactStore
from mlforge.datasets import load_csv, load_feature_csv
from mlforge.errors import PredictionSchemaError
from mlforge.inference import predict_csv, predict_frame
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import (
    LOGISTIC_REGRESSION,
    RIDGE_REGRESSION,
    TrainingConfig,
    train,
)


def _loaded_classification_artifact(tmp_path: Path) -> tuple[LoadedArtifact, pd.DataFrame]:
    path = tmp_path / "training.csv"
    rows = ["amount,region,target"]
    for index in range(30):
        rows.append(
            f"{index + 1},{'north' if index % 2 else 'south'},{'yes' if index % 3 else 'no'}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset = load_csv(path, target="target")
    result = train(
        dataset,
        TrainingConfig(task=TaskType.CLASSIFICATION, estimator=LOGISTIC_REGRESSION),
        run_store=LocalRunStore(tmp_path / "runs"),
    )
    store = LocalArtifactStore(tmp_path / "artifacts")
    store.save(result)
    return store.load(result.manifest.run_id, trusted=True), dataset.frame.drop(columns=["target"])


def test_dataframe_prediction_reorders_columns_and_matches_pipeline(tmp_path: Path) -> None:
    """Column order may vary when names and values satisfy the recorded contract."""
    artifact, features = _loaded_classification_artifact(tmp_path)
    reordered = features.loc[:, ["region", "amount"]]

    result = predict_frame(artifact, reordered)

    assert result.source_path is None
    assert result.run_id == artifact.manifest.run_id
    assert result.row_count == len(features)
    assert [record.row_number for record in result.predictions] == list(range(1, 31))
    assert [record.prediction for record in result.predictions] == list(
        artifact.pipeline.predict(features)
    )
    assert json.loads(result.to_json())["row_count"] == 30


def test_csv_prediction_returns_source_and_json_safe_records(tmp_path: Path) -> None:
    """The CSV adapter should reuse strict parsing and return structured prediction rows."""
    artifact, _ = _loaded_classification_artifact(tmp_path)
    path = tmp_path / "prediction.csv"
    path.write_text("region,amount\nwest,7\nnorth,9\n", encoding="utf-8")

    result = predict_csv(artifact, path)

    assert result.source_path == str(path.resolve())
    assert result.target == "target"
    assert result.row_count == 2
    assert all(isinstance(record.prediction, str) for record in result.predictions)
    assert json.loads(result.to_json())["predictions"][0]["row_number"] == 1


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"amount": [1]}), "missing"),
        (
            pd.DataFrame({"amount": [1], "region": ["north"], "unused": [0]}),
            "unexpected",
        ),
        (pd.DataFrame({"amount": ["not-numeric"], "region": ["north"]}), "incompatible"),
        (pd.DataFrame({"amount": [float("inf")], "region": ["north"]}), "infinite"),
        (
            pd.DataFrame({"amount": [1], "region": ["__mlforge_missing__"]}),
            "reserved",
        ),
        (pd.DataFrame(columns=["amount", "region"]), "at least one row"),
    ],
)
def test_prediction_schema_failures_are_actionable(
    tmp_path: Path,
    frame: pd.DataFrame,
    message: str,
) -> None:
    """Invalid batch inputs must fail before reaching model prediction."""
    artifact, _ = _loaded_classification_artifact(tmp_path)

    with pytest.raises(PredictionSchemaError, match=message):
        predict_frame(artifact, frame)


def test_duplicate_prediction_columns_are_rejected(tmp_path: Path) -> None:
    """Duplicate names cannot be safely aligned to the artifact contract."""
    artifact, _ = _loaded_classification_artifact(tmp_path)
    frame = pd.DataFrame([[1, 2, "north"]], columns=["amount", "amount", "region"])

    with pytest.raises(PredictionSchemaError, match="unique"):
        predict_frame(artifact, frame)


def test_regression_predictions_are_finite_json_numbers(tmp_path: Path) -> None:
    """Regression inference should normalize NumPy scalars into portable JSON numbers."""
    path = tmp_path / "regression.csv"
    rows = ["value,group,target"] + [
        f"{index},{'a' if index % 2 else 'b'},{2.5 * index + 1}" for index in range(30)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset = load_csv(path, target="target")
    trained = train(
        dataset,
        TrainingConfig(task=TaskType.REGRESSION, estimator=RIDGE_REGRESSION),
        run_store=LocalRunStore(tmp_path / "runs"),
    )
    store = LocalArtifactStore(tmp_path / "artifacts")
    store.save(trained)
    artifact = store.load(trained.manifest.run_id, trusted=True)

    result = predict_frame(artifact, pd.DataFrame({"value": [2, 8], "group": ["a", "new"]}))

    assert result.task == "regression"
    assert all(isinstance(record.prediction, float) for record in result.predictions)
    json.loads(result.to_json())


def test_target_free_feature_csv_loader_reuses_ingestion_validation(tmp_path: Path) -> None:
    """Inference CSV loading should not require or invent a supervised target column."""
    path = tmp_path / "features.csv"
    path.write_text("feature_a,feature_b\n1,x\n2,y\n", encoding="utf-8")

    frame = load_feature_csv(path)

    assert list(frame.columns) == ["feature_a", "feature_b"]
    assert len(frame) == 2
