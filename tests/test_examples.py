"""Integration tests for documented runnable examples."""

import json
import subprocess
import sys
from pathlib import Path


def test_profile_dataset_example() -> None:
    """The source example should run with the installed MLForge package."""
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, repository_root / "examples" / "profile_dataset.py"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    profile = json.loads(completed.stdout)
    assert profile["metadata"]["row_count"] == 8
    assert profile["metadata"]["target"] == "churn"
    assert profile["target"]["task_hint"] == "classification"


def test_preprocess_dataset_example() -> None:
    """The preprocessing example should fit only after a supervised split."""
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, repository_root / "examples" / "preprocess_dataset.py"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["stratified"] is True
    assert summary["train_rows"] == 6
    assert summary["validation_rows"] == 2
    assert summary["output_features"] == len(summary["feature_names"])


def test_train_customer_churn_example() -> None:
    """The end-to-end example should produce a successful terminal run manifest."""
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, repository_root / "examples" / "train_customer_churn.py"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(completed.stdout)
    assert manifest["status"] == "succeeded"
    assert manifest["configuration"]["task"] == "classification"
    assert manifest["configuration"]["estimator"] == "logistic-regression"
    assert {metric["name"] for metric in manifest["metrics"]} == {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
    }


def test_benchmark_customer_churn_example() -> None:
    """The benchmark example should emit three fair terminal classifier outcomes."""
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, repository_root / "examples" / "benchmark_customer_churn.py"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(completed.stdout)
    assert manifest["status"] == "succeeded"
    assert manifest["configuration"]["primary_metric"] == "balanced_accuracy"
    assert {entry["estimator"] for entry in manifest["entries"]} == {
        "dummy-classifier",
        "logistic-regression",
        "random-forest-classifier",
    }
    assert {entry["rank"] for entry in manifest["entries"]} == {1, 2, 3}


def test_cross_validate_customer_churn_example() -> None:
    """The cross-validation example should report fold-level aggregate evidence."""
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, repository_root / "examples" / "cross_validate_customer_churn.py"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads(completed.stdout)
    assert manifest["status"] == "succeeded"
    assert manifest["configuration"]["fold_count"] == 3
    assert len(manifest["folds"]) == 3
    assert {entry["rank"] for entry in manifest["entries"]} == {1, 2, 3}
    assert all(len(entry["folds"]) == 3 for entry in manifest["entries"])


def test_train_and_predict_example() -> None:
    """The complete artifact workflow should return two schema-validated predictions."""
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, repository_root / "examples" / "train_and_predict.py"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["artifact"]["run_id"] == output["predictions"]["run_id"]
    assert output["predictions"]["row_count"] == 2
    assert len(output["artifact"]["pipeline_sha256"]) == 64
