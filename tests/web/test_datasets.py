"""Integration tests for the local dataset upload API."""

from __future__ import annotations

import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mlforge.errors import FinalModelLineageError
from mlforge.final_models import FinalModelResult
from mlforge.web import create_app
from mlforge.web.errors import WebStorageError
from mlforge.web.settings import WebSettings
from mlforge.web.storage import DatasetStore


def _client(workspace: Path, *, max_upload_bytes: int = 100 * 1024 * 1024) -> TestClient:
    settings = WebSettings(workspace=workspace, max_upload_bytes=max_upload_bytes)
    return TestClient(create_app(settings))


def _wait_for_job(client: TestClient, job_id: str, *, timeout: float = 15.0) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] in {"complete", "failed"}:
            return cast(dict[str, object], body)
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not finish within {timeout} seconds.")


def _wait_for_finalization(
    client: TestClient,
    experiment_id: str,
    *,
    timeout: float = 15.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/experiments/{experiment_id}/finalization")
        assert response.status_code == 200
        body = response.json()
        assert body is not None
        if body["status"] in {"complete", "failed"}:
            return cast(dict[str, object], body)
        time.sleep(0.05)
    raise AssertionError(
        f"Finalization for {experiment_id} did not finish within {timeout} seconds."
    )


def test_health_probes_distinguish_process_and_storage_readiness(tmp_path: Path) -> None:
    """Deployment probes should be path-free and leave no storage artifacts."""
    workspace = tmp_path / "web"

    with _client(workspace) as client:
        live = client.get("/api/health/live")
        ready = client.get("/api/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 200
    assert live.json() == ready.json()
    assert live.json()["status"] == "ok"
    assert live.json()["version"]
    assert not list((workspace / "uploads").glob(".health-*.tmp"))


def test_readiness_failure_is_safe_and_does_not_change_liveness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storage outage should fail readiness without exposing local details."""

    def fail_readiness(_store: DatasetStore) -> None:
        raise WebStorageError("C:/private/mlforge.sqlite3 is unavailable")

    monkeypatch.setattr(DatasetStore, "check_ready", fail_readiness)

    with _client(tmp_path / "web") as client:
        live = client.get("/api/health/live")
        ready = client.get("/api/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 503
    assert ready.json() == {
        "error": {
            "code": "not_ready",
            "message": "The MLForge web workspace is unavailable.",
        }
    }
    assert "private" not in ready.text


def test_readiness_rejects_a_missing_metadata_database(tmp_path: Path) -> None:
    """A deleted database must not be silently replaced with a healthy empty file."""
    workspace = tmp_path / "web"

    with _client(workspace) as client:
        (workspace / "mlforge.sqlite3").unlink()
        ready = client.get("/api/health/ready")

    assert ready.status_code == 503
    assert ready.json()["error"]["code"] == "not_ready"


def test_upload_returns_path_free_metadata_and_publishes_uuid_file(tmp_path: Path) -> None:
    """A valid CSV should use the core parser and never expose its server path."""
    workspace = tmp_path / "web"
    content = b"age,city,outcome\n21,Kabul,yes\n34,Herat,no\n"

    with _client(workspace) as client:
        response = client.post(
            "/api/datasets",
            files={"file": ("students.csv", content, "text/csv")},
        )

    assert response.status_code == 201
    body = response.json()
    dataset_id = UUID(body["dataset_id"])
    assert body == {
        "dataset_id": str(dataset_id),
        "filename": "students.csv",
        "file_size_bytes": len(content),
        "row_count": 2,
        "column_count": 3,
        "columns": ["age", "city", "outcome"],
        "target": None,
        "created_at": body["created_at"],
    }
    assert "source_path" not in body
    uploads = list((workspace / "uploads").glob("*.csv"))
    assert uploads == [workspace / "uploads" / f"{dataset_id}.csv"]
    assert uploads[0].read_bytes() == content


def test_uploaded_dataset_can_be_loaded_and_given_an_explicit_target(tmp_path: Path) -> None:
    """Target selection should be persisted only after core validation succeeds."""
    workspace = tmp_path / "web"

    with _client(workspace) as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("training.csv", b"feature,label\n1,yes\n2,no\n", "text/csv")},
        ).json()
        dataset_id = uploaded["dataset_id"]

        selected = client.patch(
            f"/api/datasets/{dataset_id}/target",
            json={"target": "label"},
        )
        loaded = client.get(f"/api/datasets/{dataset_id}")

    assert selected.status_code == 200
    assert selected.json()["target"] == "label"
    assert loaded.status_code == 200
    assert loaded.json()["target"] == "label"


def test_invalid_target_returns_actionable_core_error(tmp_path: Path) -> None:
    """The adapter should translate target validation without duplicating it."""
    with _client(tmp_path / "web") as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("training.csv", b"feature,label\n1,yes\n", "text/csv")},
        ).json()
        response = client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "missing"},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_dataset"
    assert "was not found" in response.json()["error"]["message"]


def test_non_csv_and_malformed_uploads_leave_no_files(tmp_path: Path) -> None:
    """Rejected uploads should provide useful errors and clean temporary files."""
    workspace = tmp_path / "web"

    with _client(workspace) as client:
        wrong_extension = client.post(
            "/api/datasets",
            files={"file": ("dataset.txt", b"value,target\n1,yes\n", "text/plain")},
        )
        malformed = client.post(
            "/api/datasets",
            files={"file": ("dataset.csv", b"value,target\n1\n", "text/csv")},
        )

    assert wrong_extension.status_code == 422
    assert wrong_extension.json()["error"]["code"] == "invalid_upload"
    assert malformed.status_code == 422
    assert malformed.json()["error"]["code"] == "invalid_dataset"
    assert list((workspace / "uploads").iterdir()) == []


def test_upload_limit_is_enforced_while_streaming(tmp_path: Path) -> None:
    """The server-side byte limit must hold even when the browser check is bypassed."""
    workspace = tmp_path / "web"

    with _client(workspace, max_upload_bytes=16) as client:
        response = client.post(
            "/api/datasets",
            files={"file": ("large.csv", b"column,target\n1234567890,yes\n", "text/csv")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_upload"
    assert "16 bytes" in response.json()["error"]["message"]
    assert list((workspace / "uploads").iterdir()) == []


def test_missing_dataset_returns_structured_not_found(tmp_path: Path) -> None:
    """Unknown UUIDs should not leak storage details."""
    with _client(tmp_path / "web") as client:
        response = client.get(f"/api/datasets/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_analysis_requires_an_explicit_target(tmp_path: Path) -> None:
    """Profiling must not guess which column the user wants to predict."""
    with _client(tmp_path / "web") as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("training.csv", b"feature,label\n1,yes\n2,no\n", "text/csv")},
        ).json()
        response = client.post(f"/api/datasets/{uploaded['dataset_id']}/analysis")

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "invalid_dataset",
            "message": "Choose a target column before analyzing this dataset.",
        }
    }


def test_analysis_returns_real_path_free_core_profile(tmp_path: Path) -> None:
    """The analysis DTO should expose core evidence without local storage paths."""
    content = (
        b"student_id,score,city,outcome\n"
        b"1,90,Kabul,yes\n"
        b"2,,Herat,no\n"
        b"3,90,Kabul,yes\n"
        b"4,90,Kabul,yes\n"
    )

    with _client(tmp_path / "web") as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("students.csv", content, "text/csv")},
        ).json()
        selected = client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "outcome"},
        )
        response = client.post(f"/api/datasets/{uploaded['dataset_id']}/analysis")

    assert selected.status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert body["dataset"]["target"] == "outcome"
    assert body["missing_cell_count"] == 1
    assert body["duplicate_row_count"] == 0
    assert body["target"]["task_hint"] == "classification"
    assert body["target"]["unique_count"] == 2

    columns = {column["name"]: column for column in body["columns"]}
    assert columns["student_id"]["is_likely_identifier"] is True
    assert columns["score"]["missing_count"] == 1
    assert "Dataset contains 1 missing cells." in body["warnings"]
    assert "Likely identifier columns: student_id." in body["warnings"]
    assert "source_path" not in response.text


def test_classification_experiment_configuration_is_validated_and_persisted(
    tmp_path: Path,
) -> None:
    """Phase 5 should save core-valid configuration without starting training."""
    workspace = tmp_path / "web"
    content = (
        b"feature,target\n"
        b"1,yes\n2,no\n3,yes\n4,no\n5,yes\n6,no\n"
        b"7,yes\n8,no\n9,yes\n10,no\n11,yes\n12,no\n"
    )
    with _client(workspace) as client:
        empty_history = client.get("/api/experiments")
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("classification.csv", content, "text/csv")},
        ).json()
        client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "target"},
        )
        response = client.post(
            "/api/experiments",
            json={
                "dataset_id": uploaded["dataset_id"],
                "estimators": ["logistic-regression", "random-forest-classifier"],
                "fold_count": 3,
            },
        )
        pending_results = client.get(f"/api/experiments/{response.json()['experiment_id']}/results")
        pending_finalization = client.post(
            f"/api/experiments/{response.json()['experiment_id']}/finalize"
        )
        configured_history = client.get("/api/experiments")
        empty_models = client.get("/api/final-models")

    assert empty_history.status_code == 200
    assert empty_history.json() == {"experiments": [], "count": 0}
    assert response.status_code == 201
    body = response.json()
    assert UUID(body["experiment_id"])
    assert body["dataset_id"] == uploaded["dataset_id"]
    assert body["task"] == "classification"
    assert body["validation_strategy"] == "cross-validation"
    assert body["fold_count"] == 3
    assert body["estimators"] == ["logistic-regression", "random-forest-classifier"]
    assert body["primary_metric"] == "balanced_accuracy"
    assert "source_path" not in response.text
    assert pending_results.status_code == 409
    assert pending_results.json() == {
        "error": {
            "code": "result_not_ready",
            "message": "Run this experiment before requesting its results.",
        }
    }
    assert pending_finalization.status_code == 409
    assert pending_finalization.json()["error"]["code"] == "finalization_not_ready"
    assert configured_history.status_code == 200
    history_body = configured_history.json()
    assert history_body == {
        "experiments": [
            {
                "experiment_id": body["experiment_id"],
                "dataset_id": uploaded["dataset_id"],
                "dataset_name": "classification.csv",
                "task": "classification",
                "status": "configured",
                "model_count": 2,
                "created_at": body["created_at"],
                "updated_at": body["created_at"],
            }
        ],
        "count": 1,
    }
    assert empty_models.status_code == 200
    assert empty_models.json() == {"models": [], "count": 0}

    with sqlite3.connect(workspace / "mlforge.sqlite3") as connection:
        stored = connection.execute(
            "SELECT COUNT(*) FROM experiments WHERE experiment_id = ?",
            (body["experiment_id"],),
        ).fetchone()
        history_plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT experiment_id
            FROM experiments
            ORDER BY created_at DESC, experiment_id DESC
            """
        ).fetchall()
    assert stored == (1,)
    assert any("idx_experiments_created" in row[3] for row in history_plan)


def test_invalid_experiment_configuration_returns_core_error_without_persisting(
    tmp_path: Path,
) -> None:
    """Unsupported model choices should remain a core-owned validation decision."""
    workspace = tmp_path / "web"

    with _client(workspace) as client:
        uploaded = client.post(
            "/api/datasets",
            files={
                "file": (
                    "classification.csv",
                    b"feature,target\n1,yes\n2,no\n3,yes\n4,no\n",
                    "text/csv",
                )
            },
        ).json()
        client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "target"},
        )
        response = client.post(
            "/api/experiments",
            json={
                "dataset_id": uploaded["dataset_id"],
                "estimators": ["logistic-regression", "decision-tree"],
                "fold_count": 2,
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_experiment"
    assert "supported classification estimators" in response.json()["error"]["message"]
    with sqlite3.connect(workspace / "mlforge.sqlite3") as connection:
        stored_count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()
    assert stored_count == (0,)


def test_regression_experiment_runs_finalizes_and_predicts(tmp_path: Path) -> None:
    """Regression should follow the same durable web workflow as classification."""
    rows = "".join(f"{index},{index * 1.25 + 3}\n" for index in range(1, 31))
    content = f"feature,target\n{rows}".encode()
    workspace = tmp_path / "web"

    with _client(workspace) as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("regression.csv", content, "text/csv")},
        ).json()
        client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "target"},
        )
        response = client.post(
            "/api/experiments",
            json={
                "dataset_id": uploaded["dataset_id"],
                "estimators": ["ridge-regression", "random-forest-regressor"],
                "fold_count": 5,
            },
        )
        assert response.status_code == 201
        experiment = response.json()
        started = client.post(f"/api/experiments/{experiment['experiment_id']}/run")
        terminal = _wait_for_job(client, started.json()["job_id"])
        results = client.get(f"/api/experiments/{experiment['experiment_id']}/results")
        finalization_started = client.post(
            f"/api/experiments/{experiment['experiment_id']}/finalize"
        )
        assert finalization_started.status_code == 202
        finalized = _wait_for_finalization(client, experiment["experiment_id"])
        model_id = cast(str, finalized["final_model_id"])
        model = client.get(f"/api/final-models/{model_id}")
        prediction = client.post(
            "/api/predictions",
            data={"model_id": model_id},
            files={"file": ("future.csv", b"feature\n31\n32\n", "text/csv")},
        )
        prediction_result = client.get(f"/api/predictions/{prediction.json()['prediction_id']}")

    assert experiment["task"] == "regression"
    assert experiment["primary_metric"] == "root_mean_squared_error"
    assert terminal["status"] == "complete"
    assert results.status_code == 200
    result_body = results.json()
    assert result_body["task"] == "regression"
    assert result_body["primary_metric"] == "root_mean_squared_error"
    for entry in result_body["entries"]:
        assert {metric["name"]: metric["higher_is_better"] for metric in entry["metrics"]} == {
            "mean_absolute_error": False,
            "r2": True,
            "root_mean_squared_error": False,
        }
    assert finalized["status"] == "complete"
    assert model.status_code == 200
    assert model.json()["task"] == "regression"
    assert prediction.status_code == 201
    assert prediction_result.status_code == 200
    preview = prediction_result.json()["preview_rows"]
    assert len(preview) == 2
    assert all(float(row["prediction"]) > 0 for row in preview)


def test_experiment_job_runs_real_cross_validation_and_is_idempotent(tmp_path: Path) -> None:
    """One saved configuration should produce one durable core benchmark job."""
    workspace = tmp_path / "web"
    content = (
        b"feature,target\n"
        b"1,yes\n2,no\n3,yes\n4,no\n5,yes\n6,no\n"
        b"7,yes\n8,no\n9,yes\n10,no\n11,yes\n12,no\n"
    )
    prediction_content = b"feature\n" + b"".join(f"{value}\n".encode() for value in range(13, 35))

    with _client(workspace) as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("classification.csv", content, "text/csv")},
        ).json()
        client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "target"},
        )
        experiment = client.post(
            "/api/experiments",
            json={
                "dataset_id": uploaded["dataset_id"],
                "estimators": ["dummy-classifier", "logistic-regression"],
                "fold_count": 3,
            },
        ).json()

        loaded_experiment = client.get(f"/api/experiments/{experiment['experiment_id']}")
        started = client.post(f"/api/experiments/{experiment['experiment_id']}/run")
        assert started.status_code == 202
        terminal = _wait_for_job(client, started.json()["job_id"])
        completed_history = client.get("/api/experiments")
        results = client.get(f"/api/experiments/{experiment['experiment_id']}/results")
        finalization_started = client.post(
            f"/api/experiments/{experiment['experiment_id']}/finalize"
        )
        assert finalization_started.status_code == 202
        finalized = _wait_for_finalization(client, experiment["experiment_id"])
        final_model_id = cast(str, finalized["final_model_id"])
        final_model = client.get(f"/api/final-models/{final_model_id}")
        final_models = client.get("/api/final-models")
        repeated_finalization = client.post(
            f"/api/experiments/{experiment['experiment_id']}/finalize"
        )
        repeated = client.post(f"/api/experiments/{experiment['experiment_id']}/run")
        prediction = client.post(
            "/api/predictions",
            data={"model_id": final_model_id},
            files={"file": ("new-rows.csv", prediction_content, "text/csv")},
        )
        wrong_columns = client.post(
            "/api/predictions",
            data={"model_id": final_model_id},
            files={"file": ("wrong-columns.csv", b"other\n13\n", "text/csv")},
        )
        wrong_type = client.post(
            "/api/predictions",
            data={"model_id": final_model_id},
            files={"file": ("wrong-type.csv", b"feature\nnot-a-number\n", "text/csv")},
        )
        malformed = client.post(
            "/api/predictions",
            data={"model_id": final_model_id},
            files={"file": ("malformed.csv", b"\xff\xfe", "text/csv")},
        )
        artifact_path = workspace / "artifacts" / f"{final_model_id}.mlforge"
        artifact_path.write_bytes(b"not an MLForge artifact")
        corrupt_artifact = client.post(
            "/api/predictions",
            data={"model_id": final_model_id},
            files={"file": ("new-rows.csv", b"feature\n13\n", "text/csv")},
        )
        prediction_id = prediction.json()["prediction_id"]
        prediction_result = client.get(f"/api/predictions/{prediction_id}")
        prediction_download = client.get(f"/api/predictions/{prediction_id}/download")
        prediction_output_path = workspace / "predictions" / f"{prediction_id}.csv"
        valid_output = prediction_output_path.read_bytes()
        prediction_output_path.write_bytes(b"invalid output")
        unavailable_result = client.get(f"/api/predictions/{prediction_id}")
        prediction_output_path.write_bytes(valid_output)

    assert loaded_experiment.status_code == 200
    assert loaded_experiment.json() == experiment
    assert terminal["status"] == "complete"
    assert terminal["started_at"] is not None
    assert terminal["completed_at"] is not None
    assert terminal["benchmark_id"] is not None
    assert terminal["error_message"] is None
    assert completed_history.status_code == 200
    completed_summary = completed_history.json()["experiments"][0]
    assert completed_summary["experiment_id"] == experiment["experiment_id"]
    assert completed_summary["dataset_name"] == "classification.csv"
    assert completed_summary["status"] == "complete"
    assert completed_summary["model_count"] == 2
    assert completed_summary["updated_at"] == terminal["completed_at"]
    assert repeated.status_code == 202
    assert repeated.json()["job_id"] == started.json()["job_id"]
    assert "source_path" not in started.text

    assert results.status_code == 200
    result_body = results.json()
    assert result_body["experiment_id"] == experiment["experiment_id"]
    assert result_body["benchmark_id"] == terminal["benchmark_id"]
    assert result_body["status"] == "succeeded"
    assert result_body["task"] == "classification"
    assert result_body["target"] == "target"
    assert result_body["row_count"] == 12
    assert result_body["column_count"] == 2
    assert result_body["primary_metric"] == "balanced_accuracy"
    assert result_body["fold_count"] == 3
    assert [fold["fold_number"] for fold in result_body["folds"]] == [1, 2, 3]
    assert sorted(entry["rank"] for entry in result_body["entries"]) == [1, 2]
    for entry in result_body["entries"]:
        assert entry["status"] == "succeeded"
        assert len(entry["folds"]) == 3
        summaries = {metric["name"]: metric for metric in entry["metrics"]}
        assert set(summaries) == {
            "accuracy",
            "balanced_accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
        }
        primary = summaries["balanced_accuracy"]
        assert len(primary["fold_values"]) == 3
        assert primary["mean"] == pytest.approx(sum(primary["fold_values"]) / 3)
        assert primary["standard_deviation"] >= 0
    assert "source_path" not in results.text
    assert "partition_sha256" not in results.text

    assert finalized["status"] == "complete"
    assert finalized["final_model_id"] is not None
    assert finalized["error_message"] is None
    assert repeated_finalization.status_code == 202
    assert (
        repeated_finalization.json()["finalization_id"]
        == finalization_started.json()["finalization_id"]
    )
    assert final_model.status_code == 200
    model_body = final_model.json()
    assert model_body["final_model_id"] == finalized["final_model_id"]
    assert model_body["dataset_id"] == uploaded["dataset_id"]
    assert model_body["dataset_name"] == "classification.csv"
    assert model_body["experiment_id"] == experiment["experiment_id"]
    assert model_body["benchmark_id"] == terminal["benchmark_id"]
    assert model_body["status"] == "succeeded"
    assert model_body["task"] == "classification"
    assert model_body["fit_scope"] == "all_rows"
    assert model_body["training_rows"] == 12
    assert model_body["feature_count"] == 1
    assert model_body["primary_metric"] == "balanced_accuracy"
    assert {metric["name"] for metric in model_body["metrics"]} == {
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
    }
    assert model_body["artifact"]["filename"] == f"{finalized['final_model_id']}.mlforge"
    assert model_body["artifact"]["target"] == "target"
    assert model_body["artifact"]["features"] == [
        {"name": "feature", "pandas_dtype": "int64", "role": "numeric"}
    ]
    assert model_body["artifact"]["pipeline_size_bytes"] > 0
    assert len(model_body["artifact"]["pipeline_sha256"]) == 64
    assert set(model_body["artifact"]["environment"]) == {
        "python",
        "mlforge",
        "pandas",
        "numpy",
        "scipy",
        "scikit_learn",
    }
    assert "source_path" not in final_model.text
    assert "artifact_path" not in final_model.text
    assert "pipeline_payload" not in final_model.text

    assert final_models.status_code == 200
    assert final_models.json()["count"] == 1
    summaries = final_models.json()["models"]
    assert len(summaries) == 1
    assert summaries[0] == {
        "final_model_id": finalized["final_model_id"],
        "dataset_id": uploaded["dataset_id"],
        "dataset_name": "classification.csv",
        "experiment_id": experiment["experiment_id"],
        "estimator": model_body["estimator"],
        "task": "classification",
        "created_at": model_body["created_at"],
        "primary_metric": "balanced_accuracy",
        "primary_metric_mean": model_body["primary_metric_mean"],
        "primary_metric_standard_deviation": model_body["primary_metric_standard_deviation"],
    }
    assert "artifact" not in summaries[0]

    assert prediction.status_code == 201
    prediction_body = prediction.json()
    assert set(prediction_body) == {
        "prediction_id",
        "final_model_id",
        "input_filename",
        "status",
        "created_at",
        "completed_at",
    }
    assert prediction_body["final_model_id"] == finalized["final_model_id"]
    assert prediction_body["input_filename"] == "new-rows.csv"
    assert prediction_body["status"] == "complete"
    assert "row_count" not in prediction_body
    assert "predictions" not in prediction_body
    assert "download" not in prediction.text

    assert wrong_columns.status_code == 422
    assert wrong_columns.json()["error"]["code"] == "invalid_prediction_input"
    assert "missing ['feature']" in wrong_columns.json()["error"]["message"]
    assert "unexpected ['other']" in wrong_columns.json()["error"]["message"]
    assert wrong_type.status_code == 422
    assert wrong_type.json()["error"]["code"] == "invalid_prediction_input"
    assert (
        "Numeric feature 'feature' has incompatible pandas dtype"
        in wrong_type.json()["error"]["message"]
    )
    assert malformed.status_code == 422
    assert malformed.json()["error"]["code"] == "invalid_prediction_input"
    assert corrupt_artifact.status_code == 422
    assert corrupt_artifact.json() == {
        "error": {
            "code": "invalid_model_artifact",
            "message": "The finalized model artifact is missing, corrupt, or incompatible.",
        }
    }

    assert prediction_result.status_code == 200
    prediction_result_body = prediction_result.json()
    assert prediction_result_body["prediction_id"] == prediction_body["prediction_id"]
    assert prediction_result_body["final_model_id"] == final_model_id
    assert prediction_result_body["input_filename"] == "new-rows.csv"
    assert prediction_result_body["status"] == "complete"
    assert prediction_result_body["row_count"] == 22
    assert prediction_result_body["invalid_row_count"] == 0
    assert prediction_result_body["preview_limit"] == 20
    assert prediction_result_body["preview_truncated"] is True
    assert len(prediction_result_body["preview_rows"]) == 20
    assert [row["row_number"] for row in prediction_result_body["preview_rows"]] == list(
        range(1, 21)
    )
    assert all(isinstance(row["prediction"], str) for row in prediction_result_body["preview_rows"])
    assert "output_stored_filename" not in prediction_result_body
    assert "source_path" not in prediction_result.text

    assert prediction_download.status_code == 200
    assert prediction_download.headers["content-type"].startswith("text/csv")
    assert prediction_download.headers["cache-control"] == "no-store"
    assert prediction_download.headers["x-content-type-options"] == "nosniff"
    assert 'filename="predictions.csv"' in prediction_download.headers["content-disposition"]
    assert prediction_download.content == valid_output

    assert unavailable_result.status_code == 500
    assert unavailable_result.json()["error"]["code"] == "prediction_result_unavailable"
    assert "invalid CSV header" in unavailable_result.json()["error"]["message"]

    benchmark_path = (
        workspace / "mlbenchmarks" / "cross-validation" / f"{terminal['benchmark_id']}.json"
    )
    assert benchmark_path.is_file()
    with sqlite3.connect(workspace / "mlforge.sqlite3") as connection:
        job_count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()
        finalization_count = connection.execute("SELECT COUNT(*) FROM finalizations").fetchone()
        prediction_row = connection.execute(
            """
            SELECT input_stored_filename, output_stored_filename, row_count, status
            FROM predictions
            WHERE prediction_id = ?
            """,
            (prediction_body["prediction_id"],),
        ).fetchone()
        list_plan = connection.execute(
            """
            EXPLAIN QUERY PLAN
            SELECT finalization_id
            FROM finalizations
            WHERE status = 'complete'
            ORDER BY completed_at DESC, finalization_id DESC
            """
        ).fetchall()
    assert job_count == (1,)
    assert finalization_count == (1,)
    assert prediction_row == (
        f"{prediction_body['prediction_id']}.csv",
        f"{prediction_body['prediction_id']}.csv",
        22,
        "complete",
    )
    assert any("idx_finalizations_completed" in row[3] for row in list_plan)
    assert (workspace / "mlfinalmodels" / f"{finalized['final_model_id']}.json").is_file()
    assert (workspace / "artifacts" / f"{finalized['final_model_id']}.mlforge").is_file()
    assert (
        workspace / "prediction-inputs" / f"{prediction_body['prediction_id']}.csv"
    ).read_bytes() == prediction_content
    output_lines = (
        (workspace / "predictions" / f"{prediction_body['prediction_id']}.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert output_lines[0] == "row_number,prediction"
    assert len(output_lines) == 23
    assert output_lines[1].startswith("1,")
    assert output_lines[-1].startswith("22,")


def test_expected_training_failure_is_persisted_for_inspection(tmp_path: Path) -> None:
    """A core benchmark failure should become a readable terminal job state."""
    content = b"feature,target\ninf,yes\n1,no\n2,yes\n3,no\n4,yes\n5,no\n6,yes\n7,no\n"

    with _client(tmp_path / "web") as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("non-finite.csv", content, "text/csv")},
        ).json()
        client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "target"},
        )
        experiment = client.post(
            "/api/experiments",
            json={
                "dataset_id": uploaded["dataset_id"],
                "estimators": ["dummy-classifier", "logistic-regression"],
                "fold_count": 2,
            },
        ).json()
        started = client.post(f"/api/experiments/{experiment['experiment_id']}/run")
        terminal = _wait_for_job(client, started.json()["job_id"])
        failed_history = client.get("/api/experiments")

    assert terminal["status"] == "failed"
    assert terminal["benchmark_id"] is None
    assert terminal["completed_at"] is not None
    assert isinstance(terminal["error_message"], str)
    assert "failed because every estimator failed" in terminal["error_message"]
    assert failed_history.status_code == 200
    failed_summary = failed_history.json()["experiments"][0]
    assert failed_summary["experiment_id"] == experiment["experiment_id"]
    assert failed_summary["dataset_name"] == "non-finite.csv"
    assert failed_summary["status"] == "failed"
    assert failed_summary["updated_at"] == terminal["completed_at"]


def test_failed_finalization_is_inspectable_and_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed full-data fit should remain readable without blocking a later retry."""
    content = (
        b"feature,target\n"
        b"1,yes\n2,no\n3,yes\n4,no\n5,yes\n6,no\n"
        b"7,yes\n8,no\n9,yes\n10,no\n11,yes\n12,no\n"
    )

    with _client(tmp_path / "web") as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("classification.csv", content, "text/csv")},
        ).json()
        client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "target"},
        )
        experiment = client.post(
            "/api/experiments",
            json={
                "dataset_id": uploaded["dataset_id"],
                "estimators": ["dummy-classifier", "logistic-regression"],
                "fold_count": 3,
            },
        ).json()
        comparison = client.post(f"/api/experiments/{experiment['experiment_id']}/run").json()
        assert _wait_for_job(client, comparison["job_id"])["status"] == "complete"

        manager = cast(FastAPI, client.app).state.job_manager
        original_finalizer = manager.finalizer

        def fail_finalization(*_args: object, **_kwargs: object) -> FinalModelResult:
            raise FinalModelLineageError("intentional finalization failure")

        monkeypatch.setattr(manager, "finalizer", fail_finalization)
        first = client.post(f"/api/experiments/{experiment['experiment_id']}/finalize")
        failed = _wait_for_finalization(client, experiment["experiment_id"])

        monkeypatch.setattr(manager, "finalizer", original_finalizer)
        second = client.post(f"/api/experiments/{experiment['experiment_id']}/finalize")
        completed = _wait_for_finalization(client, experiment["experiment_id"])

    assert first.status_code == 202
    assert failed["status"] == "failed"
    assert failed["final_model_id"] is None
    assert "intentional finalization failure" in cast(str, failed["error_message"])
    assert second.status_code == 202
    assert second.json()["finalization_id"] != first.json()["finalization_id"]
    assert completed["status"] == "complete"
    assert completed["final_model_id"] is not None


def test_interrupted_job_is_failed_during_restart_recovery(tmp_path: Path) -> None:
    """Restart must never leave a lost job appearing to run forever."""
    workspace = tmp_path / "web"
    content = b"feature,target\n1,yes\n2,no\n3,yes\n4,no\n"

    with _client(workspace) as client:
        uploaded = client.post(
            "/api/datasets",
            files={"file": ("classification.csv", content, "text/csv")},
        ).json()
        client.patch(
            f"/api/datasets/{uploaded['dataset_id']}/target",
            json={"target": "target"},
        )
        experiment = client.post(
            "/api/experiments",
            json={
                "dataset_id": uploaded["dataset_id"],
                "estimators": ["dummy-classifier", "logistic-regression"],
                "fold_count": 2,
            },
        ).json()

    job_id = uuid4()
    finalization_id = uuid4()
    with sqlite3.connect(workspace / "mlforge.sqlite3") as connection:
        connection.execute(
            """
            INSERT INTO jobs (job_id, experiment_id, status, created_at, started_at)
            VALUES (?, ?, 'running', ?, ?)
            """,
            (
                str(job_id),
                experiment["experiment_id"],
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.execute(
            """
            INSERT INTO finalizations (
                finalization_id,
                experiment_id,
                status,
                created_at,
                started_at
            ) VALUES (?, ?, 'running', ?, ?)
            """,
            (
                str(finalization_id),
                experiment["experiment_id"],
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.commit()

    with _client(workspace) as client:
        recovered = client.get(f"/api/jobs/{job_id}")
        recovered_finalization = client.get(
            f"/api/experiments/{experiment['experiment_id']}/finalization"
        )

    assert recovered.status_code == 200
    assert recovered.json()["status"] == "failed"
    assert "API stopped before this comparison finished" in recovered.json()["error_message"]
    assert recovered_finalization.status_code == 200
    assert recovered_finalization.json()["finalization_id"] == str(finalization_id)
    assert recovered_finalization.json()["status"] == "failed"
    assert (
        "API stopped before this final model finished"
        in recovered_finalization.json()["error_message"]
    )


def test_missing_experiment_and_job_return_structured_not_found(tmp_path: Path) -> None:
    """Execution resources should preserve stable path-free not-found errors."""
    with _client(tmp_path / "web") as client:
        experiment = client.get(f"/api/experiments/{uuid4()}")
        results = client.get(f"/api/experiments/{uuid4()}/results")
        final_model = client.get(f"/api/final-models/{uuid4()}")
        prediction = client.post(
            "/api/predictions",
            data={"model_id": str(uuid4())},
            files={"file": ("rows.csv", b"feature\n1\n", "text/csv")},
        )
        missing_prediction_id = uuid4()
        prediction_result = client.get(f"/api/predictions/{missing_prediction_id}")
        prediction_download = client.get(f"/api/predictions/{missing_prediction_id}/download")
        job = client.get(f"/api/jobs/{uuid4()}")

    assert experiment.status_code == 404
    assert experiment.json()["error"]["code"] == "experiment_not_found"
    assert results.status_code == 404
    assert results.json()["error"]["code"] == "experiment_not_found"
    assert final_model.status_code == 404
    assert final_model.json()["error"]["code"] == "final_model_not_found"
    assert prediction.status_code == 404
    assert prediction.json()["error"]["code"] == "final_model_not_found"
    assert prediction_result.status_code == 404
    assert prediction_result.json()["error"]["code"] == "prediction_not_found"
    assert prediction_download.status_code == 404
    assert prediction_download.json()["error"]["code"] == "prediction_not_found"
    assert job.status_code == 404
    assert job.json()["error"]["code"] == "job_not_found"
