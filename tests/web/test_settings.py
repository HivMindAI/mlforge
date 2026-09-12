from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from mlforge.web import create_app
from mlforge.web.settings import WebSettings
from mlforge.web.storage import WEB_SCHEMA_VERSION


def _client(workspace: Path, *, max_upload_bytes: int = 100 * 1024 * 1024) -> TestClient:
    return TestClient(
        create_app(WebSettings(workspace=workspace, max_upload_bytes=max_upload_bytes))
    )


def test_settings_report_defaults_runtime_and_empty_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "operator-workspace"

    with _client(workspace, max_upload_bytes=4096) as client:
        response = client.get("/api/settings")

    assert response.status_code == 200
    body = response.json()
    assert body["preferences"] == {
        "default_fold_count": 5,
        "classification_metric": "balanced_accuracy",
        "regression_metric": "root_mean_squared_error",
        "classification_estimators": [
            "dummy-classifier",
            "logistic-regression",
            "random-forest-classifier",
        ],
        "regression_estimators": ["ridge-regression", "random-forest-regressor"],
        "updated_at": None,
    }
    assert body["workspace"]["name"] == "operator-workspace"
    assert body["workspace"]["max_upload_bytes"] == 4096
    assert body["workspace"]["counts"] == {
        "datasets": 0,
        "experiments": 0,
        "final_models": 0,
        "predictions": 0,
    }
    assert body["workspace"]["workspace_environment_variable"] == "MLFORGE_WEB_WORKSPACE"
    assert body["workspace"]["upload_limit_environment_variable"] == "MLFORGE_WEB_MAX_UPLOAD_BYTES"
    assert body["workspace"]["restart_required"] is True
    assert body["workspace"]["usage_bytes"] > 0
    assert body["system"]["web_schema_version"] == WEB_SCHEMA_VERSION
    assert body["system"]["mlforge_version"]
    assert body["system"]["python_version"]
    assert body["system"]["pandas_version"]
    assert body["system"]["scikit_learn_version"]
    assert body["diagnostics"] == {
        "api": "ready",
        "database": "ready",
        "worker": "available",
    }
    assert str(tmp_path) not in response.text


def test_settings_update_persists_and_controls_future_metric(tmp_path: Path) -> None:
    workspace = tmp_path / "web"
    settings_request = {
        "default_fold_count": 4,
        "classification_metric": "f1_macro",
        "regression_metric": "mean_absolute_error",
        "classification_estimators": ["dummy-classifier", "logistic-regression"],
        "regression_estimators": ["ridge-regression", "random-forest-regressor"],
    }

    with _client(workspace) as client:
        updated = client.put("/api/settings", json=settings_request)
        uploaded = client.post(
            "/api/datasets",
            files={
                "file": (
                    "classification.csv",
                    b"feature,target\n1,yes\n2,no\n3,yes\n4,no\n5,yes\n6,no\n",
                    "text/csv",
                )
            },
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
        )

    assert updated.status_code == 200
    assert updated.json()["preferences"] | {"updated_at": None} == settings_request | {
        "updated_at": None
    }
    assert updated.json()["preferences"]["updated_at"] is not None
    assert experiment.status_code == 201
    assert experiment.json()["primary_metric"] == "f1_macro"

    with _client(workspace) as restored_client:
        restored = restored_client.get("/api/settings")
    assert restored.status_code == 200
    assert restored.json()["preferences"]["default_fold_count"] == 4
    assert restored.json()["preferences"]["classification_metric"] == "f1_macro"


def test_invalid_settings_are_rejected_without_changing_defaults(tmp_path: Path) -> None:
    workspace = tmp_path / "web"
    request = {
        "default_fold_count": 11,
        "classification_metric": "not-a-metric",
        "regression_metric": "root_mean_squared_error",
        "classification_estimators": ["dummy-classifier", "logistic-regression"],
        "regression_estimators": ["ridge-regression"],
    }

    with _client(workspace) as client:
        rejected = client.put("/api/settings", json=request)
        unchanged = client.get("/api/settings")

    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "invalid_settings"
    assert unchanged.json()["preferences"]["default_fold_count"] == 5
    assert unchanged.json()["preferences"]["classification_metric"] == "balanced_accuracy"


def test_backup_download_contains_snapshot_and_excludes_prior_backups(tmp_path: Path) -> None:
    workspace = tmp_path / "web"

    with _client(workspace) as client:
        uploaded = client.post(
            "/api/datasets",
            files={
                "file": (
                    "training.csv",
                    b"feature,target\n1,yes\n2,no\n",
                    "text/csv",
                )
            },
        ).json()
        first = client.post("/api/settings/backup")
        second = client.post("/api/settings/backup")

    assert first.status_code == 200
    assert first.headers["content-type"] == "application/zip"
    assert first.headers["cache-control"] == "no-store"
    assert first.headers["x-content-type-options"] == "nosniff"
    with zipfile.ZipFile(io.BytesIO(second.content)) as archive:
        names = set(archive.namelist())
        assert "mlforge.sqlite3" in names
        assert f"uploads/{uploaded['dataset_id']}.csv" in names
        assert not any(name.startswith("backups/") for name in names)
        database_bytes = archive.read("mlforge.sqlite3")

    extracted_database = tmp_path / "backup.sqlite3"
    extracted_database.write_bytes(database_bytes)
    with sqlite3.connect(extracted_database) as connection:
        dataset_count = connection.execute("SELECT COUNT(*) FROM datasets").fetchone()
        settings_count = connection.execute("SELECT COUNT(*) FROM application_settings").fetchone()
        schema_version = connection.execute("PRAGMA user_version").fetchone()
    assert dataset_count == (1,)
    assert settings_count == (1,)
    assert schema_version == (WEB_SCHEMA_VERSION,)
    assert len(list((workspace / "backups").glob("*.zip"))) == 2
