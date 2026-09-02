from __future__ import annotations

import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from mlforge.web.errors import WebStorageError
from mlforge.web.storage import (
    WEB_SCHEMA_VERSION,
    DatasetRecord,
    DatasetStore,
    ExperimentRecord,
    ExperimentStore,
)


def _database_version(workspace: Path) -> int:
    with sqlite3.connect(workspace / "mlforge.sqlite3") as connection:
        row = connection.execute("PRAGMA user_version").fetchone()
    assert row is not None
    return int(row[0])


def test_unversioned_workspace_is_adopted_and_restored(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = DatasetStore(workspace)
    store.initialize()

    dataset_id = uuid4()
    uploaded_path = store.final_upload_path(dataset_id)
    uploaded_path.write_text("feature,target\n1,yes\n2,no\n", encoding="utf-8")
    record = DatasetRecord(
        dataset_id=dataset_id,
        original_filename="training.csv",
        stored_filename=uploaded_path.name,
        file_size_bytes=uploaded_path.stat().st_size,
        row_count=2,
        column_count=2,
        columns=("feature", "target"),
        target="target",
        created_at=datetime.now(UTC),
    )
    store.create(record)

    assert _database_version(workspace) == WEB_SCHEMA_VERSION

    restored_workspace = tmp_path / "restored"
    shutil.copytree(workspace, restored_workspace)
    restored_store = DatasetStore(restored_workspace)
    restored_store.initialize()

    assert _database_version(restored_workspace) == WEB_SCHEMA_VERSION
    assert restored_store.get(dataset_id) == record
    assert restored_store.path_for(record).read_bytes() == uploaded_path.read_bytes()


def test_newer_web_schema_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "future-workspace"
    workspace.mkdir()
    with sqlite3.connect(workspace / "mlforge.sqlite3") as connection:
        connection.execute(f"PRAGMA user_version = {WEB_SCHEMA_VERSION + 1}")

    with pytest.raises(WebStorageError, match="newer than supported"):
        DatasetStore(workspace).initialize()


def test_v1_schema_migrates_without_losing_dependent_lineage(tmp_path: Path) -> None:
    """The regression migration must preserve all existing classification relationships."""
    workspace = tmp_path / "v1-workspace"
    DatasetStore(workspace).initialize()
    identifiers = {name: uuid4() for name in ("dataset", "experiment", "job", "finalization")}
    now = datetime.now(UTC).isoformat()
    with sqlite3.connect(workspace / "mlforge.sqlite3") as connection:
        connection.execute(
            "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(identifiers["dataset"]),
                "legacy.csv",
                f"{identifiers['dataset']}.csv",
                20,
                4,
                2,
                '["feature", "target"]',
                "target",
                now,
            ),
        )
        connection.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(identifiers["experiment"]),
                str(identifiers["dataset"]),
                "classification",
                "cross-validation",
                2,
                '["dummy-classifier", "logistic-regression"]',
                "balanced_accuracy",
                now,
            ),
        )
        connection.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(identifiers["job"]),
                str(identifiers["experiment"]),
                "waiting",
                now,
                None,
                None,
                None,
                None,
            ),
        )
        connection.execute(
            "INSERT INTO finalizations VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(identifiers["finalization"]),
                str(identifiers["experiment"]),
                "waiting",
                now,
                None,
                None,
                None,
                None,
            ),
        )
        connection.execute("PRAGMA user_version = 1")

    DatasetStore(workspace).initialize()

    with sqlite3.connect(workspace / "mlforge.sqlite3") as connection:
        experiment_count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()
        job_count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()
        finalization_count = connection.execute("SELECT COUNT(*) FROM finalizations").fetchone()
        foreign_key_error = connection.execute("PRAGMA foreign_key_check").fetchone()
    assert _database_version(workspace) == WEB_SCHEMA_VERSION
    assert experiment_count == (1,)
    assert job_count == (1,)
    assert finalization_count == (1,)
    assert foreign_key_error is None

    regression = ExperimentRecord(
        experiment_id=uuid4(),
        dataset_id=identifiers["dataset"],
        task="regression",
        validation_strategy="cross-validation",
        fold_count=2,
        estimators=("ridge-regression", "random-forest-regressor"),
        primary_metric="root_mean_squared_error",
        created_at=datetime.now(UTC),
    )
    ExperimentStore(workspace).create(regression)
    assert ExperimentStore(workspace).get(regression.experiment_id) == regression
