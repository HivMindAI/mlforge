"""SQLite metadata and safe paths for local web datasets."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, suppress
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast
from uuid import UUID, uuid4

from mlforge.web.errors import (
    DatasetNotFoundError,
    ExperimentNotFoundError,
    FinalizationNotFoundError,
    JobNotFoundError,
    PredictionNotFoundError,
    PredictionResultUnavailableError,
    WebStorageError,
)

_DATASET_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL UNIQUE,
    file_size_bytes INTEGER NOT NULL CHECK (file_size_bytes > 0),
    row_count INTEGER NOT NULL CHECK (row_count > 0),
    column_count INTEGER NOT NULL CHECK (column_count > 0),
    columns_json TEXT NOT NULL,
    target TEXT,
    created_at TEXT NOT NULL
)
"""

_EXPERIMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    dataset_id TEXT NOT NULL,
    task TEXT NOT NULL CHECK (task = 'classification'),
    validation_strategy TEXT NOT NULL CHECK (validation_strategy = 'cross-validation'),
    fold_count INTEGER NOT NULL CHECK (fold_count BETWEEN 2 AND 10),
    estimators_json TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
)
"""

_EXPERIMENT_CREATED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_experiments_created
ON experiments(created_at DESC, experiment_id DESC)
"""

_JOB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('waiting', 'running', 'complete', 'failed')),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    benchmark_id TEXT,
    error_message TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
)
"""

_FINALIZATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS finalizations (
    finalization_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('waiting', 'running', 'complete', 'failed')),
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    final_model_id TEXT,
    error_message TEXT,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
)
"""

_FINALIZATION_EXPERIMENT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_finalizations_experiment_created
ON finalizations(experiment_id, created_at)
"""

_FINALIZATION_ACTIVE_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS idx_finalizations_one_active_or_complete
ON finalizations(experiment_id)
WHERE status IN ('waiting', 'running', 'complete')
"""

_FINALIZATION_COMPLETE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_finalizations_completed
ON finalizations(completed_at DESC, finalization_id DESC)
WHERE status = 'complete'
"""

_PREDICTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    finalization_id TEXT NOT NULL,
    final_model_id TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    input_stored_filename TEXT NOT NULL UNIQUE,
    output_stored_filename TEXT NOT NULL UNIQUE,
    input_file_size_bytes INTEGER NOT NULL CHECK (input_file_size_bytes > 0),
    row_count INTEGER NOT NULL CHECK (row_count > 0),
    status TEXT NOT NULL CHECK (status = 'complete'),
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    FOREIGN KEY (finalization_id) REFERENCES finalizations(finalization_id)
)
"""


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Web-owned metadata pointing to one immutable uploaded CSV."""

    dataset_id: UUID
    original_filename: str
    stored_filename: str
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    target: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    """Web-owned immutable configuration for a future comparison job."""

    experiment_id: UUID
    dataset_id: UUID
    task: str
    validation_strategy: str
    fold_count: int
    estimators: tuple[str, ...]
    primary_metric: str
    created_at: datetime


class JobStatus(StrEnum):
    """Honest persisted state of one background comparison job."""

    WAITING = "waiting"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class JobRecord:
    """Web-owned execution state linked to one immutable experiment configuration."""

    job_id: UUID
    experiment_id: UUID
    status: JobStatus
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    benchmark_id: UUID | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class FinalizationRecord:
    """Durable full-dataset fitting state linked to one completed experiment."""

    finalization_id: UUID
    experiment_id: UUID
    status: JobStatus
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    final_model_id: UUID | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class WebPredictionRecord:
    """Durable metadata for one successful schema-validated prediction run."""

    prediction_id: UUID
    finalization_id: UUID
    final_model_id: UUID
    original_filename: str
    input_stored_filename: str
    output_stored_filename: str
    input_file_size_bytes: int
    row_count: int
    status: Literal["complete"]
    created_at: datetime
    completed_at: datetime


class DatasetStore:
    """Persist dataset metadata without exposing local paths to API clients."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.uploads_directory = self.workspace / "uploads"
        self.database_path = self.workspace / "mlforge.sqlite3"

    def initialize(self) -> None:
        """Create the local workspace and the first web metadata table."""
        try:
            self.uploads_directory.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.execute(_DATASET_SCHEMA)
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise WebStorageError("Could not initialize the MLForge web workspace.") from error

    def check_ready(self) -> None:
        """Verify that metadata is readable and the durable upload volume is writable."""
        probe_path = self.uploads_directory / f".health-{uuid4()}.tmp"
        try:
            with closing(self._connect()) as connection:
                table = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'datasets'"
                ).fetchone()
                if table is None:
                    raise sqlite3.DatabaseError("The dataset metadata table is unavailable.")
            with probe_path.open("xb") as probe:
                probe.write(b"ready")
                probe.flush()
            probe_path.unlink()
        except (OSError, sqlite3.Error) as error:
            raise WebStorageError("The MLForge web workspace is not ready.") from error
        finally:
            if probe_path.exists():
                with suppress(OSError):
                    probe_path.unlink()

    def temporary_upload_path(self, dataset_id: UUID) -> Path:
        """Return a server-generated temporary CSV path inside the upload directory."""
        return self.uploads_directory / f".{dataset_id}.upload.csv"

    def final_upload_path(self, dataset_id: UUID) -> Path:
        """Return the immutable server-generated path for a validated CSV."""
        return self.uploads_directory / f"{dataset_id}.csv"

    def path_for(self, record: DatasetRecord) -> Path:
        """Resolve and verify the stored path represented by a metadata record."""
        candidate = (self.uploads_directory / record.stored_filename).resolve()
        if candidate.parent != self.uploads_directory or candidate.name != record.stored_filename:
            raise WebStorageError("Stored dataset metadata contains an unsafe filename.")
        return candidate

    def create(self, record: DatasetRecord) -> None:
        """Insert metadata for one newly published upload."""
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO datasets (
                        dataset_id,
                        original_filename,
                        stored_filename,
                        file_size_bytes,
                        row_count,
                        column_count,
                        columns_json,
                        target,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.dataset_id),
                        record.original_filename,
                        record.stored_filename,
                        record.file_size_bytes,
                        record.row_count,
                        record.column_count,
                        json.dumps(record.columns, ensure_ascii=False),
                        record.target,
                        record.created_at.isoformat(),
                    ),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not save dataset metadata.") from error

    def get(self, dataset_id: UUID) -> DatasetRecord:
        """Return one dataset record or an explicit not-found error."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                        dataset_id,
                        original_filename,
                        stored_filename,
                        file_size_bytes,
                        row_count,
                        column_count,
                        columns_json,
                        target,
                        created_at
                    FROM datasets
                    WHERE dataset_id = ?
                    """,
                    (str(dataset_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise WebStorageError("Could not read dataset metadata.") from error

        if row is None:
            raise DatasetNotFoundError(f"Dataset {dataset_id} was not found.")
        return self._record_from_row(row)

    def set_target(self, dataset_id: UUID, target: str) -> DatasetRecord:
        """Persist the explicitly selected target for a dataset."""
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(
                    "UPDATE datasets SET target = ? WHERE dataset_id = ?",
                    (target, str(dataset_id)),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not save the dataset target.") from error

        if cursor.rowcount != 1:
            raise DatasetNotFoundError(f"Dataset {dataset_id} was not found.")
        return replace(self.get(dataset_id), target=target)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _record_from_row(self, row: sqlite3.Row) -> DatasetRecord:
        try:
            raw_columns = json.loads(cast(str, row["columns_json"]))
            if not isinstance(raw_columns, list) or not all(
                isinstance(column, str) for column in raw_columns
            ):
                raise ValueError("Invalid columns metadata")
            target = row["target"]
            if target is not None and not isinstance(target, str):
                raise ValueError("Invalid target metadata")
            record = DatasetRecord(
                dataset_id=UUID(cast(str, row["dataset_id"])),
                original_filename=cast(str, row["original_filename"]),
                stored_filename=cast(str, row["stored_filename"]),
                file_size_bytes=cast(int, row["file_size_bytes"]),
                row_count=cast(int, row["row_count"]),
                column_count=cast(int, row["column_count"]),
                columns=tuple(raw_columns),
                target=target,
                created_at=datetime.fromisoformat(cast(str, row["created_at"])),
            )
            self.path_for(record)
            return record
        except (KeyError, TypeError, ValueError) as error:
            raise WebStorageError("Stored dataset metadata is invalid.") from error


class ExperimentStore:
    """Persist configured comparisons separately from immutable ML evidence."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.database_path = self.workspace / "mlforge.sqlite3"

    def initialize(self) -> None:
        """Create the experiment configuration table in the shared web database."""
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.execute(_EXPERIMENT_SCHEMA)
                connection.execute(_EXPERIMENT_CREATED_INDEX)
                connection.execute("PRAGMA optimize")
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise WebStorageError("Could not initialize experiment metadata storage.") from error

    def create(self, record: ExperimentRecord) -> None:
        """Insert one validated immutable comparison configuration."""
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO experiments (
                        experiment_id,
                        dataset_id,
                        task,
                        validation_strategy,
                        fold_count,
                        estimators_json,
                        primary_metric,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.experiment_id),
                        str(record.dataset_id),
                        record.task,
                        record.validation_strategy,
                        record.fold_count,
                        json.dumps(record.estimators),
                        record.primary_metric,
                        record.created_at.isoformat(),
                    ),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not save experiment configuration.") from error

    def get(self, experiment_id: UUID) -> ExperimentRecord:
        """Return one configured experiment or an explicit not-found error."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                        experiment_id,
                        dataset_id,
                        task,
                        validation_strategy,
                        fold_count,
                        estimators_json,
                        primary_metric,
                        created_at
                    FROM experiments
                    WHERE experiment_id = ?
                    """,
                    (str(experiment_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise WebStorageError("Could not read experiment configuration.") from error

        if row is None:
            raise ExperimentNotFoundError(f"Experiment {experiment_id} was not found.")
        return self._record_from_row(row)

    def list(self) -> tuple[ExperimentRecord, ...]:
        """Return all saved configurations newest first without changing their evidence."""
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT
                        experiment_id,
                        dataset_id,
                        task,
                        validation_strategy,
                        fold_count,
                        estimators_json,
                        primary_metric,
                        created_at
                    FROM experiments
                    ORDER BY created_at DESC, experiment_id DESC
                    """
                ).fetchall()
        except sqlite3.Error as error:
            raise WebStorageError("Could not list experiment configurations.") from error
        return tuple(self._record_from_row(row) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    def _record_from_row(self, row: sqlite3.Row) -> ExperimentRecord:
        try:
            raw_estimators = json.loads(cast(str, row["estimators_json"]))
            if not isinstance(raw_estimators, list) or not all(
                isinstance(estimator, str) and estimator for estimator in raw_estimators
            ):
                raise ValueError("Invalid estimator metadata")
            return ExperimentRecord(
                experiment_id=UUID(cast(str, row["experiment_id"])),
                dataset_id=UUID(cast(str, row["dataset_id"])),
                task=cast(str, row["task"]),
                validation_strategy=cast(str, row["validation_strategy"]),
                fold_count=cast(int, row["fold_count"]),
                estimators=tuple(raw_estimators),
                primary_metric=cast(str, row["primary_metric"]),
                created_at=datetime.fromisoformat(cast(str, row["created_at"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WebStorageError("Stored experiment configuration is invalid.") from error


class JobStore:
    """Persist one idempotent background job for each configured experiment."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.database_path = self.workspace / "mlforge.sqlite3"

    def initialize(self) -> None:
        """Create the job table in the shared web database."""
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.execute(_JOB_SCHEMA)
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise WebStorageError("Could not initialize job metadata storage.") from error

    def recover_interrupted(self, *, recovered_at: datetime) -> int:
        """Mark jobs lost during a process stop as failed instead of pretending they resumed."""
        message = "The MLForge API stopped before this comparison finished."
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(
                    """
                    UPDATE jobs
                    SET status = 'failed', completed_at = ?, error_message = ?
                    WHERE status IN ('waiting', 'running')
                    """,
                    (recovered_at.isoformat(), message),
                )
                connection.commit()
                return cursor.rowcount
        except sqlite3.Error as error:
            raise WebStorageError("Could not recover interrupted comparison jobs.") from error

    def create_or_get(
        self,
        experiment_id: UUID,
        *,
        created_at: datetime,
    ) -> tuple[JobRecord, bool]:
        """Create one waiting job, returning the existing record on duplicate requests."""
        job_id = uuid4()
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO jobs (
                        job_id,
                        experiment_id,
                        status,
                        created_at
                    ) VALUES (?, ?, 'waiting', ?)
                    """,
                    (str(job_id), str(experiment_id), created_at.isoformat()),
                )
                row = connection.execute(
                    """
                    SELECT
                        job_id,
                        experiment_id,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        benchmark_id,
                        error_message
                    FROM jobs
                    WHERE experiment_id = ?
                    """,
                    (str(experiment_id),),
                ).fetchone()
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not create a comparison job.") from error

        if row is None:
            raise WebStorageError("Comparison job creation did not return a stored record.")
        return self._record_from_row(row), cursor.rowcount == 1

    def get(self, job_id: UUID) -> JobRecord:
        """Return one persisted job or an explicit not-found error."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                        job_id,
                        experiment_id,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        benchmark_id,
                        error_message
                    FROM jobs
                    WHERE job_id = ?
                    """,
                    (str(job_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise WebStorageError("Could not read comparison job state.") from error

        if row is None:
            raise JobNotFoundError(f"Job {job_id} was not found.")
        return self._record_from_row(row)

    def find_for_experiment(self, experiment_id: UUID) -> JobRecord | None:
        """Return the experiment's job when one has been created."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                        job_id,
                        experiment_id,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        benchmark_id,
                        error_message
                    FROM jobs
                    WHERE experiment_id = ?
                    """,
                    (str(experiment_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise WebStorageError("Could not read experiment job state.") from error
        return self._record_from_row(row) if row is not None else None

    def mark_running(self, job_id: UUID, *, started_at: datetime) -> JobRecord:
        """Transition a waiting job to running."""
        self._transition(
            job_id,
            sql="UPDATE jobs SET status = 'running', started_at = ? "
            "WHERE job_id = ? AND status = 'waiting'",
            parameters=(started_at.isoformat(), str(job_id)),
        )
        return self.get(job_id)

    def mark_complete(
        self,
        job_id: UUID,
        *,
        benchmark_id: UUID,
        completed_at: datetime,
    ) -> JobRecord:
        """Transition a running job to complete with immutable benchmark lineage."""
        self._transition(
            job_id,
            sql="UPDATE jobs SET status = 'complete', completed_at = ?, benchmark_id = ? "
            "WHERE job_id = ? AND status = 'running'",
            parameters=(completed_at.isoformat(), str(benchmark_id), str(job_id)),
        )
        return self.get(job_id)

    def mark_failed(
        self,
        job_id: UUID,
        *,
        error_message: str,
        completed_at: datetime,
    ) -> JobRecord:
        """Transition an active job to failed with a readable error."""
        self._transition(
            job_id,
            sql="UPDATE jobs SET status = 'failed', completed_at = ?, error_message = ? "
            "WHERE job_id = ? AND status IN ('waiting', 'running')",
            parameters=(completed_at.isoformat(), error_message, str(job_id)),
        )
        return self.get(job_id)

    def _transition(
        self,
        job_id: UUID,
        *,
        sql: str,
        parameters: tuple[str, ...],
    ) -> None:
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(sql, parameters)
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not update comparison job state.") from error
        if cursor.rowcount != 1:
            raise WebStorageError(f"Comparison job {job_id} has an invalid state transition.")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    def _record_from_row(self, row: sqlite3.Row) -> JobRecord:
        try:
            benchmark_id = row["benchmark_id"]
            error_message = row["error_message"]
            if benchmark_id is not None and not isinstance(benchmark_id, str):
                raise ValueError("Invalid benchmark metadata")
            if error_message is not None and not isinstance(error_message, str):
                raise ValueError("Invalid job error metadata")
            return JobRecord(
                job_id=UUID(cast(str, row["job_id"])),
                experiment_id=UUID(cast(str, row["experiment_id"])),
                status=JobStatus(cast(str, row["status"])),
                created_at=datetime.fromisoformat(cast(str, row["created_at"])),
                started_at=(
                    datetime.fromisoformat(cast(str, row["started_at"]))
                    if row["started_at"] is not None
                    else None
                ),
                completed_at=(
                    datetime.fromisoformat(cast(str, row["completed_at"]))
                    if row["completed_at"] is not None
                    else None
                ),
                benchmark_id=UUID(benchmark_id) if benchmark_id is not None else None,
                error_message=error_message,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WebStorageError("Stored comparison job state is invalid.") from error


class FinalizationStore:
    """Persist retryable final-fit attempts while allowing only one active or complete result."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.database_path = self.workspace / "mlforge.sqlite3"

    def initialize(self) -> None:
        """Create finalization metadata and indexes from the actual lookup patterns."""
        try:
            self.workspace.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.execute(_FINALIZATION_SCHEMA)
                connection.execute(_FINALIZATION_EXPERIMENT_INDEX)
                connection.execute(_FINALIZATION_ACTIVE_INDEX)
                connection.execute(_FINALIZATION_COMPLETE_INDEX)
                connection.execute("PRAGMA optimize")
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise WebStorageError("Could not initialize finalization metadata storage.") from error

    def recover_interrupted(self, *, recovered_at: datetime) -> int:
        """Mark final fits lost during a process stop as failed and retryable."""
        message = "The MLForge API stopped before this final model finished."
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(
                    """
                    UPDATE finalizations
                    SET status = 'failed', completed_at = ?, error_message = ?
                    WHERE status IN ('waiting', 'running')
                    """,
                    (recovered_at.isoformat(), message),
                )
                connection.commit()
                return cursor.rowcount
        except sqlite3.Error as error:
            raise WebStorageError("Could not recover interrupted finalizations.") from error

    def create_or_get(
        self,
        experiment_id: UUID,
        *,
        created_at: datetime,
    ) -> tuple[FinalizationRecord, bool]:
        """Create a new attempt unless one is already active or complete."""
        finalization_id = uuid4()
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO finalizations (
                        finalization_id,
                        experiment_id,
                        status,
                        created_at
                    ) VALUES (?, ?, 'waiting', ?)
                    """,
                    (str(finalization_id), str(experiment_id), created_at.isoformat()),
                )
                row = connection.execute(
                    """
                    SELECT
                        finalization_id,
                        experiment_id,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        final_model_id,
                        error_message
                    FROM finalizations
                    WHERE experiment_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT 1
                    """,
                    (str(experiment_id),),
                ).fetchone()
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not create a finalization job.") from error

        if row is None:
            raise WebStorageError("Finalization job creation did not return a stored record.")
        return self._record_from_row(row), cursor.rowcount == 1

    def get(self, finalization_id: UUID) -> FinalizationRecord:
        """Return one persisted finalization attempt."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                        finalization_id,
                        experiment_id,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        final_model_id,
                        error_message
                    FROM finalizations
                    WHERE finalization_id = ?
                    """,
                    (str(finalization_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise WebStorageError("Could not read finalization state.") from error
        if row is None:
            raise FinalizationNotFoundError(f"Finalization {finalization_id} was not found.")
        return self._record_from_row(row)

    def find_for_experiment(self, experiment_id: UUID) -> FinalizationRecord | None:
        """Return the most recent finalization attempt for one experiment."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                        finalization_id,
                        experiment_id,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        final_model_id,
                        error_message
                    FROM finalizations
                    WHERE experiment_id = ?
                    ORDER BY created_at DESC, rowid DESC
                    LIMIT 1
                    """,
                    (str(experiment_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise WebStorageError("Could not read experiment finalization state.") from error
        return self._record_from_row(row) if row is not None else None

    def find_for_final_model(self, final_model_id: UUID) -> FinalizationRecord | None:
        """Resolve the web-owned lineage record for one completed final model."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                        finalization_id,
                        experiment_id,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        final_model_id,
                        error_message
                    FROM finalizations
                    WHERE final_model_id = ?
                    LIMIT 1
                    """,
                    (str(final_model_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise WebStorageError("Could not read final-model lineage state.") from error
        return self._record_from_row(row) if row is not None else None

    def list_completed(self) -> tuple[FinalizationRecord, ...]:
        """Return completed web-owned models, newest first."""
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT
                        finalization_id,
                        experiment_id,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        final_model_id,
                        error_message
                    FROM finalizations
                    WHERE status = 'complete'
                    ORDER BY completed_at DESC, finalization_id DESC
                    """
                ).fetchall()
        except sqlite3.Error as error:
            raise WebStorageError("Could not list completed final models.") from error
        return tuple(self._record_from_row(row) for row in rows)

    def mark_running(
        self,
        finalization_id: UUID,
        *,
        started_at: datetime,
    ) -> FinalizationRecord:
        """Transition a waiting finalization to running."""
        self._transition(
            finalization_id,
            sql="UPDATE finalizations SET status = 'running', started_at = ? "
            "WHERE finalization_id = ? AND status = 'waiting'",
            parameters=(started_at.isoformat(), str(finalization_id)),
        )
        return self.get(finalization_id)

    def mark_complete(
        self,
        finalization_id: UUID,
        *,
        final_model_id: UUID,
        completed_at: datetime,
    ) -> FinalizationRecord:
        """Transition a running finalization to a completed final-model lineage."""
        self._transition(
            finalization_id,
            sql="UPDATE finalizations SET status = 'complete', completed_at = ?, "
            "final_model_id = ? WHERE finalization_id = ? AND status = 'running'",
            parameters=(completed_at.isoformat(), str(final_model_id), str(finalization_id)),
        )
        return self.get(finalization_id)

    def mark_failed(
        self,
        finalization_id: UUID,
        *,
        error_message: str,
        completed_at: datetime,
        final_model_id: UUID | None = None,
    ) -> FinalizationRecord:
        """Persist a readable failure and optional failed core-manifest identity."""
        model_id = str(final_model_id) if final_model_id is not None else None
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(
                    """
                    UPDATE finalizations
                    SET status = 'failed', completed_at = ?, final_model_id = ?, error_message = ?
                    WHERE finalization_id = ? AND status IN ('waiting', 'running')
                    """,
                    (
                        completed_at.isoformat(),
                        model_id,
                        error_message,
                        str(finalization_id),
                    ),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not update finalization state.") from error
        if cursor.rowcount != 1:
            raise WebStorageError(
                f"Finalization {finalization_id} has an invalid state transition."
            )
        return self.get(finalization_id)

    def _transition(
        self,
        finalization_id: UUID,
        *,
        sql: str,
        parameters: tuple[str, ...],
    ) -> None:
        try:
            with closing(self._connect()) as connection:
                cursor = connection.execute(sql, parameters)
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not update finalization state.") from error
        if cursor.rowcount != 1:
            raise WebStorageError(
                f"Finalization {finalization_id} has an invalid state transition."
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    def _record_from_row(self, row: sqlite3.Row) -> FinalizationRecord:
        try:
            raw_model_id = row["final_model_id"]
            error_message = row["error_message"]
            if raw_model_id is not None and not isinstance(raw_model_id, str):
                raise ValueError("Invalid final-model metadata")
            if error_message is not None and not isinstance(error_message, str):
                raise ValueError("Invalid finalization error metadata")
            record = FinalizationRecord(
                finalization_id=UUID(cast(str, row["finalization_id"])),
                experiment_id=UUID(cast(str, row["experiment_id"])),
                status=JobStatus(cast(str, row["status"])),
                created_at=datetime.fromisoformat(cast(str, row["created_at"])),
                started_at=(
                    datetime.fromisoformat(cast(str, row["started_at"]))
                    if row["started_at"] is not None
                    else None
                ),
                completed_at=(
                    datetime.fromisoformat(cast(str, row["completed_at"]))
                    if row["completed_at"] is not None
                    else None
                ),
                final_model_id=UUID(raw_model_id) if raw_model_id is not None else None,
                error_message=error_message,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WebStorageError("Stored finalization state is invalid.") from error

        if record.status is JobStatus.COMPLETE and (
            record.final_model_id is None
            or record.completed_at is None
            or record.error_message is not None
        ):
            raise WebStorageError("Stored completed finalization state is invalid.")
        if record.status is JobStatus.FAILED and (
            record.completed_at is None or record.error_message is None
        ):
            raise WebStorageError("Stored failed finalization state is invalid.")
        if record.status in {JobStatus.WAITING, JobStatus.RUNNING} and (
            record.final_model_id is not None
            or record.completed_at is not None
            or record.error_message is not None
        ):
            raise WebStorageError("Stored active finalization state is invalid.")
        return record


class PredictionStore:
    """Persist successful local prediction inputs, outputs, and web metadata."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.inputs_directory = self.workspace / "prediction-inputs"
        self.outputs_directory = self.workspace / "predictions"
        self.database_path = self.workspace / "mlforge.sqlite3"

    def initialize(self) -> None:
        """Create prediction directories and the terminal-run metadata table."""
        try:
            self.inputs_directory.mkdir(parents=True, exist_ok=True)
            self.outputs_directory.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection:
                connection.execute(_PREDICTION_SCHEMA)
                connection.execute("PRAGMA optimize")
                connection.commit()
        except (OSError, sqlite3.Error) as error:
            raise WebStorageError("Could not initialize prediction storage.") from error

    def temporary_input_path(self, prediction_id: UUID) -> Path:
        """Return a server-owned temporary upload path."""
        return self.inputs_directory / f".{prediction_id}.upload.csv"

    def final_input_path(self, prediction_id: UUID) -> Path:
        """Return the immutable stored prediction-input path."""
        return self.inputs_directory / f"{prediction_id}.csv"

    def output_path(self, prediction_id: UUID) -> Path:
        """Return the create-only prediction-output path."""
        return self.outputs_directory / f"{prediction_id}.csv"

    def create(self, record: WebPredictionRecord) -> None:
        """Persist one completed prediction after both immutable files exist."""
        try:
            with closing(self._connect()) as connection:
                connection.execute(
                    """
                    INSERT INTO predictions (
                        prediction_id,
                        finalization_id,
                        final_model_id,
                        original_filename,
                        input_stored_filename,
                        output_stored_filename,
                        input_file_size_bytes,
                        row_count,
                        status,
                        created_at,
                        completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(record.prediction_id),
                        str(record.finalization_id),
                        str(record.final_model_id),
                        record.original_filename,
                        record.input_stored_filename,
                        record.output_stored_filename,
                        record.input_file_size_bytes,
                        record.row_count,
                        record.status,
                        record.created_at.isoformat(),
                        record.completed_at.isoformat(),
                    ),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise WebStorageError("Could not save prediction metadata.") from error

    def get(self, prediction_id: UUID) -> WebPredictionRecord:
        """Return one completed prediction record or an explicit not-found error."""
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT
                        prediction_id,
                        finalization_id,
                        final_model_id,
                        original_filename,
                        input_stored_filename,
                        output_stored_filename,
                        input_file_size_bytes,
                        row_count,
                        status,
                        created_at,
                        completed_at
                    FROM predictions
                    WHERE prediction_id = ?
                    """,
                    (str(prediction_id),),
                ).fetchone()
        except sqlite3.Error as error:
            raise WebStorageError("Could not read prediction metadata.") from error

        if row is None:
            raise PredictionNotFoundError(f"Prediction {prediction_id} was not found.")
        return self._record_from_row(row)

    def output_path_for(self, record: WebPredictionRecord) -> Path:
        """Resolve one stored result without accepting metadata-controlled paths."""
        expected_filename = f"{record.prediction_id}.csv"
        if record.output_stored_filename != expected_filename:
            raise PredictionResultUnavailableError(
                "The saved prediction output has invalid storage metadata."
            )
        candidate = self.outputs_directory / expected_filename
        if candidate.is_symlink() or not candidate.is_file():
            raise PredictionResultUnavailableError(
                "The saved prediction output file is missing or is not a regular file."
            )
        try:
            output_directory = self.outputs_directory.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise PredictionResultUnavailableError(
                "The saved prediction output file could not be resolved."
            ) from error
        if resolved.parent != output_directory or resolved.name != expected_filename:
            raise PredictionResultUnavailableError("The saved prediction output path is invalid.")
        return resolved

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        return connection

    def _record_from_row(self, row: sqlite3.Row) -> WebPredictionRecord:
        try:
            status = cast(str, row["status"])
            if status != "complete":
                raise ValueError("Invalid prediction status")
            record = WebPredictionRecord(
                prediction_id=UUID(cast(str, row["prediction_id"])),
                finalization_id=UUID(cast(str, row["finalization_id"])),
                final_model_id=UUID(cast(str, row["final_model_id"])),
                original_filename=cast(str, row["original_filename"]),
                input_stored_filename=cast(str, row["input_stored_filename"]),
                output_stored_filename=cast(str, row["output_stored_filename"]),
                input_file_size_bytes=cast(int, row["input_file_size_bytes"]),
                row_count=cast(int, row["row_count"]),
                status="complete",
                created_at=datetime.fromisoformat(cast(str, row["created_at"])),
                completed_at=datetime.fromisoformat(cast(str, row["completed_at"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WebStorageError("Stored prediction metadata is invalid.") from error
        if (
            not record.original_filename
            or record.input_file_size_bytes <= 0
            or record.row_count <= 0
            or record.completed_at < record.created_at
        ):
            raise WebStorageError("Stored prediction metadata is invalid.")
        return record
