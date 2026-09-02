"""Web workflow orchestration over the existing MLForge dataset API."""

from __future__ import annotations

import asyncio
import csv
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

from mlforge.artifacts import (
    ARTIFACT_SUFFIX,
    ArtifactManifest,
    LocalArtifactStore,
    verify_final_model_manifest,
)
from mlforge.benchmarks import (
    CrossValidationConfig,
    CrossValidationManifest,
    CrossValidationMetricSummary,
    LocalCrossValidationStore,
)
from mlforge.datasets import (
    CsvLoadOptions,
    DatasetProfile,
    TaskHint,
    load_csv,
    load_feature_csv,
    profile_dataset,
)
from mlforge.errors import (
    ArtifactError,
    BenchmarkStoreError,
    ConfigurationError,
    DatasetError,
    DatasetSplitError,
    DatasetValidationError,
    FinalModelStoreError,
    InferenceError,
    MLForgeError,
    PredictionSchemaError,
)
from mlforge.final_models import FinalModelManifest, LocalFinalModelStore
from mlforge.inference import PredictionResult, predict_csv, write_predictions_csv
from mlforge.pipelines import (
    CrossValidationSplitConfig,
    TaskType,
    split_cross_validation_folds,
)
from mlforge.web.errors import (
    ExperimentResultNotReadyError,
    ExperimentValidationError,
    FinalModelNotFoundError,
    InvalidModelArtifactError,
    PredictionExecutionError,
    PredictionInputValidationError,
    PredictionResultUnavailableError,
    UploadValidationError,
    WebStorageError,
)
from mlforge.web.settings import WebSettings
from mlforge.web.storage import (
    DatasetRecord,
    DatasetStore,
    ExperimentRecord,
    ExperimentStore,
    FinalizationRecord,
    FinalizationStore,
    JobRecord,
    JobStatus,
    JobStore,
    PredictionStore,
    WebPredictionRecord,
)

_UPLOAD_CHUNK_BYTES = 1024 * 1024
_PREDICTION_PREVIEW_LIMIT = 20


@dataclass(frozen=True, slots=True)
class DatasetAnalysis:
    """A web record paired with a freshly computed core dataset profile."""

    record: DatasetRecord
    profile: DatasetProfile


class DatasetService:
    """Adapt durable HTTP uploads to MLForge's path-based dataset functions."""

    def __init__(self, store: DatasetStore, settings: WebSettings) -> None:
        self.store = store
        self.settings = settings

    async def upload(self, upload: UploadFile) -> DatasetRecord:
        """Stream, validate, and publish one immutable CSV upload."""
        original_filename = _safe_display_filename(upload.filename)
        if Path(original_filename).suffix.lower() != ".csv":
            raise UploadValidationError("Choose a file with the .csv extension.")

        dataset_id = uuid4()
        temporary_path = self.store.temporary_upload_path(dataset_id)
        final_path = self.store.final_upload_path(dataset_id)
        bytes_written = 0
        published = False

        try:
            with temporary_path.open("xb") as destination:
                while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                    bytes_written += len(chunk)
                    if bytes_written > self.settings.max_upload_bytes:
                        raise UploadValidationError(
                            "CSV files must be "
                            f"{_format_upload_limit(self.settings.max_upload_bytes)} or smaller."
                        )
                    destination.write(chunk)

            if bytes_written == 0:
                raise UploadValidationError("The selected CSV file is empty.")

            options = CsvLoadOptions(max_file_size_bytes=self.settings.max_upload_bytes)
            frame = load_feature_csv(temporary_path, options=options)
            columns = tuple(str(column) for column in frame.columns)
            os.replace(temporary_path, final_path)
            published = True

            record = DatasetRecord(
                dataset_id=dataset_id,
                original_filename=original_filename,
                stored_filename=final_path.name,
                file_size_bytes=bytes_written,
                row_count=len(frame),
                column_count=len(columns),
                columns=columns,
                target=None,
                created_at=datetime.now(UTC),
            )
            try:
                self.store.create(record)
            except WebStorageError:
                _unlink_if_present(final_path)
                published = False
                raise
            return record
        except (MLForgeError, UploadValidationError, WebStorageError):
            raise
        except OSError as error:
            raise WebStorageError("Could not store the uploaded CSV file.") from error
        finally:
            with suppress(OSError):
                await upload.close()
            _unlink_if_present(temporary_path)
            if not published:
                _unlink_if_present(final_path)

    def get(self, dataset_id: UUID) -> DatasetRecord:
        """Load one web dataset record."""
        return self.store.get(dataset_id)

    def select_target(self, dataset_id: UUID, target: str) -> DatasetRecord:
        """Validate and persist an explicit target using the MLForge core."""
        record = self.store.get(dataset_id)
        dataset_path = self.store.path_for(record)
        options = CsvLoadOptions(max_file_size_bytes=self.settings.max_upload_bytes)
        load_csv(dataset_path, target=target, options=options)
        return self.store.set_target(dataset_id, target)

    def analyze(self, dataset_id: UUID) -> DatasetAnalysis:
        """Load and profile a target-configured dataset through the MLForge core."""
        record = self.store.get(dataset_id)
        if record.target is None:
            raise DatasetValidationError("Choose a target column before analyzing this dataset.")

        dataset_path = self.store.path_for(record)
        options = CsvLoadOptions(max_file_size_bytes=self.settings.max_upload_bytes)
        dataset = load_csv(dataset_path, target=record.target, options=options)
        return DatasetAnalysis(record=record, profile=profile_dataset(dataset))


class ExperimentService:
    """Validate and persist comparison configuration without starting training."""

    def __init__(
        self,
        dataset_store: DatasetStore,
        experiment_store: ExperimentStore,
        job_store: JobStore,
        settings: WebSettings,
    ) -> None:
        self.dataset_store = dataset_store
        self.experiment_store = experiment_store
        self.job_store = job_store
        self.settings = settings

    def get(self, experiment_id: UUID) -> ExperimentRecord:
        """Load one persisted experiment configuration."""
        return self.experiment_store.get(experiment_id)

    def list(self) -> tuple[ExperimentHistoryEntry, ...]:
        """Join web-owned display metadata and job state for history browsing."""
        return tuple(
            ExperimentHistoryEntry(
                experiment=experiment,
                dataset=self.dataset_store.get(experiment.dataset_id),
                job=self.job_store.find_for_experiment(experiment.experiment_id),
            )
            for experiment in self.experiment_store.list()
        )

    def create(
        self,
        dataset_id: UUID,
        *,
        estimators: tuple[str, ...],
        fold_count: int,
    ) -> ExperimentRecord:
        """Create one core-validated supervised comparison configuration."""
        dataset_record = self.dataset_store.get(dataset_id)
        if dataset_record.target is None:
            raise ExperimentValidationError(
                "Choose a target column before configuring an experiment."
            )

        dataset_path = self.dataset_store.path_for(dataset_record)
        options = CsvLoadOptions(max_file_size_bytes=self.settings.max_upload_bytes)
        dataset = load_csv(dataset_path, target=dataset_record.target, options=options)
        profile = profile_dataset(dataset)
        if profile.target.task_hint is TaskHint.UNDETERMINED:
            detected = profile.target.task_hint.value
            raise ExperimentValidationError(
                "Model comparison requires a classification or regression target. "
                f"MLForge detected the selected target as {detected}."
            )
        task = TaskType(profile.target.task_hint.value)
        primary_metric = (
            "balanced_accuracy" if task is TaskType.CLASSIFICATION else "root_mean_squared_error"
        )

        try:
            config = CrossValidationConfig(
                task=task,
                estimators=estimators,
                primary_metric=primary_metric,
                split=CrossValidationSplitConfig(fold_count=fold_count),
            )
            split_cross_validation_folds(dataset, task=task, config=config.split)
        except (ConfigurationError, DatasetSplitError) as error:
            raise ExperimentValidationError(str(error)) from error

        record = ExperimentRecord(
            experiment_id=uuid4(),
            dataset_id=dataset_id,
            task=task.value,
            validation_strategy="cross-validation",
            fold_count=config.split.fold_count,
            estimators=config.estimators,
            primary_metric=config.primary_metric,
            created_at=datetime.now(UTC),
        )
        self.experiment_store.create(record)
        return record


@dataclass(frozen=True, slots=True)
class ExperimentHistoryEntry:
    """One read-only experiment history row assembled from persisted web metadata."""

    experiment: ExperimentRecord
    dataset: DatasetRecord
    job: JobRecord | None


class ExperimentResultService:
    """Read terminal benchmark evidence and verify its saved experiment lineage."""

    def __init__(
        self,
        dataset_store: DatasetStore,
        experiment_store: ExperimentStore,
        job_store: JobStore,
        settings: WebSettings,
    ) -> None:
        self.dataset_store = dataset_store
        self.experiment_store = experiment_store
        self.job_store = job_store
        self.benchmark_store = LocalCrossValidationStore(
            settings.workspace / "mlbenchmarks" / "cross-validation"
        )

    def get(self, experiment_id: UUID) -> CrossValidationManifest:
        """Return the core-validated result for one completed experiment."""
        experiment = self.experiment_store.get(experiment_id)
        job = self.job_store.find_for_experiment(experiment_id)
        if job is None:
            raise ExperimentResultNotReadyError(
                "Run this experiment before requesting its results."
            )
        if job.status is not JobStatus.COMPLETE:
            if job.status is JobStatus.FAILED:
                message = "This experiment did not produce a successful comparison result."
            else:
                message = "Experiment results are available after the comparison completes."
            raise ExperimentResultNotReadyError(message)
        if job.benchmark_id is None:
            raise WebStorageError("Completed comparison job is missing its benchmark reference.")

        try:
            manifest = self.benchmark_store.read(str(job.benchmark_id))
        except BenchmarkStoreError as error:
            raise WebStorageError("Could not load the saved comparison result.") from error

        dataset = self.dataset_store.get(experiment.dataset_id)
        configuration = manifest.configuration
        lineage_matches = (
            configuration.task == experiment.task
            and configuration.estimators == experiment.estimators
            and configuration.fold_count == experiment.fold_count
            and configuration.primary_metric == experiment.primary_metric
            and manifest.dataset.row_count == dataset.row_count
            and manifest.dataset.column_count == dataset.column_count
            and manifest.dataset.target == dataset.target
        )
        if not lineage_matches:
            raise WebStorageError(
                "Saved comparison result does not match its experiment configuration."
            )
        return manifest


@dataclass(frozen=True, slots=True)
class FinalModelDetails:
    """Safe final-model and artifact manifests without executable pipeline bytes."""

    dataset: DatasetRecord
    experiment: ExperimentRecord
    finalization: FinalizationRecord
    manifest: FinalModelManifest
    artifact: ArtifactManifest
    artifact_filename: str
    metrics: tuple[CrossValidationMetricSummary, ...]


class FinalModelService:
    """Safely inspect completed local final models without loading pickle payloads."""

    def __init__(
        self,
        dataset_store: DatasetStore,
        experiment_store: ExperimentStore,
        finalization_store: FinalizationStore,
        experiment_result_service: ExperimentResultService,
        settings: WebSettings,
    ) -> None:
        self.dataset_store = dataset_store
        self.experiment_store = experiment_store
        self.finalization_store = finalization_store
        self.experiment_result_service = experiment_result_service
        self.final_model_store = LocalFinalModelStore(settings.workspace / "mlfinalmodels")
        self.artifact_store = LocalArtifactStore(settings.workspace / "artifacts")

    def get(self, final_model_id: UUID) -> FinalModelDetails:
        """Return verified display metadata for one web-owned final model."""
        finalization = self.finalization_store.find_for_final_model(final_model_id)
        if finalization is None or finalization.status is not JobStatus.COMPLETE:
            raise FinalModelNotFoundError(f"Final model {final_model_id} was not found.")
        return self._details(finalization)

    def list(self) -> tuple[FinalModelDetails, ...]:
        """Return every completed web-owned model, newest first."""
        return tuple(
            self._details(finalization) for finalization in self.finalization_store.list_completed()
        )

    def _details(self, finalization: FinalizationRecord) -> FinalModelDetails:
        final_model_id = finalization.final_model_id
        if final_model_id is None:
            raise WebStorageError("Completed finalization is missing its final-model reference.")

        try:
            manifest = self.final_model_store.read(str(final_model_id))
            artifact = self.artifact_store.inspect(str(final_model_id))
            verify_final_model_manifest(artifact, manifest)
        except (ArtifactError, FinalModelStoreError) as error:
            raise InvalidModelArtifactError(
                "The finalized model artifact is missing, corrupt, or incompatible."
            ) from error

        experiment = self.experiment_store.get(finalization.experiment_id)
        dataset = self.dataset_store.get(experiment.dataset_id)
        try:
            comparison = self.experiment_result_service.get(experiment.experiment_id)
        except ExperimentResultNotReadyError as error:
            raise WebStorageError("Final model source comparison is not available.") from error
        if manifest.selection.benchmark_id != comparison.benchmark_id:
            raise WebStorageError("Final model does not match its experiment benchmark lineage.")

        selected_entry = next(
            (
                entry
                for entry in comparison.entries
                if entry.estimator == manifest.configuration.estimator and entry.rank == 1
            ),
            None,
        )
        if selected_entry is None:
            raise WebStorageError("Final model does not match its selected comparison entry.")
        return FinalModelDetails(
            dataset=dataset,
            experiment=experiment,
            finalization=finalization,
            manifest=manifest,
            artifact=artifact,
            artifact_filename=f"{final_model_id}{ARTIFACT_SUFFIX}",
            metrics=selected_entry.metrics,
        )


class PredictionService:
    """Run the core prediction contract for one trusted web-owned final model."""

    def __init__(
        self,
        store: PredictionStore,
        final_model_service: FinalModelService,
        settings: WebSettings,
    ) -> None:
        self.store = store
        self.final_model_service = final_model_service
        self.settings = settings
        self.artifact_store = LocalArtifactStore(settings.workspace / "artifacts")

    async def create(
        self,
        final_model_id: UUID,
        upload: UploadFile,
    ) -> WebPredictionRecord:
        """Validate, predict, and durably save one CSV without exposing its results yet."""
        try:
            original_filename = _safe_display_filename(upload.filename)
        except UploadValidationError as error:
            raise PredictionInputValidationError(str(error)) from error
        if Path(original_filename).suffix.lower() != ".csv":
            raise PredictionInputValidationError("Choose a file with the .csv extension.")

        prediction_id = uuid4()
        temporary_path = self.store.temporary_input_path(prediction_id)
        final_input_path = self.store.final_input_path(prediction_id)
        output_path = self.store.output_path(prediction_id)
        bytes_written = 0
        completed = False
        created_at = datetime.now(UTC)

        try:
            # This establishes that the id belongs to a completed web finalization and
            # verifies lineage before executable bytes are loaded with trusted=True.
            details = self.final_model_service.get(final_model_id)
            with temporary_path.open("xb") as destination:
                while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                    bytes_written += len(chunk)
                    if bytes_written > self.settings.max_upload_bytes:
                        raise PredictionInputValidationError(
                            "Prediction CSV files must be "
                            f"{_format_upload_limit(self.settings.max_upload_bytes)} or smaller."
                        )
                    destination.write(chunk)

            if bytes_written == 0:
                raise PredictionInputValidationError("The selected prediction CSV file is empty.")

            result = await asyncio.to_thread(
                self._predict_and_write,
                final_model_id,
                details.manifest,
                temporary_path,
                output_path,
            )
            os.replace(temporary_path, final_input_path)
            completed_at = datetime.now(UTC)
            record = WebPredictionRecord(
                prediction_id=prediction_id,
                finalization_id=details.finalization.finalization_id,
                final_model_id=final_model_id,
                original_filename=original_filename,
                input_stored_filename=final_input_path.name,
                output_stored_filename=output_path.name,
                input_file_size_bytes=bytes_written,
                row_count=result.row_count,
                status="complete",
                created_at=created_at,
                completed_at=completed_at,
            )
            self.store.create(record)
            completed = True
            return record
        except InvalidModelArtifactError:
            raise
        except ArtifactError as error:
            raise InvalidModelArtifactError(
                "The finalized model artifact is missing, corrupt, or incompatible."
            ) from error
        except (PredictionSchemaError, DatasetError) as error:
            raise PredictionInputValidationError(
                _path_free_prediction_message(error, temporary_path)
            ) from error
        except InferenceError as error:
            raise PredictionExecutionError(
                _path_free_prediction_message(error, temporary_path)
            ) from error
        except (PredictionInputValidationError, WebStorageError):
            raise
        except OSError as error:
            raise WebStorageError("Could not store the prediction files.") from error
        finally:
            with suppress(OSError):
                await upload.close()
            _unlink_if_present(temporary_path)
            if not completed:
                _unlink_if_present(final_input_path)
                _unlink_if_present(output_path)

    def _predict_and_write(
        self,
        final_model_id: UUID,
        final_model_manifest: FinalModelManifest,
        input_path: Path,
        output_path: Path,
    ) -> PredictionResult:
        """Load only a verified local artifact, then use the public core inference API."""
        artifact = self.artifact_store.load(str(final_model_id), trusted=True)
        verify_final_model_manifest(artifact.manifest, final_model_manifest)
        options = CsvLoadOptions(max_file_size_bytes=self.settings.max_upload_bytes)
        result = predict_csv(artifact, input_path, options=options)
        write_predictions_csv(result, output_path)
        return result

    def get(self, prediction_id: UUID) -> PredictionDetails:
        """Return one validated terminal result with a bounded preview."""
        record = self.store.get(prediction_id)
        preview = self._inspect_output(record)
        return PredictionDetails(
            record=record,
            preview=preview,
            preview_limit=_PREDICTION_PREVIEW_LIMIT,
        )

    def download_path(self, prediction_id: UUID) -> Path:
        """Return a validated full result path suitable for a streaming response."""
        record = self.store.get(prediction_id)
        self._inspect_output(record)
        return self.store.output_path_for(record)

    def _inspect_output(
        self,
        record: WebPredictionRecord,
    ) -> tuple[PredictionPreviewRow, ...]:
        output_path = self.store.output_path_for(record)
        preview: list[PredictionPreviewRow] = []
        observed_rows = 0
        try:
            with output_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader)
                if header != ["row_number", "prediction"]:
                    raise PredictionResultUnavailableError(
                        "The saved prediction output has an invalid CSV header."
                    )
                for row in reader:
                    observed_rows += 1
                    if len(row) != 2:
                        raise PredictionResultUnavailableError(
                            "The saved prediction output contains an invalid row."
                        )
                    try:
                        row_number = int(row[0])
                    except ValueError as error:
                        raise PredictionResultUnavailableError(
                            "The saved prediction output contains an invalid row number."
                        ) from error
                    if row_number != observed_rows:
                        raise PredictionResultUnavailableError(
                            "The saved prediction output row numbers are not sequential."
                        )
                    if len(preview) < _PREDICTION_PREVIEW_LIMIT:
                        preview.append(
                            PredictionPreviewRow(
                                row_number=row_number,
                                prediction=row[1],
                            )
                        )
        except PredictionResultUnavailableError:
            raise
        except (OSError, UnicodeError, csv.Error, StopIteration) as error:
            raise PredictionResultUnavailableError(
                "The saved prediction output CSV is missing, malformed, or unreadable."
            ) from error

        if observed_rows != record.row_count:
            raise PredictionResultUnavailableError(
                "The saved prediction output row count does not match its metadata."
            )
        return tuple(preview)


@dataclass(frozen=True, slots=True)
class PredictionPreviewRow:
    """One display-safe CSV prediction value and its stable input row number."""

    row_number: int
    prediction: str


@dataclass(frozen=True, slots=True)
class PredictionDetails:
    """One terminal prediction record paired with a bounded validated preview."""

    record: WebPredictionRecord
    preview: tuple[PredictionPreviewRow, ...]
    preview_limit: int


def _safe_display_filename(filename: str | None) -> str:
    if filename is None:
        raise UploadValidationError("Choose a CSV file to upload.")
    normalized = filename.replace("\\", "/")
    basename = normalized.rsplit("/", maxsplit=1)[-1].strip()
    if not basename or "\0" in basename:
        raise UploadValidationError("The selected file has an invalid filename.")
    return basename


def _format_upload_limit(size_bytes: int) -> str:
    mebibyte = 1024 * 1024
    if size_bytes < mebibyte:
        return f"{size_bytes} bytes"
    size_mebibytes = size_bytes / mebibyte
    return f"{size_mebibytes:g} MB"


def _path_free_prediction_message(error: Exception, temporary_path: Path) -> str:
    """Keep actionable core messages while removing the server-owned temporary path."""
    message = str(error)
    candidates = {str(temporary_path), str(temporary_path.resolve())}
    for candidate in candidates:
        message = message.replace(candidate, "the uploaded prediction CSV")
    return message


def _unlink_if_present(path: Path) -> None:
    with suppress(OSError):
        path.unlink(missing_ok=True)
