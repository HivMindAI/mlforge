"""Bounded local execution for persisted MLForge comparison jobs."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Protocol
from uuid import UUID

from mlforge.artifacts import LocalArtifactStore
from mlforge.benchmarks import (
    CrossValidationConfig,
    CrossValidationResult,
    LocalCrossValidationStore,
    cross_validate_benchmark,
)
from mlforge.datasets import CsvLoadOptions, LoadedDataset, load_csv
from mlforge.errors import FinalModelFailedError, MLForgeError
from mlforge.final_models import FinalModelResult, LocalFinalModelStore, fit_selected_model
from mlforge.pipelines import CrossValidationSplitConfig
from mlforge.web.errors import FinalizationNotReadyError, WebError, WebStorageError
from mlforge.web.settings import WebSettings
from mlforge.web.storage import (
    DatasetStore,
    ExperimentStore,
    FinalizationRecord,
    FinalizationStore,
    JobRecord,
    JobStatus,
    JobStore,
)

_LOGGER = logging.getLogger("mlforge.web.jobs")


class BenchmarkRunner(Protocol):
    """Callable boundary used to test terminal job behavior deterministically."""

    def __call__(
        self,
        dataset: LoadedDataset,
        config: CrossValidationConfig,
        *,
        store: LocalCrossValidationStore,
    ) -> CrossValidationResult: ...


class FinalModelFitter(Protocol):
    """Callable boundary for explicit full-dataset fitting."""

    def __call__(
        self,
        dataset: LoadedDataset,
        selection: CrossValidationResult,
        *,
        final_model_store: LocalFinalModelStore,
        artifact_store: LocalArtifactStore,
    ) -> FinalModelResult: ...


class JobManager:
    """Execute comparison and final-fit work serially with durable state."""

    def __init__(
        self,
        dataset_store: DatasetStore,
        experiment_store: ExperimentStore,
        job_store: JobStore,
        finalization_store: FinalizationStore,
        settings: WebSettings,
        *,
        runner: BenchmarkRunner = cross_validate_benchmark,
        finalizer: FinalModelFitter = fit_selected_model,
    ) -> None:
        self.dataset_store = dataset_store
        self.experiment_store = experiment_store
        self.job_store = job_store
        self.finalization_store = finalization_store
        self.settings = settings
        self.runner = runner
        self.finalizer = finalizer
        self.benchmark_store = LocalCrossValidationStore(
            settings.workspace / "mlbenchmarks" / "cross-validation"
        )
        self.final_model_store = LocalFinalModelStore(settings.workspace / "mlfinalmodels")
        self.artifact_store = LocalArtifactStore(settings.workspace / "artifacts")
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlforge-job")

    def start(self, experiment_id: UUID) -> JobRecord:
        """Create an idempotent job and submit it to the one-worker executor."""
        self.experiment_store.get(experiment_id)
        job, created = self.job_store.create_or_get(
            experiment_id,
            created_at=datetime.now(UTC),
        )
        if not created:
            return job

        try:
            self.executor.submit(self._run, job.job_id)
        except RuntimeError as error:
            message = "The local comparison worker is unavailable."
            self.job_store.mark_failed(
                job.job_id,
                error_message=message,
                completed_at=datetime.now(UTC),
            )
            raise WebStorageError(message) from error
        return self.job_store.get(job.job_id)

    def get(self, job_id: UUID) -> JobRecord:
        """Return the latest persisted job state."""
        return self.job_store.get(job_id)

    def find_for_experiment(self, experiment_id: UUID) -> JobRecord | None:
        """Return an existing job while preserving experiment not-found semantics."""
        self.experiment_store.get(experiment_id)
        return self.job_store.find_for_experiment(experiment_id)

    def start_finalization(self, experiment_id: UUID) -> FinalizationRecord:
        """Idempotently submit the rank-one model for verified full-dataset fitting."""
        self.experiment_store.get(experiment_id)
        try:
            self._selection(experiment_id)
        except FinalizationNotReadyError:
            raise
        except MLForgeError as error:
            raise WebStorageError("The saved comparison result is unavailable.") from error
        finalization, created = self.finalization_store.create_or_get(
            experiment_id,
            created_at=datetime.now(UTC),
        )
        if not created:
            return finalization

        try:
            self.executor.submit(self._run_finalization, finalization.finalization_id)
        except RuntimeError as error:
            message = "The local final-model worker is unavailable."
            self.finalization_store.mark_failed(
                finalization.finalization_id,
                error_message=message,
                completed_at=datetime.now(UTC),
            )
            raise WebStorageError(message) from error
        return self.finalization_store.get(finalization.finalization_id)

    def find_finalization(self, experiment_id: UUID) -> FinalizationRecord | None:
        """Return the latest final-fit attempt while preserving experiment semantics."""
        self.experiment_store.get(experiment_id)
        return self.finalization_store.find_for_experiment(experiment_id)

    def get_finalization(self, finalization_id: UUID) -> FinalizationRecord:
        """Return one durable final-fit attempt."""
        return self.finalization_store.get(finalization_id)

    def shutdown(self) -> None:
        """Stop accepting work and cancel jobs that have not started."""
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job_id: UUID) -> None:
        try:
            job = self.job_store.mark_running(job_id, started_at=datetime.now(UTC))
            experiment = self.experiment_store.get(job.experiment_id)
            dataset_record = self.dataset_store.get(experiment.dataset_id)
            if dataset_record.target is None:
                raise WebStorageError(
                    "The experiment dataset no longer has a configured target column."
                )

            dataset = load_csv(
                self.dataset_store.path_for(dataset_record),
                target=dataset_record.target,
                options=CsvLoadOptions(max_file_size_bytes=self.settings.max_upload_bytes),
            )
            config = CrossValidationConfig(
                estimators=experiment.estimators,
                primary_metric=experiment.primary_metric,
                split=CrossValidationSplitConfig(fold_count=experiment.fold_count),
            )
            result = self.runner(dataset, config, store=self.benchmark_store)
            self.job_store.mark_complete(
                job_id,
                benchmark_id=UUID(result.manifest.benchmark_id),
                completed_at=datetime.now(UTC),
            )
        except (MLForgeError, WebError, ValueError, TypeError, OverflowError) as error:
            self._record_failure(job_id, _failure_message(error))
        except Exception as error:  # pragma: no cover - defensive worker boundary
            _LOGGER.exception("Unexpected comparison job failure", exc_info=error)
            self._record_failure(
                job_id,
                "Comparison failed because of an unexpected internal error. Check the API logs.",
            )

    def _run_finalization(self, finalization_id: UUID) -> None:
        try:
            finalization = self.finalization_store.mark_running(
                finalization_id,
                started_at=datetime.now(UTC),
            )
            experiment = self.experiment_store.get(finalization.experiment_id)
            dataset_record = self.dataset_store.get(experiment.dataset_id)
            if dataset_record.target is None:
                raise WebStorageError(
                    "The experiment dataset no longer has a configured target column."
                )
            dataset = load_csv(
                self.dataset_store.path_for(dataset_record),
                target=dataset_record.target,
                options=CsvLoadOptions(max_file_size_bytes=self.settings.max_upload_bytes),
            )
            result = self.finalizer(
                dataset,
                self._selection(finalization.experiment_id),
                final_model_store=self.final_model_store,
                artifact_store=self.artifact_store,
            )
            self.finalization_store.mark_complete(
                finalization_id,
                final_model_id=UUID(result.manifest.final_model_id),
                completed_at=datetime.now(UTC),
            )
        except FinalModelFailedError as error:
            self._record_finalization_failure(
                finalization_id,
                _failure_message(error),
                final_model_id=UUID(error.final_model_id),
            )
        except (MLForgeError, WebError, ValueError, TypeError, OverflowError) as error:
            self._record_finalization_failure(finalization_id, _failure_message(error))
        except Exception as error:  # pragma: no cover - defensive worker boundary
            _LOGGER.exception("Unexpected finalization job failure", exc_info=error)
            self._record_finalization_failure(
                finalization_id,
                "Finalization failed because of an unexpected internal error. Check the API logs.",
            )

    def _selection(self, experiment_id: UUID) -> CrossValidationResult:
        comparison = self.job_store.find_for_experiment(experiment_id)
        if comparison is None:
            raise FinalizationNotReadyError("Run this experiment before finalizing its best model.")
        if comparison.status is not JobStatus.COMPLETE or comparison.benchmark_id is None:
            raise FinalizationNotReadyError(
                "Finalize is available after the model comparison completes successfully."
            )
        manifest = self.benchmark_store.read(str(comparison.benchmark_id))
        if manifest.winner is None:
            raise FinalizationNotReadyError(
                "This comparison does not contain a successful rank-one model."
            )
        return CrossValidationResult(
            manifest=manifest,
            manifest_path=self.benchmark_store.manifest_path(manifest.benchmark_id),
        )

    def _record_failure(self, job_id: UUID, message: str) -> None:
        try:
            self.job_store.mark_failed(
                job_id,
                error_message=message,
                completed_at=datetime.now(UTC),
            )
        except WebStorageError as storage_error:  # pragma: no cover - storage outage boundary
            _LOGGER.exception("Could not persist comparison job failure", exc_info=storage_error)

    def _record_finalization_failure(
        self,
        finalization_id: UUID,
        message: str,
        *,
        final_model_id: UUID | None = None,
    ) -> None:
        try:
            self.finalization_store.mark_failed(
                finalization_id,
                error_message=message,
                completed_at=datetime.now(UTC),
                final_model_id=final_model_id,
            )
        except WebStorageError as storage_error:  # pragma: no cover - storage outage boundary
            _LOGGER.exception("Could not persist finalization failure", exc_info=storage_error)


def _failure_message(error: Exception) -> str:
    compact = " ".join(str(error).split())
    return compact[:2_000] if compact else "Comparison failed without an error message."
