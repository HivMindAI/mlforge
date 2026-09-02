"""Typed HTTP request and response contracts for the web adapter."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from mlforge.artifacts import ArtifactEnvironment, ArtifactFeature, ArtifactManifest
from mlforge.benchmarks import (
    CrossValidationEntry,
    CrossValidationFoldResult,
    CrossValidationFoldSnapshot,
    CrossValidationManifest,
    CrossValidationMetricSummary,
)
from mlforge.datasets import (
    ColumnKind,
    ColumnProfile,
    DatasetProfile,
    NumericSummary,
    TargetProfile,
    TaskHint,
    ValueFrequency,
)
from mlforge.final_models import FinalModelManifest
from mlforge.runs import MetricValue, RunFailure
from mlforge.web.services import (
    ExperimentHistoryEntry,
    PredictionDetails,
    PredictionPreviewRow,
)
from mlforge.web.storage import (
    DatasetRecord,
    ExperimentRecord,
    FinalizationRecord,
    JobRecord,
    JobStatus,
    WebPredictionRecord,
)


class HealthResponse(BaseModel):
    """Minimal process health information safe for deployment probes."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    version: str


class DatasetTargetRequest(BaseModel):
    """An explicit target selection for one uploaded dataset."""

    model_config = ConfigDict(frozen=True)

    target: str

    @field_validator("target")
    @classmethod
    def target_must_not_be_blank(cls, value: str) -> str:
        """Reject blank values without modifying legitimate column whitespace."""
        if not value.strip():
            raise ValueError("Target column must not be blank.")
        return value


class DatasetResponse(BaseModel):
    """Path-free dataset metadata safe to return to the browser."""

    model_config = ConfigDict(frozen=True)

    dataset_id: UUID
    filename: str
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    target: str | None
    created_at: datetime

    @classmethod
    def from_record(cls, record: DatasetRecord) -> DatasetResponse:
        """Create a public DTO without leaking the server-side stored filename."""
        return cls(
            dataset_id=record.dataset_id,
            filename=record.original_filename,
            file_size_bytes=record.file_size_bytes,
            row_count=record.row_count,
            column_count=record.column_count,
            columns=record.columns,
            target=record.target,
            created_at=record.created_at,
        )


class NumericSummaryResponse(BaseModel):
    """JSON-safe finite numeric summary for one column."""

    model_config = ConfigDict(frozen=True)

    minimum: float
    maximum: float
    mean: float
    median: float
    standard_deviation: float | None

    @classmethod
    def from_profile(cls, summary: NumericSummary) -> NumericSummaryResponse:
        return cls(
            minimum=summary.minimum,
            maximum=summary.maximum,
            mean=summary.mean,
            median=summary.median,
            standard_deviation=summary.standard_deviation,
        )


class ColumnProfileResponse(BaseModel):
    """Browser-safe quality and cardinality information for one column."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: ColumnKind
    pandas_dtype: str
    non_missing_count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    infinite_count: int
    is_constant: bool
    is_high_cardinality: bool
    is_likely_identifier: bool
    numeric_summary: NumericSummaryResponse | None

    @classmethod
    def from_profile(cls, column: ColumnProfile) -> ColumnProfileResponse:
        return cls(
            name=column.name,
            kind=column.kind,
            pandas_dtype=column.pandas_dtype,
            non_missing_count=column.non_missing_count,
            missing_count=column.missing_count,
            missing_ratio=column.missing_ratio,
            unique_count=column.unique_count,
            unique_ratio=column.unique_ratio,
            infinite_count=column.infinite_count,
            is_constant=column.is_constant,
            is_high_cardinality=column.is_high_cardinality,
            is_likely_identifier=column.is_likely_identifier,
            numeric_summary=(
                NumericSummaryResponse.from_profile(column.numeric_summary)
                if column.numeric_summary is not None
                else None
            ),
        )


class ValueFrequencyResponse(BaseModel):
    """One display-safe target value and its observed frequency."""

    model_config = ConfigDict(frozen=True)

    value: str
    count: int
    ratio: float

    @classmethod
    def from_profile(cls, frequency: ValueFrequency) -> ValueFrequencyResponse:
        return cls(value=frequency.value, count=frequency.count, ratio=frequency.ratio)


class TargetProfileResponse(BaseModel):
    """Task and balance hints for the selected target column."""

    model_config = ConfigDict(frozen=True)

    name: str
    task_hint: TaskHint
    non_missing_count: int
    missing_count: int
    unique_count: int
    class_distribution: tuple[ValueFrequencyResponse, ...]
    distribution_truncated: bool
    imbalance_warning: bool

    @classmethod
    def from_profile(cls, target: TargetProfile) -> TargetProfileResponse:
        return cls(
            name=target.name,
            task_hint=target.task_hint,
            non_missing_count=target.non_missing_count,
            missing_count=target.missing_count,
            unique_count=target.unique_count,
            class_distribution=tuple(
                ValueFrequencyResponse.from_profile(frequency)
                for frequency in target.class_distribution
            ),
            distribution_truncated=target.distribution_truncated,
            imbalance_warning=target.imbalance_warning,
        )


class DatasetAnalysisResponse(BaseModel):
    """Complete path-free Data Overview response."""

    model_config = ConfigDict(frozen=True)

    dataset: DatasetResponse
    missing_cell_count: int
    missing_cell_ratio: float
    duplicate_row_count: int
    columns: tuple[ColumnProfileResponse, ...]
    target: TargetProfileResponse
    warnings: tuple[str, ...]

    @classmethod
    def from_profile(
        cls,
        record: DatasetRecord,
        profile: DatasetProfile,
    ) -> DatasetAnalysisResponse:
        """Adapt a core profile while omitting its source-path metadata."""
        return cls(
            dataset=DatasetResponse.from_record(record),
            missing_cell_count=profile.missing_cell_count,
            missing_cell_ratio=profile.missing_cell_ratio,
            duplicate_row_count=profile.duplicate_row_count,
            columns=tuple(ColumnProfileResponse.from_profile(column) for column in profile.columns),
            target=TargetProfileResponse.from_profile(profile.target),
            warnings=profile.warnings,
        )


class ExperimentCreateRequest(BaseModel):
    """Supported inputs for a supervised cross-validation comparison."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: UUID
    estimators: tuple[str, ...]
    fold_count: int


class ExperimentResponse(BaseModel):
    """Path-free persisted experiment configuration; no training has started."""

    model_config = ConfigDict(frozen=True)

    experiment_id: UUID
    dataset_id: UUID
    task: Literal["classification", "regression"]
    validation_strategy: Literal["cross-validation"]
    fold_count: int
    estimators: tuple[str, ...]
    primary_metric: str
    created_at: datetime

    @classmethod
    def from_record(cls, record: ExperimentRecord) -> ExperimentResponse:
        return cls(
            experiment_id=record.experiment_id,
            dataset_id=record.dataset_id,
            task=cast(Literal["classification", "regression"], record.task),
            validation_strategy="cross-validation",
            fold_count=record.fold_count,
            estimators=record.estimators,
            primary_metric=record.primary_metric,
            created_at=record.created_at,
        )


class ExperimentHistoryItemResponse(BaseModel):
    """Compact read-only experiment state with safe dataset display metadata."""

    model_config = ConfigDict(frozen=True)

    experiment_id: UUID
    dataset_id: UUID
    dataset_name: str
    task: Literal["classification", "regression"]
    status: Literal["configured", "waiting", "running", "complete", "failed"]
    model_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_entry(cls, entry: ExperimentHistoryEntry) -> ExperimentHistoryItemResponse:
        job = entry.job
        history_status: Literal["configured", "waiting", "running", "complete", "failed"]
        if job is None:
            history_status = "configured"
            updated_at = entry.experiment.created_at
        else:
            if job.status is JobStatus.WAITING:
                history_status = "waiting"
            elif job.status is JobStatus.RUNNING:
                history_status = "running"
            elif job.status is JobStatus.COMPLETE:
                history_status = "complete"
            else:
                history_status = "failed"
            updated_at = job.completed_at or job.started_at or job.created_at

        return cls(
            experiment_id=entry.experiment.experiment_id,
            dataset_id=entry.dataset.dataset_id,
            dataset_name=entry.dataset.original_filename,
            task=cast(Literal["classification", "regression"], entry.experiment.task),
            status=history_status,
            model_count=len(entry.experiment.estimators),
            created_at=entry.experiment.created_at,
            updated_at=updated_at,
        )


class ExperimentListResponse(BaseModel):
    """Newest-first experiment history for the local web workspace."""

    model_config = ConfigDict(frozen=True)

    experiments: tuple[ExperimentHistoryItemResponse, ...]
    count: int

    @classmethod
    def from_entries(cls, entries: tuple[ExperimentHistoryEntry, ...]) -> ExperimentListResponse:
        experiments = tuple(ExperimentHistoryItemResponse.from_entry(entry) for entry in entries)
        return cls(experiments=experiments, count=len(experiments))


class JobResponse(BaseModel):
    """Persisted job-level progress without fabricated per-model states."""

    model_config = ConfigDict(frozen=True)

    job_id: UUID
    experiment_id: UUID
    status: JobStatus
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    benchmark_id: UUID | None
    error_message: str | None

    @classmethod
    def from_record(cls, record: JobRecord) -> JobResponse:
        return cls(
            job_id=record.job_id,
            experiment_id=record.experiment_id,
            status=record.status,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            benchmark_id=record.benchmark_id,
            error_message=record.error_message,
        )


class FinalizationResponse(BaseModel):
    """Durable full-dataset fitting state for the rank-one experiment model."""

    model_config = ConfigDict(frozen=True)

    finalization_id: UUID
    experiment_id: UUID
    status: JobStatus
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    final_model_id: UUID | None
    error_message: str | None

    @classmethod
    def from_record(cls, record: FinalizationRecord) -> FinalizationResponse:
        return cls(
            finalization_id=record.finalization_id,
            experiment_id=record.experiment_id,
            status=record.status,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            final_model_id=record.final_model_id,
            error_message=record.error_message,
        )


class FinalModelFeatureResponse(BaseModel):
    """One safe ordered prediction input from the artifact manifest."""

    model_config = ConfigDict(frozen=True)

    name: str
    pandas_dtype: str
    role: Literal["numeric", "categorical"]

    @classmethod
    def from_feature(cls, feature: ArtifactFeature) -> FinalModelFeatureResponse:
        return cls(
            name=feature.name,
            pandas_dtype=feature.pandas_dtype,
            role=feature.role.value,
        )


class FinalModelEnvironmentResponse(BaseModel):
    """Recorded runtime versions required by the executable artifact."""

    model_config = ConfigDict(frozen=True)

    python: str
    mlforge: str
    pandas: str
    numpy: str
    scipy: str
    scikit_learn: str

    @classmethod
    def from_environment(
        cls,
        environment: ArtifactEnvironment,
    ) -> FinalModelEnvironmentResponse:
        return cls(
            python=environment.python,
            mlforge=environment.mlforge,
            pandas=environment.pandas,
            numpy=environment.numpy,
            scipy=environment.scipy,
            scikit_learn=environment.scikit_learn,
        )


class FinalModelArtifactResponse(BaseModel):
    """Safe artifact metadata; never the executable pickle payload."""

    model_config = ConfigDict(frozen=True)

    filename: str
    created_at: datetime
    serialization_format: str
    pipeline_size_bytes: int
    pipeline_sha256: str
    target: str
    features: tuple[FinalModelFeatureResponse, ...]
    environment: FinalModelEnvironmentResponse

    @classmethod
    def from_manifest(
        cls,
        filename: str,
        artifact: ArtifactManifest,
    ) -> FinalModelArtifactResponse:
        return cls(
            filename=filename,
            created_at=datetime.fromisoformat(artifact.created_at),
            serialization_format=artifact.serialization_format,
            pipeline_size_bytes=artifact.pipeline_size_bytes,
            pipeline_sha256=artifact.pipeline_sha256,
            target=artifact.target,
            features=tuple(
                FinalModelFeatureResponse.from_feature(feature) for feature in artifact.features
            ),
            environment=FinalModelEnvironmentResponse.from_environment(artifact.environment),
        )


class FinalModelMetricResponse(BaseModel):
    """One source cross-validation metric retained for the selected model."""

    model_config = ConfigDict(frozen=True)

    name: str
    mean: float
    standard_deviation: float

    @classmethod
    def from_summary(cls, metric: CrossValidationMetricSummary) -> FinalModelMetricResponse:
        return cls(
            name=metric.name,
            mean=metric.mean,
            standard_deviation=metric.standard_deviation,
        )


class FinalModelSummaryResponse(BaseModel):
    """Compact model-list evidence without artifact internals."""

    model_config = ConfigDict(frozen=True)

    final_model_id: UUID
    dataset_id: UUID
    dataset_name: str
    experiment_id: UUID
    estimator: str
    task: Literal["classification", "regression"]
    created_at: datetime
    primary_metric: str
    primary_metric_mean: float
    primary_metric_standard_deviation: float

    @classmethod
    def from_manifests(
        cls,
        dataset: DatasetRecord,
        experiment: ExperimentRecord,
        manifest: FinalModelManifest,
    ) -> FinalModelSummaryResponse:
        return cls(
            final_model_id=UUID(manifest.final_model_id),
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.original_filename,
            experiment_id=experiment.experiment_id,
            estimator=manifest.configuration.estimator,
            task=cast(Literal["classification", "regression"], manifest.configuration.task),
            created_at=datetime.fromisoformat(manifest.completed_at),
            primary_metric=manifest.selection.primary_metric,
            primary_metric_mean=manifest.selection.primary_metric_mean,
            primary_metric_standard_deviation=(
                manifest.selection.primary_metric_standard_deviation
            ),
        )


class FinalModelListResponse(BaseModel):
    """Newest-first collection of completed web-owned models."""

    model_config = ConfigDict(frozen=True)

    models: tuple[FinalModelSummaryResponse, ...]
    count: int


class FinalModelResponse(BaseModel):
    """Relevant verified final-model evidence without local paths or executable bytes."""

    model_config = ConfigDict(frozen=True)

    final_model_id: UUID
    dataset_id: UUID
    dataset_name: str
    experiment_id: UUID
    benchmark_id: UUID
    status: Literal["succeeded"]
    estimator: str
    task: Literal["classification", "regression"]
    created_at: datetime
    fit_scope: Literal["all_rows"]
    training_rows: int
    feature_count: int
    primary_metric: str
    primary_metric_mean: float
    primary_metric_standard_deviation: float
    metrics: tuple[FinalModelMetricResponse, ...]
    warnings: tuple[str, ...]
    artifact: FinalModelArtifactResponse

    @classmethod
    def from_manifests(
        cls,
        dataset: DatasetRecord,
        experiment: ExperimentRecord,
        manifest: FinalModelManifest,
        artifact: ArtifactManifest,
        artifact_filename: str,
        metrics: tuple[CrossValidationMetricSummary, ...],
    ) -> FinalModelResponse:
        return cls(
            final_model_id=UUID(manifest.final_model_id),
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.original_filename,
            experiment_id=experiment.experiment_id,
            benchmark_id=UUID(manifest.selection.benchmark_id),
            status="succeeded",
            estimator=manifest.configuration.estimator,
            task=cast(Literal["classification", "regression"], manifest.configuration.task),
            created_at=datetime.fromisoformat(manifest.completed_at),
            fit_scope="all_rows",
            training_rows=manifest.training_rows,
            feature_count=manifest.feature_count,
            primary_metric=manifest.selection.primary_metric,
            primary_metric_mean=manifest.selection.primary_metric_mean,
            primary_metric_standard_deviation=(
                manifest.selection.primary_metric_standard_deviation
            ),
            metrics=tuple(FinalModelMetricResponse.from_summary(metric) for metric in metrics),
            warnings=manifest.warnings,
            artifact=FinalModelArtifactResponse.from_manifest(artifact_filename, artifact),
        )


class PredictionCreatedResponse(BaseModel):
    """Minimal Phase 10 acknowledgement without preview or downloadable results."""

    model_config = ConfigDict(frozen=True)

    prediction_id: UUID
    final_model_id: UUID
    input_filename: str
    status: Literal["complete"]
    created_at: datetime
    completed_at: datetime

    @classmethod
    def from_record(cls, record: WebPredictionRecord) -> PredictionCreatedResponse:
        return cls(
            prediction_id=record.prediction_id,
            final_model_id=record.final_model_id,
            input_filename=record.original_filename,
            status="complete",
            created_at=record.created_at,
            completed_at=record.completed_at,
        )


class PredictionPreviewRowResponse(BaseModel):
    """One CSV-safe preview value from a completed batch."""

    model_config = ConfigDict(frozen=True)

    row_number: int
    prediction: str

    @classmethod
    def from_row(cls, row: PredictionPreviewRow) -> PredictionPreviewRowResponse:
        return cls(row_number=row.row_number, prediction=row.prediction)


class PredictionResponse(BaseModel):
    """Completed prediction summary with a bounded preview and no local paths."""

    model_config = ConfigDict(frozen=True)

    prediction_id: UUID
    final_model_id: UUID
    input_filename: str
    status: Literal["complete"]
    row_count: int
    invalid_row_count: Literal[0]
    preview_rows: tuple[PredictionPreviewRowResponse, ...]
    preview_limit: int
    preview_truncated: bool
    created_at: datetime
    completed_at: datetime

    @classmethod
    def from_details(cls, details: PredictionDetails) -> PredictionResponse:
        record = details.record
        return cls(
            prediction_id=record.prediction_id,
            final_model_id=record.final_model_id,
            input_filename=record.original_filename,
            status="complete",
            row_count=record.row_count,
            invalid_row_count=0,
            preview_rows=tuple(
                PredictionPreviewRowResponse.from_row(row) for row in details.preview
            ),
            preview_limit=details.preview_limit,
            preview_truncated=record.row_count > len(details.preview),
            created_at=record.created_at,
            completed_at=record.completed_at,
        )


class ResultFoldPlanResponse(BaseModel):
    """Useful row counts for one shared validation fold."""

    model_config = ConfigDict(frozen=True)

    fold_number: int
    train_rows: int
    validation_rows: int

    @classmethod
    def from_snapshot(cls, fold: CrossValidationFoldSnapshot) -> ResultFoldPlanResponse:
        return cls(
            fold_number=fold.fold_number,
            train_rows=fold.train_rows,
            validation_rows=fold.validation_rows,
        )


class ResultMetricValueResponse(BaseModel):
    """One observed metric value within an estimator fold."""

    model_config = ConfigDict(frozen=True)

    name: str
    value: float

    @classmethod
    def from_metric(cls, metric: MetricValue) -> ResultMetricValueResponse:
        return cls(name=metric.name, value=metric.value)


class ResultEstimatorFoldResponse(BaseModel):
    """One estimator's actual metrics and warnings for a completed fold."""

    model_config = ConfigDict(frozen=True)

    fold_number: int
    metrics: tuple[ResultMetricValueResponse, ...]
    duration_seconds: float
    warnings: tuple[str, ...]

    @classmethod
    def from_result(cls, fold: CrossValidationFoldResult) -> ResultEstimatorFoldResponse:
        return cls(
            fold_number=fold.fold_number,
            metrics=tuple(ResultMetricValueResponse.from_metric(metric) for metric in fold.metrics),
            duration_seconds=fold.duration_seconds,
            warnings=fold.warnings,
        )


class ResultMetricSummaryResponse(BaseModel):
    """Core-computed mean, population deviation, and observed fold values."""

    model_config = ConfigDict(frozen=True)

    name: str
    fold_values: tuple[float, ...]
    mean: float
    standard_deviation: float
    higher_is_better: bool

    @classmethod
    def from_summary(
        cls,
        summary: CrossValidationMetricSummary,
    ) -> ResultMetricSummaryResponse:
        return cls(
            name=summary.name,
            fold_values=summary.fold_values,
            mean=summary.mean,
            standard_deviation=summary.standard_deviation,
            higher_is_better=summary.higher_is_better,
        )


class ResultFailureResponse(BaseModel):
    """Safe estimator failure details without traceback or dataset values."""

    model_config = ConfigDict(frozen=True)

    error_type: str
    message: str

    @classmethod
    def from_failure(cls, failure: RunFailure) -> ResultFailureResponse:
        return cls(error_type=failure.error_type, message=failure.message)


class ResultEntryResponse(BaseModel):
    """One model's ranked cross-validation outcome."""

    model_config = ConfigDict(frozen=True)

    estimator: str
    status: Literal["succeeded", "failed"]
    rank: int | None
    primary_metric_mean: float | None
    primary_metric_standard_deviation: float | None
    metrics: tuple[ResultMetricSummaryResponse, ...]
    folds: tuple[ResultEstimatorFoldResponse, ...]
    duration_seconds: float
    failure_fold: int | None
    failure: ResultFailureResponse | None

    @classmethod
    def from_entry(cls, entry: CrossValidationEntry) -> ResultEntryResponse:
        return cls(
            estimator=entry.estimator,
            status=entry.status.value,
            rank=entry.rank,
            primary_metric_mean=entry.primary_metric_mean,
            primary_metric_standard_deviation=entry.primary_metric_standard_deviation,
            metrics=tuple(
                ResultMetricSummaryResponse.from_summary(metric) for metric in entry.metrics
            ),
            folds=tuple(ResultEstimatorFoldResponse.from_result(fold) for fold in entry.folds),
            duration_seconds=entry.duration_seconds,
            failure_fold=entry.failure_fold,
            failure=(
                ResultFailureResponse.from_failure(entry.failure)
                if entry.failure is not None
                else None
            ),
        )


class ExperimentResultResponse(BaseModel):
    """Path-free terminal evidence for a completed experiment comparison."""

    model_config = ConfigDict(frozen=True)

    experiment_id: UUID
    benchmark_id: UUID
    status: Literal["succeeded", "partial", "failed"]
    started_at: datetime
    completed_at: datetime
    task: Literal["classification", "regression"]
    target: str
    row_count: int
    column_count: int
    primary_metric: str
    fold_count: int
    warnings: tuple[str, ...]
    folds: tuple[ResultFoldPlanResponse, ...]
    entries: tuple[ResultEntryResponse, ...]

    @classmethod
    def from_manifest(
        cls,
        experiment_id: UUID,
        manifest: CrossValidationManifest,
    ) -> ExperimentResultResponse:
        return cls(
            experiment_id=experiment_id,
            benchmark_id=UUID(manifest.benchmark_id),
            status=manifest.status.value,
            started_at=datetime.fromisoformat(manifest.started_at),
            completed_at=datetime.fromisoformat(manifest.completed_at),
            task=cast(Literal["classification", "regression"], manifest.configuration.task),
            target=manifest.dataset.target,
            row_count=manifest.dataset.row_count,
            column_count=manifest.dataset.column_count,
            primary_metric=manifest.configuration.primary_metric,
            fold_count=manifest.configuration.fold_count,
            warnings=manifest.warnings,
            folds=tuple(ResultFoldPlanResponse.from_snapshot(fold) for fold in manifest.folds),
            entries=tuple(ResultEntryResponse.from_entry(entry) for entry in manifest.entries),
        )


class ErrorDetail(BaseModel):
    """Stable structured error payload for expected failures."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Top-level API error response."""

    error: ErrorDetail
