"""FastAPI application construction and web error translation."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from mlforge import __version__
from mlforge.errors import DatasetError
from mlforge.web.api import (
    experiment_router,
    final_model_router,
    job_router,
    prediction_router,
    router,
)
from mlforge.web.errors import (
    DatasetNotFoundError,
    ExperimentNotFoundError,
    ExperimentResultNotReadyError,
    ExperimentValidationError,
    FinalizationNotFoundError,
    FinalizationNotReadyError,
    FinalModelNotFoundError,
    InvalidModelArtifactError,
    JobNotFoundError,
    PredictionExecutionError,
    PredictionInputValidationError,
    PredictionNotFoundError,
    PredictionResultUnavailableError,
    UploadValidationError,
    WebStorageError,
)
from mlforge.web.jobs import JobManager
from mlforge.web.services import (
    DatasetService,
    ExperimentResultService,
    ExperimentService,
    FinalModelService,
    PredictionService,
)
from mlforge.web.settings import WebSettings
from mlforge.web.storage import (
    DatasetStore,
    ExperimentStore,
    FinalizationStore,
    JobStore,
    PredictionStore,
)

_LOGGER = logging.getLogger("mlforge.web")


def _error_response(*, status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def create_app(settings: WebSettings | None = None) -> FastAPI:
    """Construct the single-process local API with injected settings for tests."""
    resolved_settings = settings if settings is not None else WebSettings.from_environment()
    store = DatasetStore(resolved_settings.workspace)
    store.initialize()
    experiment_store = ExperimentStore(resolved_settings.workspace)
    experiment_store.initialize()
    job_store = JobStore(resolved_settings.workspace)
    job_store.initialize()
    job_store.recover_interrupted(recovered_at=datetime.now(UTC))
    finalization_store = FinalizationStore(resolved_settings.workspace)
    finalization_store.initialize()
    finalization_store.recover_interrupted(recovered_at=datetime.now(UTC))
    job_manager = JobManager(
        store,
        experiment_store,
        job_store,
        finalization_store,
        resolved_settings,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            job_manager.shutdown()

    app = FastAPI(
        title="MLForge local API",
        description="Thin single-user web adapter over the MLForge Python core.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.dataset_service = DatasetService(store, resolved_settings)
    app.state.experiment_service = ExperimentService(
        store,
        experiment_store,
        job_store,
        resolved_settings,
    )
    experiment_result_service = ExperimentResultService(
        store,
        experiment_store,
        job_store,
        resolved_settings,
    )
    app.state.experiment_result_service = experiment_result_service
    app.state.final_model_service = FinalModelService(
        store,
        experiment_store,
        finalization_store,
        experiment_result_service,
        resolved_settings,
    )
    prediction_store = PredictionStore(resolved_settings.workspace)
    prediction_store.initialize()
    app.state.prediction_service = PredictionService(
        prediction_store,
        app.state.final_model_service,
        resolved_settings,
    )
    app.state.job_manager = job_manager
    app.include_router(router, prefix="/api")
    app.include_router(experiment_router, prefix="/api")
    app.include_router(job_router, prefix="/api")
    app.include_router(final_model_router, prefix="/api")
    app.include_router(prediction_router, prefix="/api")

    @app.exception_handler(UploadValidationError)
    async def handle_upload_validation(
        _request: Request,
        error: UploadValidationError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="invalid_upload",
            message=str(error),
        )

    @app.exception_handler(DatasetError)
    async def handle_dataset_error(
        _request: Request,
        error: DatasetError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="invalid_dataset",
            message=str(error),
        )

    @app.exception_handler(DatasetNotFoundError)
    async def handle_dataset_not_found(
        _request: Request,
        error: DatasetNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="dataset_not_found",
            message=str(error),
        )

    @app.exception_handler(ExperimentValidationError)
    async def handle_experiment_validation(
        _request: Request,
        error: ExperimentValidationError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="invalid_experiment",
            message=str(error),
        )

    @app.exception_handler(ExperimentNotFoundError)
    async def handle_experiment_not_found(
        _request: Request,
        error: ExperimentNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="experiment_not_found",
            message=str(error),
        )

    @app.exception_handler(ExperimentResultNotReadyError)
    async def handle_experiment_result_not_ready(
        _request: Request,
        error: ExperimentResultNotReadyError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="result_not_ready",
            message=str(error),
        )

    @app.exception_handler(FinalizationNotReadyError)
    async def handle_finalization_not_ready(
        _request: Request,
        error: FinalizationNotReadyError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_409_CONFLICT,
            code="finalization_not_ready",
            message=str(error),
        )

    @app.exception_handler(FinalizationNotFoundError)
    async def handle_finalization_not_found(
        _request: Request,
        error: FinalizationNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="finalization_not_found",
            message=str(error),
        )

    @app.exception_handler(FinalModelNotFoundError)
    async def handle_final_model_not_found(
        _request: Request,
        error: FinalModelNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="final_model_not_found",
            message=str(error),
        )

    @app.exception_handler(PredictionInputValidationError)
    async def handle_prediction_input_validation(
        _request: Request,
        error: PredictionInputValidationError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="invalid_prediction_input",
            message=str(error),
        )

    @app.exception_handler(InvalidModelArtifactError)
    async def handle_invalid_model_artifact(
        _request: Request,
        error: InvalidModelArtifactError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="invalid_model_artifact",
            message=str(error),
        )

    @app.exception_handler(PredictionExecutionError)
    async def handle_prediction_execution(
        _request: Request,
        error: PredictionExecutionError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="prediction_failed",
            message=str(error),
        )

    @app.exception_handler(PredictionNotFoundError)
    async def handle_prediction_not_found(
        _request: Request,
        error: PredictionNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="prediction_not_found",
            message=str(error),
        )

    @app.exception_handler(PredictionResultUnavailableError)
    async def handle_prediction_result_unavailable(
        _request: Request,
        error: PredictionResultUnavailableError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="prediction_result_unavailable",
            message=str(error),
        )

    @app.exception_handler(JobNotFoundError)
    async def handle_job_not_found(
        _request: Request,
        error: JobNotFoundError,
    ) -> JSONResponse:
        return _error_response(
            status_code=status.HTTP_404_NOT_FOUND,
            code="job_not_found",
            message=str(error),
        )

    @app.exception_handler(WebStorageError)
    async def handle_web_storage_error(
        _request: Request,
        error: WebStorageError,
    ) -> JSONResponse:
        _LOGGER.exception("MLForge web storage failed", exc_info=error)
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="storage_error",
            message="The local MLForge workspace is unavailable.",
        )

    return app
