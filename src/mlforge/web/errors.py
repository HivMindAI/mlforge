"""Expected failures owned by the MLForge web adapter."""


class WebError(Exception):
    """Base class for expected web-adapter failures."""


class UploadValidationError(WebError):
    """Raised when an uploaded file violates the HTTP upload contract."""


class DatasetNotFoundError(WebError):
    """Raised when a web dataset id has no stored record."""


class ExperimentNotFoundError(WebError):
    """Raised when a web experiment id has no stored record."""


class JobNotFoundError(WebError):
    """Raised when a web job id has no stored record."""


class ExperimentResultNotReadyError(WebError):
    """Raised when a configured experiment has no completed benchmark evidence."""


class FinalizationNotFoundError(WebError):
    """Raised when a finalization id has no stored record."""


class FinalizationNotReadyError(WebError):
    """Raised when an experiment has no finalizable rank-one result."""


class FinalModelNotFoundError(WebError):
    """Raised when a web final-model id has no completed local record."""


class PredictionInputValidationError(WebError):
    """Raised when an uploaded prediction CSV violates the model input contract."""


class InvalidModelArtifactError(WebError):
    """Raised when a finalized model artifact cannot be safely inspected or loaded."""


class PredictionExecutionError(WebError):
    """Raised when a trusted model cannot produce a valid prediction output."""


class PredictionNotFoundError(WebError):
    """Raised when a prediction id has no completed local record."""


class PredictionResultUnavailableError(WebError):
    """Raised when a saved prediction output is missing or invalid."""


class ExperimentValidationError(WebError):
    """Raised when a comparison configuration is unsupported or invalid."""


class WebStorageError(WebError):
    """Raised when local web metadata or upload storage is unavailable."""
