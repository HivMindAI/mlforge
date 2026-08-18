"""Domain exceptions raised by MLForge."""

__all__ = [
    "ArtifactCompatibilityError",
    "ArtifactError",
    "ArtifactFormatError",
    "ArtifactIntegrityError",
    "ArtifactPathError",
    "ArtifactTrustError",
    "BenchmarkError",
    "BenchmarkFailedError",
    "BenchmarkStoreError",
    "ConfigurationError",
    "DatasetError",
    "DatasetFormatError",
    "DatasetPathError",
    "DatasetSplitError",
    "DatasetValidationError",
    "InferenceError",
    "MLForgeError",
    "PipelineError",
    "PredictionSchemaError",
    "PreprocessingError",
    "RunComparisonError",
    "RunError",
    "RunStoreError",
    "TrainingError",
    "TrainingFailedError",
]


class MLForgeError(Exception):
    """Base class for expected MLForge failures."""


class ConfigurationError(MLForgeError):
    """Raised when application configuration is invalid."""


class DatasetError(MLForgeError):
    """Base class for expected dataset failures."""


class DatasetPathError(DatasetError):
    """Raised when a dataset path cannot be accessed safely."""


class DatasetFormatError(DatasetError):
    """Raised when dataset bytes are not a supported CSV representation."""


class DatasetValidationError(DatasetError):
    """Raised when a parsed dataset violates MLForge dataset requirements."""


class PipelineError(MLForgeError):
    """Base class for expected splitting and preprocessing failures."""


class DatasetSplitError(PipelineError):
    """Raised when a dataset cannot produce the requested supervised split."""


class PreprocessingError(PipelineError):
    """Raised when features cannot form a safe preprocessing pipeline."""


class TrainingError(MLForgeError):
    """Base class for expected estimator fitting and evaluation failures."""


class TrainingFailedError(TrainingError):
    """Raised after a failed training attempt has been recorded."""

    def __init__(self, message: str, *, run_id: str, manifest_path: str) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.manifest_path = manifest_path


class RunError(MLForgeError):
    """Base class for expected run-record failures."""


class RunStoreError(RunError):
    """Raised when a local run manifest cannot be stored or read safely."""


class RunComparisonError(RunError):
    """Raised when selected run manifests cannot be compared meaningfully."""


class BenchmarkError(MLForgeError):
    """Base class for expected local benchmark failures."""


class BenchmarkStoreError(BenchmarkError):
    """Raised when a local benchmark manifest cannot be stored or read safely."""


class BenchmarkFailedError(BenchmarkError):
    """Raised after every requested benchmark estimator failed and was recorded."""

    def __init__(self, message: str, *, benchmark_id: str, manifest_path: str) -> None:
        super().__init__(message)
        self.benchmark_id = benchmark_id
        self.manifest_path = manifest_path


class ArtifactError(MLForgeError):
    """Base class for expected model-artifact failures."""


class ArtifactPathError(ArtifactError):
    """Raised when an artifact path cannot be accessed safely."""


class ArtifactFormatError(ArtifactError):
    """Raised when an artifact archive or manifest has an unsupported structure."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when artifact bytes do not match their recorded digest or size."""


class ArtifactTrustError(ArtifactError):
    """Raised when executable artifact loading was not explicitly trusted."""


class ArtifactCompatibilityError(ArtifactError):
    """Raised before loading an artifact from an incompatible Python environment."""


class InferenceError(MLForgeError):
    """Base class for expected batch-inference failures."""


class PredictionSchemaError(InferenceError):
    """Raised when prediction inputs violate the artifact's feature contract."""
