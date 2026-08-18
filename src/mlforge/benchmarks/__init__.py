"""Local classification benchmark public API."""

from mlforge.benchmarks.cross_validation_service import cross_validate_benchmark
from mlforge.benchmarks.cross_validation_store import LocalCrossValidationStore
from mlforge.benchmarks.cross_validation_types import (
    CROSS_VALIDATION_MANIFEST_SCHEMA_VERSION,
    CrossValidationConfig,
    CrossValidationConfiguration,
    CrossValidationEntry,
    CrossValidationFoldResult,
    CrossValidationFoldSnapshot,
    CrossValidationManifest,
    CrossValidationMetricSummary,
    CrossValidationResult,
)
from mlforge.benchmarks.service import benchmark
from mlforge.benchmarks.store import LocalBenchmarkStore
from mlforge.benchmarks.types import (
    BENCHMARK_MANIFEST_SCHEMA_VERSION,
    DEFAULT_CLASSIFICATION_BENCHMARK_ESTIMATORS,
    BenchmarkConfig,
    BenchmarkConfiguration,
    BenchmarkEntry,
    BenchmarkManifest,
    BenchmarkResult,
    BenchmarkStatus,
)

__all__ = [
    "BENCHMARK_MANIFEST_SCHEMA_VERSION",
    "CROSS_VALIDATION_MANIFEST_SCHEMA_VERSION",
    "DEFAULT_CLASSIFICATION_BENCHMARK_ESTIMATORS",
    "BenchmarkConfig",
    "BenchmarkConfiguration",
    "BenchmarkEntry",
    "BenchmarkManifest",
    "BenchmarkResult",
    "BenchmarkStatus",
    "CrossValidationConfig",
    "CrossValidationConfiguration",
    "CrossValidationEntry",
    "CrossValidationFoldResult",
    "CrossValidationFoldSnapshot",
    "CrossValidationManifest",
    "CrossValidationMetricSummary",
    "CrossValidationResult",
    "LocalBenchmarkStore",
    "LocalCrossValidationStore",
    "benchmark",
    "cross_validate_benchmark",
]
