"""Exercise the installed MLForge wheel without relying on repository package sources."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from mlforge import __version__
from mlforge.artifacts import LocalArtifactStore, inspect_artifact, load_artifact
from mlforge.benchmarks import (
    BenchmarkConfig,
    CrossValidationConfig,
    LocalBenchmarkStore,
    LocalCrossValidationStore,
    benchmark,
    cross_validate_benchmark,
)
from mlforge.datasets import load_csv
from mlforge.inference import predict_frame, write_predictions_csv
from mlforge.pipelines import CrossValidationSplitConfig, TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig, train

TRAINING_CSV = """age,monthly_spend,region,churn
24,42.50,north,no
31,68.20,south,no
45,95.10,east,yes
28,51.00,west,no
52,110.75,north,yes
39,,south,no
34,73.40,east,no
61,125.00,west,yes
"""


def main() -> None:
    """Run one complete installed-package workflow in an isolated temporary directory."""
    assert __version__ == version("hivmind-mlforge")
    version_command = subprocess.run(
        [sys.executable, "-m", "mlforge", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert version_command.stdout.strip() == f"mlforge {__version__}"

    with TemporaryDirectory(prefix="mlforge-wheel-smoke-") as directory:
        workspace = Path(directory)
        source = workspace / "training.csv"
        source.write_text(TRAINING_CSV, encoding="utf-8", newline="\n")

        dataset = load_csv(source, target="churn")
        result = train(
            dataset,
            TrainingConfig(
                task=TaskType.CLASSIFICATION,
                estimator=LOGISTIC_REGRESSION,
            ),
            run_store=LocalRunStore(workspace / "runs"),
        )
        saved = LocalArtifactStore(workspace / "artifacts").save(result)
        inspected = inspect_artifact(saved.path)
        loaded = load_artifact(saved.path, trusted=True)
        predictions = predict_frame(
            loaded,
            pd.DataFrame(
                {
                    "age": [29, 57],
                    "monthly_spend": [62.0, 118.5],
                    "region": ["south", "north"],
                }
            ),
        )
        prediction_output = write_predictions_csv(predictions, workspace / "predictions.csv")
        benchmark_result = benchmark(
            dataset,
            BenchmarkConfig(),
            run_store=LocalRunStore(workspace / "benchmark-runs"),
            benchmark_store=LocalBenchmarkStore(workspace / "benchmarks"),
        )
        cross_validation_result = cross_validate_benchmark(
            dataset,
            CrossValidationConfig(
                split=CrossValidationSplitConfig(fold_count=3, random_seed=42),
            ),
            store=LocalCrossValidationStore(workspace / "cross-validation"),
        )

        assert inspected.run_id == result.manifest.run_id
        assert predictions.run_id == result.manifest.run_id
        assert predictions.row_count == 2
        assert list(pd.read_csv(prediction_output).columns) == ["row_number", "prediction"]
        assert benchmark_result.manifest.winner is not None
        assert benchmark_result.manifest_path.is_file()
        assert {entry.rank for entry in benchmark_result.manifest.entries} == {1, 2, 3}
        assert cross_validation_result.manifest.winner is not None
        assert cross_validation_result.manifest_path.is_file()
        assert len(cross_validation_result.manifest.folds) == 3
        assert {entry.rank for entry in cross_validation_result.manifest.entries} == {1, 2, 3}
        assert (
            LocalBenchmarkStore(workspace / "benchmarks").read(
                benchmark_result.manifest.benchmark_id
            )
            == benchmark_result.manifest
        )
        assert (
            LocalCrossValidationStore(workspace / "cross-validation").read(
                cross_validation_result.manifest.benchmark_id
            )
            == cross_validation_result.manifest
        )
        print(
            json.dumps(
                {
                    "artifact": str(saved.path),
                    "benchmark_winner": benchmark_result.manifest.winner.estimator,
                    "cross_validation_winner": (cross_validation_result.manifest.winner.estimator),
                    "prediction_output": str(prediction_output),
                    "predictions": predictions.to_dict(),
                    "version": __version__,
                },
                allow_nan=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
