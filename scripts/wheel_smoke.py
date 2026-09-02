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
from mlforge.final_models import LocalFinalModelStore, fit_selected_model
from mlforge.inference import predict_frame, write_predictions_csv
from mlforge.pipelines import CrossValidationSplitConfig, TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import (
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_REGRESSOR,
    RIDGE_REGRESSION,
    TrainingConfig,
    train,
)

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

REGRESSION_CSV = """area,price
10,29.5
12,34.0
14,38.5
16,43.0
18,47.5
20,52.0
22,56.5
24,61.0
26,65.5
28,70.0
30,74.5
32,79.0
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
        regression_source = workspace / "regression.csv"
        regression_source.write_text(REGRESSION_CSV, encoding="utf-8", newline="\n")

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
        final_model_result = fit_selected_model(
            dataset,
            cross_validation_result,
            final_model_store=LocalFinalModelStore(workspace / "final-models"),
            artifact_store=LocalArtifactStore(workspace / "artifacts"),
        )
        assert final_model_result.artifact_path is not None
        final_inspected = inspect_artifact(final_model_result.artifact_path)
        final_loaded = load_artifact(final_model_result.artifact_path, trusted=True)
        final_predictions = predict_frame(
            final_loaded,
            pd.DataFrame(
                {
                    "age": [29, 57],
                    "monthly_spend": [62.0, 118.5],
                    "region": ["south", "north"],
                }
            ),
        )
        regression_dataset = load_csv(regression_source, target="price")
        regression_selection = cross_validate_benchmark(
            regression_dataset,
            CrossValidationConfig(
                task=TaskType.REGRESSION,
                estimators=(RIDGE_REGRESSION, RANDOM_FOREST_REGRESSOR),
                primary_metric="root_mean_squared_error",
                split=CrossValidationSplitConfig(fold_count=3, random_seed=42),
            ),
            store=LocalCrossValidationStore(workspace / "regression-cross-validation"),
        )
        regression_final = fit_selected_model(
            regression_dataset,
            regression_selection,
            final_model_store=LocalFinalModelStore(workspace / "regression-final-models"),
            artifact_store=LocalArtifactStore(workspace / "artifacts"),
        )
        assert regression_final.artifact_path is not None
        regression_loaded = load_artifact(regression_final.artifact_path, trusted=True)
        regression_predictions = predict_frame(
            regression_loaded,
            pd.DataFrame({"area": [34, 36]}),
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
        assert final_model_result.manifest.training_rows == len(dataset.frame)
        assert final_model_result.manifest.fit_scope == "all_rows"
        assert final_model_result.manifest.selection.benchmark_id == (
            cross_validation_result.manifest.benchmark_id
        )
        assert final_model_result.manifest.artifact is not None
        assert (
            final_model_result.manifest.artifact.pipeline_sha256 == final_inspected.pipeline_sha256
        )
        assert (
            final_model_result.manifest.artifact.pipeline_size_bytes
            == final_inspected.pipeline_size_bytes
        )
        assert final_inspected.model_id == final_model_result.manifest.final_model_id
        assert final_predictions.run_id == final_model_result.manifest.final_model_id
        assert final_predictions.row_count == 2
        assert regression_selection.manifest.configuration.task == "regression"
        assert regression_selection.manifest.winner is not None
        assert regression_final.manifest.configuration.task == "regression"
        assert regression_predictions.row_count == 2
        assert all(
            isinstance(record.prediction, float) for record in regression_predictions.predictions
        )
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
                    "final_model_id": final_model_result.manifest.final_model_id,
                    "prediction_output": str(prediction_output),
                    "predictions": predictions.to_dict(),
                    "regression_final_model_id": regression_final.manifest.final_model_id,
                    "version": __version__,
                },
                allow_nan=False,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
