"""Select with cross-validation, refit every row, save, trust, and predict."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from mlforge.artifacts import LocalArtifactStore
from mlforge.benchmarks import (
    CrossValidationConfig,
    LocalCrossValidationStore,
    cross_validate_benchmark,
)
from mlforge.datasets import load_csv
from mlforge.final_models import LocalFinalModelStore, fit_selected_model
from mlforge.inference import predict_csv
from mlforge.pipelines import CrossValidationSplitConfig


def main() -> None:
    """Run the complete explicit model-selection-to-final-artifact workflow."""
    examples_directory = Path(__file__).resolve().parent
    dataset = load_csv(examples_directory / "customer_churn.csv", target="churn")
    with TemporaryDirectory(prefix="mlforge-final-model-") as directory:
        workspace = Path(directory)
        selection = cross_validate_benchmark(
            dataset,
            CrossValidationConfig(
                primary_metric="balanced_accuracy",
                split=CrossValidationSplitConfig(fold_count=3, random_seed=42),
            ),
            store=LocalCrossValidationStore(workspace / "cross-validation"),
        )
        artifacts = LocalArtifactStore(workspace / "artifacts")
        final_model = fit_selected_model(
            dataset,
            selection,
            final_model_store=LocalFinalModelStore(workspace / "final-models"),
            artifact_store=artifacts,
        )
        if final_model.artifact_path is None:
            raise RuntimeError("Final-model artifact was not persisted.")
        artifact_manifest = artifacts.inspect(final_model.manifest.final_model_id)

        # Pickle loading executes code; trust only artifacts whose source is established.
        loaded = artifacts.load(final_model.manifest.final_model_id, trusted=True)
        predictions = predict_csv(
            loaded,
            examples_directory / "prediction_customers.csv",
        )
        print(
            json.dumps(
                {
                    "selection": selection.manifest.to_dict(),
                    "final_model": final_model.manifest.to_dict(),
                    "artifact": artifact_manifest.to_dict(),
                    "predictions": predictions.to_dict(),
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
