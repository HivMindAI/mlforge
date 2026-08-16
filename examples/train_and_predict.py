"""Train, persist, explicitly trust, and batch-predict with one local artifact."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from mlforge.artifacts import LocalArtifactStore
from mlforge.datasets import load_csv
from mlforge.inference import predict_csv
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig, train


def main() -> None:
    """Exercise the complete supported local workflow without leaving generated files."""
    examples_directory = Path(__file__).resolve().parent
    dataset = load_csv(examples_directory / "customer_churn.csv", target="churn")
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        estimator=LOGISTIC_REGRESSION,
    )
    with TemporaryDirectory(prefix="mlforge-workflow-") as directory:
        workspace = Path(directory)
        trained = train(
            dataset,
            config,
            run_store=LocalRunStore(workspace / "runs"),
        )
        artifact_store = LocalArtifactStore(workspace / "artifacts")
        saved = artifact_store.save(trained)

        # Loading a pickle-based pipeline is executable. Only opt in for a verified source.
        trusted_artifact = artifact_store.load(trained.manifest.run_id, trusted=True)
        predictions = predict_csv(
            trusted_artifact,
            examples_directory / "prediction_customers.csv",
        )
        print(
            json.dumps(
                {
                    "artifact": saved.manifest.to_dict(),
                    "predictions": predictions.to_dict(),
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
