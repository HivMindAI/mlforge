"""Train and evaluate a local baseline while recording its complete run manifest."""

from pathlib import Path
from tempfile import TemporaryDirectory

from mlforge.datasets import load_csv
from mlforge.pipelines import TaskType
from mlforge.runs import LocalRunStore
from mlforge.training import LOGISTIC_REGRESSION, TrainingConfig, train


def main() -> None:
    """Run the bundled dataset through MLForge's complete in-memory training workflow."""
    dataset_path = Path(__file__).with_name("customer_churn.csv")
    dataset = load_csv(dataset_path, target="churn")
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        estimator=LOGISTIC_REGRESSION,
    )
    with TemporaryDirectory(prefix="mlforge-example-") as directory:
        result = train(dataset, config, run_store=LocalRunStore(Path(directory)))
        print(result.manifest.to_json())


if __name__ == "__main__":
    main()
