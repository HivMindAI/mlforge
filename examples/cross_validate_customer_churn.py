"""Compare local classification baselines with shared stratified folds."""

from pathlib import Path
from tempfile import TemporaryDirectory

from mlforge.benchmarks import (
    CrossValidationConfig,
    LocalCrossValidationStore,
    cross_validate_benchmark,
)
from mlforge.datasets import load_csv
from mlforge.pipelines import CrossValidationSplitConfig


def main() -> None:
    dataset = load_csv(Path(__file__).with_name("customer_churn.csv"), target="churn")
    with TemporaryDirectory(prefix="mlforge-cross-validation-") as directory:
        result = cross_validate_benchmark(
            dataset,
            CrossValidationConfig(
                primary_metric="balanced_accuracy",
                split=CrossValidationSplitConfig(fold_count=3, random_seed=42),
            ),
            store=LocalCrossValidationStore(Path(directory)),
        )
        print(result.manifest.to_json())


if __name__ == "__main__":
    main()
