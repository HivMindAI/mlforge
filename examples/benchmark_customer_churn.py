"""Run the default local classification benchmark on the bundled churn data."""

from pathlib import Path
from tempfile import TemporaryDirectory

from mlforge.benchmarks import BenchmarkConfig, LocalBenchmarkStore, benchmark
from mlforge.datasets import load_csv
from mlforge.runs import LocalRunStore


def main() -> None:
    dataset = load_csv(Path(__file__).with_name("customer_churn.csv"), target="churn")
    with TemporaryDirectory(prefix="mlforge-benchmark-") as directory:
        root = Path(directory)
        result = benchmark(
            dataset,
            BenchmarkConfig(primary_metric="balanced_accuracy"),
            run_store=LocalRunStore(root / "runs"),
            benchmark_store=LocalBenchmarkStore(root / "benchmarks"),
        )
        print(result.manifest.to_json())


if __name__ == "__main__":
    main()
