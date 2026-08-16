"""Profile the small CSV bundled with the MLForge source repository."""

from pathlib import Path

from mlforge.datasets import load_csv, profile_dataset


def main() -> None:
    """Load the example dataset and print its deterministic JSON profile."""
    dataset_path = Path(__file__).with_name("customer_churn.csv")
    dataset = load_csv(dataset_path, target="churn")
    print(profile_dataset(dataset).to_json())


if __name__ == "__main__":
    main()
