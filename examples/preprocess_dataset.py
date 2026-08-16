"""Split and preprocess the bundled dataset without leaking validation state."""

import json
from pathlib import Path

from mlforge.datasets import load_csv
from mlforge.pipelines import TaskType, build_preprocessor, split_dataset


def main() -> None:
    """Fit preprocessing on training rows and summarize the transformed partitions."""
    dataset_path = Path(__file__).with_name("customer_churn.csv")
    dataset = load_csv(dataset_path, target="churn")
    split = split_dataset(dataset, task=TaskType.CLASSIFICATION)
    preprocessor = build_preprocessor(split)

    transformed_train = preprocessor.fit_transform(split.train_features, split.train_target)
    transformed_validation = preprocessor.transform(split.validation_features)
    summary = {
        "stratified": split.stratified,
        "train_rows": transformed_train.shape[0],
        "validation_rows": transformed_validation.shape[0],
        "output_features": transformed_train.shape[1],
        "feature_names": preprocessor.get_feature_names_out().tolist(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
