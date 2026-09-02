"""Behavioral tests for task-aware train/validation splitting."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from mlforge.datasets import load_csv
from mlforge.errors import ConfigurationError, DatasetSplitError, DatasetValidationError
from mlforge.pipelines import (
    CrossValidationSplitConfig,
    SplitConfig,
    TaskType,
    split_classification_folds,
    split_dataset,
    split_partition_sha256,
    split_regression_folds,
)


def _write_dataset(tmp_path: Path, rows: list[str], *, header: str = "value,target") -> Path:
    path = tmp_path / "training.csv"
    path.write_text("\n".join([header, *rows, ""]), encoding="utf-8")
    return path


def test_classification_split_is_deterministic_disjoint_and_stratified(tmp_path: Path) -> None:
    """A fixed seed should reproduce indices while preserving every row exactly once."""
    rows = [f"{index},{'yes' if index % 2 else 'no'}" for index in range(20)]
    dataset = load_csv(_write_dataset(tmp_path, rows), target="target")
    config = SplitConfig(validation_fraction=0.25, random_seed=7)

    first = split_dataset(dataset, task=TaskType.CLASSIFICATION, config=config)
    second = split_dataset(dataset, task=TaskType.CLASSIFICATION, config=config)

    assert first.train_features.index.tolist() == second.train_features.index.tolist()
    assert first.validation_features.index.tolist() == second.validation_features.index.tolist()
    assert set(first.train_features.index).isdisjoint(first.validation_features.index)
    assert set(first.train_features.index) | set(first.validation_features.index) == set(
        dataset.frame.index
    )
    assert "target" not in first.train_features
    assert first.train_target.index.equals(first.train_features.index)
    assert first.validation_target.index.equals(first.validation_features.index)
    assert first.stratified is True
    assert sorted(first.validation_target.value_counts().tolist()) == [2, 3]


def test_regression_defaults_to_an_unstratified_split(tmp_path: Path) -> None:
    """Regression should not treat continuous target values as strata."""
    rows = [f"{index},{index / 10}" for index in range(10)]
    dataset = load_csv(_write_dataset(tmp_path, rows), target="target")

    result = split_dataset(dataset, task=TaskType.REGRESSION)

    assert result.stratified is False
    assert len(result.train_features) == 8
    assert len(result.validation_features) == 2


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: SplitConfig(validation_fraction=0),
        lambda: SplitConfig(validation_fraction=1),
        lambda: SplitConfig(validation_fraction=True),
        lambda: SplitConfig(random_seed=True),
        lambda: SplitConfig(random_seed=-1),
        lambda: SplitConfig(random_seed=2**32),
        lambda: SplitConfig(stratify="yes"),  # type: ignore[arg-type]
    ],
)
def test_invalid_split_configuration_is_rejected(
    config_factory: Callable[[], SplitConfig],
) -> None:
    """Configuration errors should fail before a dataset operation begins."""
    with pytest.raises(ConfigurationError):
        config_factory()


def test_task_must_be_an_explicit_enum(tmp_path: Path) -> None:
    """A profile hint or arbitrary string must not silently choose training semantics."""
    dataset = load_csv(_write_dataset(tmp_path, ["1,no", "2,yes", "3,no"]), target="target")

    with pytest.raises(DatasetSplitError, match="TaskType"):
        split_dataset(dataset, task="classification")  # type: ignore[arg-type]


def test_missing_target_values_are_rejected(tmp_path: Path) -> None:
    """MLForge must not silently discard supervised examples with missing labels."""
    dataset = load_csv(_write_dataset(tmp_path, ["1,yes", "2,", "3,no"]), target="target")

    with pytest.raises(DatasetSplitError, match="missing values"):
        split_dataset(
            dataset,
            task=TaskType.CLASSIFICATION,
            config=SplitConfig(stratify=False),
        )


def test_classification_requires_two_classes(tmp_path: Path) -> None:
    """A constant target cannot define a classification problem."""
    dataset = load_csv(_write_dataset(tmp_path, ["1,yes", "2,yes", "3,yes"]), target="target")

    with pytest.raises(DatasetSplitError, match="at least two"):
        split_dataset(dataset, task=TaskType.CLASSIFICATION)


def test_stratification_requires_repeated_classes(tmp_path: Path) -> None:
    """A singleton class cannot be placed in both train and validation partitions."""
    dataset = load_csv(
        _write_dataset(tmp_path, ["1,common", "2,common", "3,common", "4,rare"]),
        target="target",
    )

    with pytest.raises(DatasetSplitError, match="at least two rows"):
        split_dataset(dataset, task=TaskType.CLASSIFICATION)


def test_stratification_requires_space_for_each_class(tmp_path: Path) -> None:
    """Both partitions must be large enough to contain every requested stratum."""
    dataset = load_csv(
        _write_dataset(
            tmp_path,
            ["1,a", "2,a", "3,b", "4,b", "5,c", "6,c"],
        ),
        target="target",
    )

    with pytest.raises(DatasetSplitError, match="too small"):
        split_dataset(
            dataset,
            task=TaskType.CLASSIFICATION,
            config=SplitConfig(validation_fraction=0.2),
        )


def test_tiny_unstratified_classification_rejects_one_class_training_split(
    tmp_path: Path,
) -> None:
    """Tiny inputs should fail explicitly when the training partition is unusable."""
    dataset = load_csv(_write_dataset(tmp_path, ["1,no", "2,yes"]), target="target")

    with pytest.raises(DatasetSplitError, match="training partition"):
        split_dataset(
            dataset,
            task=TaskType.CLASSIFICATION,
            config=SplitConfig(stratify=False),
        )


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (["1,low", "2,high", "3,medium"], "numeric"),
        (["1,1.5", "2,inf", "3,2.5"], "finite"),
    ],
)
def test_regression_target_must_be_numeric_and_finite(
    tmp_path: Path,
    rows: list[str],
    message: str,
) -> None:
    """Regression target validation should fail before an estimator sees invalid labels."""
    dataset = load_csv(_write_dataset(tmp_path, rows), target="target")

    with pytest.raises(DatasetSplitError, match=message):
        split_dataset(dataset, task=TaskType.REGRESSION)


def test_numeric_classification_target_must_be_finite(tmp_path: Path) -> None:
    """Infinite numeric labels should fail at the task boundary, not during estimator fitting."""
    dataset = load_csv(
        _write_dataset(tmp_path, ["1,0", "2,inf", "3,1", "4,0"]),
        target="target",
    )

    with pytest.raises(DatasetSplitError, match="finite"):
        split_dataset(dataset, task=TaskType.CLASSIFICATION)


def test_regression_cannot_request_target_stratification(tmp_path: Path) -> None:
    """Continuous-target stratification is not a supported implicit policy."""
    dataset = load_csv(
        _write_dataset(tmp_path, ["1,1.1", "2,2.2", "3,3.3", "4,4.4"]),
        target="target",
    )

    with pytest.raises(DatasetSplitError, match="only for classification"):
        split_dataset(
            dataset,
            task=TaskType.REGRESSION,
            config=SplitConfig(stratify=True),
        )


def test_supervised_split_requires_a_feature_column(tmp_path: Path) -> None:
    """A target-only table cannot train a useful supervised estimator."""
    dataset = load_csv(
        _write_dataset(tmp_path, ["no", "yes", "no"], header="target"),
        target="target",
    )

    with pytest.raises(DatasetSplitError, match="feature column"):
        split_dataset(
            dataset,
            task=TaskType.CLASSIFICATION,
            config=SplitConfig(stratify=False),
        )


def test_duplicate_target_shape_is_rejected_as_corrupt_loaded_data(tmp_path: Path) -> None:
    """A target must never resolve to a two-dimensional duplicate-column selection."""
    dataset = load_csv(_write_dataset(tmp_path, ["1,no", "2,yes"]), target="target")
    dataset.frame = pd.concat([dataset.frame[["target"]], dataset.frame[["target"]]], axis=1)
    dataset.frame.columns = ["target", "target"]

    with pytest.raises(DatasetValidationError, match="validated metadata"):
        split_dataset(dataset, task=TaskType.CLASSIFICATION)


def test_classification_folds_are_deterministic_disjoint_and_exhaustive(
    tmp_path: Path,
) -> None:
    """Every source row should validate once under a reproducible shared fold plan."""
    rows = [f"{index},{('a', 'b', 'c')[index % 3]}" for index in range(30)]
    dataset = load_csv(_write_dataset(tmp_path, rows), target="target")
    config = CrossValidationSplitConfig(fold_count=5, random_seed=17)

    first = split_classification_folds(dataset, config=config)
    second = split_classification_folds(dataset, config=config)

    assert len(first) == 5
    assert tuple(map(split_partition_sha256, first)) == tuple(map(split_partition_sha256, second))
    validation_indices: list[int] = []
    for split in first:
        assert set(split.train_features.index).isdisjoint(split.validation_features.index)
        assert set(split.train_features.index) | set(split.validation_features.index) == set(
            dataset.frame.index
        )
        assert split.train_target.index.equals(split.train_features.index)
        assert split.validation_target.index.equals(split.validation_features.index)
        assert "target" not in split.train_features
        assert split.stratified is True
        assert split.validation_target.value_counts().to_dict() == {"a": 2, "b": 2, "c": 2}
        validation_indices.extend(split.validation_features.index.tolist())
    assert sorted(validation_indices) == dataset.frame.index.tolist()


def test_regression_folds_are_deterministic_unstratified_and_exhaustive(
    tmp_path: Path,
) -> None:
    """Continuous targets should use reproducible K-fold partitions without stratification."""
    rows = [f"{index},{index * 1.75 + 4}" for index in range(30)]
    dataset = load_csv(_write_dataset(tmp_path, rows), target="target")
    config = CrossValidationSplitConfig(fold_count=5, random_seed=29)

    first = split_regression_folds(dataset, config=config)
    second = split_regression_folds(dataset, config=config)

    assert tuple(map(split_partition_sha256, first)) == tuple(map(split_partition_sha256, second))
    assert all(split.task is TaskType.REGRESSION for split in first)
    assert all(split.stratified is False for split in first)
    assert all(len(split.validation_features) >= 2 for split in first)
    validation_indices = [
        index for split in first for index in split.validation_features.index.tolist()
    ]
    assert sorted(validation_indices) == dataset.frame.index.tolist()


def test_regression_folds_require_two_validation_rows_per_fold(tmp_path: Path) -> None:
    """Regression folds must be large enough for finite R-squared evidence."""
    dataset = load_csv(
        _write_dataset(tmp_path, [f"{index},{index * 2.0}" for index in range(7)]),
        target="target",
    )

    with pytest.raises(DatasetSplitError, match="two validation rows"):
        split_regression_folds(
            dataset,
            config=CrossValidationSplitConfig(fold_count=4),
        )


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: CrossValidationSplitConfig(fold_count=1),
        lambda: CrossValidationSplitConfig(fold_count=11),
        lambda: CrossValidationSplitConfig(fold_count=True),
        lambda: CrossValidationSplitConfig(random_seed=True),
        lambda: CrossValidationSplitConfig(random_seed=-1),
        lambda: CrossValidationSplitConfig(random_seed=2**32),
    ],
)
def test_invalid_cross_validation_configuration_is_rejected(
    config_factory: Callable[[], CrossValidationSplitConfig],
) -> None:
    """Fold plans should reject ambiguous counts and invalid random seeds early."""
    with pytest.raises(ConfigurationError):
        config_factory()


def test_classification_folds_require_each_class_in_every_fold(tmp_path: Path) -> None:
    """The requested fold count cannot exceed the smallest class population."""
    dataset = load_csv(
        _write_dataset(
            tmp_path,
            ["1,a", "2,a", "3,a", "4,a", "5,b", "6,b", "7,b"],
        ),
        target="target",
    )

    with pytest.raises(DatasetSplitError, match="one row per fold"):
        split_classification_folds(
            dataset,
            config=CrossValidationSplitConfig(fold_count=4),
        )
