"""Integration and metric tests for local baseline training."""

import math
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from sklearn.utils.validation import check_is_fitted

from mlforge.datasets import LoadedDataset, load_csv
from mlforge.errors import ConfigurationError, TrainingError, TrainingFailedError
from mlforge.pipelines import SplitConfig, TaskType
from mlforge.runs import LocalRunStore, RunStatus
from mlforge.training import (
    LOGISTIC_REGRESSION,
    RANDOM_FOREST_CLASSIFIER,
    RANDOM_FOREST_REGRESSOR,
    RIDGE_REGRESSION,
    TrainingConfig,
    evaluate_predictions,
    train,
)


def _classification_dataset(tmp_path: Path) -> LoadedDataset:
    rows = ["row_id,age,region,target"]
    for index in range(40):
        target = "yes" if (index % 5 in {0, 1}) else "no"
        region = ("north", "south", "east")[index % 3]
        age = "" if index in {4, 17} else str(20 + index)
        rows.append(f"customer-{index},{age},{region},{target}")
    path = tmp_path / "classification.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_csv(path, target="target")


def _regression_dataset(tmp_path: Path) -> LoadedDataset:
    rows = ["row_id,value,group,target"]
    for index in range(40):
        group = "a" if index % 2 else "b"
        rows.append(f"{index},{index / 2},{group},{3 * index + 2}")
    path = tmp_path / "regression.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return load_csv(path, target="target")


@pytest.mark.parametrize(
    ("task", "estimator", "dataset_factory", "metric_names"),
    [
        (
            TaskType.CLASSIFICATION,
            LOGISTIC_REGRESSION,
            _classification_dataset,
            {"accuracy", "balanced_accuracy", "f1_weighted"},
        ),
        (
            TaskType.CLASSIFICATION,
            RANDOM_FOREST_CLASSIFIER,
            _classification_dataset,
            {"accuracy", "balanced_accuracy", "f1_weighted"},
        ),
        (
            TaskType.REGRESSION,
            RIDGE_REGRESSION,
            _regression_dataset,
            {"mean_absolute_error", "r2", "root_mean_squared_error"},
        ),
        (
            TaskType.REGRESSION,
            RANDOM_FOREST_REGRESSOR,
            _regression_dataset,
            {"mean_absolute_error", "r2", "root_mean_squared_error"},
        ),
    ],
)
def test_supported_estimators_train_evaluate_and_record(
    tmp_path: Path,
    task: TaskType,
    estimator: str,
    dataset_factory: Callable[[Path], LoadedDataset],
    metric_names: set[str],
) -> None:
    """Every supported baseline should complete the same public training contract."""
    dataset = dataset_factory(tmp_path)
    store = LocalRunStore(tmp_path / "runs")

    result = train(
        dataset,
        TrainingConfig(task=task, estimator=estimator),
        run_store=store,
    )

    assert result.manifest.status is RunStatus.SUCCEEDED
    assert {metric.name for metric in result.manifest.metrics} == metric_names
    assert result.manifest.dataset.sha256 == dataset.metadata.sha256
    assert result.manifest.dataset.encoding == dataset.metadata.encoding
    assert result.manifest.dataset.delimiter == dataset.metadata.delimiter
    assert result.manifest.configuration.estimator == estimator
    assert result.manifest.environment.scikit_learn
    assert result.manifest.split is not None
    assert result.manifest.split.train_rows + result.manifest.split.validation_rows == 40
    assert len(result.manifest.split.partition_sha256) == 64
    assert result.manifest_path.is_file()
    assert store.read(result.manifest.run_id) == result.manifest
    check_is_fitted(result.pipeline)


def test_fixed_seed_reproduces_metrics_and_predictions(tmp_path: Path) -> None:
    """Repeated same-data runs should vary in identity, not learned outputs."""
    dataset = _classification_dataset(tmp_path)
    config = TrainingConfig(
        task=TaskType.CLASSIFICATION,
        estimator=RANDOM_FOREST_CLASSIFIER,
        split=SplitConfig(random_seed=19),
    )
    store = LocalRunStore(tmp_path / "runs")

    first = train(dataset, config, run_store=store)
    second = train(dataset, config, run_store=store)

    assert first.manifest.run_id != second.manifest.run_id
    assert first.manifest.metrics == second.manifest.metrics
    assert first.manifest.configuration == second.manifest.configuration
    assert first.manifest.split is not None
    assert second.manifest.split is not None
    assert first.manifest.split.partition_sha256 == second.manifest.split.partition_sha256


def test_metric_values_match_known_classification_and_regression_examples() -> None:
    """Metric wrappers should retain scikit-learn's documented numeric definitions."""
    classification = evaluate_predictions(
        task=TaskType.CLASSIFICATION,
        actual=pd.Series(["a", "a", "b", "b"]),
        predicted=["a", "b", "b", "b"],
    )
    regression = evaluate_predictions(
        task=TaskType.REGRESSION,
        actual=pd.Series([1.0, 2.0, 3.0]),
        predicted=[1.0, 2.0, 5.0],
    )

    classification_values = {metric.name: metric.value for metric in classification}
    regression_values = {metric.name: metric.value for metric in regression}
    assert classification_values["accuracy"] == 0.75
    assert classification_values["balanced_accuracy"] == 0.75
    assert classification_values["f1_weighted"] == pytest.approx(0.7333333333)
    assert regression_values["mean_absolute_error"] == pytest.approx(2 / 3)
    assert regression_values["root_mean_squared_error"] == pytest.approx(math.sqrt(4 / 3))
    assert regression_values["r2"] == -1.0


def test_evaluation_rejects_shape_mismatch_and_tiny_regression_validation() -> None:
    """Metrics should not silently truncate rows or emit undefined R-squared values."""
    with pytest.raises(TrainingError, match="row counts"):
        evaluate_predictions(
            task=TaskType.CLASSIFICATION,
            actual=pd.Series(["yes", "no"]),
            predicted=["yes"],
        )
    with pytest.raises(TrainingError, match="at least two"):
        evaluate_predictions(
            task=TaskType.REGRESSION,
            actual=pd.Series([1.0]),
            predicted=[1.0],
        )


@pytest.mark.parametrize(
    ("task", "estimator"),
    [
        (TaskType.CLASSIFICATION, RIDGE_REGRESSION),
        (TaskType.REGRESSION, LOGISTIC_REGRESSION),
        (TaskType.CLASSIFICATION, "unknown"),
    ],
)
def test_unsupported_estimator_task_combinations_fail_early(
    task: TaskType,
    estimator: str,
) -> None:
    """A run cannot be configured with an incompatible or unknown estimator."""
    with pytest.raises(ConfigurationError):
        TrainingConfig(task=task, estimator=estimator)


def test_expected_training_failure_is_persisted_before_raising(tmp_path: Path) -> None:
    """A data/preprocessing failure should leave one terminal failed manifest, never a partial."""
    path = tmp_path / "invalid.csv"
    rows = [
        "value,target",
        "1,no",
        "2,yes",
        "inf,no",
        "4,yes",
        "5,no",
        "6,yes",
        "7,no",
        "8,yes",
        "9,no",
        "10,yes",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    dataset = load_csv(path, target="target")
    store = LocalRunStore(tmp_path / "runs")

    with pytest.raises(TrainingFailedError) as captured:
        train(
            dataset,
            TrainingConfig(task=TaskType.CLASSIFICATION, estimator=LOGISTIC_REGRESSION),
            run_store=store,
        )

    error = captured.value
    manifest = store.read(error.run_id)
    assert Path(error.manifest_path).is_file()
    assert manifest.status is RunStatus.FAILED
    assert manifest.failure is not None
    assert manifest.failure.error_type == "PreprocessingError"
    assert manifest.metrics == ()
    assert manifest.split is not None
    assert not list((tmp_path / "runs").glob("*.tmp"))
