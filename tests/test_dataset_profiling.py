"""Tests for deterministic dataset profiling."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mlforge.datasets import ColumnKind, TaskHint, load_csv, profile_dataset
from mlforge.errors import DatasetValidationError


def _write_csv(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8", newline="")
    return path


def test_profile_reports_quality_without_mutating_data(tmp_path: Path) -> None:
    """Profiling should measure missingness, duplicates, and constants without mutation."""
    path = _write_csv(
        tmp_path / "quality.csv",
        "id,value,constant,target\n1,1.5,x,yes\n2,,x,no\n2,,x,no\n",
    )
    dataset = load_csv(path, target="target")
    original = dataset.frame.copy(deep=True)

    profile = profile_dataset(dataset)

    pd.testing.assert_frame_equal(dataset.frame, original)
    assert profile.missing_cell_count == 2
    assert profile.missing_cell_ratio == pytest.approx(1 / 6)
    assert profile.duplicate_row_count == 1
    assert profile.target.task_hint is TaskHint.CLASSIFICATION
    assert profile.target.imbalance_warning is False
    constant_profile = next(column for column in profile.columns if column.name == "constant")
    assert constant_profile.is_constant is True
    assert any("Constant columns:" in warning for warning in profile.warnings)
    assert any("duplicate rows" in warning for warning in profile.warnings)


def test_profile_detects_identifiers_cardinality_and_imbalance(tmp_path: Path) -> None:
    """Documented heuristics should flag likely IDs, cardinality, and class imbalance."""
    rows = [
        f"{index},code-{index},{'minority' if index < 5 else 'majority'}" for index in range(60)
    ]
    path = _write_csv(
        tmp_path / "heuristics.csv",
        "record_id,external_code,target\n" + "\n".join(rows) + "\n",
    )

    profile = profile_dataset(load_csv(path, target="target"))
    profiles = {column.name: column for column in profile.columns}

    assert profiles["record_id"].kind is ColumnKind.INTEGER
    assert profiles["record_id"].is_likely_identifier is True
    assert profiles["external_code"].is_high_cardinality is True
    assert profile.target.imbalance_warning is True
    assert profile.target.class_distribution[0].value == "majority"
    assert profile.target.class_distribution[0].count == 55
    assert any("Likely identifier columns: record_id" in warning for warning in profile.warnings)
    assert any("High-cardinality columns: external_code" in warning for warning in profile.warnings)
    assert any("appears imbalanced" in warning for warning in profile.warnings)


def test_numeric_target_can_hint_regression(tmp_path: Path) -> None:
    """A sufficiently varied numeric target should be identified as regression-like."""
    rows = [f"{index},{index / 10}" for index in range(25)]
    path = _write_csv(
        tmp_path / "regression.csv",
        "feature,target\n" + "\n".join(rows) + "\n",
    )

    profile = profile_dataset(load_csv(path, target="target"))

    assert profile.target.task_hint is TaskHint.REGRESSION
    assert profile.target.class_distribution == ()
    assert profile.target.distribution_truncated is False


def test_mixed_value_column_is_profiled_as_string(tmp_path: Path) -> None:
    """Mixed lexical values should remain visible instead of being coerced to numbers."""
    path = _write_csv(tmp_path / "mixed.csv", "feature,target\n1,yes\nunknown,no\n")

    profile = profile_dataset(load_csv(path, target="target"))
    feature = next(column for column in profile.columns if column.name == "feature")

    assert feature.kind is ColumnKind.STRING


def test_numeric_summary_excludes_non_finite_values_and_json_is_standard(tmp_path: Path) -> None:
    """Infinity should be counted explicitly and never leak invalid JSON numbers."""
    path = _write_csv(
        tmp_path / "nonfinite.csv",
        "value,target\ninf,0\n-inf,1\n,0\n",
    )

    profile = profile_dataset(load_csv(path, target="target"))
    value_profile = next(column for column in profile.columns if column.name == "value")
    serialized = profile.to_json()

    assert value_profile.infinite_count == 2
    assert value_profile.numeric_summary is None
    assert any(
        "Columns containing infinite values: value" in warning for warning in profile.warnings
    )
    assert "Infinity" not in serialized
    assert "NaN" not in serialized
    assert json.loads(serialized)["target"]["task_hint"] == "classification"


def test_target_missingness_and_undetermined_task_are_reported(tmp_path: Path) -> None:
    """An unusable target should remain inspectable and emit explicit warnings."""
    path = _write_csv(tmp_path / "target.csv", "feature,target\n1,\n2,\n")

    profile = profile_dataset(load_csv(path, target="target"))

    assert profile.target.task_hint is TaskHint.UNDETERMINED
    assert profile.target.missing_count == 2
    assert any("contains missing values" in warning for warning in profile.warnings)
    assert any("could not be inferred" in warning for warning in profile.warnings)


def test_profile_rejects_a_mutated_loaded_dataframe(tmp_path: Path) -> None:
    """Metadata drift should fail clearly instead of producing a misleading profile."""
    path = _write_csv(tmp_path / "data.csv", "feature,target\n1,0\n2,1\n")
    dataset = load_csv(path, target="target")
    dataset.frame.drop(columns=["feature"], inplace=True)

    with pytest.raises(DatasetValidationError, match="no longer matches"):
        profile_dataset(dataset)


def test_profile_dictionary_is_deterministic(tmp_path: Path) -> None:
    """Repeated profiling should produce equal serialized domain values."""
    path = _write_csv(tmp_path / "data.csv", "feature,target\n1,a\n2,b\n")
    dataset = load_csv(path, target="target")

    assert profile_dataset(dataset).to_dict() == profile_dataset(dataset).to_dict()
