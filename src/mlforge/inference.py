"""Schema-validated local batch prediction over explicitly trusted artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd

from mlforge.artifacts import FeatureRole, LoadedArtifact
from mlforge.datasets import CsvLoadOptions, load_feature_csv
from mlforge.errors import InferenceError, PredictionSchemaError

PredictionValue: TypeAlias = str | int | float | bool

__all__ = [
    "PredictionRecord",
    "PredictionResult",
    "PredictionValue",
    "predict_csv",
    "predict_frame",
]


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One JSON-safe prediction identified by stable one-based input row number."""

    row_number: int
    prediction: PredictionValue

    def __post_init__(self) -> None:
        if isinstance(self.row_number, bool) or not isinstance(self.row_number, int):
            raise InferenceError("Prediction row number must be an integer.")
        if self.row_number <= 0:
            raise InferenceError("Prediction row number must be positive.")
        if isinstance(self.prediction, float) and not math.isfinite(self.prediction):
            raise InferenceError("Prediction values must be finite.")
        if not isinstance(self.prediction, (str, int, float, bool)):
            raise InferenceError("Prediction values must be JSON scalar values.")

    def to_dict(self) -> dict[str, PredictionValue | int]:
        return {"row_number": self.row_number, "prediction": self.prediction}


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Structured batch result tied to one artifact and optional source CSV."""

    run_id: str
    task: str
    target: str
    source_path: str | None
    predictions: tuple[PredictionRecord, ...]

    @property
    def row_count(self) -> int:
        return len(self.predictions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "target": self.target,
            "source_path": self.source_path,
            "row_count": self.row_count,
            "predictions": [record.to_dict() for record in self.predictions],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent, sort_keys=True)


def _validate_columns(artifact: LoadedArtifact, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise PredictionSchemaError("Prediction input must be a pandas DataFrame.")
    if frame.empty:
        raise PredictionSchemaError("Prediction input must contain at least one row.")
    if not frame.columns.is_unique:
        raise PredictionSchemaError("Prediction feature names must be unique.")
    if any(not isinstance(name, str) or not name.strip() for name in frame.columns):
        raise PredictionSchemaError("Prediction feature names must be non-blank strings.")

    expected = tuple(feature.name for feature in artifact.manifest.features)
    actual = tuple(str(name) for name in frame.columns)
    missing = tuple(name for name in expected if name not in actual)
    extra = tuple(name for name in actual if name not in expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {list(missing)!r}")
        if extra:
            details.append(f"unexpected {list(extra)!r}")
        raise PredictionSchemaError(
            "Prediction columns do not match the artifact schema: " + "; ".join(details) + "."
        )
    ordered = frame.loc[:, list(expected)]
    return ordered.copy()


def _validate_values(artifact: LoadedArtifact, frame: pd.DataFrame) -> None:
    for feature in artifact.manifest.features:
        series = frame[feature.name]
        if feature.role is FeatureRole.NUMERIC:
            if (
                pd.api.types.is_bool_dtype(series.dtype)
                or not pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_complex_dtype(series.dtype)
            ):
                raise PredictionSchemaError(
                    f"Numeric feature {feature.name!r} has incompatible pandas dtype "
                    f"{series.dtype!s}; training dtype was {feature.pandas_dtype}."
                )
            if any(not math.isfinite(float(value)) for value in series.dropna().tolist()):
                raise PredictionSchemaError(
                    f"Numeric feature {feature.name!r} contains infinite values."
                )
            continue

        if pd.api.types.is_complex_dtype(series.dtype):
            raise PredictionSchemaError(
                f"Categorical feature {feature.name!r} must not contain complex numbers."
            )
        unsupported = [
            value for value in series.dropna().tolist() if not pd.api.types.is_scalar(value)
        ]
        if unsupported:
            raise PredictionSchemaError(
                f"Categorical feature {feature.name!r} contains non-scalar values."
            )
        fill_value = artifact.manifest.categorical_fill_value
        if bool(series.eq(fill_value).fillna(False).any()):
            raise PredictionSchemaError(
                f"Categorical feature {feature.name!r} contains the reserved missing-value "
                f"marker {fill_value!r}."
            )


def _prediction_value(value: object) -> PredictionValue:
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, bool):
        return scalar
    if isinstance(scalar, int):
        return scalar
    if isinstance(scalar, float):
        if not math.isfinite(scalar):
            raise InferenceError("Model produced a non-finite prediction.")
        return scalar
    if isinstance(scalar, str):
        return scalar
    raise InferenceError(
        f"Model produced an unsupported prediction value: {type(scalar).__name__}."
    )


def predict_frame(artifact: LoadedArtifact, frame: pd.DataFrame) -> PredictionResult:
    """Validate, reorder, and predict one in-memory feature batch."""
    if not isinstance(artifact, LoadedArtifact):
        raise InferenceError("artifact must be a LoadedArtifact from explicit trusted loading.")
    validated = _validate_columns(artifact, frame)
    _validate_values(artifact, validated)
    try:
        raw_predictions = artifact.pipeline.predict(validated)
    except (TypeError, ValueError, OverflowError) as error:
        raise InferenceError(
            f"Fitted pipeline could not predict the validated batch: {error}"
        ) from error
    if len(raw_predictions) != len(validated):
        raise InferenceError("Model prediction count does not match the input row count.")
    records = tuple(
        PredictionRecord(row_number=index, prediction=_prediction_value(value))
        for index, value in enumerate(raw_predictions, start=1)
    )
    return PredictionResult(
        run_id=artifact.manifest.run_id,
        task=artifact.manifest.task,
        target=artifact.manifest.target,
        source_path=None,
        predictions=records,
    )


def predict_csv(
    artifact: LoadedArtifact,
    path: str | PathLike[str],
    *,
    options: CsvLoadOptions | None = None,
) -> PredictionResult:
    """Load a strict local CSV and return schema-validated structured predictions."""
    frame = load_feature_csv(path, options=options)
    result = predict_frame(artifact, frame)
    try:
        source_path = str(Path(path).expanduser().resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise InferenceError(f"Could not resolve prediction source path: {path}") from error
    return PredictionResult(
        run_id=result.run_id,
        task=result.task,
        target=result.target,
        source_path=source_path,
        predictions=result.predictions,
    )
