"""Strict metadata and results for trusted local model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from sklearn.pipeline import Pipeline

from mlforge.errors import ArtifactFormatError

ARTIFACT_MANIFEST_SCHEMA_VERSION = 2
TRAINING_RUN_ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
ARTIFACT_SERIALIZATION_FORMAT = "pickle-protocol-5"


class FeatureRole(StrEnum):
    """Preprocessing role a prediction feature must satisfy."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class ArtifactLineageKind(StrEnum):
    """Immutable record type whose digest anchors an artifact."""

    TRAINING_RUN = "training-run"
    FINAL_MODEL = "final-model"


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ArtifactFormatError(f"Artifact manifest {label} must be a JSON object.")
    return cast(dict[str, object], value)


def _keys(value: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    details: list[str] = []
    if missing:
        details.append(f"missing {missing!r}")
    if extra:
        details.append(f"unexpected {extra!r}")
    raise ArtifactFormatError(
        f"Artifact manifest {label} has invalid fields: {', '.join(details)}."
    )


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactFormatError(f"Artifact manifest {label} must be a non-blank string.")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactFormatError(f"Artifact manifest {label} must be an integer.")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactFormatError(f"Artifact manifest {label} must be a lowercase SHA-256.")
    return digest


def _array(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ArtifactFormatError(f"Artifact manifest {label} must be a JSON array.")
    return cast(list[object], value)


@dataclass(frozen=True, slots=True)
class ArtifactEnvironment:
    """Exact runtime versions required before executable model loading."""

    python: str
    mlforge: str
    pandas: str
    numpy: str
    scipy: str
    scikit_learn: str

    def __post_init__(self) -> None:
        for name, value in (
            ("python", self.python),
            ("mlforge", self.mlforge),
            ("pandas", self.pandas),
            ("numpy", self.numpy),
            ("scipy", self.scipy),
            ("scikit_learn", self.scikit_learn),
        ):
            _string(value, f"environment {name}")

    def to_dict(self) -> dict[str, str]:
        return {
            "python": self.python,
            "mlforge": self.mlforge,
            "pandas": self.pandas,
            "numpy": self.numpy,
            "scipy": self.scipy,
            "scikit_learn": self.scikit_learn,
        }

    @classmethod
    def from_object(cls, value: object) -> ArtifactEnvironment:
        data = _object(value, "environment")
        expected = {"python", "mlforge", "pandas", "numpy", "scipy", "scikit_learn"}
        _keys(data, expected, "environment")
        return cls(
            python=_string(data["python"], "environment python"),
            mlforge=_string(data["mlforge"], "environment mlforge"),
            pandas=_string(data["pandas"], "environment pandas"),
            numpy=_string(data["numpy"], "environment numpy"),
            scipy=_string(data["scipy"], "environment scipy"),
            scikit_learn=_string(data["scikit_learn"], "environment scikit_learn"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactFeature:
    """One ordered raw input feature and its validation role."""

    name: str
    pandas_dtype: str
    role: FeatureRole

    def __post_init__(self) -> None:
        _string(self.name, "feature name")
        _string(self.pandas_dtype, f"feature {self.name!r} pandas_dtype")
        if not isinstance(self.role, FeatureRole):
            raise ArtifactFormatError("Artifact feature role must be a FeatureRole value.")

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "pandas_dtype": self.pandas_dtype,
            "role": self.role.value,
        }

    @classmethod
    def from_object(cls, value: object) -> ArtifactFeature:
        data = _object(value, "feature")
        _keys(data, {"name", "pandas_dtype", "role"}, "feature")
        raw_role = _string(data["role"], "feature role")
        try:
            role = FeatureRole(raw_role)
        except ValueError as error:
            raise ArtifactFormatError(
                f"Unsupported artifact feature role: {raw_role!r}."
            ) from error
        return cls(
            name=_string(data["name"], "feature name"),
            pandas_dtype=_string(data["pandas_dtype"], "feature pandas_dtype"),
            role=role,
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Safe-to-inspect metadata for one executable fitted-pipeline payload."""

    schema_version: int
    run_id: str
    created_at: str
    serialization_format: str
    pipeline_sha256: str
    pipeline_size_bytes: int
    run_manifest_sha256: str
    task: str
    target: str
    categorical_fill_value: str
    features: tuple[ArtifactFeature, ...]
    environment: ArtifactEnvironment
    lineage_kind: ArtifactLineageKind = ArtifactLineageKind.TRAINING_RUN

    def __post_init__(self) -> None:
        schema_version = _integer(self.schema_version, "schema_version")
        if schema_version not in {
            TRAINING_RUN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            ARTIFACT_MANIFEST_SCHEMA_VERSION,
        }:
            raise ArtifactFormatError(
                f"Unsupported artifact manifest schema version: {self.schema_version}."
            )
        if not isinstance(self.lineage_kind, ArtifactLineageKind):
            raise ArtifactFormatError("Artifact lineage_kind must be an ArtifactLineageKind value.")
        if (
            schema_version == TRAINING_RUN_ARTIFACT_MANIFEST_SCHEMA_VERSION
            and self.lineage_kind is not ArtifactLineageKind.TRAINING_RUN
        ):
            raise ArtifactFormatError("Version 1 artifacts can reference training runs only.")
        run_id = _string(self.run_id, "run_id")
        try:
            parsed_id = UUID(run_id)
        except (ValueError, AttributeError, TypeError) as error:
            raise ArtifactFormatError("Artifact run_id must be a UUID.") from error
        if str(parsed_id) != run_id:
            raise ArtifactFormatError("Artifact run_id must use canonical lowercase UUID form.")
        created_at = _string(self.created_at, "created_at")
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError as error:
            raise ArtifactFormatError("Artifact created_at must use ISO 8601 format.") from error
        if created.tzinfo is None:
            raise ArtifactFormatError("Artifact created_at must be timezone-aware.")
        if self.serialization_format != ARTIFACT_SERIALIZATION_FORMAT:
            raise ArtifactFormatError(
                f"Unsupported artifact serialization format: {self.serialization_format!r}."
            )
        _sha256(self.pipeline_sha256, "pipeline_sha256")
        _sha256(self.run_manifest_sha256, "run_manifest_sha256")
        if _integer(self.pipeline_size_bytes, "pipeline_size_bytes") <= 0:
            raise ArtifactFormatError("Artifact pipeline_size_bytes must be positive.")
        if self.task not in {"classification", "regression"}:
            raise ArtifactFormatError(f"Unsupported artifact task: {self.task!r}.")
        _string(self.target, "target")
        _string(self.categorical_fill_value, "categorical_fill_value")
        if not isinstance(self.features, tuple) or not self.features:
            raise ArtifactFormatError("Artifact features must be a non-empty immutable tuple.")
        if any(not isinstance(feature, ArtifactFeature) for feature in self.features):
            raise ArtifactFormatError("Artifact features must contain ArtifactFeature values.")
        names = tuple(feature.name for feature in self.features)
        if len(set(names)) != len(names):
            raise ArtifactFormatError("Artifact feature names must be unique.")
        if self.target in names:
            raise ArtifactFormatError("Artifact target must not appear in its input features.")
        if not isinstance(self.environment, ArtifactEnvironment):
            raise ArtifactFormatError("Artifact environment is invalid.")

    @property
    def model_id(self) -> str:
        """Return the generic model identity (the legacy name is ``run_id``)."""
        return self.run_id

    @property
    def lineage_manifest_sha256(self) -> str:
        """Return the generic lineage digest (legacy name: ``run_manifest_sha256``)."""
        return self.run_manifest_sha256

    def to_dict(self) -> dict[str, Any]:
        shared: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "serialization_format": self.serialization_format,
            "pipeline_sha256": self.pipeline_sha256,
            "pipeline_size_bytes": self.pipeline_size_bytes,
            "task": self.task,
            "target": self.target,
            "categorical_fill_value": self.categorical_fill_value,
            "features": [feature.to_dict() for feature in self.features],
            "environment": self.environment.to_dict(),
        }
        if self.schema_version == TRAINING_RUN_ARTIFACT_MANIFEST_SCHEMA_VERSION:
            return {
                **shared,
                "run_id": self.run_id,
                "run_manifest_sha256": self.run_manifest_sha256,
            }
        return {
            **shared,
            "model_id": self.model_id,
            "lineage_kind": self.lineage_kind.value,
            "lineage_manifest_sha256": self.lineage_manifest_sha256,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministic standards-compliant artifact metadata."""
        return json.dumps(self.to_dict(), allow_nan=False, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, content: str) -> ArtifactManifest:
        """Parse and fully validate an untrusted artifact manifest without loading code."""
        try:
            raw: object = json.loads(content)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ArtifactFormatError(f"Artifact manifest is not valid JSON: {error}") from error
        data = _object(raw, "root")
        schema_version = _integer(data.get("schema_version"), "schema_version")
        shared = {
            "schema_version",
            "created_at",
            "serialization_format",
            "pipeline_sha256",
            "pipeline_size_bytes",
            "task",
            "target",
            "categorical_fill_value",
            "features",
            "environment",
        }
        if schema_version == TRAINING_RUN_ARTIFACT_MANIFEST_SCHEMA_VERSION:
            expected = shared | {"run_id", "run_manifest_sha256"}
            run_id = _string(data.get("run_id"), "run_id")
            lineage_digest = _sha256(data.get("run_manifest_sha256"), "run_manifest_sha256")
            lineage_kind = ArtifactLineageKind.TRAINING_RUN
        elif schema_version == ARTIFACT_MANIFEST_SCHEMA_VERSION:
            expected = shared | {"model_id", "lineage_kind", "lineage_manifest_sha256"}
            run_id = _string(data.get("model_id"), "model_id")
            lineage_digest = _sha256(data.get("lineage_manifest_sha256"), "lineage_manifest_sha256")
            raw_kind = _string(data.get("lineage_kind"), "lineage_kind")
            try:
                lineage_kind = ArtifactLineageKind(raw_kind)
            except ValueError as error:
                raise ArtifactFormatError(
                    f"Unsupported artifact lineage kind: {raw_kind!r}."
                ) from error
        else:
            raise ArtifactFormatError(
                f"Unsupported artifact manifest schema version: {schema_version}."
            )
        _keys(data, expected, "root")
        return cls(
            schema_version=schema_version,
            run_id=run_id,
            created_at=_string(data["created_at"], "created_at"),
            serialization_format=_string(data["serialization_format"], "serialization_format"),
            pipeline_sha256=_sha256(data["pipeline_sha256"], "pipeline_sha256"),
            pipeline_size_bytes=_integer(data["pipeline_size_bytes"], "pipeline_size_bytes"),
            run_manifest_sha256=lineage_digest,
            task=_string(data["task"], "task"),
            target=_string(data["target"], "target"),
            categorical_fill_value=_string(
                data["categorical_fill_value"], "categorical_fill_value"
            ),
            features=tuple(
                ArtifactFeature.from_object(item) for item in _array(data["features"], "features")
            ),
            environment=ArtifactEnvironment.from_object(data["environment"]),
            lineage_kind=lineage_kind,
        )


@dataclass(frozen=True, slots=True)
class SavedArtifact:
    """A newly persisted artifact and its safe manifest."""

    path: Path
    manifest: ArtifactManifest


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """An explicitly trusted, integrity-checked fitted pipeline."""

    path: Path
    manifest: ArtifactManifest
    pipeline: Pipeline
