"""Atomic local persistence with a hard trust boundary around pipeline deserialization."""

from __future__ import annotations

import hashlib
import os
import pickle
import stat
import sys
import warnings
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from sklearn.exceptions import InconsistentVersionWarning  # type: ignore[attr-defined]
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from mlforge.artifacts.types import (
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ARTIFACT_SERIALIZATION_FORMAT,
    TRAINING_RUN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
    ArtifactEnvironment,
    ArtifactFeature,
    ArtifactLineageKind,
    ArtifactManifest,
    FeatureRole,
    LoadedArtifact,
    SavedArtifact,
)
from mlforge.errors import (
    ArtifactCompatibilityError,
    ArtifactFormatError,
    ArtifactIntegrityError,
    ArtifactPathError,
    ArtifactTrustError,
    FinalModelStoreError,
    RunStoreError,
)
from mlforge.runs import RunManifest, RunStatus
from mlforge.training import TrainingResult

if TYPE_CHECKING:
    from mlforge.final_models.types import FinalModelManifest, FinalModelResult

ARTIFACT_SUFFIX = ".mlforge"
MANIFEST_MEMBER = "manifest.json"
PIPELINE_MEMBER = "pipeline.pkl"
MAX_MANIFEST_SIZE_BYTES = 1024 * 1024
MAX_PIPELINE_SIZE_BYTES = 1024 * 1024 * 1024
MAX_ARTIFACT_SIZE_BYTES = MAX_MANIFEST_SIZE_BYTES + MAX_PIPELINE_SIZE_BYTES + 64 * 1024
_HASH_CHUNK_SIZE = 1024 * 1024


def _canonical_run_id(run_id: str) -> str:
    try:
        parsed = UUID(run_id)
    except (ValueError, AttributeError, TypeError) as error:
        raise ArtifactPathError("Artifact run ID must be a canonical UUID.") from error
    canonical = str(parsed)
    if run_id != canonical:
        raise ArtifactPathError("Artifact run ID must be a canonical lowercase UUID.")
    return canonical


def _current_environment() -> ArtifactEnvironment:
    return ArtifactEnvironment(
        python=sys.version.split()[0],
        mlforge=version("hivmind-mlforge"),
        pandas=version("pandas"),
        numpy=version("numpy"),
        scipy=version("scipy"),
        scikit_learn=version("scikit-learn"),
    )


def _safe_artifact_path(path: Path) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ArtifactPathError(f"Artifact path must not be a symbolic link: {candidate}")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ArtifactPathError(
            f"Artifact does not exist or cannot be resolved: {candidate}"
        ) from error
    if not resolved.is_file():
        raise ArtifactPathError(f"Artifact path is not a regular file: {resolved}")
    if resolved.suffix.lower() != ARTIFACT_SUFFIX:
        raise ArtifactPathError(f"Artifact must use the {ARTIFACT_SUFFIX} extension: {resolved}")
    try:
        size = resolved.stat().st_size
    except OSError as error:
        raise ArtifactPathError(f"Could not inspect artifact file: {resolved}") from error
    if size <= 0:
        raise ArtifactFormatError(f"Artifact file is empty: {resolved}")
    if size > MAX_ARTIFACT_SIZE_BYTES:
        raise ArtifactPathError(
            f"Artifact exceeds the {MAX_ARTIFACT_SIZE_BYTES}-byte size limit: {resolved}"
        )
    return resolved


def _validate_members(archive: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, zipfile.ZipInfo]:
    infos = archive.infolist()
    if [item.filename for item in infos] != [MANIFEST_MEMBER, PIPELINE_MEMBER]:
        raise ArtifactFormatError(
            f"Artifact archive must contain exactly {MANIFEST_MEMBER!r} and {PIPELINE_MEMBER!r}."
        )
    if any(
        item.is_dir() or stat.S_ISLNK(item.external_attr >> 16) or item.flag_bits & 0x1
        for item in infos
    ):
        raise ArtifactFormatError("Artifact archive members must be unencrypted regular files.")
    if any(item.compress_type != zipfile.ZIP_STORED for item in infos):
        raise ArtifactFormatError("Artifact archive members must use the stored ZIP format.")
    manifest_info, pipeline_info = infos
    if manifest_info.file_size > MAX_MANIFEST_SIZE_BYTES:
        raise ArtifactFormatError("Artifact manifest exceeds its size limit.")
    if pipeline_info.file_size <= 0 or pipeline_info.file_size > MAX_PIPELINE_SIZE_BYTES:
        raise ArtifactFormatError("Artifact pipeline payload has an invalid size.")
    return manifest_info, pipeline_info


def _read_archive(path: Path, *, include_pipeline: bool) -> tuple[ArtifactManifest, bytes | None]:
    resolved = _safe_artifact_path(path)
    try:
        with zipfile.ZipFile(resolved, mode="r") as archive:
            manifest_info, pipeline_info = _validate_members(archive)
            manifest_bytes = archive.read(manifest_info)
            try:
                manifest_text = manifest_bytes.decode("utf-8", errors="strict")
            except UnicodeError as error:
                raise ArtifactFormatError("Artifact manifest must be valid UTF-8 text.") from error
            manifest = ArtifactManifest.from_json(manifest_text)
            if manifest.model_id != resolved.stem:
                raise ArtifactFormatError("Artifact model ID does not match its filename.")
            if manifest.pipeline_size_bytes != pipeline_info.file_size:
                raise ArtifactIntegrityError("Artifact pipeline size does not match its manifest.")

            digest = hashlib.sha256()
            payload = bytearray() if include_pipeline else None
            with archive.open(pipeline_info, mode="r") as stream:
                for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                    digest.update(chunk)
                    if payload is not None:
                        payload.extend(chunk)
            if digest.hexdigest() != manifest.pipeline_sha256:
                raise ArtifactIntegrityError(
                    "Artifact pipeline checksum does not match its manifest."
                )
    except ArtifactFormatError:
        raise
    except ArtifactIntegrityError:
        raise
    except (OSError, EOFError, zipfile.BadZipFile, RuntimeError) as error:
        raise ArtifactFormatError(f"Could not read artifact archive: {resolved}") from error
    return manifest, bytes(payload) if payload is not None else None


def _features(result: TrainingResult | FinalModelResult) -> tuple[ArtifactFeature, ...]:
    numeric = set(result.feature_schema.numeric_features)
    categorical = set(result.feature_schema.categorical_features)
    return tuple(
        ArtifactFeature(
            name=name,
            pandas_dtype=pandas_dtype,
            role=FeatureRole.NUMERIC if name in numeric else FeatureRole.CATEGORICAL,
        )
        for name, pandas_dtype in result.feature_dtypes
        if name in numeric or name in categorical
    )


def _manifest(result: TrainingResult, pipeline_bytes: bytes) -> ArtifactManifest:
    environment = result.manifest.environment
    return ArtifactManifest(
        schema_version=TRAINING_RUN_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        run_id=result.manifest.run_id,
        created_at=datetime.now(UTC).isoformat(timespec="microseconds"),
        serialization_format=ARTIFACT_SERIALIZATION_FORMAT,
        pipeline_sha256=hashlib.sha256(pipeline_bytes).hexdigest(),
        pipeline_size_bytes=len(pipeline_bytes),
        run_manifest_sha256=hashlib.sha256(
            result.manifest.to_json(indent=None).encode("utf-8")
        ).hexdigest(),
        task=result.manifest.configuration.task,
        target=result.manifest.dataset.target,
        categorical_fill_value=result.manifest.configuration.categorical_fill_value,
        features=_features(result),
        environment=ArtifactEnvironment(
            python=environment.python,
            mlforge=environment.mlforge,
            pandas=environment.pandas,
            numpy=environment.numpy,
            scipy=environment.scipy,
            scikit_learn=environment.scikit_learn,
        ),
        lineage_kind=ArtifactLineageKind.TRAINING_RUN,
    )


def _final_model_manifest(
    result: FinalModelResult,
    pipeline_bytes: bytes,
) -> ArtifactManifest:
    environment = result.manifest.environment
    artifact_contract = result.manifest.artifact
    if artifact_contract is None:
        raise ArtifactIntegrityError("Final-model manifest is missing its artifact contract.")
    pipeline_sha256 = hashlib.sha256(pipeline_bytes).hexdigest()
    if (
        artifact_contract.artifact_id != result.manifest.final_model_id
        or artifact_contract.serialization_format != ARTIFACT_SERIALIZATION_FORMAT
        or artifact_contract.pipeline_sha256 != pipeline_sha256
        or artifact_contract.pipeline_size_bytes != len(pipeline_bytes)
    ):
        raise ArtifactIntegrityError(
            "Serialized final pipeline does not match its immutable artifact contract."
        )
    return ArtifactManifest(
        schema_version=ARTIFACT_MANIFEST_SCHEMA_VERSION,
        run_id=result.manifest.final_model_id,
        created_at=datetime.now(UTC).isoformat(timespec="microseconds"),
        serialization_format=ARTIFACT_SERIALIZATION_FORMAT,
        pipeline_sha256=pipeline_sha256,
        pipeline_size_bytes=len(pipeline_bytes),
        run_manifest_sha256=hashlib.sha256(
            result.manifest.to_json(indent=None).encode("utf-8")
        ).hexdigest(),
        task=result.manifest.configuration.task,
        target=result.manifest.dataset.target,
        categorical_fill_value=result.manifest.configuration.categorical_fill_value,
        features=_features(result),
        environment=ArtifactEnvironment(
            python=environment.python,
            mlforge=environment.mlforge,
            pandas=environment.pandas,
            numpy=environment.numpy,
            scipy=environment.scipy,
            scikit_learn=environment.scikit_learn,
        ),
        lineage_kind=ArtifactLineageKind.FINAL_MODEL,
    )


def verify_run_manifest(artifact: ArtifactManifest, run: RunManifest) -> None:
    """Verify that a safe artifact manifest references the exact canonical run record."""
    if not isinstance(artifact, ArtifactManifest) or not isinstance(run, RunManifest):
        raise ArtifactIntegrityError(
            "Artifact and run manifests must be validated manifest values."
        )
    if artifact.lineage_kind is not ArtifactLineageKind.TRAINING_RUN:
        raise ArtifactIntegrityError("Artifact does not reference a training run manifest.")
    if artifact.run_id != run.run_id:
        raise ArtifactIntegrityError("Artifact and run manifest identifiers do not match.")
    digest = hashlib.sha256(run.to_json(indent=None).encode("utf-8")).hexdigest()
    if artifact.run_manifest_sha256 != digest:
        raise ArtifactIntegrityError("Artifact does not reference the supplied run manifest.")


def verify_final_model_manifest(
    artifact: ArtifactManifest,
    final_model: FinalModelManifest,
) -> None:
    """Verify that an artifact references the exact canonical final-model record."""
    from mlforge.final_models.types import FinalModelManifest

    if not isinstance(artifact, ArtifactManifest) or not isinstance(
        final_model, FinalModelManifest
    ):
        raise ArtifactIntegrityError(
            "Artifact and final-model manifests must be validated manifest values."
        )
    if artifact.lineage_kind is not ArtifactLineageKind.FINAL_MODEL:
        raise ArtifactIntegrityError("Artifact does not reference a final-model manifest.")
    if artifact.model_id != final_model.final_model_id:
        raise ArtifactIntegrityError("Artifact and final-model manifest identifiers do not match.")
    digest = hashlib.sha256(final_model.to_json(indent=None).encode("utf-8")).hexdigest()
    if artifact.lineage_manifest_sha256 != digest:
        raise ArtifactIntegrityError(
            "Artifact does not reference the supplied final-model manifest."
        )
    contract = final_model.artifact
    if contract is None:
        raise ArtifactIntegrityError("Final-model manifest is missing its artifact contract.")
    if (
        contract.artifact_id != artifact.model_id
        or contract.serialization_format != artifact.serialization_format
        or contract.pipeline_sha256 != artifact.pipeline_sha256
        or contract.pipeline_size_bytes != artifact.pipeline_size_bytes
    ):
        raise ArtifactIntegrityError(
            "Artifact payload does not match the supplied final-model artifact contract."
        )


def _persisted_run(result: TrainingResult) -> RunManifest:
    path = result.manifest_path
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError(
            "Training result must reference its persisted regular-file run manifest."
        )
    try:
        content = path.read_text(encoding="utf-8")
        persisted = RunManifest.from_json(content)
    except (OSError, UnicodeError, RunStoreError) as error:
        raise ArtifactIntegrityError(
            "Could not validate the persisted training run manifest."
        ) from error
    if persisted != result.manifest:
        raise ArtifactIntegrityError("Training result does not match its persisted run manifest.")
    return persisted


def _persisted_final_model(result: FinalModelResult) -> FinalModelManifest:
    from mlforge.final_models.types import FinalModelManifest

    path = result.manifest_path
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError(
            "Final-model result must reference its persisted regular-file manifest."
        )
    try:
        content = path.read_text(encoding="utf-8")
        persisted = FinalModelManifest.from_json(content)
    except (OSError, UnicodeError, FinalModelStoreError) as error:
        raise ArtifactIntegrityError(
            "Could not validate the persisted final-model manifest."
        ) from error
    if persisted != result.manifest:
        raise ArtifactIntegrityError(
            "Final-model result does not match its persisted final-model manifest."
        )
    if path.stem != persisted.final_model_id:
        raise ArtifactIntegrityError(
            "Final-model manifest ID does not match its persisted filename."
        )
    return persisted


def _validate_pipeline_contract(result: TrainingResult | FinalModelResult) -> None:
    try:
        check_is_fitted(result.pipeline)
        actual_features = tuple(str(name) for name in result.pipeline.feature_names_in_)
    except (TypeError, ValueError, AttributeError) as error:
        raise ArtifactFormatError("Artifact pipeline must be fitted on named features.") from error
    if actual_features != result.feature_schema.all_features:
        raise ArtifactIntegrityError(
            "Fitted pipeline feature names do not match the training result schema."
        )


def _check_environment(manifest: ArtifactManifest) -> None:
    current = _current_environment()
    if manifest.environment == current:
        return
    mismatches = [
        f"{name}: trained={getattr(manifest.environment, name)!r}, "
        f"current={getattr(current, name)!r}"
        for name in (
            "python",
            "mlforge",
            "pandas",
            "numpy",
            "scipy",
            "scikit_learn",
        )
        if getattr(manifest.environment, name) != getattr(current, name)
    ]
    raise ArtifactCompatibilityError(
        "Artifact dependency versions must exactly match before loading: " + "; ".join(mismatches)
    )


@dataclass(frozen=True, slots=True)
class LocalArtifactStore:
    """Create-only local store for one atomic artifact archive per successful run."""

    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise ArtifactPathError("Artifact store root must be a pathlib.Path.")

    def _resolved_root(self, *, create: bool) -> Path:
        candidate = self.root.expanduser()
        try:
            if create:
                candidate.mkdir(parents=True, exist_ok=True)
            resolved = candidate.resolve(strict=not create)
        except OSError as error:
            raise ArtifactPathError(f"Could not access artifact store: {candidate}") from error
        if resolved.exists() and not resolved.is_dir():
            raise ArtifactPathError(f"Artifact store path is not a directory: {resolved}")
        return resolved

    def artifact_path(self, run_id: str) -> Path:
        """Return the canonical archive path for one successful run UUID."""
        canonical = _canonical_run_id(run_id)
        return self._resolved_root(create=False) / f"{canonical}{ARTIFACT_SUFFIX}"

    def save(self, result: TrainingResult) -> SavedArtifact:
        """Serialize and atomically publish a fitted pipeline without overwriting."""
        if not isinstance(result, TrainingResult):
            raise ArtifactFormatError("Artifact input must be a TrainingResult.")
        if result.manifest.status is not RunStatus.SUCCEEDED:
            raise ArtifactFormatError("Only a successful training result can become an artifact.")
        persisted_run = _persisted_run(result)
        _validate_pipeline_contract(result)
        try:
            pipeline_bytes = pickle.dumps(result.pipeline, protocol=5)
        except (TypeError, ValueError, pickle.PickleError) as error:
            raise ArtifactFormatError(f"Could not serialize fitted pipeline: {error}") from error
        manifest = _manifest(result, pipeline_bytes)
        verify_run_manifest(manifest, persisted_run)
        return self._write(manifest, pipeline_bytes)

    def save_final(self, result: FinalModelResult) -> SavedArtifact:
        """Serialize an explicitly fitted full-dataset model with verified selection lineage."""
        from mlforge.final_models.types import FinalModelResult

        if not isinstance(result, FinalModelResult):
            raise ArtifactFormatError("Final artifact input must be a FinalModelResult.")
        if result.manifest.status is not RunStatus.SUCCEEDED:
            raise ArtifactFormatError(
                "Only a successful final-model result can become an artifact."
            )
        persisted = _persisted_final_model(result)
        _validate_pipeline_contract(result)
        try:
            pipeline_bytes = pickle.dumps(result.pipeline, protocol=5)
        except (TypeError, ValueError, pickle.PickleError) as error:
            raise ArtifactFormatError(f"Could not serialize fitted pipeline: {error}") from error
        manifest = _final_model_manifest(result, pipeline_bytes)
        verify_final_model_manifest(manifest, persisted)
        return self._write(manifest, pipeline_bytes)

    def _write(self, manifest: ArtifactManifest, pipeline_bytes: bytes) -> SavedArtifact:
        """Atomically publish one already validated manifest and payload."""
        root = self._resolved_root(create=True)
        final_path = root / f"{manifest.model_id}{ARTIFACT_SUFFIX}"
        if final_path.exists():
            raise ArtifactPathError(f"Artifact already exists and is immutable: {final_path}")
        temporary_path = root / f".{manifest.model_id}.{uuid4()}.tmp"
        try:
            with temporary_path.open("x+b") as stream:
                with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
                    archive.writestr(MANIFEST_MEMBER, manifest.to_json().encode("utf-8"))
                    archive.writestr(PIPELINE_MEMBER, pipeline_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, final_path)
        except FileExistsError as error:
            raise ArtifactPathError(
                f"Artifact already exists and is immutable: {final_path}"
            ) from error
        except OSError as error:
            raise ArtifactPathError(f"Could not atomically write artifact: {final_path}") from error
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise ArtifactPathError(
                    f"Could not clean up temporary artifact: {temporary_path}"
                ) from cleanup_error
        return SavedArtifact(path=final_path, manifest=manifest)

    def inspect(self, run_id: str) -> ArtifactManifest:
        """Safely verify and return artifact metadata without executing its payload."""
        manifest, _ = _read_archive(self.artifact_path(run_id), include_pipeline=False)
        return manifest

    def load(self, run_id: str, *, trusted: bool = False) -> LoadedArtifact:
        """Load executable model bytes only after an explicit trust decision."""
        return load_artifact(self.artifact_path(run_id), trusted=trusted)


def inspect_artifact(path: Path) -> ArtifactManifest:
    """Safely inspect and checksum one artifact without deserializing Python objects."""
    if not isinstance(path, Path):
        raise ArtifactPathError("Artifact path must be a pathlib.Path.")
    manifest, _ = _read_archive(path, include_pipeline=False)
    return manifest


def load_artifact(path: Path, *, trusted: bool = False) -> LoadedArtifact:
    """Integrity-check and deserialize a trusted local artifact in a matching environment."""
    if trusted is not True:
        raise ArtifactTrustError(
            "Artifact loading can execute Python code. Pass trusted=True only for a local artifact "
            "whose source you have verified."
        )
    if not isinstance(path, Path):
        raise ArtifactPathError("Artifact path must be a pathlib.Path.")
    manifest, payload = _read_archive(path, include_pipeline=True)
    if payload is None:
        raise ArtifactFormatError("Artifact pipeline payload was not read.")
    _check_environment(manifest)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            value = pickle.loads(payload)
    except InconsistentVersionWarning as error:
        raise ArtifactCompatibilityError(
            "Artifact contains a scikit-learn object from an incompatible version."
        ) from error
    except (
        pickle.PickleError,
        AttributeError,
        EOFError,
        ImportError,
        IndexError,
        TypeError,
        ValueError,
    ) as error:
        raise ArtifactFormatError(
            f"Trusted artifact payload could not be loaded: {error}"
        ) from error
    if not isinstance(value, Pipeline):
        raise ArtifactFormatError("Trusted artifact payload is not a scikit-learn Pipeline.")
    try:
        check_is_fitted(value)
    except (TypeError, AttributeError) as error:
        raise ArtifactFormatError("Trusted artifact pipeline is not fitted.") from error
    expected_features = tuple(feature.name for feature in manifest.features)
    actual_features = tuple(str(name) for name in value.feature_names_in_)
    if actual_features != expected_features:
        raise ArtifactIntegrityError(
            "Loaded pipeline feature names do not match the artifact manifest."
        )
    return LoadedArtifact(path=_safe_artifact_path(path), manifest=manifest, pipeline=value)
