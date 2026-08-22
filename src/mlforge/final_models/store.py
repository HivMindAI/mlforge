"""Create-only local storage for strict final-model manifests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from mlforge.errors import FinalModelStoreError
from mlforge.final_models.types import FinalModelManifest

MAX_FINAL_MODEL_MANIFEST_SIZE_BYTES = 1024 * 1024


def _canonical_final_model_id(final_model_id: str) -> str:
    try:
        parsed = UUID(final_model_id)
    except (ValueError, AttributeError, TypeError) as error:
        raise FinalModelStoreError("Final-model ID must be a canonical UUID.") from error
    canonical = str(parsed)
    if final_model_id != canonical:
        raise FinalModelStoreError("Final-model ID must be a canonical lowercase UUID.")
    return canonical


@dataclass(frozen=True, slots=True)
class LocalFinalModelStore:
    """Filesystem-backed final-model records that are created once and never overwritten."""

    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise FinalModelStoreError("Final-model store root must be a pathlib.Path.")

    def _resolved_root(self, *, create: bool) -> Path:
        candidate = self.root.expanduser()
        try:
            if create:
                candidate.mkdir(parents=True, exist_ok=True)
            resolved = candidate.resolve(strict=not create)
        except OSError as error:
            raise FinalModelStoreError(
                f"Could not access final-model store directory: {candidate}"
            ) from error
        if resolved.exists() and not resolved.is_dir():
            raise FinalModelStoreError(f"Final-model store path is not a directory: {resolved}")
        return resolved

    def manifest_path(self, final_model_id: str) -> Path:
        canonical = _canonical_final_model_id(final_model_id)
        return self._resolved_root(create=False) / f"{canonical}.json"

    def write(self, manifest: FinalModelManifest) -> Path:
        if not isinstance(manifest, FinalModelManifest):
            raise FinalModelStoreError("Final-model store input must be a FinalModelManifest.")
        root = self._resolved_root(create=True)
        final_path = root / f"{manifest.final_model_id}.json"
        if final_path.exists():
            raise FinalModelStoreError(
                f"Final-model manifest already exists and is immutable: {final_path}"
            )
        temporary_path = root / f".{manifest.final_model_id}.{uuid4()}.tmp"
        try:
            with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(manifest.to_json())
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, final_path)
        except FileExistsError as error:
            raise FinalModelStoreError(
                f"Final-model manifest already exists and is immutable: {final_path}"
            ) from error
        except OSError as error:
            raise FinalModelStoreError(
                f"Could not atomically write final-model manifest: {final_path}"
            ) from error
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise FinalModelStoreError(
                    f"Could not clean up temporary final-model manifest: {temporary_path}"
                ) from cleanup_error
        return final_path

    def read(self, final_model_id: str) -> FinalModelManifest:
        path = self.manifest_path(final_model_id)
        if path.is_symlink() or not path.is_file():
            raise FinalModelStoreError(
                f"Final-model manifest does not exist or is not a regular file: {path}"
            )
        try:
            size = path.stat().st_size
        except OSError as error:
            raise FinalModelStoreError(f"Could not inspect final-model manifest: {path}") from error
        if size > MAX_FINAL_MODEL_MANIFEST_SIZE_BYTES:
            raise FinalModelStoreError(
                f"Final-model manifest exceeds the {MAX_FINAL_MODEL_MANIFEST_SIZE_BYTES}-byte "
                f"size limit: {path}"
            )
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise FinalModelStoreError(
                f"Could not read UTF-8 final-model manifest: {path}"
            ) from error
        manifest = FinalModelManifest.from_json(content)
        if manifest.final_model_id != final_model_id:
            raise FinalModelStoreError(
                f"Final-model manifest ID does not match its filename: {path.name}"
            )
        return manifest

    def list_manifests(self) -> tuple[FinalModelManifest, ...]:
        try:
            root = self._resolved_root(create=False)
        except FinalModelStoreError:
            if not self.root.expanduser().exists():
                return ()
            raise
        manifests: list[FinalModelManifest] = []
        for path in sorted(root.glob("*.json"), key=lambda item: item.name):
            try:
                final_model_id = _canonical_final_model_id(path.stem)
            except FinalModelStoreError:
                continue
            manifests.append(self.read(final_model_id))
        return tuple(sorted(manifests, key=lambda item: (item.started_at, item.final_model_id)))
