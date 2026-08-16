"""Atomic, immutable local storage for validated run manifests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from mlforge.errors import RunStoreError
from mlforge.runs.types import RunManifest

MAX_MANIFEST_SIZE_BYTES = 1024 * 1024


def _canonical_run_id(run_id: str) -> str:
    try:
        parsed = UUID(run_id)
    except (ValueError, AttributeError) as error:
        raise RunStoreError("Run ID must be a canonical UUID.") from error
    canonical = str(parsed)
    if run_id != canonical:
        raise RunStoreError("Run ID must be a canonical lowercase UUID.")
    return canonical


@dataclass(frozen=True, slots=True)
class LocalRunStore:
    """Filesystem-backed run records that are created once and never overwritten."""

    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise RunStoreError("Run store root must be a pathlib.Path.")

    def _resolved_root(self, *, create: bool) -> Path:
        candidate = self.root.expanduser()
        try:
            if create:
                candidate.mkdir(parents=True, exist_ok=True)
            resolved = candidate.resolve(strict=not create)
        except OSError as error:
            raise RunStoreError(f"Could not access run store directory: {candidate}") from error
        if resolved.exists() and not resolved.is_dir():
            raise RunStoreError(f"Run store path is not a directory: {resolved}")
        return resolved

    def manifest_path(self, run_id: str) -> Path:
        """Return the safe canonical path for one run identifier."""
        canonical = _canonical_run_id(run_id)
        root = self._resolved_root(create=False)
        return root / f"{canonical}.json"

    def write(self, manifest: RunManifest) -> Path:
        """Atomically create one manifest without overwriting an existing run."""
        root = self._resolved_root(create=True)
        final_path = root / f"{manifest.run_id}.json"
        if final_path.exists():
            raise RunStoreError(f"Run manifest already exists and is immutable: {final_path}")

        temporary_path = root / f".{manifest.run_id}.{uuid4()}.tmp"
        try:
            with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(manifest.to_json())
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, final_path)
        except FileExistsError as error:
            raise RunStoreError(
                f"Run manifest already exists and is immutable: {final_path}"
            ) from error
        except OSError as error:
            raise RunStoreError(f"Could not atomically write run manifest: {final_path}") from error
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise RunStoreError(
                    f"Could not clean up temporary run manifest: {temporary_path}"
                ) from cleanup_error
        return final_path

    def read(self, run_id: str) -> RunManifest:
        """Read one size-limited manifest and validate its complete schema."""
        path = self.manifest_path(run_id)
        if path.is_symlink() or not path.is_file():
            raise RunStoreError(f"Run manifest does not exist or is not a regular file: {path}")
        try:
            size = path.stat().st_size
        except OSError as error:
            raise RunStoreError(f"Could not inspect run manifest: {path}") from error
        if size > MAX_MANIFEST_SIZE_BYTES:
            raise RunStoreError(
                f"Run manifest exceeds the {MAX_MANIFEST_SIZE_BYTES}-byte size limit: {path}"
            )
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise RunStoreError(f"Could not read UTF-8 run manifest: {path}") from error
        manifest = RunManifest.from_json(content)
        if manifest.run_id != run_id:
            raise RunStoreError(f"Run manifest ID does not match its filename: {path.name}")
        return manifest

    def list_manifests(self) -> tuple[RunManifest, ...]:
        """Return all validated manifests in chronological order."""
        try:
            root = self._resolved_root(create=False)
        except RunStoreError:
            if not self.root.expanduser().exists():
                return ()
            raise

        paths = sorted(root.glob("*.json"), key=lambda path: path.name)
        manifests: list[RunManifest] = []
        for path in paths:
            try:
                run_id = path.stem
                _canonical_run_id(run_id)
            except RunStoreError:
                continue
            manifests.append(self.read(run_id))
        return tuple(sorted(manifests, key=lambda item: (item.started_at, item.run_id)))
