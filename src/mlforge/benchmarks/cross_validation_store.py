"""Create-only local storage for strict cross-validation benchmark manifests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from mlforge.benchmarks.cross_validation_types import CrossValidationManifest
from mlforge.errors import BenchmarkStoreError

MAX_CROSS_VALIDATION_MANIFEST_SIZE_BYTES = 4 * 1024 * 1024


def _canonical_benchmark_id(benchmark_id: str) -> str:
    try:
        parsed = UUID(benchmark_id)
    except (ValueError, AttributeError) as error:
        raise BenchmarkStoreError(
            "Cross-validation benchmark ID must be a canonical UUID."
        ) from error
    canonical = str(parsed)
    if benchmark_id != canonical:
        raise BenchmarkStoreError(
            "Cross-validation benchmark ID must be a canonical lowercase UUID."
        )
    return canonical


@dataclass(frozen=True, slots=True)
class LocalCrossValidationStore:
    """Filesystem-backed cross-validation records with immutable publication."""

    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise BenchmarkStoreError("Cross-validation store root must be a pathlib.Path.")

    def _resolved_root(self, *, create: bool) -> Path:
        candidate = self.root.expanduser()
        try:
            if create:
                candidate.mkdir(parents=True, exist_ok=True)
            resolved = candidate.resolve(strict=not create)
        except OSError as error:
            raise BenchmarkStoreError(
                f"Could not access cross-validation store directory: {candidate}"
            ) from error
        if resolved.exists() and not resolved.is_dir():
            raise BenchmarkStoreError(f"Cross-validation store path is not a directory: {resolved}")
        return resolved

    def manifest_path(self, benchmark_id: str) -> Path:
        canonical = _canonical_benchmark_id(benchmark_id)
        return self._resolved_root(create=False) / f"{canonical}.json"

    def write(self, manifest: CrossValidationManifest) -> Path:
        root = self._resolved_root(create=True)
        final_path = root / f"{manifest.benchmark_id}.json"
        if final_path.exists():
            raise BenchmarkStoreError(
                f"Cross-validation manifest already exists and is immutable: {final_path}"
            )

        temporary_path = root / f".{manifest.benchmark_id}.{uuid4()}.tmp"
        try:
            with temporary_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(manifest.to_json())
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, final_path)
        except FileExistsError as error:
            raise BenchmarkStoreError(
                f"Cross-validation manifest already exists and is immutable: {final_path}"
            ) from error
        except OSError as error:
            raise BenchmarkStoreError(
                f"Could not atomically write cross-validation manifest: {final_path}"
            ) from error
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise BenchmarkStoreError(
                    f"Could not clean up temporary cross-validation manifest: {temporary_path}"
                ) from cleanup_error
        return final_path

    def read(self, benchmark_id: str) -> CrossValidationManifest:
        path = self.manifest_path(benchmark_id)
        if path.is_symlink() or not path.is_file():
            raise BenchmarkStoreError(
                f"Cross-validation manifest does not exist or is not a regular file: {path}"
            )
        try:
            size = path.stat().st_size
        except OSError as error:
            raise BenchmarkStoreError(
                f"Could not inspect cross-validation manifest: {path}"
            ) from error
        if size > MAX_CROSS_VALIDATION_MANIFEST_SIZE_BYTES:
            raise BenchmarkStoreError(
                "Cross-validation manifest exceeds the "
                f"{MAX_CROSS_VALIDATION_MANIFEST_SIZE_BYTES}-byte size limit: {path}"
            )
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise BenchmarkStoreError(
                f"Could not read UTF-8 cross-validation manifest: {path}"
            ) from error
        manifest = CrossValidationManifest.from_json(content)
        if manifest.benchmark_id != benchmark_id:
            raise BenchmarkStoreError(
                f"Cross-validation manifest ID does not match its filename: {path.name}"
            )
        return manifest

    def list_manifests(self) -> tuple[CrossValidationManifest, ...]:
        try:
            root = self._resolved_root(create=False)
        except BenchmarkStoreError:
            if not self.root.expanduser().exists():
                return ()
            raise

        manifests: list[CrossValidationManifest] = []
        for path in sorted(root.glob("*.json"), key=lambda item: item.name):
            try:
                benchmark_id = path.stem
                _canonical_benchmark_id(benchmark_id)
            except BenchmarkStoreError:
                continue
            manifests.append(self.read(benchmark_id))
        return tuple(sorted(manifests, key=lambda item: (item.started_at, item.benchmark_id)))
