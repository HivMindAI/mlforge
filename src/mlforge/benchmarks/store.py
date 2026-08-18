"""Atomic, immutable local storage for validated benchmark manifests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from mlforge.benchmarks.types import BenchmarkManifest
from mlforge.errors import BenchmarkStoreError

MAX_BENCHMARK_MANIFEST_SIZE_BYTES = 1024 * 1024


def _canonical_benchmark_id(benchmark_id: str) -> str:
    try:
        parsed = UUID(benchmark_id)
    except (ValueError, AttributeError) as error:
        raise BenchmarkStoreError("Benchmark ID must be a canonical UUID.") from error
    canonical = str(parsed)
    if benchmark_id != canonical:
        raise BenchmarkStoreError("Benchmark ID must be a canonical lowercase UUID.")
    return canonical


@dataclass(frozen=True, slots=True)
class LocalBenchmarkStore:
    """Filesystem-backed benchmark records that are created once and never overwritten."""

    root: Path

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            raise BenchmarkStoreError("Benchmark store root must be a pathlib.Path.")

    def _resolved_root(self, *, create: bool) -> Path:
        candidate = self.root.expanduser()
        try:
            if create:
                candidate.mkdir(parents=True, exist_ok=True)
            resolved = candidate.resolve(strict=not create)
        except OSError as error:
            raise BenchmarkStoreError(
                f"Could not access benchmark store directory: {candidate}"
            ) from error
        if resolved.exists() and not resolved.is_dir():
            raise BenchmarkStoreError(f"Benchmark store path is not a directory: {resolved}")
        return resolved

    def manifest_path(self, benchmark_id: str) -> Path:
        """Return the safe canonical path for one benchmark identifier."""
        canonical = _canonical_benchmark_id(benchmark_id)
        return self._resolved_root(create=False) / f"{canonical}.json"

    def write(self, manifest: BenchmarkManifest) -> Path:
        """Atomically create one manifest without overwriting an existing benchmark."""
        root = self._resolved_root(create=True)
        final_path = root / f"{manifest.benchmark_id}.json"
        if final_path.exists():
            raise BenchmarkStoreError(
                f"Benchmark manifest already exists and is immutable: {final_path}"
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
                f"Benchmark manifest already exists and is immutable: {final_path}"
            ) from error
        except OSError as error:
            raise BenchmarkStoreError(
                f"Could not atomically write benchmark manifest: {final_path}"
            ) from error
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise BenchmarkStoreError(
                    f"Could not clean up temporary benchmark manifest: {temporary_path}"
                ) from cleanup_error
        return final_path

    def read(self, benchmark_id: str) -> BenchmarkManifest:
        """Read one size-limited manifest and validate its complete schema."""
        path = self.manifest_path(benchmark_id)
        if path.is_symlink() or not path.is_file():
            raise BenchmarkStoreError(
                f"Benchmark manifest does not exist or is not a regular file: {path}"
            )
        try:
            size = path.stat().st_size
        except OSError as error:
            raise BenchmarkStoreError(f"Could not inspect benchmark manifest: {path}") from error
        if size > MAX_BENCHMARK_MANIFEST_SIZE_BYTES:
            raise BenchmarkStoreError(
                "Benchmark manifest exceeds the "
                f"{MAX_BENCHMARK_MANIFEST_SIZE_BYTES}-byte size limit: {path}"
            )
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise BenchmarkStoreError(f"Could not read UTF-8 benchmark manifest: {path}") from error
        manifest = BenchmarkManifest.from_json(content)
        if manifest.benchmark_id != benchmark_id:
            raise BenchmarkStoreError(
                f"Benchmark manifest ID does not match its filename: {path.name}"
            )
        return manifest

    def list_manifests(self) -> tuple[BenchmarkManifest, ...]:
        """Return all validated benchmark manifests in chronological order."""
        try:
            root = self._resolved_root(create=False)
        except BenchmarkStoreError:
            if not self.root.expanduser().exists():
                return ()
            raise

        manifests: list[BenchmarkManifest] = []
        for path in sorted(root.glob("*.json"), key=lambda item: item.name):
            try:
                benchmark_id = path.stem
                _canonical_benchmark_id(benchmark_id)
            except BenchmarkStoreError:
                continue
            manifests.append(self.read(benchmark_id))
        return tuple(sorted(manifests, key=lambda item: (item.started_at, item.benchmark_id)))
