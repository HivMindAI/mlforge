"""Configuration for the local single-user web adapter."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from mlforge.datasets import CsvLoadOptions
from mlforge.errors import ConfigurationError

WEB_WORKSPACE_ENVIRONMENT_VARIABLE = "MLFORGE_WEB_WORKSPACE"
WEB_MAX_UPLOAD_BYTES_ENVIRONMENT_VARIABLE = "MLFORGE_WEB_MAX_UPLOAD_BYTES"
DEFAULT_WEB_MAX_UPLOAD_BYTES = CsvLoadOptions().max_file_size_bytes


@dataclass(frozen=True, slots=True)
class WebSettings:
    """Filesystem and resource settings for one local web process."""

    workspace: Path = Path(".mlforge-web")
    max_upload_bytes: int = DEFAULT_WEB_MAX_UPLOAD_BYTES

    def __post_init__(self) -> None:
        """Reject invalid settings before creating storage."""
        if not isinstance(self.workspace, Path):
            raise ConfigurationError("MLForge web workspace must be a pathlib.Path value.")
        if (
            isinstance(self.max_upload_bytes, bool)
            or not isinstance(self.max_upload_bytes, int)
            or self.max_upload_bytes <= 0
        ):
            raise ConfigurationError("MLForge web upload limit must be greater than zero bytes.")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> WebSettings:
        """Build web settings from an injected mapping or the process environment."""
        source = os.environ if environ is None else environ
        raw_workspace = source.get(WEB_WORKSPACE_ENVIRONMENT_VARIABLE, ".mlforge-web")
        raw_max_upload_bytes = source.get(
            WEB_MAX_UPLOAD_BYTES_ENVIRONMENT_VARIABLE,
            str(DEFAULT_WEB_MAX_UPLOAD_BYTES),
        )

        if not raw_workspace.strip():
            raise ConfigurationError("MLForge web workspace must not be blank.")
        try:
            max_upload_bytes = int(raw_max_upload_bytes)
        except ValueError as error:
            raise ConfigurationError(
                "MLForge web upload limit must be a whole number of bytes."
            ) from error

        return cls(workspace=Path(raw_workspace), max_upload_bytes=max_upload_bytes)
