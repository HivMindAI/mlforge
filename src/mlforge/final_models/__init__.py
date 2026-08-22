"""Explicit cross-validation selection to full-dataset model public API."""

from mlforge.final_models.service import fit_selected_model
from mlforge.final_models.store import LocalFinalModelStore
from mlforge.final_models.types import (
    FINAL_MODEL_FIT_SCOPE,
    FINAL_MODEL_MANIFEST_SCHEMA_VERSION,
    FinalModelArtifact,
    FinalModelConfiguration,
    FinalModelManifest,
    FinalModelResult,
    FinalModelSelection,
)

__all__ = [
    "FINAL_MODEL_FIT_SCOPE",
    "FINAL_MODEL_MANIFEST_SCHEMA_VERSION",
    "FinalModelArtifact",
    "FinalModelConfiguration",
    "FinalModelManifest",
    "FinalModelResult",
    "FinalModelSelection",
    "LocalFinalModelStore",
    "fit_selected_model",
]
