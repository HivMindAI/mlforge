"""Tests for typed MLForge application configuration."""

import pytest

from mlforge.config import ApplicationConfig, LogLevel
from mlforge.errors import ConfigurationError


def test_default_configuration() -> None:
    """Defaults should be deterministic when no environment values exist."""
    assert ApplicationConfig.from_environment({}) == ApplicationConfig(log_level=LogLevel.WARNING)


def test_environment_log_level_is_case_insensitive() -> None:
    """Environment input should be normalized into the typed enum."""
    config = ApplicationConfig.from_environment({"MLFORGE_LOG_LEVEL": " info "})

    assert config.log_level is LogLevel.INFO


def test_explicit_override_takes_precedence() -> None:
    """An explicit entrypoint value should override environment-derived settings."""
    config = ApplicationConfig.from_environment({"MLFORGE_LOG_LEVEL": "INFO"})

    overridden = config.with_overrides(log_level=LogLevel.DEBUG)

    assert overridden.log_level is LogLevel.DEBUG
    assert config.log_level is LogLevel.INFO


@pytest.mark.parametrize("value", ["", "verbose", "info,debug"])
def test_invalid_environment_log_level(value: str) -> None:
    """Invalid environment values should fail with an actionable domain error."""
    with pytest.raises(ConfigurationError, match="Invalid log level"):
        ApplicationConfig.from_environment({"MLFORGE_LOG_LEVEL": value})
