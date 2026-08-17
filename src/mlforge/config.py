"""Typed application configuration."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from mlforge.errors import ConfigurationError

LOG_LEVEL_ENVIRONMENT_VARIABLE = "MLFORGE_LOG_LEVEL"

__all__ = ["ApplicationConfig", "LOG_LEVEL_ENVIRONMENT_VARIABLE", "LogLevel"]


class LogLevel(StrEnum):
    """Supported MLForge logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def parse(cls, value: str) -> "LogLevel":
        """Parse a case-insensitive log level or raise a domain error."""
        normalized = value.strip().upper()
        try:
            return cls(normalized)
        except ValueError as error:
            expected = ", ".join(level.value for level in cls)
            raise ConfigurationError(
                f"Invalid log level {value!r}; expected one of: {expected}."
            ) from error


@dataclass(frozen=True, slots=True)
class ApplicationConfig:
    """Process-level settings resolved before an application operation starts."""

    log_level: LogLevel = LogLevel.WARNING

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> "ApplicationConfig":
        """Build settings from an injected mapping or the process environment."""
        source = os.environ if environ is None else environ
        raw_log_level = source.get(LOG_LEVEL_ENVIRONMENT_VARIABLE)
        if raw_log_level is None:
            return cls()
        return cls(log_level=LogLevel.parse(raw_log_level))

    def with_overrides(self, *, log_level: LogLevel | None = None) -> "ApplicationConfig":
        """Return settings with explicit values taking precedence."""
        if log_level is None:
            return self
        return ApplicationConfig(log_level=log_level)
