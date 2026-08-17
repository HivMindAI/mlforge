"""Explicit logging configuration for MLForge application entrypoints."""

import logging
from typing import TextIO

from mlforge.config import LogLevel

LOGGER_NAME = "mlforge"
_CLI_HANDLER_NAME = "mlforge-cli"
_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"

__all__ = ["LOGGER_NAME", "configure_logging"]


def configure_logging(level: LogLevel, *, stream: TextIO | None = None) -> logging.Logger:
    """Configure and return the MLForge logger for a CLI process."""
    logger = logging.getLogger(LOGGER_NAME)

    for handler in tuple(logger.handlers):
        if handler.get_name() == _CLI_HANDLER_NAME:
            logger.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream)
    handler.set_name(_CLI_HANDLER_NAME)
    handler.setLevel(level.value)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level.value)
    logger.propagate = False
    return logger
