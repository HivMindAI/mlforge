"""Tests for explicit MLForge logging configuration."""

import logging
from io import StringIO

from mlforge.config import LogLevel
from mlforge.logging_config import LOGGER_NAME, configure_logging


def test_configure_logging_formats_package_records() -> None:
    """The entrypoint formatter should produce concise package-owned records."""
    logger = logging.getLogger(LOGGER_NAME)
    original_handlers = tuple(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    stream = StringIO()

    try:
        configure_logging(LogLevel.INFO, stream=stream)
        logging.getLogger("mlforge.training").info("training started")

        assert stream.getvalue() == "INFO mlforge.training: training started\n"
    finally:
        for handler in tuple(logger.handlers):
            if handler not in original_handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_configure_logging_is_idempotent() -> None:
    """Repeated entrypoint setup should not duplicate MLForge-owned handlers."""
    logger = logging.getLogger(LOGGER_NAME)
    original_handlers = tuple(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    try:
        configure_logging(LogLevel.INFO, stream=StringIO())
        configure_logging(LogLevel.DEBUG, stream=StringIO())

        configured_handlers = [
            handler for handler in logger.handlers if handler.get_name() == "mlforge-cli"
        ]
        assert len(configured_handlers) == 1
        assert logger.level == logging.DEBUG
    finally:
        for handler in tuple(logger.handlers):
            if handler not in original_handlers:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(original_level)
        logger.propagate = original_propagate
