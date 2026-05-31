"""Shared runtime logger for console and rotating file output."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "camera_recognition"
LOG_DIRECTORY = Path("logs")
LOG_FILE_PATH = LOG_DIRECTORY / "runtime.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
MAX_LOG_BYTES = 10 * 1024 * 1024
BACKUP_COUNT = 5


def _configure_logger() -> logging.Logger:
    """Configure and return the project-wide runtime logger."""
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)

    shared_logger = logging.getLogger(LOGGER_NAME)
    shared_logger.setLevel(logging.INFO)
    shared_logger.propagate = False

    if shared_logger.handlers:
        return shared_logger

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        LOG_FILE_PATH,
        maxBytes=MAX_LOG_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    shared_logger.addHandler(console_handler)
    shared_logger.addHandler(file_handler)
    return shared_logger


logger = _configure_logger()
