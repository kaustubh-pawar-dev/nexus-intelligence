"""Logging setup for NEXUS Intelligence."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from nexus.config import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOGGER_NAME = "nexus"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger.

    The root ``nexus`` logger gets a console handler and a rotating file handler
    exactly once. Child loggers (``nexus.data``, ...) inherit them.
    """
    settings = get_settings()
    base = logging.getLogger(_LOGGER_NAME)

    if not base.handlers:
        base.setLevel(settings.log_level)
        formatter = logging.Formatter(_LOG_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        base.addHandler(console)

        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.logs_dir / "nexus.log",
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        base.addHandler(file_handler)

        base.propagate = False

    if name and name != _LOGGER_NAME:
        suffix = name.removeprefix(_LOGGER_NAME + ".")
        return logging.getLogger(f"{_LOGGER_NAME}.{suffix}")
    return base