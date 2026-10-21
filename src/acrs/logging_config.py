from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

from acrs.config import PROJECT_ROOT


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger("acrs")

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
    )
    console_handler.setLevel(level)

    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    target_log = log_file or PROJECT_ROOT / "logs" / "acrs.log"
    target_log.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        target_log,
        maxBytes=10_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
