"""
src/utils/logger.py
-------------------
Centralised logging. All modules call get_logger(__name__).

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting fetch...")
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    """Return a configured logger. Safe to call multiple times (no duplicate handlers)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
