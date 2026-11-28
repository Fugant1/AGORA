"""Structured logging utility for AGORA experiments."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "agora", level: int = logging.INFO) -> logging.Logger:
    """Creates a formatted console logger for research experiments."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
