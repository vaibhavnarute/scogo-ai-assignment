"""Concise human-readable console logging, separate from JSONL traces."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("harness")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger

