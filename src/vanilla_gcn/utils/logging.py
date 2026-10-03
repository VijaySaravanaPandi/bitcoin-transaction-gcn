"""
logging.py — Centralised logging configuration for Vanilla GCN.

Uses Python's standard ``logging`` module.  All modules in this package
use ``logging.getLogger(__name__)`` — this module configures the root
handler so that output is formatted consistently.

Usage
-----
    from vanilla_gcn.utils.logging import setup_logging
    setup_logging(level="INFO")

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Training started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str | int = "INFO",
    log_file: Optional[str | Path] = None,
    *,
    force: bool = False,
) -> None:
    """Configure the root logger for the vanilla_gcn package.

    Parameters
    ----------
    level : str or int
        Logging level.  One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, ``"CRITICAL"`` or the corresponding integer constants.
    log_file : str or Path, optional
        If provided, also write logs to this file.
    force : bool
        If ``True``, remove existing handlers before adding new ones.
        Useful in interactive / notebook environments.

    Examples
    --------
    >>> setup_logging("DEBUG")
    >>> setup_logging("INFO", log_file="outputs/logs/run.log")
    """
    root_logger = logging.getLogger("vanilla_gcn")

    if force:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    numeric_level = (
        level if isinstance(level, int) else getattr(logging, level.upper(), logging.INFO)
    )
    root_logger.setLevel(numeric_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``vanilla_gcn``.

    Parameters
    ----------
    name : str
        Sub-name for the logger (e.g. ``__name__`` of the calling module).

    Returns
    -------
    logging.Logger

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Hello from %s", __name__)
    """
    return logging.getLogger(f"vanilla_gcn.{name}" if not name.startswith("vanilla_gcn") else name)
