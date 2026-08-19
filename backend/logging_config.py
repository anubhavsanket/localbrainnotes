"""Centralised logging configuration for LocalBrain.

Call ``setup_logging()`` once at startup (in ``main.py``) to configure the
root logger.  Every module then just does::

    import logging
    logger = logging.getLogger(__name__)

Level is controlled by ``settings.LOG_LEVEL`` (default ``INFO``).
"""
import logging
import sys

from config import settings


def setup_logging() -> None:
    """Configure the root logger once at process start."""
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    fmt = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called more than once.
    if not root.handlers:
        root.addHandler(handler)
    # Quiet noisy third-party loggers.
    for name in ("httpx", "httpcore", "chromadb", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
