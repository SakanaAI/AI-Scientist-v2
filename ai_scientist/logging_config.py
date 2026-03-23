"""Central logging setup for AI-Scientist. Call :func:`setup_logging` once from entry points."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: Optional[int] = None, *, force: bool = True) -> None:
    """Configure root logging (stderr). Level from *level* or env ``AI_SCIENTIST_LOG_LEVEL`` (default INFO)."""
    if level is None:
        name = os.environ.get("AI_SCIENTIST_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, name, logging.INFO)
    kwargs = dict(
        level=level,
        format=_DEFAULT_FORMAT,
        stream=sys.stderr,
    )
    # Python 3.8+: replace existing handlers so repeated calls take effect
    if sys.version_info >= (3, 8):
        kwargs["force"] = force
    logging.basicConfig(**kwargs)
    # Ensure common noisy libraries don't hide our logs unless DEBUG
    if level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
