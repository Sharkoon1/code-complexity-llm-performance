import logging
import sys
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def setup_logging(log_path: Path = None, level: str = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]

    level = level.upper()
    if level not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: '{level}'. Must be one of {sorted(VALID_LOG_LEVELS)}"
        )

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    logging.info("Logging initialized")