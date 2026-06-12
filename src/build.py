import pandas as pd
import logging
from pathlib import Path
from src.config import PATHS
from src.patches import annotate_patches, filter_patches

logger = logging.getLogger(__name__)


def _build_filtered_dataset(tasks_path: Path) -> pd.DataFrame:
    try:
        tasks = pd.read_parquet(tasks_path)
    except OSError as e:
        logger.error(f"Error reading {tasks_path}: {e}")
        raise

    filtered = filter_patches(annotate_patches(tasks))
    logger.info(
        f"{len(filtered)}/{len(tasks)} tasks kept (single Python file, no new file)."
    )
    return filtered


def build_swe_bench_filtered_dataset() -> pd.DataFrame:
    return _build_filtered_dataset(PATHS.SWEBENCH_TASKS)


def build_swe_bench_live_filtered_dataset() -> pd.DataFrame:
    return _build_filtered_dataset(PATHS.SWEBENCH_LIVE_TASKS)
