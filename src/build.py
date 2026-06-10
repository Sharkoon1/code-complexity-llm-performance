import pandas as pd
import logging
from src.config import PATHS

logger = logging.getLogger(__name__)


def build_labeled_dataset() -> pd.DataFrame:
    try:
        tasks = pd.read_parquet(PATHS.SWEBENCH_TASKS)
    except OSError as e:
        logger.error(f"Error reading {PATHS.SWEBENCH_TASKS}: {e}")
        raise
    try:
        labels = pd.read_parquet(PATHS.AGENTLESS_LABELS)
    except OSError as e:
        logger.error(f"Error reading {PATHS.AGENTLESS_LABELS}: {e}")
        raise

    df = tasks.merge(labels, on="instance_id", how="left")

    unknown_label_ids = set(labels["instance_id"]) - set(tasks["instance_id"])
    if unknown_label_ids:
        raise ValueError(
            f"{len(unknown_label_ids)} label ids do not exist in the swe-bench-dataset"
            f"check dataset version"
        )

    n_missing = df["status"].isna().sum()
    df["status"] = df["status"].fillna("unresolved")
    df["resolved"] = df["resolved"].fillna(False).astype(bool)

    logger.info(f"{n_missing} Tasks without explicit label -> mark as 'unresolved'")
    logger.info(f"Status distribution:\n{df['status'].value_counts().to_string()}")
    logger.info(f"Resolve rate: {df['resolved'].mean():.1%}")

    try:
        difficulty = pd.read_parquet(PATHS.TASK_DIFFICULTY)
    except OSError as e:
        logger.error(f"Error reading {PATHS.TASK_DIFFICULTY}: {e}")
        raise

    df = df.merge(difficulty, on="instance_id", how="left")
    n_missing_rate = df["resolution_rate"].isna().sum()
    if n_missing_rate:
        logger.warning(
            f"{n_missing_rate} tasks without a resolution_rate. Check the predictions loader"
        )
    logger.info(
        f"Mean resolution_rate across all agents: {df['resolution_rate'].mean():.1%}"
    )

    return df
