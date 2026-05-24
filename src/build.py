import pandas as pd
import logging
from src.config import PATHS

logger = logging.getLogger(__name__)

def build_labeled_dataset() -> pd.DataFrame:
    tasks = pd.read_parquet(PATHS.SWEBENCH_TASKS)
    labels = pd.read_parquet(PATHS.AGENTLESS_LABELS)

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

    logger.info(f"{n_missing} Tasks without explicit labeel -> mark as 'unresolved'")
    logger.info(f"Status distribution:\n{df['status'].value_counts().to_string()}")
    logger.info(f"Resolve rate: {df['resolved'].mean():.1%}")

    return df