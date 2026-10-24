import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def sink(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    logger.info(f"Wrote {len(df)} rows to {path.name}.")
    return df
