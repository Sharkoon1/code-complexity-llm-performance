"""SWE-bench pipeline step: metrics per task from the cached pre-patch source files."""

import logging

import pandas as pd
from tqdm import tqdm

from code_complexity.metrics import metrics_for_code
from code_complexity.patches import extract_patched_functions
from code_complexity.preprocess import normalize
from code_complexity.shared import cache_path

logger = logging.getLogger(__name__)


def compute_whole_file_metrics(filtered_dataset: pd.DataFrame) -> pd.DataFrame:
    metrics_list = []

    for _, row in tqdm(filtered_dataset.iterrows(), total=len(filtered_dataset)):
        cache = cache_path(row["repo"], row["base_commit"], row["python_files"][0])
        try:
            raw_code = cache.read_text()
        except OSError as e:
            logger.error(f"Error reading cache for {row['instance_id']}: {e}")
            raise
        metrics = {"instance_id": row["instance_id"]}

        try:
            code = normalize(raw_code)
            if code is None:
                raise ValueError("preprocessing (black) failed")
            metrics.update({"parsable": True, **metrics_for_code(code)})
        except Exception as e:
            logger.error(f"Error computing metrics for {row['instance_id']}: {e}")
            metrics["parsable"] = False
            metrics["error"] = f"{type(e).__name__}: {e}"

        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)


def compute_function_metrics(filtered_dataset: pd.DataFrame) -> pd.DataFrame:
    metrics_list = []

    for _, row in tqdm(filtered_dataset.iterrows(), total=len(filtered_dataset)):
        metrics = {"instance_id": row["instance_id"]}
        try:
            raw_code = cache_path(
                row["repo"], row["base_commit"], row["python_files"][0]
            ).read_text()
            function_source = extract_patched_functions(
                row["patch"], raw_code, row["python_files"][0]
            )
            if function_source is None:
                metrics.update(has_patched_function=False, parsable=True)
            else:
                code = normalize(function_source)
                if code is None:
                    raise ValueError("preprocessing (black) failed")
                metrics.update(
                    has_patched_function=True,
                    parsable=True,
                    **metrics_for_code(code),
                )
        except Exception as error:
            logger.error(
                f"Error computing function metrics for {row['instance_id']}: {error}"
            )
            metrics["parsable"] = False
            metrics["error"] = f"{type(error).__name__}: {error}"

        metrics_list.append(metrics)

    excluded = sum(1 for entry in metrics_list if not entry.get("has_patched_function"))
    logger.info(
        f"Function-level: {len(metrics_list) - excluded}/{len(metrics_list)} tasks "
        f"have a patched function ({excluded} excluded)."
    )
    return pd.DataFrame(metrics_list)
