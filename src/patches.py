import logging
import pandas as pd
from unidiff import PatchSet

logger = logging.getLogger(__name__)


def filter_patches(labeled_dataset: pd.DataFrame) -> pd.DataFrame:
    filtered = labeled_dataset[
        (labeled_dataset["n_python_files"] == 1)
        & (~labeled_dataset["has_new_python_file"])
        & (labeled_dataset["status"] != "no_generation")
    ]
    return filtered.copy()


def annotate_patches(labeled_dataset: pd.DataFrame) -> pd.DataFrame:
    df = labeled_dataset.copy()
    parsed = df["patch"].apply(_parse)
    parsed_df = pd.DataFrame(parsed.tolist())
    return pd.concat([df, parsed_df], axis=1)


def _parse(patch_text: str) -> dict:
    try:
        ps = PatchSet(patch_text)
    except Exception as e:
        logger.error(f"Error parsing patch: {e}")
        raise
    python_files = [pf for pf in ps if pf.path.endswith(".py") and not pf.is_added_file]

    return {
        "n_python_files": len(python_files),
        "python_files": [pf.path for pf in python_files],
        "has_new_python_file": any(
            pf.is_added_file and pf.path.endswith(".py") for pf in ps
        ),
    }
