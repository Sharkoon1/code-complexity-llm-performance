import pandas as pd
import ast
import logging
from dataclasses import dataclass
from src.config import PATHS
from src.shared import cache_path
from src.lm_cc import compute_lm_cc
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class MetricResult:
    cyclomatic: float
  

def compute_metrics(filtered_dataset: pd.DataFrame) -> pd.DataFrame:
    metrics_list = []

    for _, row in tqdm(filtered_dataset.iterrows(), total=len(filtered_dataset)):
        cache = cache_path(row["repo"], row["base_commit"], row["python_files"][0])
        code = cache.read_text()

        metrics = {"instance_id": row["instance_id"]}
        
        try:
            metrics.update({
                "parsable": True,
                **_compute_cyclomatic(code),
                **_compute_halstead(code),
                **_compute_loc(code),
                **compute_lm_cc(code),
            })
        except Exception as e:
            metrics["parsable"] = False
            metrics["error"] = f"{type(e).__name__}: {e}"
        
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_parquet(PATHS.RESULTS / "result.parquet")
    return metrics_df

def _compute_cyclomatic(code: str) -> dict:
    results = cc_visit(code)

    if not results:
        # file without functions only imports / constants
        return {"cc_avg": 0, "cc_max": 0, "cc_sum": 0}

    complexities = [r.complexity for r in results]
    return {
        "cc_avg": sum(complexities) / len(complexities),
        "cc_max": max(complexities),
        "cc_sum": sum(complexities),
    }


def _compute_halstead(code: str) -> dict:
    h = h_visit(code).total
    return {
        "halstead_volume": h.volume,
        "halstead_difficulty": h.difficulty,
        "halstead_effort": h.effort,
    }


def _compute_loc(code: str) -> dict:
    raw = analyze(code)
    return {
        "loc": raw.loc,
        "lloc": raw.lloc,
        "sloc": raw.sloc,
    }
