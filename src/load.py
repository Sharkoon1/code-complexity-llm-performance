from datasets import load_dataset
import requests
import pandas as pd
from src.config import PATHS
import logging
from tqdm import tqdm
from src.shared import cache_path

logger = logging.getLogger(__name__)

def load_swe_bench():
    ds = load_dataset(
        "princeton-nlp/SWE-bench_Verified",
        split="test",
        revision=PATHS.REVISION,
    )
    df = ds.to_pandas()
    df.to_parquet("data/raw/swebench_verified.parquet")


def fetch_model_bench_predictions(predicition_set: str):
    url = (
        "https://raw.githubusercontent.com/SWE-bench/experiments/"
        "main/evaluation/verified/"
        f"{predicition_set}"
        "results/results.json"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    results = response.json()

    resolved = set(results.get("resolved", []))
    no_generation = set(results.get("no_generation", []))
    no_logs = set(results.get("no_logs", []))
    all_ids = resolved | no_generation | no_logs

    def status_for(iid):
        if iid in resolved:
            return "resolved"
        if iid in no_generation:
            return "no_generation"
        if iid in no_logs:
            return "no_logs"
        return "unresolved"

    df = pd.DataFrame(
        {
            "instance_id": sorted(all_ids),
            "status": [status_for(i) for i in sorted(all_ids)],
        }
    )
    df["resolved"] = df["status"] == "resolved"

    df.to_parquet("data/raw/agentless_claude35sonnet_labels.parquet")


def _fetch_github_pre_patch_file(repo: str, base_commit: str, file_path: str) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{file_path}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def fetch_all_pre_patch_files(labeled_dataset: pd.DataFrame) -> None:
    for _, row in tqdm(labeled_dataset.iterrows(), total=len(labeled_dataset)):
        cache = cache_path(row["repo"], row["base_commit"], row["python_files"][0])
        
        if cache.exists():
            continue     
        try:
            code = _fetch_github_pre_patch_file(
                repo=row["repo"],
                base_commit=row["base_commit"],
                file_path=row["python_files"][0],
            )
        except requests.RequestException as e:
            logger.warning(f"Error on {row['instance_id']}: {e}")
            continue
        
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(code, encoding="utf-8")
