from datasets import load_dataset
import requests
import pandas as pd
from src.config import PATHS
import logging
from tqdm import tqdm
from src.shared import cache_path

logger = logging.getLogger(__name__)


def load_swe_bench(split: str = PATHS.SWEBENCH_SPLIT) -> pd.DataFrame:
    try:
        ds = load_dataset(PATHS.SWEBENCH_DATASET, split=split, revision=PATHS.REVISION)
    except Exception as e:
        logger.error(f"Error fetching {PATHS.SWEBENCH_DATASET} ({split}): {e}")
        raise
    return ds.to_pandas()


def load_swe_bench_live(split: str = PATHS.SWEBENCH_LIVE_SPLIT) -> pd.DataFrame:
    try:
        ds = load_dataset(PATHS.SWEBENCH_LIVE_DATASET, split=split)
    except Exception as e:
        logger.error(f"Error fetching {PATHS.SWEBENCH_LIVE_DATASET} ({split}): {e}")
        raise
    return ds.to_pandas()


def fetch_swe_bench_submission_labels(predicition_set: str) -> pd.DataFrame:
    url = (
        "https://raw.githubusercontent.com/SWE-bench/experiments/"
        f"{PATHS.EXPERIMENTS_REVISION}/evaluation/verified/"
        f"{predicition_set}"
        "results/results.json"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        results = response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching predictions for {predicition_set}: {e}")
        raise

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

    return df


def fetch_swe_bench_live_submission_labels(
    submission: str, revision: str = PATHS.SUBMISSION_REPO_REVISION
) -> pd.DataFrame:
    base = (
        "https://raw.githubusercontent.com/SWE-bench-Live/submission/"
        f"{revision}/{submission}"
    )
    try:
        preds_response = requests.get(f"{base}/preds.json", timeout=30)
        preds_response.raise_for_status()
        results_response = requests.get(f"{base}/results.json", timeout=30)
        results_response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error fetching submission {submission}: {e}")
        raise

    preds = preds_response.json()
    results = results_response.json()
    resolved = set(results.get("resolved_ids", []))
    empty_patch = set(results.get("empty_patch_ids", []))

    def status_for(iid):
        if iid in resolved:
            return "resolved"
        if iid in empty_patch or not (preds[iid].get("model_patch") or "").strip():
            return "no_generation"
        return "unresolved"

    instance_ids = sorted(preds)
    df = pd.DataFrame(
        {
            "instance_id": instance_ids,
            "status": [status_for(i) for i in instance_ids],
        }
    )
    df["resolved"] = df["status"] == "resolved"

    return df


def list_verified_submissions(revision: str = PATHS.EXPERIMENTS_REVISION) -> list[str]:
    url = (
        "https://api.github.com/repos/SWE-bench/experiments/"
        f"contents/evaluation/verified?ref={revision}"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        entries = response.json()
    except requests.RequestException as e:
        logger.error(f"Error listing verified submissions: {e}")
        raise
    return sorted(entry["name"] for entry in entries if entry.get("type") == "dir")


def _fetch_resolved_ids(submission: str, revision: str) -> set[str] | None:
    url = (
        "https://raw.githubusercontent.com/SWE-bench/experiments/"
        f"{revision}/evaluation/verified/{submission}/results/results.json"
    )
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return set(response.json().get("resolved", []))
    except (requests.RequestException, ValueError) as e:
        logger.warning(f"Skipping {submission}: no usable results.json ({e}).")
        return None


def fetch_all_swe_bench_submissions(
    task_ids: list[str],
    revision: str = PATHS.EXPERIMENTS_REVISION,
    force: bool = False,
) -> pd.DataFrame:
    cache = PATHS.PREDICTIONS_CACHE / f"verified_{revision}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    rows = []
    for submission in tqdm(list_verified_submissions(revision)):
        resolved = _fetch_resolved_ids(submission, revision)
        if resolved is None:
            continue
        rows.extend(
            {"instance_id": iid, "model": submission, "resolved": iid in resolved}
            for iid in task_ids
        )

    predictions = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(cache)
    logger.info(
        f"Loaded {predictions['model'].nunique()} verified submissions "
        f"over {len(task_ids)} tasks -> {cache.name}."
    )
    return predictions


def aggregate_resolution_rate(predictions: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        predictions.groupby("instance_id")["resolved"]
        .agg(n_models="count", n_resolved="sum")
        .reset_index()
    )
    aggregate["resolution_rate"] = aggregate["n_resolved"] / aggregate["n_models"]
    return aggregate


def _fetch_github_pre_patch_file(repo: str, base_commit: str, file_path: str) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{file_path}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Error fetching {file_path} from {repo}@{base_commit}: {e}")
        raise
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
            logger.error(f"Error fetching {row['instance_id']}: {e}")
            continue

        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(code, encoding="utf-8")
