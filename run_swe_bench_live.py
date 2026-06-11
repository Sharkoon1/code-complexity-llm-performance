from src.load import (
    load_swe_bench_live,
    fetch_swe_bench_live_submission_labels,
    fetch_all_pre_patch_files,
)
from src.build import build_swe_bench_live_filtered_dataset
from src.logging import setup_logging
from src.config import PATHS
from src.metrics import compute_whole_file_metrics, compute_function_metrics
from src.sink import sink


def run_swe_bench_live_pipeline():
    setup_logging(PATHS.LOGS, "info")

    load_swe_bench_live().pipe(sink, PATHS.SWEBENCH_LIVE_TASKS)
    fetch_swe_bench_live_submission_labels(PATHS.SWEAGENT_CLAUDE37_LIVE_SUBMISSION).pipe(
        sink, PATHS.SWEAGENT_CLAUDE37_LIVE_LABELS
    )
    fetch_swe_bench_live_submission_labels(PATHS.OPENHANDS_QWEN3CODER_LIVE_SUBMISSION).pipe(
        sink, PATHS.OPENHANDS_QWEN3CODER_LIVE_LABELS
    )

    filtered = build_swe_bench_live_filtered_dataset().pipe(sink, PATHS.LIVE_FILTERED_DATASET)

    fetch_all_pre_patch_files(filtered)
    compute_whole_file_metrics(filtered).pipe(sink, PATHS.LIVE_WHOLE_FILE_RESULT_DATASET)
    compute_function_metrics(filtered).pipe(sink, PATHS.LIVE_FUNCTION_RESULT_DATASET)


if __name__ == "__main__":
    run_swe_bench_live_pipeline()
