from src.load import (
    load_swe_bench,
    fetch_swe_bench_submission_labels,
    fetch_all_swe_bench_submissions,
    aggregate_resolution_rate,
    fetch_all_pre_patch_files,
)
from src.build import build_swe_bench_filtered_dataset
from src.logging import setup_logging
from src.config import PATHS
from src.metrics import compute_whole_file_metrics, compute_function_metrics
from src.sink import sink


def run_swe_bench_pipeline():
    setup_logging(PATHS.LOGS, "info")

    tasks = load_swe_bench().pipe(sink, PATHS.SWEBENCH_TASKS)
    fetch_swe_bench_submission_labels(PATHS.SWEAGENT_CLAUDE37_VERIFIED_SUBMISSION).pipe(
        sink, PATHS.SWEAGENT_CLAUDE37_VERIFIED_LABELS
    )
    fetch_swe_bench_submission_labels(PATHS.OPENHANDS_QWEN3CODER_VERIFIED_SUBMISSION).pipe(
        sink, PATHS.OPENHANDS_QWEN3CODER_VERIFIED_LABELS
    )

    predictions = fetch_all_swe_bench_submissions(tasks["instance_id"].tolist())
    predictions.pipe(sink, PATHS.ALL_PREDICTIONS)
    aggregate_resolution_rate(predictions).pipe(sink, PATHS.TASK_DIFFICULTY)

    filtered = build_swe_bench_filtered_dataset().pipe(sink, PATHS.VERIFIED_FILTERED_DATASET)

    fetch_all_pre_patch_files(filtered)
    compute_whole_file_metrics(filtered).pipe(sink, PATHS.VERIFIED_WHOLE_FILE_RESULT_DATASET)
    compute_function_metrics(filtered).pipe(sink, PATHS.VERIFIED_FUNCTION_RESULT_DATASET)


if __name__ == "__main__":
    run_swe_bench_pipeline()
