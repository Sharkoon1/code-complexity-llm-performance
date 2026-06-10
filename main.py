from src.load import (
    load_swe_bench,
    fetch_model_bench_predictions,
    fetch_all_model_predictions,
    aggregate_resolution_rate,
    fetch_all_pre_patch_files,
)
from src.build import build_labeled_dataset
from src.patches import  annotate_patches, filter_patches
from src.logging import setup_logging
from src.config import  PATHS
from src.metrics import compute_whole_file_metrics, compute_function_metrics
from src.sink import sink
import pandas as pd

def main():
    setup_logging(PATHS.LOGS, "info")

    tasks = load_swe_bench().pipe(sink, PATHS.SWEBENCH_TASKS)
    fetch_model_bench_predictions("20241202_agentless-1.5_claude-3.5-sonnet-20241022/").pipe(sink, PATHS.AGENTLESS_LABELS)
    fetch_model_bench_predictions("20240620_sweagent_claude3.5sonnet/").pipe(sink, PATHS.SWEAGENT_LABELS)

    predictions = fetch_all_model_predictions(tasks["instance_id"].tolist())
    predictions.pipe(sink, PATHS.ALL_PREDICTIONS)
    aggregate_resolution_rate(predictions).pipe(sink, PATHS.TASK_DIFFICULTY)

    filtered_labeled_dataset = (
        build_labeled_dataset()
        .pipe(sink, PATHS.LABELED_DATASET)
        .pipe(annotate_patches)
        .pipe(filter_patches)
        .pipe(sink, PATHS.FILTERED_DATASET)
        )   
    
    fetch_all_pre_patch_files(filtered_labeled_dataset)
    compute_whole_file_metrics(filtered_labeled_dataset).pipe(sink, PATHS.WHOLE_FILE_RESULT_DATASET)
    compute_function_metrics(filtered_labeled_dataset).pipe(sink, PATHS.FUNCTION_RESULT_DATASET)


if __name__ == "__main__":
    main()
