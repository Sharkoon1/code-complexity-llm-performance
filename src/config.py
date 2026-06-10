from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    ROOT: Path = Path(__file__).resolve().parent.parent

    DATA: Path = ROOT / "data"
    RAW: Path = DATA / "raw"
    INTERIM: Path = DATA / "interim"
    PROCESSED: Path = DATA / "processed"
    ENTROPIES: Path = DATA / "entropies"
    CODE_CACHE: Path = DATA / "cache" / "code"
    PREDICTIONS_CACHE: Path = DATA / "cache" / "predictions"
    LOGS: Path = ROOT / "logs" / "pipeline.log"

    NOTEBOOKS: Path = ROOT / "notebooks"
    RESULT: Path = ROOT / "results"

    REVISION: str = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
    EXPERIMENTS_REVISION: str = "82be2b5ac4967af9559484e7bffef227e97348af"

    SWEBENCH_TASKS: Path = RAW / "swebench_verified.parquet"
    AGENTLESS_LABELS: Path = RAW / "agentless_claude35sonnet_labels.parquet"
    ALL_PREDICTIONS: Path = RAW / "swebench_all_predictions.parquet"
    TASK_DIFFICULTY: Path = INTERIM / "swebench_task_difficulty.parquet"
    LABELED_DATASET: Path = INTERIM / "swebench_claude35sonnet.parquet"
    FILTERED_DATASET: Path = INTERIM / "swebench_claude35sonnet_filtered.parquet"
    WHOLE_FILE_RESULT_DATASET: Path = RESULT / "swe_bench_whole_file_result.parquet"
    FUNCTION_RESULT_DATASET: Path = RESULT / "swe_bench_function_result.parquet"


PATHS = Config()
