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
    SWEBENCH_DATASET: str = "princeton-nlp/SWE-bench_Verified"
    SWEBENCH_SPLIT: str = "test"

    SWEBENCH_TASKS: Path = RAW / "swebench_verified.parquet"
    ALL_PREDICTIONS: Path = RAW / "swebench_all_predictions.parquet"
    TASK_DIFFICULTY: Path = INTERIM / "swebench_task_difficulty.parquet"

    SWEAGENT_CLAUDE37_VERIFIED_SUBMISSION: str = "20250225_sweagent_claude-3-7-sonnet/"
    OPENHANDS_QWEN3CODER_VERIFIED_SUBMISSION: str = "20250805_openhands-Qwen3-Coder-480B-A35B-Instruct/"
    SWEAGENT_CLAUDE37_VERIFIED_LABELS: Path = RAW / "sweagent_claude37_verified_labels.parquet"
    OPENHANDS_QWEN3CODER_VERIFIED_LABELS: Path = RAW / "openhands_qwen3coder_verified_labels.parquet"

    VERIFIED_FILTERED_DATASET: Path = INTERIM / "swebench_verified_filtered.parquet"
    VERIFIED_WHOLE_FILE_RESULT_DATASET: Path = RESULT / "swe_bench_verified_whole_file_result.parquet"
    VERIFIED_FUNCTION_RESULT_DATASET: Path = RESULT / "swe_bench_verified_function_result.parquet"

    SWEBENCH_LIVE_DATASET: str = "SWE-bench-Live/SWE-bench-Live"
    SWEBENCH_LIVE_SPLIT: str = "lite"
    SUBMISSION_REPO_REVISION: str = "main"
    SWEBENCH_LIVE_TASKS: Path = RAW / "swebench_live_lite.parquet"

    SWEAGENT_CLAUDE37_LIVE_SUBMISSION: str = "submissions/lite/20250501-sweagent-claude37"
    OPENHANDS_QWEN3CODER_LIVE_SUBMISSION: str = "submissions/lite/20250725-openhands-Qwen3-Coder-480B-A35B"
    SWEAGENT_CLAUDE37_LIVE_LABELS: Path = RAW / "sweagent_claude37_live_labels.parquet"
    OPENHANDS_QWEN3CODER_LIVE_LABELS: Path = RAW / "openhands_qwen3coder_live_labels.parquet"

    LIVE_FILTERED_DATASET: Path = INTERIM / "swebench_live_filtered.parquet"
    LIVE_WHOLE_FILE_RESULT_DATASET: Path = RESULT / "swe_bench_live_whole_file_result.parquet"
    LIVE_FUNCTION_RESULT_DATASET: Path = RESULT / "swe_bench_live_function_result.parquet"


PATHS = Config()
