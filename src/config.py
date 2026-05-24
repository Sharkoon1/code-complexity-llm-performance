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
    LOGS: Path = ROOT / "logs" / "pipeline.log"

    NOTEBOOKS: Path = ROOT / "notebooks"
    RESULTS: Path = ROOT / "results"

    REVISION: str = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"

    SWEBENCH_TASKS: Path = RAW / "swebench_verified.parquet"
    AGENTLESS_LABELS: Path = RAW / "agentless_claude35sonnet_labels.parquet"
    LABELED_DATASET: Path = INTERIM / "swebench_claude35sonnet.parquet"
    RESULTS_DATASET: Path = RESULTS / "result.parquet"
    FILTERED_DATASET: Path = INTERIM / "swebench_claude35sonnet_filtered.parquet" 
    RESULT_DATASET: Path = RESULTS / "result.parquet"
PATHS = Config()  