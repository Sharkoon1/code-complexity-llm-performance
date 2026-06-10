from src.config import PATHS
from pathlib import Path


def cache_path(repo: str, base_commit: str, file_path: str) -> Path:
    safe_repo = repo.replace("/", "__")
    safe_file = file_path.replace("/", "__")
    return PATHS.CODE_CACHE / safe_repo / base_commit / safe_file
