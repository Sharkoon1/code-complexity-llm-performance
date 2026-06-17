import pandas as pd
import ast
import logging
import tokenize
import io
from dataclasses import dataclass
from src.config import PATHS
from src.shared import cache_path
from src.graphs import compute_graph_metrics
from src.lm_cc import compute_lm_cc
from src.preprocess import normalize
from src.patches import extract_patched_functions
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
from complexipy import code_complexity
from tqdm import tqdm

logger = logging.getLogger(__name__)


def metrics_for_code(code: str) -> dict:
    """All complexity metrics (classical + LM-CC + graph) for a single code string."""
    return {
        **_compute_cyclomatic(code),
        **_compute_cognitive(code),
        **_compute_maintainability(code),
        **_compute_halstead(code),
        **_compute_loc(code),
        **_compute_nesting(code),
        **_compute_token_count(code),
        **_compute_function_count(code),
        **_compute_branches(code),
        **compute_lm_cc(code),
        **compute_graph_metrics(code),
    }


def compute_whole_file_metrics(filtered_dataset: pd.DataFrame) -> pd.DataFrame:
    metrics_list = []

    for _, row in tqdm(filtered_dataset.iterrows(), total=len(filtered_dataset)):
        cache = cache_path(row["repo"], row["base_commit"], row["python_files"][0])
        try:
            raw_code = cache.read_text()
        except OSError as e:
            logger.error(f"Error reading cache for {row['instance_id']}: {e}")
            raise
        metrics = {"instance_id": row["instance_id"]}

        try:
            code = normalize(raw_code)
            if code is None:
                raise ValueError("preprocessing (black) failed")
            metrics.update({"parsable": True, **metrics_for_code(code)})
        except Exception as e:
            logger.error(f"Error computing metrics for {row['instance_id']}: {e}")
            metrics["parsable"] = False
            metrics["error"] = f"{type(e).__name__}: {e}"

        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)


def compute_function_metrics(filtered_dataset: pd.DataFrame) -> pd.DataFrame:
    metrics_list = []

    for _, row in tqdm(filtered_dataset.iterrows(), total=len(filtered_dataset)):
        metrics = {"instance_id": row["instance_id"]}
        try:
            raw_code = cache_path(
                row["repo"], row["base_commit"], row["python_files"][0]
            ).read_text()
            function_source = extract_patched_functions(
                row["patch"], raw_code, row["python_files"][0]
            )
            if function_source is None:
                metrics.update(has_patched_function=False, parsable=True)
            else:
                code = normalize(function_source)
                if code is None:
                    raise ValueError("preprocessing (black) failed")
                metrics.update(
                    has_patched_function=True,
                    parsable=True,
                    **metrics_for_code(code),
                )
        except Exception as error:
            logger.error(
                f"Error computing function metrics for {row['instance_id']}: {error}"
            )
            metrics["parsable"] = False
            metrics["error"] = f"{type(error).__name__}: {error}"

        metrics_list.append(metrics)

    excluded = sum(1 for entry in metrics_list if not entry.get("has_patched_function"))
    logger.info(
        f"Function-level: {len(metrics_list) - excluded}/{len(metrics_list)} tasks "
        f"have a patched function ({excluded} excluded)."
    )
    return pd.DataFrame(metrics_list)


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


def _compute_cognitive(code: str) -> dict:
    result = code_complexity(code)
    per_function = [f.complexity for f in result.functions]

    if not per_function:
        return {"cognitive_avg": 0, "cognitive_max": 0, "cognitive_sum": result.complexity}

    return {
        "cognitive_avg": sum(per_function) / len(per_function),
        "cognitive_max": max(per_function),
        "cognitive_sum": result.complexity,
    }


def _compute_maintainability(code: str) -> dict:
    return {"maintainability_index": mi_visit(code, True)}

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


def _max_nesting_depth(node, current_depth=0):
    max_depth = current_depth

    nesting_types = (ast.If, ast.For, ast.While, ast.AsyncFor, ast.Try, ast.With)

    for child in ast.iter_child_nodes(node):
        if isinstance(child, nesting_types):
            child_depth = _max_nesting_depth(child, current_depth + 1)
        else:
            child_depth = _max_nesting_depth(child, current_depth)
        max_depth = max(max_depth, child_depth)

    return max_depth


def _compute_nesting(code: str) -> dict:
    tree = ast.parse(code)
    max_depth = _max_nesting_depth(tree)
    return {"nesting_max": max_depth}


def _compute_token_count(code: str) -> dict:
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    meaningful_types = {tokenize.NAME, tokenize.OP, tokenize.NUMBER, tokenize.STRING}
    meaningful_tokens = [t for t in tokens if t.type in meaningful_types]

    return {
        "n_tokens_total": len(tokens),
        "n_tokens_meaningful": len(meaningful_tokens),
    }


def _compute_function_count(code: str) -> dict:
    tree = ast.parse(code)

    n_funcs = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    n_classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

    return {
        "n_functions": n_funcs,
        "n_classes": n_classes,
    }


def _compute_branches(code: str) -> dict:
    tree = ast.parse(code)

    branch_types = (ast.If, ast.For, ast.While, ast.Try, ast.AsyncFor)
    n_branches = sum(1 for node in ast.walk(tree) if isinstance(node, branch_types))

    return {"n_branches": n_branches}
