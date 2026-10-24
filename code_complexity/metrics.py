"""Complexity metrics for Python source code (whole file or function source).

Core metrics -- classical (radon, complexipy), AST-based (nesting, tokens, counts) and
graph-based (PyCG call graph, Scalpel program dependence graph) -- need only the core
dependencies. LM-CC is optional (``code-complexity[lmcc]``) and is imported on demand.
"""

import ast
import io
import logging
import tokenize
from pathlib import Path

from complexipy import code_complexity
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze

from code_complexity.graphs import compute_graph_metrics
from code_complexity.preprocess import normalize as normalize_source

logger = logging.getLogger(__name__)


def metrics_for_code(code: str, lm_cc: bool = True) -> dict:
    """All complexity metrics (classical + LM-CC + graph) for a single code string.

    ``code`` is expected to be normalized already (see :func:`measure_code`). With
    ``lm_cc=False`` the LM-CC features are skipped and torch is never imported.
    """
    metrics = {
        **_compute_cyclomatic(code),
        **_compute_cognitive(code),
        **_compute_maintainability(code),
        **_compute_halstead(code),
        **_compute_loc(code),
        **_compute_nesting(code),
        **_compute_token_count(code),
        **_compute_function_count(code),
        **_compute_branches(code),
    }
    if lm_cc:
        from code_complexity.lm_cc import compute_lm_cc  # optional extra: torch, transformers

        metrics.update(compute_lm_cc(code))
    metrics.update(compute_graph_metrics(code))
    return metrics


def measure_code(code: str, *, normalize: bool = True, lm_cc: bool = False) -> dict:
    """Whole-file metrics for a Python source string, as computed in the project work.

    With ``normalize=True`` (default) the source is first stripped of docstrings and
    decorators and formatted with black, exactly like the SWE-bench pipeline; all
    metrics (``loc``, ``nesting_max``, ``pdg_max_fan_in``, ``cg_max_out_degree``, ...)
    refer to the normalized code. LM-CC is off by default.

    Raises ``ValueError`` when the source cannot be normalized (black failed or nothing
    is left); parser errors of un-normalized code propagate.
    """
    if normalize:
        code = normalize_source(code)
        if code is None:
            raise ValueError("preprocessing (black) failed")
    return metrics_for_code(code, lm_cc=lm_cc)


def measure_file(path: str | Path, **kwargs) -> dict:
    """Whole-file metrics for a Python file (read as UTF-8); see :func:`measure_code`."""
    return measure_code(Path(path).read_text(encoding="utf-8"), **kwargs)


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
