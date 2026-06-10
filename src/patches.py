import ast
import logging
import pandas as pd
from unidiff import PatchSet

logger = logging.getLogger(__name__)


def filter_patches(labeled_dataset: pd.DataFrame) -> pd.DataFrame:
    filtered = labeled_dataset[
        (labeled_dataset["n_python_files"] == 1)
        & (~labeled_dataset["has_new_python_file"])
        & (labeled_dataset["status"] != "no_generation")
    ]
    return filtered.copy()


def annotate_patches(labeled_dataset: pd.DataFrame) -> pd.DataFrame:
    df = labeled_dataset.copy()
    parsed = df["patch"].apply(_parse)
    parsed_df = pd.DataFrame(parsed.tolist())
    return pd.concat([df, parsed_df], axis=1)


def _parse(patch_text: str) -> dict:
    try:
        ps = PatchSet(patch_text)
    except Exception as e:
        logger.error(f"Error parsing patch: {e}")
        raise
    python_files = [pf for pf in ps if pf.path.endswith(".py") and not pf.is_added_file]

    return {
        "n_python_files": len(python_files),
        "python_files": [pf.path for pf in python_files],
        "has_new_python_file": any(
            pf.is_added_file and pf.path.endswith(".py") for pf in ps
        ),
    }


def extract_patched_functions(
    patch_text: str, pre_patch_source: str, file_path: str
) -> str | None:
    patch_set = PatchSet(patch_text)
    affected_lines = set()
    for patched_file in patch_set:
        if patched_file.path != file_path:
            continue
        for hunk in patched_file:
            removed = {
                line.source_line_no
                for line in hunk
                if line.is_removed and line.source_line_no
            }
            affected_lines |= removed or {
                line.source_line_no
                for line in hunk
                if line.is_context and line.source_line_no
            }
    if not affected_lines:
        return None

    try:
        tree = ast.parse(pre_patch_source)
    except SyntaxError:
        return None

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    top_level_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and isinstance(parents.get(node), (ast.Module, ast.ClassDef))
    ]
    selected = sorted(
        (
            function
            for function in top_level_functions
            if affected_lines
            & set(range(function.lineno, (function.end_lineno or function.lineno) + 1))
        ),
        key=lambda function: function.lineno,
    )
    if not selected:
        return None
    return "\n\n".join(
        ast.get_source_segment(pre_patch_source, function) for function in selected
    )
