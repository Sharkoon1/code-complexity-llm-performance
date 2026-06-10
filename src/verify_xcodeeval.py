"""Verfication of the LM-CC implementation to the paper's xCodeEval-APR numbers.

model-free: cached entropies through our block-tree + features.
with-model: recompute entropy with CodeLlama (needs LM_CC_MODEL=codellama/CodeLlama-7b-hf).
"""

import argparse
import json

import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from src.config import PATHS
from src.block_tree import get_code_with_boundaries, CodeBlockProcessor
from src.lm_cc import compute_lm_cc, _get_lmcc
from src.correlation import partial_spearman, best_subgroup_result

DATA_DIR = PATHS.DATA / "verification" / "xcodeeval_apr"
THRESHOLD = 0.67


def _loc(code):
    return len([line for line in code.splitlines() if line.strip()])


def _tree_sig(node):
    """Child-count shape of the tree"""
    return tuple(_tree_sig(child) for child in node.get("children", []))


def _fmt(result):
    if result["n_bins"] is None or pd.isna(result["rho"]):
        return "—"
    return f"{result['rho']:+.3f} (p={result['p']:.4f}, {result['n_bins']} groups)"


def _report_correlation(df):
    rho0, p0 = spearmanr(df["lm_cc"], df["pass1"])
    rhop, pp = partial_spearman(df, "lm_cc", "pass1", "loc")
    sub0 = best_subgroup_result(df, "lm_cc", "pass1", control=None)
    subp = best_subgroup_result(df, "lm_cc", "pass1", control="loc")
    print(
        f"  sample   zero {rho0:+.3f} (p={p0:.3f}), partial|loc {rhop:+.3f} (p={pp:.3f})"
    )
    print(f"  subgroup zero {_fmt(sub0)}, partial|loc {_fmt(subp)}")


def verify_model_free():
    cache = json.loads((DATA_DIR / "reference_cache.json").read_text())
    pass1 = json.loads((DATA_DIR / "pass1.json").read_text())

    tree_match = lmcc_match = 0
    rows = []
    for entry in cache:
        code, _, spans = get_code_with_boundaries(
            entry["tokens"], entry["entropies"], threshold=THRESHOLD
        )
        tree = CodeBlockProcessor().parse_code_blocks(code, entry["tokens"], spans)
        if _tree_sig(tree) == _tree_sig(entry["block_tree"]):
            tree_match += 1
        if abs(_get_lmcc(tree) - _get_lmcc(entry["block_tree"])) < 1e-9:
            lmcc_match += 1
        rows.append(
            {
                "lm_cc": _get_lmcc(tree),
                "pass1": pass1[entry["task_id"]]["pass@1"],
                "loc": _loc(entry["code"]),
            }
        )

    n = len(cache)
    print(f"\nmodel-free, {n} programs")
    print(f"  block-tree match: {tree_match}/{n}, LM-CC match: {lmcc_match}/{n}")
    _report_correlation(pd.DataFrame(rows))


def verify_with_model():
    samples = [
        json.loads(line)
        for line in (DATA_DIR / "samples.jsonl").read_text().splitlines()
        if line.strip()
    ]

    rows = []
    lmcc_match = 0
    max_diff = 0.0
    for s in tqdm(samples, desc="LM-CC (CodeLlama)"):
        lmcc = compute_lm_cc(s["code"])["lm_cc_score"]
        max_diff = max(max_diff, abs(lmcc - s["ref_lmcc"]))
        if abs(lmcc - s["ref_lmcc"]) < 1e-6:
            lmcc_match += 1
        rows.append({"lm_cc": lmcc, "pass1": s["pass1"], "loc": _loc(s["code"])})

    n = len(samples)
    print(f"\nwith model, {n} programs")
    print(f"  LM-CC matches reference: {lmcc_match}/{n} (max diff {max_diff:.4g})")
    _report_correlation(pd.DataFrame(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["model-free", "with-model"], default="model-free"
    )
    args = parser.parse_args()
    if args.mode == "model-free":
        verify_model_free()
    else:
        verify_with_model()


if __name__ == "__main__":
    main()
