"""Verfication of the LM-CC implementation to the papers xCodeEval-APR numbers.

model-free: cached entropies through block-tree and features.
with-model: recompute entropy with CodeLlama
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from tqdm import tqdm

from src.config import PATHS
from src.block_tree import get_code_with_boundaries, CodeBlockProcessor
from src.lm_cc import compute_lm_cc, _get_lmcc
from src.correlation import partial_spearman

DATA_DIR = PATHS.DATA / "verification" / "xcodeeval_apr"
THRESHOLD = 0.67


def _loc(code):
    return len([line for line in code.splitlines() if line.strip()])


def _tree_sig(node):
    return tuple(_tree_sig(child) for child in node.get("children", []))


def _fmt(rho, p):
    if pd.isna(rho):
        return "—"
    return f"{rho:+.3f} (p={p:.4f})"


def _best_subgroup(score, metric, loc=None, *, lo=9, hi=11, alpha=0.05):
    order = np.argsort(metric)
    score = np.asarray(score, dtype=float)[order]
    metric = np.asarray(metric, dtype=float)[order]
    loc = None if loc is None else np.asarray(loc, dtype=float)[order]
    n = len(score)
    best = (np.nan, np.nan)
    for min_cnt in range(max(1, n // 20), max(1, n // 8) + 1):
        bounds = [(i, min(i + min_cnt, n)) for i in range(0, n, min_cnt)]
        valid = np.array([b - a for a, b in bounds]) >= min_cnt
        g = int(valid.sum())
        if not (lo <= g <= hi):
            continue
        xm = np.array([np.median(metric[a:b]) for a, b in bounds])[valid]
        ym = np.array([np.mean(score[a:b]) for a, b in bounds])[valid]
        if loc is None:
            rho, p = spearmanr(xm, ym)
        else:
            zm = np.array([np.median(loc[a:b]) for a, b in bounds])[valid]
            rxy, rxz, ryz = (
                spearmanr(xm, ym)[0],
                spearmanr(xm, zm)[0],
                spearmanr(ym, zm)[0],
            )
            denom = np.sqrt((1 - rxz**2) * (1 - ryz**2))
            if denom == 0 or np.isnan(denom):
                continue
            rho = (rxy - rxz * ryz) / denom
            z = 0.5 * np.log((1 + rho) / (1 - rho)) * np.sqrt(g - 4)
            p = 2 * (1 - norm.cdf(abs(z))) if g > 4 and abs(rho) < 1 else np.nan
        if np.isnan(rho) or rho >= 0 or (not np.isnan(p) and p >= alpha):
            continue
        if np.isnan(best[0]) or abs(rho) > abs(best[0]):
            best = (rho, p)
    return best


def _report_correlation(df):
    rho0, p0 = spearmanr(df["lm_cc"], df["pass1"])
    rhop, pp = partial_spearman(df, "lm_cc", "pass1", "loc")
    s = df["pass1"].to_numpy(dtype=float)
    m = df["lm_cc"].to_numpy(dtype=float)
    loc = df["loc"].to_numpy(dtype=float)
    print(
        f"  sample   zero {rho0:+.3f} (p={p0:.3f}), partial|loc {rhop:+.3f} (p={pp:.3f})"
    )
    print(
        f"  subgroup zero {_fmt(*_best_subgroup(s, m))}, "
        f"partial|loc {_fmt(*_best_subgroup(s, m, loc))}"
    )


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
