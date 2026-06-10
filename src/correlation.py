"""Subgroup correlation analysis"""

import warnings

import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import norm, spearmanr


def partial_spearman(df: pd.DataFrame, x: str, y: str, z: str) -> tuple[float, float]:
    """Partial Spearman of x vs y given z, p-value via Fisher z."""
    rho_xy, _ = spearmanr(df[x], df[y])
    rho_xz, _ = spearmanr(df[x], df[z])
    rho_yz, _ = spearmanr(df[y], df[z])

    denom = np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2))
    if denom == 0 or np.isnan(denom):
        return np.nan, np.nan

    partial_rho = (rho_xy - rho_xz * rho_yz) / denom
    n = len(df)
    if n <= 4 or abs(partial_rho) >= 1.0:
        return partial_rho, np.nan

    z_stat = 0.5 * np.log((1 + partial_rho) / (1 - partial_rho)) * np.sqrt(n - 4)
    return partial_rho, 2 * (1 - norm.cdf(abs(z_stat)))


def _group_by_metric(score, metric, loc, min_cnt):
    order = np.argsort(metric)
    score, metric = score[order], metric[order]
    loc = None if loc is None else loc[order]
    bounds = [(i, min(i + min_cnt, len(score))) for i in range(0, len(score), min_cnt)]
    mean_score = np.array([np.nanmean(score[a:b]) for a, b in bounds])
    median_metric = np.array([np.nanmedian(metric[a:b]) for a, b in bounds])
    counts = np.array([b - a for a, b in bounds], dtype=int)
    median_loc = (
        None if loc is None else np.array([np.nanmedian(loc[a:b]) for a, b in bounds])
    )
    return mean_score, median_metric, median_loc, counts


def _grouped_corr(score, metric, loc, min_cnt):
    mean_score, median_metric, median_loc, counts = _group_by_metric(
        score, metric, loc, min_cnt
    )
    valid = ~np.isnan(mean_score) & ~np.isnan(median_metric) & (counts >= min_cnt)
    if median_loc is not None:
        valid &= ~np.isnan(median_loc)
    n_valid = int(valid.sum())
    if n_valid < 2:
        return np.nan, np.nan, n_valid

    x, y = median_metric[valid], mean_score[valid]
    if loc is None:
        res = spearmanr(x, y)
        return float(res.correlation), float(res.pvalue), n_valid

    z = median_loc[valid]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pg.partial_corr(
                data=pd.DataFrame({"m": x, "s": y, "l": z}),
                x="m",
                y="s",
                covar="l",
                method="spearman",
            )
        pcol = "p-val" if "p-val" in out.columns else "p_val"
        return float(out["r"].iloc[0]), float(out[pcol].iloc[0]), n_valid
    except Exception:
        return np.nan, np.nan, n_valid


def get_grouped_partial_corr(
    score, metric, loc=None, *, group_range=(9, 11), alpha=0.05, negative_only=True
) -> dict | None:
    score = np.asarray(score, dtype=float)
    metric = np.asarray(metric, dtype=float)
    loc = None if loc is None else np.asarray(loc, dtype=float)
    n = len(score)
    lo_groups, hi_groups = group_range

    best = None
    for min_cnt in range(max(1, n // 20), max(1, n // 8) + 1):
        rho, p, groups = _grouped_corr(score, metric, loc, min_cnt)
        if np.isnan(rho) or np.isnan(p) or not (lo_groups <= groups <= hi_groups):
            continue
        if p >= alpha or (negative_only and rho >= 0):
            continue
        if best is None or abs(rho) > abs(best["rho"]):
            best = {"rho": rho, "p": p, "n_bins": groups, "min_cnt": min_cnt}
    return best


def best_subgroup_result(
    df: pd.DataFrame,
    metric: str,
    score: str,
    control: str | None = None,
    *,
    alpha: float = 0.05,
    negative_only: bool = True,
) -> dict:
    cols = [metric, score] + ([control] if control else [])
    sub = df[cols].dropna()
    loc = None if control is None else sub[control].to_numpy(dtype=float)
    best = get_grouped_partial_corr(
        sub[score].to_numpy(dtype=float),
        sub[metric].to_numpy(dtype=float),
        loc,
        alpha=alpha,
        negative_only=negative_only,
    )
    if best is None:
        return {"rho": np.nan, "p": np.nan, "n_bins": None, "significant": False}
    return {
        "rho": best["rho"],
        "p": best["p"],
        "n_bins": best["n_bins"],
        "significant": True,
    }


def _fmt_subgroup(result: dict) -> str:
    if result["n_bins"] is None or np.isnan(result["rho"]):
        return "—"
    return f"{result['rho']:+.3f} (p={result['p']:.3f}, n={result['n_bins']})"


def _fmt_sample(rho: float, p: float, alpha: float) -> str:
    if np.isnan(rho) or np.isnan(p) or p >= alpha:
        return "—"
    return f"{rho:+.3f} (p={p:.3f})"


def full_correlation_table(
    df: pd.DataFrame,
    metrics: list[str],
    score: str = "resolved",
    control: str = "loc",
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-metric correlations (sample + subgroup, zero-order + partial).

    "—" means no significant result: p >= alpha, an undefined partial (metric
    collinear with the control), or no significant-negative subgroup result.
    """
    rows = []
    for metric in metrics:
        rho_s0, p_s0 = spearmanr(df[metric], df[score])
        sample_zero = _fmt_sample(rho_s0, p_s0, alpha)

        if metric == control:
            sample_partial = "—"
        else:
            sample_partial = _fmt_sample(
                *partial_spearman(df, metric, score, control), alpha
            )

        subgroup_zero = _fmt_subgroup(
            best_subgroup_result(df, metric, score, control=None, alpha=alpha)
        )
        if metric == control:
            subgroup_partial = "—"
        else:
            subgroup_partial = _fmt_subgroup(
                best_subgroup_result(df, metric, score, control=control, alpha=alpha)
            )

        rows.append(
            {
                "metric": metric,
                "sample_zero": sample_zero,
                f"sample_partial_{control}": sample_partial,
                "subgroup_zero": subgroup_zero,
                f"subgroup_partial_{control}": subgroup_partial,
            }
        )

    return pd.DataFrame(rows)


def per_agent_correlation_table(
    df: pd.DataFrame,
    metric: str = "lm_cc_score",
    score: str = "resolved",
    control: str = "loc",
) -> pd.DataFrame:
    """One row per agent with its resolve rate and the four correlations of `metric`.

    Agents that resolved none or all tasks are skipped.
    """
    rows = []
    for agent, group in df.groupby("model"):
        group = group.dropna(subset=[metric, score, control])
        if len(group) < 30 or group[score].nunique() < 2:
            continue
        rho_s0, _ = spearmanr(group[metric], group[score])
        rho_sp, _ = partial_spearman(group, metric, score, control)
        sub_zero = best_subgroup_result(group, metric, score, control=None)
        sub_partial = best_subgroup_result(group, metric, score, control=control)
        rows.append(
            {
                "agent": agent,
                "resolve_rate": group[score].mean(),
                "sample_zero": rho_s0,
                "sample_partial_loc": rho_sp,
                "subgroup_zero": sub_zero["rho"],
                "subgroup_partial_loc": sub_partial["rho"],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("resolve_rate", ascending=False)
        .reset_index(drop=True)
    )
