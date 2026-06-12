"""Correlation analysis"""

import numpy as np
import pandas as pd
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


def fixed_subgroup_corr(
    score, metric, loc=None, *, group_range: tuple[int, int] = (9, 11)
) -> tuple[float, float]:
    score = np.asarray(score, dtype=float)
    metric = np.asarray(metric, dtype=float)
    n = len(score)
    low, high = group_range
    if n < low:
        return np.nan, np.nan

    order = np.argsort(metric)
    score, metric = score[order], metric[order]
    n_groups = min(range(low, high + 1), key=lambda count: (n % count, abs(count - 10)))
    groups = np.array_split(np.arange(n), n_groups)

    metric_median = np.array([np.median(metric[idx]) for idx in groups])
    score_mean = np.array([np.mean(score[idx]) for idx in groups])

    if loc is None:
        rho, p = spearmanr(metric_median, score_mean)
        return float(rho), float(p)

    loc = np.asarray(loc, dtype=float)[order]
    loc_median = np.array([np.median(loc[idx]) for idx in groups])
    rho_metric_score, _ = spearmanr(metric_median, score_mean)
    rho_metric_loc, _ = spearmanr(metric_median, loc_median)
    rho_score_loc, _ = spearmanr(score_mean, loc_median)
    denom = np.sqrt((1 - rho_metric_loc**2) * (1 - rho_score_loc**2))
    if denom == 0 or np.isnan(denom):
        return np.nan, np.nan

    partial_rho = (rho_metric_score - rho_metric_loc * rho_score_loc) / denom
    if n_groups <= 4 or abs(partial_rho) >= 1.0:
        return float(partial_rho), np.nan

    z_stat = 0.5 * np.log((1 + partial_rho) / (1 - partial_rho)) * np.sqrt(n_groups - 4)
    return float(partial_rho), float(2 * (1 - norm.cdf(abs(z_stat))))


def _format_corr(rho: float, p: float, alpha: float) -> str:
    if np.isnan(rho):
        return "—"
    text = f"{rho:+.3f}"
    if text in ("+0.000", "-0.000"):
        text = "0.000"
    return text + ("*" if (not np.isnan(p) and p < alpha) else "")


def full_correlation_table(
    df: pd.DataFrame,
    metrics: list[str],
    score: str = "resolved",
    control: str = "loc",
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-metric correlations (sample + subgroup, zero-order + partial)."""
    rows = []
    for metric in metrics:
        present = [col for col in (metric, score, control) if col in df.columns]
        data = df.dropna(subset=present)
        score_vals = data[score].to_numpy(dtype=float)
        metric_vals = data[metric].to_numpy(dtype=float)
        loc_vals = (
            data[control].to_numpy(dtype=float) if control in data.columns else None
        )
        is_control = metric == control

        rows.append(
            {
                "metric": metric,
                "sample_zero": _format_corr(
                    *spearmanr(metric_vals, score_vals), alpha
                ),
                f"sample_partial_{control}": (
                    "—"
                    if is_control
                    else _format_corr(
                        *partial_spearman(data, metric, score, control), alpha
                    )
                ),
                "subgroup_zero": _format_corr(
                    *fixed_subgroup_corr(score_vals, metric_vals), alpha
                ),
                f"subgroup_partial_{control}": (
                    "—"
                    if is_control
                    else _format_corr(
                        *fixed_subgroup_corr(score_vals, metric_vals, loc_vals), alpha
                    )
                ),
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
        score_vals = group[score].to_numpy(dtype=float)
        metric_vals = group[metric].to_numpy(dtype=float)
        loc_vals = group[control].to_numpy(dtype=float)

        sample_rho, _ = spearmanr(group[metric], group[score])
        partial_rho, _ = partial_spearman(group, metric, score, control)
        subgroup_rho, _ = fixed_subgroup_corr(score_vals, metric_vals)
        subgroup_partial_rho, _ = fixed_subgroup_corr(score_vals, metric_vals, loc_vals)

        rows.append(
            {
                "agent": agent,
                "resolve_rate": group[score].mean(),
                "sample_zero": sample_rho,
                "sample_partial_loc": partial_rho,
                "subgroup_zero": subgroup_rho,
                "subgroup_partial_loc": subgroup_partial_rho,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("resolve_rate", ascending=False)
        .reset_index(drop=True)
    )
