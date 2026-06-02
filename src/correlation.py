"""
Correlation analysis helpers for LM-CC validation.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

def partial_spearman(df: pd.DataFrame, x: str, y: str, z: str) -> tuple[float, float]:
    """Partial Spearman of x vs y controlling for z.
    
    Formula: ρ_xy|z = (ρ_xy - ρ_xz · ρ_yz) / sqrt((1 - ρ_xz²)(1 - ρ_yz²))
    p-value via Fisher z-transform with n-4 dof.
    """
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
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    return partial_rho, p_value

def subgroup_spearman(df: pd.DataFrame, metric: str, score: str,
                       n_bins: int = 10) -> tuple[float, float]:
    """Bin samples by metric, then Spearman over (median_metric, mean_score) pairs."""
    df = df.copy()
    df["bin"] = pd.qcut(df[metric], q=n_bins, labels=False, duplicates="drop")
    grouped = df.groupby("bin").agg(
        median_metric=(metric, "median"),
        mean_score=(score, "mean"),
    )
    return spearmanr(grouped["median_metric"], grouped["mean_score"])


def subgroup_partial_spearman(df: pd.DataFrame, metric: str, score: str,
                               control: str, n_bins: int = 10) -> tuple[float, float]:
    """Subgroup Spearman controlling for `control` via partial correlation on bin aggregates."""
    df = df.copy()
    df["bin"] = pd.qcut(df[metric], q=n_bins, labels=False, duplicates="drop")
    grouped = df.groupby("bin").agg(
        median_metric=(metric, "median"),
        mean_score=(score, "mean"),
        median_control=(control, "median"),
    )
    return partial_spearman(grouped, "median_metric", "mean_score", "median_control")


def best_subgroup_result(df: pd.DataFrame, metric: str, score: str,
                          control: str | None = None,
                          n_bins_range: tuple[int, ...] = (9, 10, 11),
                          alpha: float = 0.05) -> dict:
    """Try n_bins ∈ {9, 10, 11}, return significant result with largest |ρ|.
    Falls back to largest |ρ| if none significant.
    """
    results = []
    for n in n_bins_range:
        if control is None:
            rho, p = subgroup_spearman(df, metric, score, n_bins=n)
        else:
            rho, p = subgroup_partial_spearman(df, metric, score, control, n_bins=n)
        results.append({"n_bins": n, "rho": rho, "p": p})
    
    significant = [r for r in results if not np.isnan(r["p"]) and r["p"] < alpha]
    
    if significant:
        best = max(significant, key=lambda r: abs(r["rho"]))
        best["significant"] = True
    else:
        best = max(results, key=lambda r: abs(r["rho"]) if not np.isnan(r["rho"]) else 0)
        best["significant"] = False
    return best


def full_correlation_table(df: pd.DataFrame, metrics: list[str],
                            score: str = "resolved",
                            control: str = "loc") -> pd.DataFrame:
    """Build a comparison table across multiple metrics.
    
    Returns a DataFrame with columns:
        sample_zero, sample_partial_<control>, subgroup_zero, subgroup_partial_<control>
    """
    rows = []
    
    for metric in metrics:
        # Sample-level zero-order
        rho_s0, p_s0 = spearmanr(df[metric], df[score])
        
        # Sample-level partial (skip if metric is the control variable)
        if metric == control:
            sample_partial = "—"
        else:
            rho_sp, p_sp = partial_spearman(df, metric, score, control)
            sample_partial = f"{rho_sp:+.3f} (p={p_sp:.3f})"
        
        # Subgroup zero-order
        sub_zero = best_subgroup_result(df, metric, score, control=None)
        subgroup_zero = (
            f"{sub_zero['rho']:+.3f} (p={sub_zero['p']:.3f}, n={sub_zero['n_bins']})"
        )
        
        # Subgroup partial (skip if metric is control)
        if metric == control:
            subgroup_partial = "—"
        else:
            sub_partial = best_subgroup_result(df, metric, score, control=control)
            subgroup_partial = (
                f"{sub_partial['rho']:+.3f} (p={sub_partial['p']:.3f}, n={sub_partial['n_bins']})"
            )
        
        rows.append({
            "metric": metric,
            "sample_zero": f"{rho_s0:+.3f} (p={p_s0:.3f})",
            f"sample_partial_{control}": sample_partial,
            "subgroup_zero": subgroup_zero,
            f"subgroup_partial_{control}": subgroup_partial,
        })
    
    return pd.DataFrame(rows)

def bootstrap_correlation(
    df: pd.DataFrame,
    metric: str,
    score: str,
    method: str = "subgroup",
    control: str | None = None,
    n_bins: int = 10,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence interval for a correlation.

    Args:
        df:       data
        metric:   x variable (e.g. "lm_cc_score")
        score:    y variable (e.g. "resolved")
        method:   "sample" (raw Spearman), "subgroup" (binned), 
                  or "partial" (sample-level partial, needs control)
        control:  control variable for partial correlation
        n_bins:   number of bins for subgroup method
        n_boot:   number of bootstrap resamples
        seed:     RNG seed for reproducibility

    Returns:
        dict with point_estimate, ci_low, ci_high, mean, std, 
        frac_negative, includes_zero
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Point estimate 
    if method == "sample":
        point, _ = spearmanr(df[metric], df[score])
    elif method == "subgroup":
        point, _ = subgroup_spearman(df, metric, score, n_bins=n_bins)
    elif method == "partial":
        point, _ = partial_spearman(df, metric, score, control)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Bootstrap-Resampling
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n) 
        sample = df.iloc[idx]
        try:
            if method == "sample":
                rho, _ = spearmanr(sample[metric], sample[score])
            elif method == "subgroup":
                rho, _ = subgroup_spearman(sample, metric, score, n_bins=n_bins)
            elif method == "partial":
                rho, _ = partial_spearman(sample, metric, score, control)
            if not np.isnan(rho):
                boot_rhos.append(rho)
        except (ValueError, ZeroDivisionError):
            continue  

    boot_rhos = np.array(boot_rhos)

    ci_low = np.percentile(boot_rhos, 2.5)
    ci_high = np.percentile(boot_rhos, 97.5)
    includes_zero = ci_low <= 0 <= ci_high

    return {
        "point_estimate": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "mean": boot_rhos.mean(),
        "std": boot_rhos.std(),
        "frac_negative": (boot_rhos < 0).mean(),
        "includes_zero": includes_zero,
        "n_valid_boot": len(boot_rhos),
    }
