"""Analysis helpers shared by the SWE-bench result notebooks."""

from pathlib import Path
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

from src.config import PATHS
from src.analysis.correlation import (
    _spearman,
    fixed_subgroup_corr,
    full_correlation_table,
    partial_spearman,
    per_agent_correlation_table,
)

CLASSICAL = [
    "cc_avg",
    "cc_max",
    "cc_sum",
    "cognitive_avg",
    "cognitive_max",
    "cognitive_sum",
    "maintainability_index",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "loc",
    "lloc",
    "nesting_max",
    "token_count",
    "n_tokens_meaningful",
    "function_count",
    "n_functions",
    "n_branches",
]
LM_CC_FEATURES = [
    "lm_cc_max_comp",
    "lm_cc_avg_comp",
    "lm_cc_total_comp",
    "lm_cc_max_branch",
    "lm_cc_avg_branch",
    "lm_cc_total_branch",
]
LM_CC = ["lm_cc_score"]
GRAPH = [
    "cg_n_nodes",
    "cg_n_edges",
    "cg_max_out_degree",
    "cg_longest_chain",
    "cg_density",
    "cg_n_cycles",
    "pdg_longest_chain",
    "pdg_max_defuse_distance",
    "pdg_density",
    "pdg_max_fan_in",
    "pdg_max_fan_out",
    "pdg_mean_degree",
    "pdg_slice_mean",
    "pdg_slice_max",
    "pdg_n_cycles",
    "pdg_control_data_ratio",
    "pdg_control_span_max",
    "pdg_control_span_mean",
    "pdg_unconditional_fraction",
    "pdg_n_control_regions",
    "pdg_control_betweenness_max",
]
ALL_METRICS = CLASSICAL + LM_CC_FEATURES + LM_CC + GRAPH
LONG_METRIC_COLS = ["lm_cc_score", "loc", "cc_sum", "halstead_volume", "nesting_max"]

# Curated, de-duplicated metric selection for the per-metric lollipop plot:
# one representative per collinear family (e.g. cc_sum stands in for cc_avg/max),
# grouped LM-CC -> classical -> call graph -> program dependence graph.
LOLLIPOP_METRICS_NO_DUPLICATES = [
    "lm_cc_score",
    "cc_sum",
    "cognitive_sum",
    "maintainability_index",
    "halstead_difficulty",
    "nesting_max",
    "cg_n_nodes",
    "cg_n_edges",
    "cg_max_out_degree",
    "cg_longest_chain",
    "cg_density",
    "cg_n_cycles",
    "pdg_longest_chain",
    "pdg_max_defuse_distance",
    "pdg_density",
    "pdg_max_fan_in",
    "pdg_max_fan_out",
    "pdg_slice_mean",
    "pdg_n_cycles",
    "pdg_control_data_ratio",
    "pdg_control_span_mean",
    "pdg_unconditional_fraction",
    "pdg_n_control_regions",
    "pdg_control_betweenness_max",
]


def _graph_failed(df: pd.DataFrame) -> pd.Series:
    needed = {"cg_n_nodes", "pdg_unconditional_fraction", "n_functions"}
    if not needed <= set(df.columns):
        return pd.Series(False, index=df.index)
    return (
        (df["cg_n_nodes"].fillna(0) == 0)
        | (df["pdg_unconditional_fraction"].fillna(0) == 0)
    ) & (df["n_functions"].fillna(0) > 0)


def load_metrics(result_path, labels_path=None):
    df = pd.read_parquet(result_path)
    mask = df["parsable"].fillna(False) & df["lm_cc_score"].notna()
    if "has_patched_function" in df.columns:
        mask &= df["has_patched_function"].fillna(False)

    failed = _graph_failed(df)
    mask &= ~failed

    clean = df[mask].copy()
    if labels_path is not None:
        labels = pd.read_parquet(labels_path)[["instance_id", "resolved"]]
        clean = clean.merge(labels, on="instance_id", how="left")
        clean["resolved"] = clean["resolved"].fillna(False)
    return clean


def _available(clean):
    return [m for m in ALL_METRICS if m in clean.columns]


def correlation_report(clean, score="resolved", control="loc"):
    """Full correlation table per metric (sample + subgroup, zero-order + partial)."""
    return full_correlation_table(
        clean, _available(clean), score=score, control=control
    )

def correlation_heatmap(clean, title="", metrics=None, method="spearman", annot_limit=30):
    if metrics is None:
        metrics = _available(clean)

    if "loc" in metrics:
        metrics = ["loc"] + [m for m in metrics if m != "loc"]

    corr = clean[metrics].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    annot = len(metrics) <= annot_limit
    size = max(10, 0.45 * len(metrics))

    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(
        corr,
        mask=mask,
        annot=annot,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.75, "label": f"{method.capitalize()} ρ"},
    )

    plt.title(f"{method.capitalize()} correlation heatmap {title}".strip())
    plt.tight_layout()
    plt.show()

    return corr

def extremes(clean, n=5):
    cols = [
        c
        for c in ["instance_id", "lm_cc_score", "loc", "cc_sum", "nesting_max"]
        if c in clean.columns
    ]
    for label, mask in [
        ("resolved", clean["resolved"] == 1),
        ("not resolved", clean["resolved"] == 0),
    ]:
        for which, sub in [
            ("High", clean[mask].nlargest(n, "lm_cc_score")),
            ("Low", clean[mask].nsmallest(n, "lm_cc_score")),
        ]:
            print(f"=== {which} LM-CC, {label} ===")
            print(sub[cols].to_string(index=False))
            print()


def resolved_vs_not(clean):
    res = clean[clean["resolved"] == 1]["lm_cc_score"]
    fail = clean[clean["resolved"] == 0]["lm_cc_score"]
    print(f"LM-CC resolved:     median {res.median():.1f}, mean {res.mean():.1f}")
    print(f"LM-CC not resolved: median {fail.median():.1f}, mean {fail.mean():.1f}")
    print(f"Mann-Whitney-U: p = {mannwhitneyu(res, fail).pvalue:.3f}")


def collinearity(clean):
    metrics = [
        "lm_cc_score",
        "lm_cc_total_comp",
        "lm_cc_total_branch",
        "cc_sum",
        "cognitive_sum",
        "maintainability_index",
        "halstead_volume",
        "lloc",
        "nesting_max",
        "n_branches",
        *GRAPH,
    ]
    print(f"{'Metric':<30} {'rho with LOC':>12}")
    for m in metrics:
        if m in clean.columns:
            rho, _ = _spearman(clean[m], clean["loc"])
            shown = f"{rho:>+12.3f}" if not pd.isna(rho) else f"{'-':>12}"
            print(f"{m:<30} {shown}")


def descriptive(clean):
    metrics = [
        "lm_cc_score",
        "lm_cc_total_comp",
        "lm_cc_total_branch",
        "lm_cc_max_comp",
        "cc_sum",
        "halstead_volume",
        "loc",
        "lloc",
        "nesting_max",
        "n_branches",
    ]
    metrics = [m for m in metrics if m in clean.columns]
    desc = (
        clean[metrics]
        .describe()
        .T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    )
    print(desc.round(2).to_string())


def metrics_overview(clean, title=""):
    print(clean["lm_cc_score"].describe().to_string())
    clean["lm_cc_score"].hist(bins=50)
    plt.xlabel("LM-CC")
    plt.ylabel("count")
    plt.title(f"LM-CC distribution {title}".strip())
    plt.show()

    print("\n=== Collinearity with LOC ===")
    collinearity(clean)
    print("\n=== Descriptive statistics ===")
    descriptive(clean)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, m in zip(axes, ["lm_cc_score", "nesting_max"]):
        rho = spearmanr(clean[m], clean["loc"]).correlation
        ax.scatter(clean["loc"], clean[m], alpha=0.4, s=15)
        ax.set_xlabel("LOC")
        ax.set_ylabel(m)
        ax.set_title(f"{m} vs LOC (rho={rho:.3f})")
    plt.tight_layout()
    plt.show()

    print("\n=== Metric correlation heatmap ===")
    correlation_heatmap(clean, title)



def per_agent_table(long, top=10):
    table = per_agent_correlation_table(
        long, metric="lm_cc_score", score="resolved", control="loc"
    )

    def _fmt(df):
        df = df.copy()
        df["agent"] = df["agent"].map(
            lambda n: n.split("_", 1)[1] if n[:8].isdigit() else n
        )
        df["resolve_rate"] = (df["resolve_rate"] * 100).round(1).astype(str) + " %"
        for c in [
            "sample_zero",
            "sample_partial_loc",
            "subgroup_zero",
            "subgroup_partial_loc",
        ]:
            df[c] = df[c].map(lambda v: f"{v:+.3f}")
        return df

    print(f"Per-agent LM-CC across all {len(table)} agents")
    print(_fmt(table).to_string(index=False))
    print(f"\nTop {top} agents by strongest negative correlation (sample_zero)")
    print(_fmt(table.sort_values("sample_zero").head(top)).to_string(index=False))
    return table


def all_agents_report(result_path, title=""):
    clean = load_metrics(result_path)
    difficulty = pd.read_parquet(PATHS.TASK_DIFFICULTY)
    all_predictions = pd.read_parquet(PATHS.ALL_PREDICTIONS)
    clean["resolution_rate"] = clean["instance_id"].map(
        difficulty.set_index("instance_id")["resolution_rate"]
    )
    clean = clean[clean["resolution_rate"].notna()]
    long = all_predictions.merge(
        clean[["instance_id", *LONG_METRIC_COLS]], on="instance_id", how="inner"
    )
    n_agents = all_predictions["model"].nunique()
    print(f"{title}: {len(clean)} tasks, {n_agents} agents, long {len(long):,} rows\n")

    print(clean["resolution_rate"].describe().to_string())
    clean["resolution_rate"].hist(bins=30)
    plt.xlabel("resolution rate")
    plt.ylabel("count")
    plt.title(f"Task difficulty across {n_agents} agents")
    plt.show()

    print(correlation_report(clean, score="resolution_rate").to_string(index=False))

    clean["tier"] = pd.cut(
        clean["resolution_rate"],
        [-0.01, 0.33, 0.66, 1.01],
        labels=["hard", "medium", "easy"],
    )
    print(
        clean.groupby("tier", observed=True)["lm_cc_score"]
        .agg(["count", "mean", "median"])
        .to_string()
    )
    groups = [g["lm_cc_score"].values for _, g in clean.groupby("tier", observed=True)]
    print(f"Kruskal-Wallis across tiers: p = {kruskal(*groups).pvalue:.3f}")
    clean.boxplot(column="lm_cc_score", by="tier")
    plt.suptitle("")
    plt.title(f"LM-CC by difficulty tier {title}".strip())
    plt.ylabel("LM-CC")
    plt.show()

    return per_agent_table(long)


def whole_vs_function_report():
    file_df = pd.read_parquet(PATHS.VERIFIED_WHOLE_FILE_RESULT_DATASET)
    function_df = pd.read_parquet(PATHS.VERIFIED_FUNCTION_RESULT_DATASET)
    difficulty = pd.read_parquet(PATHS.TASK_DIFFICULTY)
    merged = file_df.merge(
        function_df, on="instance_id", suffixes=("_file", "_fn")
    ).merge(
        difficulty[["instance_id", "resolution_rate"]], on="instance_id", how="left"
    )
    clean = merged[
        merged["has_patched_function"].fillna(False)
        & merged["parsable_file"].fillna(False)
        & merged["parsable_fn"].fillna(False)
        & merged["lm_cc_score_file"].notna()
        & merged["lm_cc_score_fn"].notna()
        & merged["resolution_rate"].notna()
    ].copy()
    print(f"Tasks with both file & function available: {len(clean)}\n")

    print("rho(LM-CC, LOC):")
    print(
        f"  whole file : {spearmanr(clean['lm_cc_score_file'], clean['loc_file']).correlation:+.3f}"
    )
    print(
        f"  function   : {spearmanr(clean['lm_cc_score_fn'], clean['loc_fn']).correlation:+.3f}"
    )

    pf_file, _ = partial_spearman(
        clean, "lm_cc_score_file", "resolution_rate", "loc_file"
    )
    pf_fn, _ = partial_spearman(clean, "lm_cc_score_fn", "resolution_rate", "loc_fn")
    print("\nLM-CC partial(resolution_rate | LOC):")
    print(f"  whole file : {pf_file:+.3f}")
    print(f"  function   : {pf_fn:+.3f}")

    print("\n=== WHOLE FILE (control = loc_file) ===")
    print(
        full_correlation_table(
            clean,
            ["lm_cc_score_file", "cc_sum_file", "nesting_max_file"],
            score="resolution_rate",
            control="loc_file",
        ).to_string(index=False)
    )
    print("\n=== FUNCTION (control = loc_fn) ===")
    print(
        full_correlation_table(
            clean,
            ["lm_cc_score_fn", "cc_sum_fn", "nesting_max_fn"],
            score="resolution_rate",
            control="loc_fn",
        ).to_string(index=False)
    )


COMPARISON_AGENTS = [
    (
        "SWE-agent / Claude-3.7",
        PATHS.SWEAGENT_CLAUDE37_VERIFIED_LABELS,
        PATHS.SWEAGENT_CLAUDE37_LIVE_LABELS,
    ),
    (
        "OpenHands / Qwen3-Coder",
        PATHS.OPENHANDS_QWEN3CODER_VERIFIED_LABELS,
        PATHS.OPENHANDS_QWEN3CODER_LIVE_LABELS,
    ),
]

COMPARISON_RESULTS = {
    "verified": {
        "whole_file": PATHS.VERIFIED_WHOLE_FILE_RESULT_DATASET,
        "function": PATHS.VERIFIED_FUNCTION_RESULT_DATASET,
    },
    "live": {
        "whole_file": PATHS.LIVE_WHOLE_FILE_RESULT_DATASET,
        "function": PATHS.LIVE_FUNCTION_RESULT_DATASET,
    },
}


def lm_cc_distribution(level="whole_file", metric="lm_cc_score", save=None):
    """LM-CC distribution for Verified vs Live in one figure.

    Left: density-normalised histogram + KDE (density, not count, so the two
    datasets are comparable despite different task counts). Right: box + strip
    plot showing median, spread and outliers.
    """
    colors = {"verified": "#3b6fb0", "live": "#e07a3f"}
    order = ["verified", "live"]
    frames = []
    for name in order:
        clean = load_metrics(COMPARISON_RESULTS[name][level])
        frames.append(pd.DataFrame({metric: clean[metric].to_numpy(), "dataset": name}))
    data = pd.concat(frames, ignore_index=True)
    counts = data.groupby("dataset")[metric].size()

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 2]}
    )

    for name in order:
        sub = data.loc[data["dataset"] == name, metric]
        ax_l.hist(
            sub,
            bins=40,
            density=True,
            alpha=0.30,
            color=colors[name],
            label=f"{name} (n={counts[name]})",
        )
    sns.kdeplot(
        data=data,
        x=metric,
        hue="dataset",
        hue_order=order,
        palette=colors,
        common_norm=False,
        linewidth=2,
        ax=ax_l,
        legend=False,
    )
    ax_l.set_xlim(left=0)
    ax_l.set_xlabel("LM-CC")
    ax_l.set_ylabel("density")
    ax_l.set_title("Normalised distribution")
    ax_l.legend(title="")

    sns.boxplot(
        data=data,
        x="dataset",
        y=metric,
        order=order,
        hue="dataset",
        hue_order=order,
        palette=colors,
        width=0.5,
        showfliers=False,
        boxprops=dict(alpha=0.45),
        legend=False,
        ax=ax_r,
    )
    sns.stripplot(
        data=data,
        x="dataset",
        y=metric,
        order=order,
        hue="dataset",
        hue_order=order,
        palette=colors,
        size=3,
        alpha=0.45,
        jitter=0.25,
        legend=False,
        ax=ax_r,
    )
    for i, name in enumerate(order):
        med = data.loc[data["dataset"] == name, metric].median()
        ax_r.annotate(
            f"median {med:.0f}",
            (i, med),
            textcoords="offset points",
            xytext=(14, 0),
            va="center",
            fontsize=9,
            color="0.2",
        )
    ax_r.set_xlabel("")
    ax_r.set_ylabel("LM-CC")
    ax_r.set_title("Median, spread & outliers")

    fig.suptitle(f"LM-CC: Verified vs Live ({level})", fontsize=13, y=1.02)
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return data


_LOLLIPOP_POS, _LOLLIPOP_NEG = "#2a788e", "#c0392b"


def _draw_lollipop_ax(ax, g, order, y, rho_col, p_col, alpha):
    """Draw one lollipop column: a stem+dot per metric, coloured by sign of the
    correlation, faded where it is not significant (p >= alpha)."""
    g = g.set_index("metric").reindex(order)
    for m in order:
        r = g.loc[m, rho_col]
        if pd.isna(r):
            continue
        sig = not pd.isna(g.loc[m, p_col]) and g.loc[m, p_col] < alpha
        color = _LOLLIPOP_POS if r >= 0 else _LOLLIPOP_NEG
        yi = y[m]
        ax.plot(
            [0, r],
            [yi, yi],
            color=color,
            lw=1.8,
            alpha=1.0 if sig else 0.3,
            solid_capstyle="round",
            zorder=1,
        )
        ax.scatter(
            r,
            yi,
            s=46 if sig else 26,
            color=color,
            alpha=1.0 if sig else 0.35,
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
    ax.axvline(0, color="0.55", lw=0.9, zorder=0)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def per_metric_agent_correlations(level="whole_file", control="loc", metrics=None):
    """Numeric per-metric correlations for every agent x benchmark.

    Returns one row per (agent, benchmark, metric) with four correlations of
    `metric` vs resolved and their p-values: the sample (per-instance)
    zero-order and partial-`control` Spearman, and the subgroup (9-11 bins)
    zero-order and partial-`control` Spearman. This is the numeric counterpart
    of `benchmark_comparison`, but across all metrics instead of a single one.
    """
    rows = []
    for agent, verified_labels, live_labels in COMPARISON_AGENTS:
        for benchmark, labels in [("verified", verified_labels), ("live", live_labels)]:
            clean = load_metrics(COMPARISON_RESULTS[benchmark][level], labels)
            cols = metrics if metrics is not None else _available(clean)
            for metric in cols:
                if metric not in clean.columns:
                    continue
                data = clean.dropna(
                    subset=[c for c in (metric, "resolved", control) if c in clean.columns]
                )
                score_vals = data["resolved"].to_numpy(dtype=float)
                metric_vals = data[metric].to_numpy(dtype=float)
                loc_vals = data[control].to_numpy(dtype=float)

                rho, p = _spearman(metric_vals, score_vals)
                sub_rho, sub_p = fixed_subgroup_corr(score_vals, metric_vals)
                if metric == control:
                    partial_rho, partial_p = np.nan, np.nan
                    sub_partial_rho, sub_partial_p = np.nan, np.nan
                else:
                    partial_rho, partial_p = partial_spearman(
                        data, metric, "resolved", control
                    )
                    sub_partial_rho, sub_partial_p = fixed_subgroup_corr(
                        score_vals, metric_vals, loc_vals
                    )
                rows.append(
                    {
                        "agent": agent,
                        "benchmark": benchmark,
                        "metric": metric,
                        "n": len(data),
                        "rho": rho,
                        "p": p,
                        "partial_rho": partial_rho,
                        "partial_p": partial_p,
                        "sub_rho": sub_rho,
                        "sub_p": sub_p,
                        "sub_partial_rho": sub_partial_rho,
                        "sub_partial_p": sub_partial_p,
                    }
                )
    return pd.DataFrame(rows)


_LOLLIPOP_VALUES = {
    "zero": ("rho", "p", "Spearman ρ"),
    "partial": ("partial_rho", "partial_p", "partial ρ | {control}"),
    "subgroup": ("sub_rho", "sub_p", "subgroup ρ (9-11 bins)"),
    "subgroup_partial": (
        "sub_partial_rho",
        "sub_partial_p",
        "subgroup partial ρ | {control} (9-11 bins)",
    ),
}


def lollipop_correlations(
    level="whole_file",
    value="subgroup_partial",
    control="loc",
    metrics=None,
    alpha=0.05,
    table=None,
    agent=None,
    col_width=3.4,
    save=None,
):
    """Lollipop plot of per-metric correlations, one column per agent x benchmark.

    ``value`` selects which correlation of metric vs resolved is shown:
    ``"zero"`` (sample Spearman rho), ``"partial"`` (sample partial rho |
    control), ``"subgroup"`` (subgroup rho, 9-11 bins) or ``"subgroup_partial"``
    (subgroup partial rho | control, the size-adjusted grouped estimate).
    y-axis: metrics; one small-multiple column per agent + dataset. Sign is
    encoded by colour (teal positive / red negative) and redundantly by position
    around the zero line; significant cells (p < alpha) are drawn solid, the
    rest faded.

    Pass ``agent`` to keep only that agent's columns (verified + live), which
    makes each column wider/more legible; ``col_width`` sets the inches per
    column.
    """
    if table is None:
        table = per_metric_agent_correlations(
            level=level, control=control, metrics=metrics
        )
    if agent is not None:
        table = table[table["agent"] == agent]
        if table.empty:
            available = [a for a, *_ in COMPARISON_AGENTS]
            raise ValueError(f"no rows for agent {agent!r}; available: {available}")

    if value not in _LOLLIPOP_VALUES:
        raise ValueError(f"value must be one of {list(_LOLLIPOP_VALUES)}, got {value!r}")
    rho_col, p_col, xlabel = _LOLLIPOP_VALUES[value]
    xlabel = xlabel.format(control=control)

    order = list(dict.fromkeys(table["metric"]))
    y = {m: i for i, m in enumerate(order)}
    facets = list(table.groupby(["agent", "benchmark"], sort=False))

    n = len(facets)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(col_width * n, max(6, 0.42 * len(order))),
        sharey=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, ((agent, benchmark), g) in zip(axes, facets):
        _draw_lollipop_ax(ax, g, order, y, rho_col, p_col, alpha)
        ax.set_title(f"{agent}\n{benchmark}", fontsize=13)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.tick_params(axis="x", labelsize=11)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=12)
    axes[0].invert_yaxis()
    grouped = "subgroup" in value
    kind = "subgroup (9-11 bins)" if grouped else "sample"
    fig.suptitle(
        f"Per-metric {kind} correlation with resolved ({level}) - solid = p<{alpha}",
        y=1.005,
        fontsize=15,
    )
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return table


def per_metric_loc_correlations(level="whole_file", metrics=None):
    """Per-metric Spearman rho(metric, LOC) for each dataset (Verified, Live).

    Collinearity with LOC is a property of the code metrics only, so — unlike
    the resolved-correlation table — it does not depend on the agent and is
    faceted by dataset alone.
    """
    rows = []
    for benchmark in ["verified", "live"]:
        clean = load_metrics(COMPARISON_RESULTS[benchmark][level])
        cols = metrics if metrics is not None else _available(clean)
        for metric in cols:
            if metric not in clean.columns or metric == "loc":
                continue
            data = clean.dropna(subset=[c for c in (metric, "loc") if c in clean.columns])
            rho, p = _spearman(data[metric], data["loc"])
            rows.append(
                {
                    "benchmark": benchmark,
                    "metric": metric,
                    "n": len(data),
                    "rho": rho,
                    "p": p,
                }
            )
    return pd.DataFrame(rows)


def lollipop_loc_collinearity(
    level="whole_file", metrics=None, alpha=0.05, table=None, col_width=4.5, save=None
):
    """Lollipop plot of each metric's collinearity with LOC, one column per dataset.

    x-axis: Spearman rho(metric, LOC); y-axis: metrics; columns: Verified / Live.
    Same visual grammar as ``lollipop_correlations`` (sign by colour + position,
    faded when p >= alpha).
    """
    if table is None:
        table = per_metric_loc_correlations(level=level, metrics=metrics)

    order = list(dict.fromkeys(table["metric"]))
    y = {m: i for i, m in enumerate(order)}
    facets = list(table.groupby("benchmark", sort=False))

    n = len(facets)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(col_width * n, max(6, 0.42 * len(order))),
        sharey=True,
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for ax, (benchmark, g) in zip(axes, facets):
        _draw_lollipop_ax(ax, g, order, y, "rho", "p", alpha)
        ax.set_title(benchmark, fontsize=13)
        ax.set_xlabel("Spearman ρ(metric, LOC)", fontsize=12)
        ax.tick_params(axis="x", labelsize=11)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=12)
    axes[0].invert_yaxis()
    fig.suptitle(
        f"Per-metric collinearity with LOC ({level}) — solid = p<{alpha}",
        y=1.005,
        fontsize=15,
    )
    plt.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()
    return table


def benchmark_comparison(metric="lm_cc_score", level="whole_file", control="loc"):
    rows = []
    for agent, verified_labels, live_labels in COMPARISON_AGENTS:
        for benchmark, labels in [("verified", verified_labels), ("live", live_labels)]:
            clean = load_metrics(COMPARISON_RESULTS[benchmark][level], labels)
            rho, p = spearmanr(clean[metric], clean["resolved"])
            partial_rho, partial_p = partial_spearman(
                clean, metric, "resolved", control
            )
            rows.append(
                {
                    "agent": agent,
                    "benchmark": benchmark,
                    "n": len(clean),
                    "resolve_rate": clean["resolved"].mean(),
                    "rho": rho,
                    "p": p,
                    f"partial_{control}": partial_rho,
                    "partial_p": partial_p,
                }
            )
    return pd.DataFrame(rows)


def benchmark_comparison_report(
    level="whole_file", metric="lm_cc_score", control="loc", alpha=0.05
):
    table = benchmark_comparison(metric=metric, level=level, control=control)

    def stars(rho, p):
        return f"{rho:+.3f}" + ("*" if p < alpha else " ")

    show = pd.DataFrame(
        {
            "agent": table["agent"],
            "benchmark": table["benchmark"],
            "n": table["n"],
            "resolve_rate": (table["resolve_rate"] * 100).round(1).astype(str) + " %",
            "rho": [stars(r, p) for r, p in zip(table["rho"], table["p"])],
            f"partial_{control}": [
                stars(r, p)
                for r, p in zip(table[f"partial_{control}"], table["partial_p"])
            ],
        }
    )
    print(
        f"{metric} vs resolved, {level} level (partial controls for {control}; * = p<{alpha})"
    )
    print(show.to_string(index=False))
    print(
        f"\nsignificant zero-order cells: {int((table['p'] < alpha).sum())}/{len(table)}"
    )

    pivot = table.pivot(index="agent", columns="benchmark", values="rho")[
        ["verified", "live"]
    ]
    sig = (
        table.pivot(index="agent", columns="benchmark", values="p")[
            ["verified", "live"]
        ]
        < alpha
    )
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    for container, benchmark in zip(ax.containers, ["verified", "live"]):
        for bar, agent in zip(container, pivot.index):
            if not bool(sig.loc[agent, benchmark]):
                bar.set_alpha(0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Spearman rho ({metric} vs resolved)")
    ax.set_xlabel("")
    ax.set_title(f"{metric}: Verified vs Live ({level})  (faded = p>={alpha})")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return table
