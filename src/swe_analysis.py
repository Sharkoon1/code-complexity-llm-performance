"""Analysis helpers shared by the SWE-bench result notebooks."""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

from src.config import PATHS
from src.correlation import (
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


def load_metrics(result_path, labels_path=None):
    df = pd.read_parquet(result_path)
    mask = df["parsable"].fillna(False) & df["lm_cc_score"].notna()
    if "has_patched_function" in df.columns:
        mask &= df["has_patched_function"].fillna(False)
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
        "halstead_volume",
        "lloc",
        "nesting_max",
        "n_branches",
    ]
    print(f"{'Metric':<22} {'rho with LOC':>12}")
    for m in metrics:
        if m in clean.columns:
            print(f"{m:<22} {spearmanr(clean[m], clean['loc']).correlation:>+12.3f}")


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
