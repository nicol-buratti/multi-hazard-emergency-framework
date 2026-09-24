#!/usr/bin/env python3
"""
aggregate_kpis.py
-----------------
Aggregates per-run agent KPI data into thesis-ready tables.

Usage
-----
    python aggregate_kpis.py kpi_evaluation_n_50_seed_3_snapshots_5_eval_3_key.csv
    python aggregate_kpis.py run_a.csv run_b.csv --outdir results --no-tests

Input columns (one row per run):
    run_id, scenario_type, latency_ms, tool_overhead_ms, tool_call_count,
    distinct_tools_used, llm_calls, prompt_tokens, completion_tokens,
    total_tokens, assessment_count, schema_adherence, routing_correct,
    danger_score, api_failed

Output (in --outdir, default: ./tables):
    T1_overview, T2_reliability, T3_latency, T4_resources, T5_danger,
    T6_routing_conditional, T7_stat_tests
    each as .csv, .tex (booktabs) and .md
"""

import argparse
import itertools
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats

    HAVE_SCIPY = True
except ImportError:  # tests are optional
    HAVE_SCIPY = False

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SCENARIO_ORDER = ["normal", "fire", "earthquake", "fire_earthquake"]
SCENARIO_LABELS = {
    "normal": "Normal",
    "fire": "Fire",
    "earthquake": "Earthquake",
    "fire_earthquake": "Fire + Earthquake",
}
BOOL_COLS = ["schema_adherence", "routing_correct", "api_failed"]
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (returns fractions)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_pct_ci(k, n):
    lo, hi = wilson_ci(k, n)
    return f"{100 * k / n:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def fmt_mean_sd(s, dec=1):
    return f"{s.mean():.{dec}f} ± {s.std(ddof=1):.{dec}f}"


def with_overall(df):
    """Append an 'Overall' pseudo-scenario so every table has a total row."""
    all_ = df.copy()
    all_["scenario_type"] = "Overall"
    return pd.concat([df, all_], ignore_index=True)


def scenario_index(present):
    order = [s for s in SCENARIO_ORDER if s in present]
    order += [s for s in present if s not in order and s != "Overall"]
    if "Overall" in present:
        order.append("Overall")
    return order


def label(s):
    return SCENARIO_LABELS.get(s, s)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def cliffs_delta(a, b):
    """Cliff's delta effect size (-1..1) via the Mann-Whitney U statistic."""
    a, b = np.asarray(a), np.asarray(b)
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return 2 * u / (len(a) * len(b)) - 1


def effect_label(d):
    d = abs(d)
    return (
        "negligible"
        if d < 0.147
        else "small"
        if d < 0.33
        else "medium"
        if d < 0.474
        else "large"
    )


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def latex_escape(x):
    s = str(x)
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("±", r"$\pm$"),
        ("<", r"$<$"),
        (">", r"$>$"),
    ]:
        s = s.replace(a, b)
    return s


def to_latex(df, caption, label_):
    cols = list(df.columns)
    header = " & ".join(latex_escape(c) for c in cols) + r" \\"
    rows = [
        " & ".join(latex_escape(v) for v in r) + r" \\" for r in df.astype(str).values
    ]
    # separate 'Overall' row if present
    for i, r in enumerate(df.astype(str).values):
        if r[0] == "Overall":
            rows[i] = r"\midrule" + "\n" + rows[i]
    spec = "l" + "r" * (len(cols) - 1)
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\small",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{tab:{label_}}}",
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )


def to_markdown(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values]
    return "\n".join(lines) + "\n"


def save(df, name, caption, outdir):
    df.to_csv(outdir / f"{name}.csv", index=False)
    (outdir / f"{name}.tex").write_text(to_latex(df, caption, name.lower()))
    (outdir / f"{name}.md").write_text(to_markdown(df))
    print(f"\n=== {name}: {caption} ===")
    print(df.to_string(index=False))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def table_overview(df):
    d = with_overall(df)
    rows = []
    for s in scenario_index(d.scenario_type.unique()):
        g = d[d.scenario_type == s]
        rows.append(
            {
                "Scenario": label(s),
                "Runs (n)": len(g),
                "Mean tool calls": f"{g.tool_call_count.mean():.2f}",
                "Mean distinct tools": f"{g.distinct_tools_used.mean():.2f}",
                "Mean LLM calls": f"{g.llm_calls.mean():.2f}",
                "Runs with ≥1 tool call (%)": f"{100 * (g.tool_call_count > 0).mean():.1f}",
            }
        )
    return pd.DataFrame(rows)


def table_reliability(df):
    """Correctness / robustness KPIs with 95% Wilson CIs."""
    d = with_overall(df)
    rows = []
    for s in scenario_index(d.scenario_type.unique()):
        g = d[d.scenario_type == s]
        n = len(g)
        rows.append(
            {
                "Scenario": label(s),
                "n": n,
                "Routing correct % [95% CI]": fmt_pct_ci(
                    int(g.routing_correct.sum()), n
                ),
                "Schema adherence % [95% CI]": fmt_pct_ci(
                    int(g.schema_adherence.sum()), n
                ),
                "API failures % [95% CI]": fmt_pct_ci(int(g.api_failed.sum()), n),
            }
        )
    return pd.DataFrame(rows)


def table_latency(df):
    d = with_overall(df)
    rows = []
    for s in scenario_index(d.scenario_type.unique()):
        g = d[d.scenario_type == s]
        lat = g.latency_ms
        rows.append(
            {
                "Scenario": label(s),
                "Mean (s)": f"{lat.mean() / 1000:.2f}",
                "SD (s)": f"{lat.std(ddof=1) / 1000:.2f}",
                "Median (s)": f"{lat.median() / 1000:.2f}",
                "P90 (s)": f"{lat.quantile(0.90) / 1000:.2f}",
                "P95 (s)": f"{lat.quantile(0.95) / 1000:.2f}",
                "Min (s)": f"{lat.min() / 1000:.2f}",
                "Max (s)": f"{lat.max() / 1000:.2f}",
                "Tool overhead (ms)": fmt_mean_sd(g.tool_overhead_ms, 1),
                "Overhead / latency (%)": f"{100 * g.tool_overhead_ms.sum() / lat.sum():.2f}",
            }
        )
    return pd.DataFrame(rows)


def table_resources(df, price_in=None, price_out=None):
    d = with_overall(df)
    rows = []
    for s in scenario_index(d.scenario_type.unique()):
        g = d[d.scenario_type == s]
        row = {
            "Scenario": label(s),
            "Prompt tok.": fmt_mean_sd(g.prompt_tokens, 0),
            "Completion tok.": fmt_mean_sd(g.completion_tokens, 0),
            "Total tok.": fmt_mean_sd(g.total_tokens, 0),
            "Tok. per LLM call": f"{(g.total_tokens / g.llm_calls).mean():.0f}",
            "Latency per 1k tok. (s)": f"{(g.latency_ms / 1000 / (g.total_tokens / 1000)).mean():.2f}",
        }
        if price_in is not None and price_out is not None:
            cost = (g.prompt_tokens * price_in + g.completion_tokens * price_out) / 1e6
            row["Cost / run (USD)"] = f"{cost.mean():.5f}"
        rows.append(row)
    return pd.DataFrame(rows)


def table_danger(df):
    d = with_overall(df)
    rows = []
    for s in scenario_index(d.scenario_type.unique()):
        g = d[d.scenario_type == s].danger_score
        rows.append(
            {
                "Scenario": label(s),
                "Mean": f"{g.mean():.3f}",
                "SD": f"{g.std(ddof=1):.3f}",
                "Median": f"{g.median():.2f}",
                "Min": f"{g.min():.2f}",
                "Max": f"{g.max():.2f}",
                "Score ≥ 0.5 (%)": f"{100 * (g >= 0.5).mean():.1f}",
            }
        )
    return pd.DataFrame(rows)


def table_routing_conditional(df):
    """
    Splits each scenario by routing_correct. Only scenarios that contain both
    outcomes are informative (here: earthquake) – this is where the cost of
    tool use vs. the effect on the danger score shows up.
    """
    rows = []
    for s in scenario_index(df.scenario_type.unique()):
        sub = df[df.scenario_type == s]
        if sub.routing_correct.nunique() < 2:
            continue
        for ok, g in sub.groupby("routing_correct"):
            rows.append(
                {
                    "Scenario": label(s),
                    "Routing correct": "Yes" if ok else "No",
                    "n": len(g),
                    "Tool calls": f"{g.tool_call_count.mean():.2f}",
                    "LLM calls": f"{g.llm_calls.mean():.2f}",
                    "Latency (s)": fmt_mean_sd(g.latency_ms / 1000, 2),
                    "Total tok.": fmt_mean_sd(g.total_tokens, 0),
                    "Danger score": fmt_mean_sd(g.danger_score, 3),
                }
            )
    return pd.DataFrame(rows)


def table_tests(df):
    """
    Omnibus Kruskal-Wallis across scenarios + pairwise Mann-Whitney U
    (Holm-corrected) with Cliff's delta, for latency, total tokens, danger score.
    """
    if not HAVE_SCIPY:
        print("scipy not installed – skipping statistical tests.")
        return None
    scen = [s for s in scenario_index(df.scenario_type.unique())]
    metrics = {
        "latency_ms": "Latency",
        "total_tokens": "Total tokens",
        "danger_score": "Danger score",
    }
    rows = []
    for col, name in metrics.items():
        groups = [df.loc[df.scenario_type == s, col].values for s in scen]
        if all(np.ptp(g) == 0 for g in groups) and len({g[0] for g in groups}) == 1:
            continue  # constant metric, nothing to test
        H, p_kw = stats.kruskal(*groups)
        pairs = list(itertools.combinations(scen, 2))
        raw_p, deltas = [], []
        for a, b in pairs:
            xa = df.loc[df.scenario_type == a, col]
            xb = df.loc[df.scenario_type == b, col]
            try:
                raw_p.append(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
            except ValueError:  # identical constant samples
                raw_p.append(1.0)
            deltas.append(cliffs_delta(xa, xb))
        adj = holm(raw_p)
        for (a, b), p, pa, d in zip(pairs, raw_p, adj, deltas):
            rows.append(
                {
                    "Metric": name,
                    "Comparison": f"{label(a)} vs {label(b)}",
                    "Kruskal-Wallis p": fmt_p(p_kw),
                    "MWU p (Holm)": fmt_p(pa),
                    "Cliff's δ": f"{d:+.2f}",
                    "Effect": effect_label(d),
                    "Sig.": "*" if pa < ALPHA else "",
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def load(paths):
    frames = []
    for p in paths:
        f = pd.read_csv(p)
        f["source"] = Path(p).name
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    for c in BOOL_COLS:  # robust against "True"/"False" strings
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype(str).str.lower().map({"true": True, "false": False})
        df[c] = df[c].astype(bool)
    return df


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", nargs="+", help="one or more KPI csv files (concatenated)")
    ap.add_argument("--outdir", default="tables")
    ap.add_argument("--no-tests", action="store_true", help="skip statistical tests")
    ap.add_argument(
        "--price-in", type=float, help="USD per 1M prompt tokens (adds a cost column)"
    )
    ap.add_argument("--price-out", type=float, help="USD per 1M completion tokens")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load(args.csv)

    meta = re.findall(r"(n|seed|snapshots|eval)_(\d+)", " ".join(args.csv))
    print(
        f"Loaded {len(df)} runs from {len(args.csv)} file(s); "
        f"scenarios: {df.scenario_type.value_counts().to_dict()}"
    )
    if meta:
        print("Parsed from filename:", dict(meta))

    save(
        table_overview(df),
        "T1_overview",
        "Overview of evaluation runs per scenario",
        outdir,
    )
    save(
        table_reliability(df),
        "T2_reliability",
        "Reliability KPIs per scenario (95% Wilson CI)",
        outdir,
    )
    save(
        table_latency(df),
        "T3_latency",
        "End-to-end latency and tool overhead per scenario",
        outdir,
    )
    save(
        table_resources(df, args.price_in, args.price_out),
        "T4_resources",
        "Token consumption per scenario (mean ± SD)",
        outdir,
    )
    save(
        table_danger(df), "T5_danger", "Danger score distribution per scenario", outdir
    )

    t6 = table_routing_conditional(df)
    if not t6.empty:
        save(t6, "T6_routing_conditional", "Runs split by routing correctness", outdir)

    if not args.no_tests:
        t7 = table_tests(df)
        if t7 is not None and not t7.empty:
            save(
                t7,
                "T7_stat_tests",
                "Pairwise scenario comparisons (Mann-Whitney U, Holm-corrected)",
                outdir,
            )

    print(f"\nAll tables written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
