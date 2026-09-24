#!/usr/bin/env python3
"""
aggregate_openrouter.py
-----------------------
Aggregates an OpenRouter activity export and (optionally) joins it with the
per-run KPI csv to derive cost / cache / latency-decomposition tables.

Usage
-----
    uv run python -m  evaluation.aggregate_openrouter openrouter_activity_2026-09-23.csv \
                          --kpi kpi_evaluation_n_50_seed_3_snapshots_5_eval_3_key.csv --outdir tables

Requires aggregate_kpis.py in the same folder (re-uses its table/export helpers).

How the join works
------------------
OpenRouter exports one row per LLM generation, with no run_id. Generations are
sorted by `created_at`, and the KPI file says how many LLM calls each run made
(`llm_calls`), so generations are assigned to runs in run_id order:
    run 0 gets the first llm_calls[0] generations, run 1 the next ones, ...
This assumes runs were executed sequentially and the export contains exactly
this evaluation's generations. The assumption is VALIDATED: for every run the
summed prompt and completion tokens must equal the KPI file's values. If they
don't, the script stops the joined tables and tells you (standalone tables are
still produced).

Assumptions about OpenRouter columns
------------------------------------
* cost_total is the net cost charged; cost_cache is a (negative) cache
  discount, so gross cost = cost_total - cost_cache.
* generation_time_ms / time_to_first_token_ms are per generation.

Output: O1_overview, O2_cost, O3_cost_by_routing, O4_call_latency_cache,
        O5_call_position, O6_latency_decomposition  (.csv / .tex / .md)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.aggregate_kpis import (
    fmt_mean_sd,
    label,
    load,
    save,
    scenario_index,
    with_overall,
)


# --------------------------------------------------------------------------- #
# Loading / joining
# --------------------------------------------------------------------------- #
def load_openrouter(path, api_key=None):
    o = pd.read_csv(path, parse_dates=["created_at"])
    if api_key:
        o = o[o.api_key_name == api_key]
    o = o.sort_values("created_at").reset_index(drop=True)
    o["cost_cache"] = o["cost_cache"].fillna(0.0)  # NaN = no cache discount
    o["cost_gross"] = o["cost_total"] - o["cost_cache"]  # cache discount is negative
    o["cache_hit"] = o["tokens_cached"] > 0
    return o


def join_runs(o, k):
    """Assign generations to runs; return (joined_calls, ok, message)."""
    if len(o) != int(k.llm_calls.sum()):
        return (
            None,
            False,
            (
                f"{len(o)} generations vs {int(k.llm_calls.sum())} LLM calls in KPI file "
                "- cannot align."
            ),
        )
    k = k.sort_values("run_id").reset_index(drop=True)
    o = o.copy()
    o["run_id"] = np.repeat(k.run_id.values, k.llm_calls.values)
    o["call_idx"] = o.groupby("run_id").cumcount() + 1
    chk = o.groupby("run_id").agg(
        p=("tokens_prompt", "sum"), c=("tokens_completion", "sum")
    )
    chk = k.set_index("run_id").join(chk)
    p_ok = (chk.prompt_tokens == chk.p).mean()
    c_ok = (chk.completion_tokens == chk.c).mean()
    if p_ok < 1.0 or c_ok < 1.0:
        return (
            None,
            False,
            (
                f"token validation failed (prompt match {p_ok:.1%}, completion match "
                f"{c_ok:.1%}) - runs were probably not sequential."
            ),
        )
    scen = k[["run_id", "scenario_type", "routing_correct"]]
    return (
        o.merge(scen, on="run_id"),
        True,
        ("join validated: prompt & completion tokens match for 100% of runs"),
    )


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def usd(x, dec=5):
    return f"{x:.{dec}f}"


def table_overview(o):
    prov = ", ".join(f"{p} ({n})" for p, n in o.provider_name.value_counts().items())
    fin = ", ".join(
        f"{f} ({n})" for f, n in o.finish_reason_normalized.value_counts().items()
    )
    rows = [
        ("Model", ", ".join(sorted(o.model_permaslug.unique()))),
        ("Generations (n)", len(o)),
        (
            "Time window",
            f"{o.created_at.min():%Y-%m-%d %H:%M:%S} - {o.created_at.max():%H:%M:%S}",
        ),
        ("Providers (generations)", prov),
        ("Finish reasons", fin),
        ("Streamed (%)", f"{100 * o.streamed.mean():.1f}"),
        ("Cancelled (%)", f"{100 * o.cancelled.mean():.1f}"),
        ("Prompt tokens (total)", f"{o.tokens_prompt.sum():,}"),
        ("Completion tokens (total)", f"{o.tokens_completion.sum():,}"),
        ("Reasoning tokens (total)", f"{o.tokens_reasoning.sum():,}"),
        (
            "Cache hit rate, token-weighted (%)",
            f"{100 * o.tokens_cached.sum() / o.tokens_prompt.sum():.1f}",
        ),
        ("Generations with cache hit (%)", f"{100 * o.cache_hit.mean():.1f}"),
        ("Total net cost (USD)", usd(o.cost_total.sum(), 4)),
        ("Total gross cost without cache (USD)", usd(o.cost_gross.sum(), 4)),
        (
            "Cache saving (%)",
            f"{100 * (1 - o.cost_total.sum() / o.cost_gross.sum()):.1f}",
        ),
        ("Mean cost per generation (USD)", usd(o.cost_total.mean(), 6)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def table_cost(j):
    """Per-run cost (summing all LLM calls of a run)."""
    runs = j.groupby(["run_id", "scenario_type"], as_index=False).agg(
        net=("cost_total", "sum"),
        gross=("cost_gross", "sum"),
        calls=("call_idx", "size"),
    )
    d = with_overall(runs)
    out = []
    for s in scenario_index(d.scenario_type.unique()):
        g = d[d.scenario_type == s]
        out.append(
            {
                "Scenario": label(s),
                "Runs": len(g),
                "Net cost / run (USD)": f"{g.net.mean():.5f} ± {g.net.std(ddof=1):.5f}",
                "Gross cost / run (USD)": usd(g.gross.mean()),
                "Cache saving (%)": f"{100 * (1 - g.net.sum() / g.gross.sum()):.1f}",
                "Cost / 1,000 runs (USD)": f"{1000 * g.net.mean():.2f}",
                "Total (USD)": usd(g.net.sum(), 4),
                "Share of total cost (%)": (
                    f"{100 * g.net.sum() / runs.net.sum():.1f}"
                    if s != "Overall"
                    else "100.0"
                ),
            }
        )
    return pd.DataFrame(out)


def table_cost_by_routing(j):
    runs = j.groupby(
        ["run_id", "scenario_type", "routing_correct"], as_index=False
    ).agg(net=("cost_total", "sum"), calls=("call_idx", "size"))
    rows = []
    for s in scenario_index(runs.scenario_type.unique()):
        sub = runs[runs.scenario_type == s]
        if sub.routing_correct.nunique() < 2:
            continue
        for ok, g in sub.groupby("routing_correct"):
            rows.append(
                {
                    "Scenario": label(s),
                    "Routing correct": "Yes" if ok else "No",
                    "n": len(g),
                    "LLM calls / run": f"{g.calls.mean():.2f}",
                    "Net cost / run (USD)": f"{g.net.mean():.5f} ± {g.net.std(ddof=1):.5f}",
                }
            )
    return pd.DataFrame(rows)


def table_call_latency_cache(j):
    d = with_overall(j)
    rows = []
    for s in scenario_index(d.scenario_type.unique()):
        g = d[d.scenario_type == s]
        rows.append(
            {
                "Scenario": label(s),
                "LLM calls": len(g),
                "TTFT mean (ms)": f"{g.time_to_first_token_ms.mean():.0f}",
                "TTFT median (ms)": f"{g.time_to_first_token_ms.median():.0f}",
                "Gen. time mean (ms)": f"{g.generation_time_ms.mean():.0f}",
                "Gen. time P95 (ms)": f"{g.generation_time_ms.quantile(0.95):.0f}",
                "Gen. time max (ms)": f"{g.generation_time_ms.max():.0f}",
                "Cache hit rate (%)": f"{100 * g.tokens_cached.sum() / g.tokens_prompt.sum():.1f}",
                "Calls with cache hit (%)": f"{100 * g.cache_hit.mean():.1f}",
            }
        )
    return pd.DataFrame(rows)


def table_call_position(j):
    """How each successive LLM call in the agent loop behaves (context growth, caching)."""
    rows = []
    for i, g in j.groupby("call_idx"):
        rows.append(
            {
                "Call # in run": i,
                "n": len(g),
                "Prompt tok.": (
                    fmt_mean_sd(g.tokens_prompt, 0)
                    if len(g) > 1
                    else f"{g.tokens_prompt.iloc[0]}"
                ),
                "Completion tok.": f"{g.tokens_completion.mean():.0f}",
                "Cache hit rate (%)": f"{100 * g.tokens_cached.sum() / g.tokens_prompt.sum():.1f}",
                "Gen. time mean (ms)": f"{g.generation_time_ms.mean():.0f}",
                "Cost / call (USD)": usd(g.cost_total.mean(), 6),
            }
        )
    return pd.DataFrame(rows)


def table_latency_decomposition(j, k):
    """End-to-end latency = LLM generation time + tool overhead + residual (framework/network)."""
    gen = j.groupby("run_id").generation_time_ms.sum().rename("llm_ms")
    r = k.set_index("run_id").join(gen)
    r["residual_ms"] = r.latency_ms - r.llm_ms - r.tool_overhead_ms
    d = with_overall(r.reset_index())
    rows = []
    for s in scenario_index(d.scenario_type.unique()):
        g = d[d.scenario_type == s]
        rows.append(
            {
                "Scenario": label(s),
                "End-to-end (ms)": f"{g.latency_ms.mean():.0f}",
                "LLM generation (ms)": f"{g.llm_ms.mean():.0f}",
                "Tool overhead (ms)": f"{g.tool_overhead_ms.mean():.0f}",
                "Residual (ms)": f"{g.residual_ms.mean():.0f}",
                "LLM share (%)": f"{100 * g.llm_ms.sum() / g.latency_ms.sum():.1f}",
                "Residual share (%)": f"{100 * g.residual_ms.sum() / g.latency_ms.sum():.1f}",
                "Mean ms / LLM call": f"{g.llm_ms.sum() / g.llm_calls.sum():.0f}",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("openrouter_csv")
    ap.add_argument("--kpi", help="KPI csv (enables joined tables O2, O3, O5, O6)")
    ap.add_argument(
        "--api-key", help="only keep rows with this api_key_name (e.g. eval_3)"
    )
    ap.add_argument("--outdir", default="tables")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    o = load_openrouter(args.openrouter_csv, args.api_key)
    print(f"Loaded {len(o)} OpenRouter generations")

    save(table_overview(o), "O1_overview", "OpenRouter generation overview", outdir)

    if not args.kpi:
        print("\nNo --kpi given: skipping per-scenario tables.")
        return
    k = load([args.kpi])
    j, ok, msg = join_runs(o, k)
    print(f"\n{msg}")
    if not ok:
        print("Skipping joined tables.")
        return

    save(table_cost(j), "O2_cost", "Inference cost per run and scenario", outdir)
    t3 = table_cost_by_routing(j)
    if not t3.empty:
        save(
            t3,
            "O3_cost_by_routing",
            "Cost per run split by routing correctness",
            outdir,
        )
    save(
        table_call_latency_cache(j),
        "O4_call_latency_cache",
        "Per-call latency and prompt-cache hit rate per scenario",
        outdir,
    )
    save(
        table_call_position(j),
        "O5_call_position",
        "Behaviour of successive LLM calls within a run",
        outdir,
    )
    save(
        table_latency_decomposition(j, k),
        "O6_latency_decomposition",
        "Decomposition of end-to-end latency",
        outdir,
    )
    print(f"\nAll tables written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
