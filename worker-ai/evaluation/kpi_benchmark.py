"""
kpi_benchmark.py
=================

END-TO-END KPI benchmark/testing for HazardMapReduceManager against the
REAL stack: a real LLM (ChatOpenAI via `create_agent`), real tools from
`memgraph_custom_tools.get_memgraph_tools()`, and the real Memgraph MCP server
configured in that function. No component is mocked: to run this script,
you need a reachable Memgraph MCP server and real LLM credentials (read by
`agent_graph.AppSettings`, typically from a .env file), exactly as in the
normal execution of your agent.

Tested entry point: `HazardMapReduceManager.analyze_data(data)`.
If that method is still called `process_data` in your project, the script
will detect it automatically and use it as a fallback — see `_resolve_entrypoint()`.

WHAT IS MEASURED AND HOW (since there is no mock to read internal states from,
every KPI is observed externally):

  1. End-to-end latency: perf_counter() around the real call.
  2. Tool overhead: each tool returned by get_memgraph_tools() is instrumented
     (its real `.coroutine` is wrapped) to sum up the time spent in real
     calls to the MCP/Memgraph server.
  3. Schema adherence: if the call raises a Pydantic ValidationError (or an
     error whose message refers to it), the run is marked as non-compliant
     on the first attempt.
  4. Routing accuracy: heuristics on the returned `danger_type` (since we
     don't intercept the internal routing edges with the real graph):
     "fire" -> expected danger_type in {fire, smoke, heat};
     "earthquake" -> {earthquake}; "fire_earthquake" -> any of the two;
     "normal" -> no high-risk assessment expected.
  5. Danger_score variance: calculated at the end of the run (stdev) on the
     values actually returned by the LLM for each scenario.
  6. API drop rate: NO fake failures are injected here (it wouldn't make
     sense against a real system) — any network/API failure genuinely observed
     during the runs is simply classified and counted.
  7. Tool call count: number of real tool invocations per run (plus the number
     of distinct tools used) — same instrumentation as the overhead.
  8. LLM calls and tokens: by hooking a callback directly to `manager.model`
     (the only ChatOpenAI instance shared by Triage/Fire/Earthquake), every
     real call to the model for the run is intercepted, even if the three
     agents do not exchange messages with each other. If your `llm_base_url`
     is a gateway/proxy that doesn't return `usage` in the response,
     `llm_calls` will be > 0 but tokens will stay at 0 (the script will warn you).
  9. Assessment count: how many ThreatAssessments each run produces — useful
     for measuring how often the system propagates the evaluation to adjacent
     rooms instead of just the primary one.

TO BE CONFIGURED BEFORE USE
------------------------------
`SCENARIO_ROOMS` below and `DEPARTMENT` use placeholders. If your real
department/room in the graph is different, replace them — otherwise, the real
tools will return "no data found" for every run.

AGENT INPUT
------------------
REAL format confirmed via manual test (we don't change it, we adapt):

    {
        "room": "<department>:<room>",
        "sensor_data": [
            {"co2": ..., "temperature": ..., "humidity": ..., "tvoc": ...,
             "eco2": ..., "room": "<department>:<room>", "timestamp": "..."},
            ...  # typically 5, one for each of the last 5 minutes,
                 # from oldest to newest
        ],
    }

Each element in `sensor_data` is already the MAXIMUM value of that minute.
The seismic sensors (vibration_g/acceleration_g) don't have a real sensor
connected yet: they remain synthetic, appended to the same readings.
The lists are dynamically generated at each run by a per-scenario function
(`SCENARIO_GENERATORS`), so each run is a slightly different test case.

USAGE
---
    uv run python -m evaluation.kpi_benchmark --n 5 --snapshots 5 --seed 42 --output kpi_evaluation.csv

Tip: start with a low --n (3-5) for an initial real smoke test, since each
run implies a real LLM call + real graph query.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import re
import statistics
import sys
import time
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import ValidationError
from langchain_core.callbacks import AsyncCallbackHandler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent_graph import HazardMapReduceManager  # noqa: E402
from src.memgraph_custom_tools import get_memgraph_tools  # noqa: E402

# ----------------------------------------------------------------------------
# 1. DYNAMIC GENERATION OF TELEMETRY WINDOWS
# ----------------------------------------------------------------------------
# REAL format confirmed via manual testing: `ainvoke`/`analyze_data` expects
# a dict {"room": "<department>:<room>", "sensor_data": [...]}, where each
# element of sensor_data is the ALREADY aggregated (max) reading of a minute.
# The seismic fields do not have a real sensor connected yet: they remain
# synthetic as before.
#
# Each scenario is a function that returns a new payload upon each call (with
# random noise/trends via `rng`), so every run is slightly different.

DEPARTMENT = "departmenta"

# Room used for each scenario (must exist in the graph).
SCENARIO_ROOMS: dict[str, str] = {
    "normal": "ab1",
    "fire": "ab2",
    "earthquake": "ab3",
    "fire_earthquake": "ab2",  # (e.g. explosion)
}

SCENARIO_EXPECTED_DANGER_TYPES: dict[str, set[str]] = {
    "normal": {"none"},
    "fire": {"fire", "smoke", "heat"},
    "earthquake": {"earthquake"},
    "fire_earthquake": {"fire", "smoke", "heat", "earthquake", "other"},
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _trend_series(
    rng: random.Random,
    start: float,
    end: float,
    n: int,
    lo: float,
    hi: float,
    noise_frac: float = 0.05,
) -> list[float]:
    """Series of n values with linear trend start->end (event gradually worsening/improving) plus noise."""
    span = abs(end - start)
    values = []
    for i in range(n):
        t = i / max(n - 1, 1)
        base = start + (end - start) * t
        noisy = base + rng.gauss(0, span * noise_frac + 1e-6)
        values.append(round(_clamp(noisy, lo, hi), 3))
    return values


def _step_series(
    rng: random.Random,
    baseline: float,
    spike: float,
    n: int,
    onset_index: int,
    lo: float,
    hi: float,
    noise_frac: float = 0.05,
) -> list[float]:
    """Series with a sudden jump at minute `onset_index` (sudden event, e.g. explosion)."""
    span = abs(spike - baseline)
    values = []
    for i in range(n):
        base = baseline if i < onset_index else spike
        noisy = base + rng.gauss(0, span * noise_frac + 1e-6)
        values.append(round(_clamp(noisy, lo, hi), 3))
    return values


def _flat_series(
    rng: random.Random,
    level: float,
    n: int,
    lo: float,
    hi: float,
    noise_frac: float = 0.05,
) -> list[float]:
    return [
        round(_clamp(level + rng.gauss(0, level * noise_frac + 1e-6), lo, hi), 3)
        for _ in range(n)
    ]


def _assemble_window(
    room: str,
    department: str,
    n: int,
    co2: list[float],
    temperature: list[float],
    humidity: list[float],
    tvoc: list[float],
    eco2: list[float],
    vibration_g: list[float],
    acceleration_g: list[float],
) -> dict[str, Any]:
    """Builds the REAL payload: {"room": "<department>:<room>", "sensor_data": [...]},
    from oldest minute (index 0) to newest (index n-1)."""
    combined_room = f"{department}:{room}"
    now = datetime.now()
    sensor_data = [
        {
            "co2": round(co2[i]),
            "temperature": temperature[i],
            "humidity": humidity[i],
            "tvoc": round(tvoc[i]),
            "eco2": round(eco2[i]),
            "vibration_g": vibration_g[i],
            "acceleration_g": acceleration_g[i],
            "room": combined_room,
            "timestamp": (now - timedelta(minutes=(n - 1 - i))).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        }
        for i in range(n)
    ]
    return {"room": combined_room, "sensor_data": sensor_data}


def generate_normal_window(
    rng: random.Random, room: str, department: str, n: int = 5
) -> dict[str, Any]:
    return _assemble_window(
        room,
        department,
        n,
        co2=_flat_series(rng, 420, n, 380, 500),
        temperature=_flat_series(rng, 24.0, n, 18, 28),
        humidity=_flat_series(rng, 55.0, n, 40, 65),
        tvoc=_flat_series(rng, 80, n, 0, 250),
        eco2=_flat_series(rng, 450, n, 400, 600),
        vibration_g=_flat_series(rng, 0.01, n, 0.0, 0.03),
        acceleration_g=_flat_series(rng, 0.02, n, 0.0, 0.05),
    )


def generate_fire_window(
    rng: random.Random, room: str, department: str, n: int = 5
) -> dict[str, Any]:
    # Gradual escalation: a fire developing minute by minute.
    return _assemble_window(
        room,
        department,
        n,
        co2=_trend_series(rng, 600, 2400, n, 380, 5000),
        temperature=_trend_series(rng, 30.0, 90.0, n, 18, 300),
        humidity=_trend_series(rng, 55.0, 15.0, n, 0, 100),
        tvoc=_trend_series(rng, 300, 59000, n, 0, 60000),
        eco2=_trend_series(rng, 600, 59000, n, 400, 60000),
        vibration_g=_flat_series(rng, 0.01, n, 0.0, 0.03),
        acceleration_g=_flat_series(rng, 0.02, n, 0.0, 0.05),
    )


def generate_earthquake_window(
    rng: random.Random, room: str, department: str, n: int = 5
) -> dict[str, Any]:
    # Gradual seismic escalation; air/gas remain at baseline.
    return _assemble_window(
        room,
        department,
        n,
        co2=_flat_series(rng, 420, n, 380, 500),
        temperature=_flat_series(rng, 24.0, n, 18, 28),
        humidity=_flat_series(rng, 55.0, n, 40, 65),
        tvoc=_flat_series(rng, 80, n, 0, 250),
        eco2=_flat_series(rng, 450, n, 400, 600),
        vibration_g=_trend_series(rng, 0.05, 0.55, n, 0.0, 2.0),
        acceleration_g=_trend_series(rng, 0.05, 0.60, n, 0.0, 2.0),
    )


def generate_fire_and_earthquake_window(
    rng: random.Random, room: str, department: str, n: int = 5
) -> dict[str, Any]:
    # Sudden and simultaneous event across all sensors (e.g. explosion).
    onset = max(1, n // 2)
    return _assemble_window(
        room,
        department,
        n,
        co2=_step_series(rng, 420, 2600, n, onset, 380, 5000),
        temperature=_step_series(rng, 21.0, 95.0, n, onset, 18, 300),
        humidity=_step_series(rng, 55.0, 10.0, n, onset, 0, 100),
        tvoc=_step_series(rng, 80, 59500, n, onset, 0, 60000),
        eco2=_step_series(rng, 450, 59500, n, onset, 400, 60000),
        vibration_g=_step_series(rng, 0.01, 0.70, n, onset, 0.0, 2.0),
        acceleration_g=_step_series(rng, 0.02, 0.75, n, onset, 0.0, 2.0),
    )


SCENARIO_GENERATORS: dict[
    str, Callable[[random.Random, str, str, int], dict[str, Any]]
] = {
    "normal": generate_normal_window,
    "fire": generate_fire_window,
    "earthquake": generate_earthquake_window,
    "fire_earthquake": generate_fire_and_earthquake_window,
}

# Recognizes combined IDs like "departmenta:ab1" and simple IDs like "ab1"/"R101".
ROOM_ID_PATTERN = re.compile(r"\b([A-Za-z][\w-]*:[\w-]+|[A-Za-z]{1,4}\d{1,4})\b")

CSV_COLUMNS = [
    "run_id",
    "scenario_type",
    "latency_ms",
    "tool_overhead_ms",
    "tool_call_count",
    "distinct_tools_used",
    "llm_calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "assessment_count",
    "schema_adherence",
    "routing_correct",
    "danger_score",
    "api_failed",
]

# ----------------------------------------------------------------------------
# 2. REAL TOOLS INSTRUMENTATION (overhead + call counting)
# ----------------------------------------------------------------------------

# Each element: (tool_name, duration_ms) — one entry per real invocation.
_tool_calls: ContextVar[list[tuple[str, float]]] = ContextVar("tool_calls")


def instrument_tools(tools: list[Any]) -> list[Any]:
    """Wraps the real `.coroutine` of each tool to sum its duration and count calls."""
    for t in tools:
        if getattr(t, "coroutine", None) is None:
            continue  # synchronous-only tool: ignored here
        original: Callable[..., Awaitable[Any]] = t.coroutine
        tool_name = t.name

        async def wrapped(
            *args: Any, _original=original, _name=tool_name, **kwargs: Any
        ) -> Any:
            t0 = time.perf_counter()
            try:
                return await _original(*args, **kwargs)
            finally:
                bucket = _tool_calls.get(None)
                if bucket is not None:
                    bucket.append((_name, (time.perf_counter() - t0) * 1000))

        t.coroutine = wrapped
    return tools


class TokenUsageCallback(AsyncCallbackHandler):
    """Callback hooked directly to `manager.model` (the single ChatOpenAI instance).
    Intercepts EVERY real LLM call of the current run."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    async def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.llm_calls += 1
        usage: dict[str, Any] | None = None

        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage")

        if not usage:
            # Fallback: AIMessage.usage_metadata (newer langchain-core format).
            try:
                message = response.generations[0][0].message
                usage_meta = getattr(message, "usage_metadata", None)
                if usage_meta:
                    usage = {
                        "prompt_tokens": usage_meta.get("input_tokens", 0),
                        "completion_tokens": usage_meta.get("output_tokens", 0),
                        "total_tokens": usage_meta.get("total_tokens", 0),
                    }
            except (IndexError, AttributeError, TypeError):
                usage = None

        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
            self.completion_tokens += usage.get("completion_tokens", 0) or 0
            self.total_tokens += usage.get("total_tokens", 0) or 0


# ----------------------------------------------------------------------------
# 3. MANAGER ENTRYPOINT — uses analyze_data, with fallback to process_data
# ----------------------------------------------------------------------------


def _resolve_entrypoint(
    manager: HazardMapReduceManager,
) -> Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]]:
    entrypoint = getattr(manager, "analyze_data", None)
    if entrypoint is not None:
        return entrypoint
    entrypoint = getattr(manager, "process_data", None)
    if entrypoint is not None:
        print("[INFO] 'analyze_data' not found: using 'process_data' as entrypoint.")
        return entrypoint
    raise AttributeError(
        "HazardMapReduceManager exposes neither 'analyze_data' nor 'process_data'."
    )


# ----------------------------------------------------------------------------
# 4. REAL ERROR CLASSIFICATION (KPI 3 / KPI 6)
# ----------------------------------------------------------------------------


def classify_exception(exc: Exception) -> tuple[bool, bool]:
    """Returns (schema_adherence_ok, api_failed) starting from a real exception."""
    if isinstance(exc, ValidationError):
        return False, False
    message = str(exc).lower()
    if any(k in message for k in ("validation", "schema", "pydantic")):
        return False, False
    if any(
        k in message
        for k in (
            "timeout",
            "connection",
            "network",
            "unreachable",
            "rate limit",
            "429",
            "econn",
        )
    ):
        return True, True
    # Unclassified failure: treat it as a generic drop.
    return True, True


# ----------------------------------------------------------------------------
# 5. EXECUTION OF A SINGLE REAL RUN
# ----------------------------------------------------------------------------


async def run_single(
    entrypoint: Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]],
    model: Any,
    scenario_key: str,
    run_id: int,
    room: str,
    rng: random.Random,
    n_snapshots: int,
) -> dict[str, Any]:
    token = _tool_calls.set([])
    token_handler = TokenUsageCallback()
    model.callbacks = [token_handler]  # hooked to the single shared ChatOpenAI instance

    row: dict[str, Any] = {
        "run_id": run_id,
        "scenario_type": scenario_key,
        "latency_ms": None,
        "tool_overhead_ms": 0.0,
        "tool_call_count": 0,
        "distinct_tools_used": 0,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "assessment_count": 0,
        "schema_adherence": True,
        "routing_correct": False,
        "danger_score": None,
        "api_failed": False,
    }

    # {"room": "<department>:<room>", "sensor_data": [...]}, generated ad-hoc for this run.
    window = SCENARIO_GENERATORS[scenario_key](rng, room, DEPARTMENT, n_snapshots)

    t0 = time.perf_counter()
    try:
        assessments = await entrypoint(window)
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        calls = _tool_calls.get([])
        row["tool_overhead_ms"] = round(sum(ms for _, ms in calls), 2)
        row["tool_call_count"] = len(calls)
        row["distinct_tools_used"] = len({name for name, _ in calls})

        row["llm_calls"] = token_handler.llm_calls
        row["prompt_tokens"] = token_handler.prompt_tokens
        row["completion_tokens"] = token_handler.completion_tokens
        row["total_tokens"] = token_handler.total_tokens

        row["assessment_count"] = len(assessments)
        row["danger_score"] = assessments[0]["danger_score"] if assessments else None

        # --- KPI 4: routing accuracy (observed danger_type heuristic) ---
        expected_types = SCENARIO_EXPECTED_DANGER_TYPES[scenario_key]
        observed_types = {a.get("danger_type") for a in assessments}
        if expected_types == {"none"}:
            row["routing_correct"] = not observed_types & {
                "fire",
                "smoke",
                "heat",
                "earthquake",
                "other",
            }
        else:
            row["routing_correct"] = bool(observed_types & expected_types)

    except Exception as exc:
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        calls = _tool_calls.get([])
        row["tool_overhead_ms"] = round(sum(ms for _, ms in calls), 2)
        row["tool_call_count"] = len(calls)
        row["distinct_tools_used"] = len({name for name, _ in calls})
        row["llm_calls"] = token_handler.llm_calls
        row["prompt_tokens"] = token_handler.prompt_tokens
        row["completion_tokens"] = token_handler.completion_tokens
        row["total_tokens"] = token_handler.total_tokens
        schema_ok, api_failed = classify_exception(exc)
        row["schema_adherence"] = schema_ok
        row["api_failed"] = api_failed
        print(
            f"[run {run_id}] {scenario_key}: real exception -> {type(exc).__name__}: {exc}"
        )
    finally:
        _tool_calls.reset(token)

    return row


# ----------------------------------------------------------------------------
# 6. MAIN HARNESS
# ----------------------------------------------------------------------------


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== KPI Summary by Scenario (real stack) ===")
    for scenario_key in SCENARIO_GENERATORS:
        subset = [r for r in rows if r["scenario_type"] == scenario_key]
        if not subset:
            continue
        n = len(subset)
        latencies = [r["latency_ms"] for r in subset if r["latency_ms"] is not None]
        overheads = [r["tool_overhead_ms"] for r in subset]
        tool_calls = [r["tool_call_count"] for r in subset]
        llm_calls = [r["llm_calls"] for r in subset]
        total_tokens = [r["total_tokens"] for r in subset]
        assessment_counts = [r["assessment_count"] for r in subset]
        scores = [r["danger_score"] for r in subset if r["danger_score"] is not None]
        schema_rate = sum(r["schema_adherence"] for r in subset) / n
        routing_rate = sum(r["routing_correct"] for r in subset) / n
        api_fail_rate = sum(r["api_failed"] for r in subset) / n
        score_stdev = statistics.pstdev(scores) if len(scores) >= 1 else float("nan")

        print(f"\n--- Scenario: {scenario_key} (n={n}) ---")
        print(
            f"  Average latency:          {statistics.mean(latencies):.1f} ms"
            if latencies
            else "  Average latency:          n/a"
        )
        print(f"  Average tool overhead:    {statistics.mean(overheads):.1f} ms")
        print(f"  Average tool calls:       {statistics.mean(tool_calls):.1f}")
        print(f"  Average LLM calls:        {statistics.mean(llm_calls):.1f}")
        print(
            f"  Average total tokens:     {statistics.mean(total_tokens):.1f}"
            + (
                "  [WARNING: always 0 -> the LLM endpoint likely does not expose usage]"
                if sum(total_tokens) == 0 and sum(llm_calls) > 0
                else ""
            )
        )
        print(f"  Average assessments/run:  {statistics.mean(assessment_counts):.2f}")
        print(f"  Schema adherence rate:    {schema_rate:.1%}")
        print(f"  Routing accuracy:         {routing_rate:.1%}")
        print(f"  Danger score stdev:       {score_stdev:.4f}")
        print(f"  Observed API drop rate:   {api_fail_rate:.1%}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="END-TO-END KPI benchmark (real stack) for HazardMapReduceManager"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of runs per scenario (real LLM calls)",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=5,
        help="Objects per window (aggregated minutes) sent to the agent",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed to reproduce the generated windows",
    )
    parser.add_argument("--output", type=str, default="kpi_evaluation.csv")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    manager = HazardMapReduceManager()
    tools = await get_memgraph_tools()  # REAL connection to Memgraph MCP server
    tools = instrument_tools(tools)
    await manager.initialize_graph(tools=tools)
    entrypoint = _resolve_entrypoint(manager)

    rows: list[dict[str, Any]] = []
    run_id = 0
    output_path = Path(args.output)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        f.flush()

        for scenario_key, room in SCENARIO_ROOMS.items():
            for _ in range(args.n):
                row = await run_single(
                    entrypoint,
                    manager.model,
                    scenario_key,
                    run_id,
                    room,
                    rng,
                    args.snapshots,
                )
                rows.append(row)
                writer.writerow(row)
                f.flush()  # written immediately to disk, not buffered until the end
                print(
                    f"[run {run_id}] {scenario_key}: written to CSV "
                    f"(latency {row['latency_ms']} ms, {row['llm_calls']} LLM calls, "
                    f"{row['tool_call_count']} tool calls, {row['total_tokens']} tokens)"
                )
                run_id += 1

    print(f"\nWritten {len(rows)} runs to {output_path.resolve()}")
    print_summary(rows)


if __name__ == "__main__":
    asyncio.run(main())
