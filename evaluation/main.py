"""
Evaluation script for static vs. adaptive (hazard-aware) routing on a Memgraph
graph of Places/Hazards.

Requirements:
    pip install neo4j pandas

Assumptions made (see conversation for full list; the ones NOT explicitly
confirmed by the user are marked with [ASSUMPTION]):

  - Connection: bolt://localhost:7687, no auth (Memgraph in Docker, default config).
  - Best path per origin = the path with the MINIMUM total_weight among all
    rows returned by the query (i.e. among all reachable gathering points),
    regardless of which target it leads to.
  - Hazard exposure = number of nodes in the path (origin AND target included)
    whose id is also returned by DANGERZONES_QUERY.
  - Route success/valid = the query returned at least one row for that origin.
  - Hazard-free route = a successful route whose hazard exposure == 0.
  - HFR (Hazard-Free Rate)   = hazard_free_adaptive_routes / valid_adaptive_routes * 100
  - RSR (Route Feasibility)  = origins_with_safe_adaptive_route / total_origins * 100
        where "safe route" = adaptive route found AND hazard exposure == 0.
  - Safe Path Cost Overhead  = (W_adaptive - W_static) / W_static
        -> "N/A" if either static or adaptive path is missing for that origin.
  - Node name in the path string = the `name` property of each Place node,
    joined with "-" (e.g. "A1-A2-E1"), origin and target included.
  - Timing: one warm-up query is executed before the loop starts (not
    measured); each subsequent static/adaptive query is timed individually
    with time.perf_counter().
  - [ASSUMPTION] `allowed_ids` for ADAPTIVE_MOVEMENT_QUERY = ids of the nodes
    returned by SAFEZONES_QUERY (i.e. places NOT affected by a medium/high
    hazard). This parameter is required by the query but was not specified
    by the user; change GET_SAFEZONE_IDS usage below if a different set of
    "allowed" nodes was intended.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from neo4j import GraphDatabase, Record

from graph_injection import plain_graph

# --------------------------------------------------------------------------
# Connection settings
# --------------------------------------------------------------------------
MEMGRAPH_URI = "bolt://localhost:7687"
MEMGRAPH_AUTH = ("", "")  # no auth by default on a local Memgraph docker container

# --------------------------------------------------------------------------
# Cypher queries (from main.py)
# --------------------------------------------------------------------------
SAFEZONES_QUERY = """
MATCH (p:Place) WHERE NOT EXISTS {   MATCH (p)-[:AFFECTS]-(h:Hazard) WHERE NOT h.level IN ['low', 'none'] } RETURN DISTINCT p;
"""
DANGERZONES_QUERY = """
MATCH (p:Place) WHERE EXISTS { MATCH (p)-[:AFFECTS]-(h:Hazard) WHERE h.level IN ['medium', 'high']} RETURN DISTINCT p;
"""
BUILDING_QUERY = """
MATCH (p:Place) WHERE p.amenity = 'university' RETURN p;
"""
GATHERINGPOINTS_QUERY = """
MATCH (p:Place) WHERE p.amenity = 'gatheringPoint' RETURN p;
"""
# Topological degree (distinct neighbours) of every origin room. Rooms with
# degree 1 are structural "single points of failure" (SPOF): if their one
# and only corridor becomes hazardous, NO route can ever exist for them,
# regardless of how good the adaptive algorithm is. Counting DISTINCT
# neighbours (not relationships) matters here because every edge in this
# graph is stored twice (A->B and B->A), so counting relationships directly
# would double every degree.
NODE_DEGREE_QUERY = """
MATCH (p:Place)-[:CONNECTED_TO]-(n:Place)
WHERE p.amenity = 'university'
RETURN p.name AS name, count(DISTINCT n) AS degree;
"""

SHORTEST_PATH_QUERY = """
MATCH (source:Place)
WHERE source.name = $start_name
MATCH path = (source)-[edges:CONNECTED_TO *WSHORTEST (e, v | toFloat(v.weight)) total_weight]->(target:Place)
WHERE id(target) IN $target_ids
RETURN target AS target_id, path, total_weight;
"""

ADAPTIVE_MOVEMENT_QUERY = """
MATCH (source:Place)
WHERE source.name = $start_name
AND id(source) IN $allowed_ids
MATCH path = (source)-[edges:CONNECTED_TO *WSHORTEST (e, v |
    CASE
        WHEN id(v) IN $allowed_ids THEN v.weight
        ELSE 999999999
    END
) total_weight]->(target:Place)
WHERE id(target) IN $target_ids AND total_weight < 999999999
RETURN target AS target_id, path, total_weight;
"""


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------
@dataclass
class PathResult:
    found: bool
    node_names: str = "N/A"
    node_ids: list = field(default_factory=list)
    weight: Optional[float] = None
    hazard_exposure: Optional[int] = None
    time_sec: Optional[float] = None  # mean over n_timing_repeats runs
    time_std_sec: Optional[float] = None  # 0.0 if n_timing_repeats == 1


@dataclass
class OriginRow:
    origin: str
    static: PathResult = field(default_factory=lambda: PathResult(found=False))
    adaptive: PathResult = field(default_factory=lambda: PathResult(found=False))
    degree: Optional[int] = None
    structural_class: str = "unknown"
    origin_in_hazard_zone: bool = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def get_place_names(records: list[Record]) -> list[str]:
    """Extract p.name from records shaped as RETURN p / RETURN DISTINCT p."""
    return [r["p"]["name"] for r in records]


def get_place_ids(records: list[Record]) -> list[int]:
    """Extract the internal Memgraph node id from records shaped as RETURN p.

    IMPORTANT: we must use `.id` (plain integer), NOT `.element_id`.
    The neo4j Python driver's `.element_id` returns a Neo4j-5-style string
    (e.g. "4:xxxxx:12"), which is meaningless to Memgraph's `id()` Cypher
    function used in the WHERE clauses (`id(target) IN $target_ids`, etc.).
    Passing element_id strings there makes every `IN` comparison fail
    silently, which is what produces all-N/A results.
    """
    return [r["p"].id for r in records]


def get_node_degrees(session) -> dict:
    """name -> topological degree (distinct CONNECTED_TO neighbours), for
    every 'university' Place. Rooms not returned by this query at all (no
    CONNECTED_TO relationship whatsoever) are treated as degree 0 by the
    caller via dict.get(name, 0)."""
    records = list(session.run(NODE_DEGREE_QUERY))
    return {r["name"]: r["degree"] for r in records}


def classify_degree(degree: Optional[int]) -> str:
    """Structural class used throughout the evaluation tables:
    - SPOF (degree 1): a single corridor connects this room to the rest
      of the building. If that corridor becomes hazardous, evacuation is
      topologically impossible, independent of the routing algorithm.
    - branch (degree 2): exactly two ways out; hazard on one alone does
      not necessarily strand the origin, but hazard on both does.
    - hub (degree >=3): multiple alternatives, the case the adaptive
      framework is best positioned to actually demonstrate value on.
    """
    if degree is None:
        return "unknown"
    if degree <= 1:
        return "SPOF (degree 1)"
    if degree == 2:
        return "branch (degree 2)"
    return "hub (degree >=3)"


def build_hazard_query(room_name: str, level: str = "high") -> str:
    """Single source of truth for 'create a hazard on this room' Cypher,
    reused by every scenario below (previously each scenario duplicated
    this block by hand, which is how S1/S3/S2 ended up with slightly
    different payload text for no real reason)."""
    return f"""
    MERGE (p:Place {{name: "{room_name}"}})
    CREATE (h:Hazard {{
      department: "departmentA",
      room: "{room_name}",
      warning: "pre-alert",
      level: "{level}",
      type: "smoke",
      score: 0.85,
      justification: "Elevated TVOC (60,000 ppm) and CO2 (445-447 ppm) levels indicate potential combustion o..."
    }})
    CREATE (h)-[:AFFECTS]->(p);
    """


def pick_best_record(records: list[Record]) -> Optional[Record]:
    """Among all rows returned (one per reachable target), keep the one
    with the minimum total_weight, independent of which target it reaches."""
    if not records:
        return None
    return min(records, key=lambda r: r["total_weight"])


def build_path_result(
    record: Optional[Record],
    dangerzone_ids: set,
    elapsed: float,
    elapsed_std: float = 0.0,
) -> PathResult:
    if record is None:
        return PathResult(found=False, time_sec=elapsed, time_std_sec=elapsed_std)

    path = record["path"]
    nodes = path.nodes  # neo4j.graph.Path -> tuple of Node objects, in traversal order

    names = [n.get("name", "?") for n in nodes]
    node_ids = [
        n.id for n in nodes
    ]  # see note in get_place_ids() about element_id vs id

    hazard_count = sum(1 for nid in node_ids if nid in dangerzone_ids)

    return PathResult(
        found=True,
        node_names="-".join(names),
        node_ids=node_ids,
        weight=record["total_weight"],
        hazard_exposure=hazard_count,
        time_sec=elapsed,
        time_std_sec=elapsed_std,
    )


def timed_query(session, query: str, params: dict) -> tuple[list[Record], float]:
    start = time.perf_counter()
    result = list(session.run(query, params))
    elapsed = time.perf_counter() - start
    return result, elapsed


def timed_query_repeated(
    session, query: str, params: dict, n_repeats: int = 1
) -> tuple[list[Record], float, float]:
    """Run the same query n_repeats times and return (records, mean_sec,
    std_sec). The records returned are from the LAST run (the topology
    does not change between repeats within a single evaluate() call, so
    this is safe and avoids re-parsing paths n_repeats times).

    This directly addresses the thesis-draft limitation "each query is
    timed only once, without repetitions/averaging" -- with n_repeats=1
    behaviour is identical to the previous single-shot timed_query()."""
    times = []
    result: list[Record] = []
    for _ in range(max(1, n_repeats)):
        start = time.perf_counter()
        result = list(session.run(query, params))
        times.append(time.perf_counter() - start)

    mean_t = sum(times) / len(times)
    std_t = (
        (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
        if len(times) > 1
        else 0.0
    )
    return result, mean_t, std_t


# --------------------------------------------------------------------------
# Main evaluation logic
# --------------------------------------------------------------------------
def evaluate(
    session, n_timing_repeats: int = 5, node_degrees: Optional[dict] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    n_timing_repeats: each static/adaptive query is run this many times per
        origin and the MEAN/STD is reported, instead of a single
        perf_counter() sample. This directly addresses the "adaptive
        faster than static on a single, unrepeated measurement" limitation
        -- with n_timing_repeats=1 the behaviour is identical to before.
    node_degrees: precomputed {name: degree} map (see get_node_degrees).
        If None, computed once here. Passed explicitly by
        run_dynamic_hazard_scenario() so it is only computed once for the
        whole S2 timeline instead of once per step (degree is a property
        of the building graph and does not change across hazard steps).
    """
    # --- Reference sets -----------------------------------------------
    origins_records = list(session.run(BUILDING_QUERY))
    origins = get_place_names(origins_records)
    origin_name_to_id = dict(zip(origins, get_place_ids(origins_records)))

    gathering_records = list(session.run(GATHERINGPOINTS_QUERY))
    target_ids = get_place_ids(gathering_records)

    dangerzone_records = list(session.run(DANGERZONES_QUERY))
    dangerzone_ids = set(get_place_ids(dangerzone_records))

    # [ASSUMPTION] allowed_ids for the adaptive query = safe-zone node ids
    safezone_records = list(session.run(SAFEZONES_QUERY))
    allowed_ids = get_place_ids(safezone_records)

    if node_degrees is None:
        node_degrees = get_node_degrees(session)

    print(
        f"[debug] origins={len(origins)} "
        f"target_ids={len(target_ids)} "
        f"dangerzone_ids={len(dangerzone_ids)} "
        f"allowed_ids(safezones)={len(allowed_ids)}"
    )

    if not origins:
        raise RuntimeError("BUILDING_QUERY returned no origins.")
    if not target_ids:
        raise RuntimeError("GATHERINGPOINTS_QUERY returned no targets.")

    # --- Warm-up (not timed) -------------------------------------------
    warmup_params = {"start_name": origins[0], "target_ids": target_ids}
    session.run(SHORTEST_PATH_QUERY, warmup_params).consume()

    rows: list[OriginRow] = []

    for origin_name in origins:
        row = OriginRow(origin=origin_name)
        row.degree = node_degrees.get(origin_name, 0)
        row.structural_class = classify_degree(row.degree)
        row.origin_in_hazard_zone = origin_name_to_id.get(origin_name) in dangerzone_ids

        # --- Static shortest path ---------------------------------------
        static_params = {"start_name": origin_name, "target_ids": target_ids}
        static_records, static_time, static_std = timed_query_repeated(
            session, SHORTEST_PATH_QUERY, static_params, n_timing_repeats
        )
        best_static = pick_best_record(static_records)
        row.static = build_path_result(
            best_static, dangerzone_ids, static_time, static_std
        )

        # --- Adaptive (hazard-aware) shortest path ----------------------
        adaptive_params = {
            "start_name": origin_name,
            "target_ids": target_ids,
            "allowed_ids": allowed_ids,
        }
        adaptive_records, adaptive_time, adaptive_std = timed_query_repeated(
            session, ADAPTIVE_MOVEMENT_QUERY, adaptive_params, n_timing_repeats
        )
        best_adaptive = pick_best_record(adaptive_records)
        row.adaptive = build_path_result(
            best_adaptive, dangerzone_ids, adaptive_time, adaptive_std
        )

        rows.append(row)

    # --- Build the per-origin DataFrame ---------------------------------
    records_for_df = []
    for row in rows:
        s, a = row.static, row.adaptive

        # Safe Path Cost Overhead = (W_adaptive - W_static) / W_static
        if s.found and a.found and s.weight not in (None, 0):
            overhead = (a.weight - s.weight) / s.weight
        else:
            overhead = "N/A"

        records_for_df.append(
            {
                "origin": row.origin,
                "Origin Degree": row.degree,
                "Structural Class": row.structural_class,
                "Origin In Hazard Zone": row.origin_in_hazard_zone,
                "Static Path": s.node_names,
                "Static Cost": float(s.weight) if s.found else "N/A",
                "Static Hazard Exposure": int(s.hazard_exposure) if s.found else "N/A",
                "Static Success Rate": s.found,
                "Static Time (ms)": (
                    round(s.time_sec * 1000, 3) if s.time_sec is not None else "N/A"
                ),
                "Static Time Std (ms)": (
                    round(s.time_std_sec * 1000, 3)
                    if s.time_std_sec is not None
                    else "N/A"
                ),
                "Adaptive Path": a.node_names,
                "Adaptive Cost": float(a.weight) if a.found else "N/A",
                "Adaptive Hazard Exposure": (
                    int(a.hazard_exposure) if a.found else "N/A"
                ),
                "Adaptive Success Rate": a.found,
                "Adaptive Time (ms)": (
                    round(a.time_sec * 1000, 3) if a.time_sec is not None else "N/A"
                ),
                "Adaptive Time Std (ms)": (
                    round(a.time_std_sec * 1000, 3)
                    if a.time_std_sec is not None
                    else "N/A"
                ),
                "Safety Overhead (%)": (
                    round(float(overhead) * 100, 2)
                    if overhead not in (None, "N/A", "")
                    else "N/A"
                ),
            }
        )

    df = pd.DataFrame(records_for_df)

    # --- Aggregate metrics (dataset-wide, NOT per-origin) ---------------
    # These describe the whole run, not a single row, so they live in a
    # separate one-row summary DataFrame instead of being repeated on
    # every line of the per-origin table.
    total_origins = len(rows)
    valid_adaptive = sum(1 for r in rows if r.adaptive.found)
    hazard_free_adaptive = sum(
        1 for r in rows if r.adaptive.found and r.adaptive.hazard_exposure == 0
    )
    origins_with_safe_route = hazard_free_adaptive  # adaptive found AND hazard-free

    # Origins where the hazard sits ON the origin room itself are a trivial/
    # degenerate case (the route is "unsafe" from step 0 by definition, no
    # routing algorithm can fix that) and were previously silently mixed
    # into the same RSR/HFR denominator as genuine reroute cases, which is
    # exactly what produced the "RSR 33% on the affected subset vs 85% on
    # the total" discrepancy noted in the thesis draft. Report both.
    trivial_self_hazard = sum(1 for r in rows if r.origin_in_hazard_zone)
    non_trivial_total = total_origins - trivial_self_hazard

    hfr = (hazard_free_adaptive / valid_adaptive * 100) if valid_adaptive > 0 else 0.0
    rsr = (origins_with_safe_route / total_origins * 100) if total_origins > 0 else 0.0
    rsr_excl_trivial = (
        (origins_with_safe_route / non_trivial_total * 100)
        if non_trivial_total > 0
        else "N/A"
    )

    spof_origins = sum(1 for r in rows if r.structural_class == "SPOF (degree 1)")

    summary_df = pd.DataFrame(
        [
            {
                "Total Origins": total_origins,
                "SPOF Origins (degree 1)": spof_origins,
                "Trivial Self-Hazard Origins": trivial_self_hazard,
                "Valid Adaptive Routes": valid_adaptive,
                "Hazard-free adaptive routes": hazard_free_adaptive,
                "Origins with safe route": origins_with_safe_route,
                "Hazard-free route rate HFR%": round(hfr, 2),
                "Route feasibility RSR% (all origins)": round(rsr, 2),
                "Route feasibility RSR% (excl. trivial self-hazard)": (
                    round(rsr_excl_trivial, 2) if rsr_excl_trivial != "N/A" else "N/A"
                ),
            }
        ]
    )

    return df, summary_df


def create_df(df, summary_df, suffix):
    output_path = "route_evaluation" + suffix + ".csv"
    summary_path = "route_evaluation_summary" + suffix + ".csv"

    df.to_csv(output_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Per-origin evaluation table written to {output_path}")
    print(df.to_string(index=False))
    print()
    print(f"Aggregate summary written to {summary_path}")
    print(summary_df.to_string(index=False))


# --------------------------------------------------------------------------
# S2 — Dynamic Hazard / Hazard Propagation + Route Adaptation Rate (RAR)
# --------------------------------------------------------------------------
# The static-scenario runs above (no_hazard / hazard_on_C1 / multi_hazard)
# are three INDEPENDENT snapshots of the graph: each one calls plain_graph()
# and evaluates from scratch. That is enough for S0, S1 and S3, but S2 in
# the thesis text is explicitly about a hazard that GROWS OVER TIME on the
# same run, where the framework must notice that a previously-safe route
# has become unsafe and recompute it. That requires comparing the route
# used at step t-1 against the route computed at step t for the SAME
# origin, which the functions above don't do. This function adds exactly
# that, without changing anything in evaluate()/create_df().
#
# Route Adaptation Rate (RAR), as defined in the thesis text:
#
#     RAR = correctly_adapted_routes / routes_requiring_adaptation * 100
#
# For a given origin and a given step transition (t-1 -> t):
#   - "requires adaptation"  := the adaptive route used at t-1 passed through
#     at least one node that became hazardous exactly at step t (i.e. it is
#     in current_dangerzone_ids but was NOT in prev_dangerzone_ids).
#   - "correctly adapted"    := at step t the framework returns
#         * a new adaptive route with hazard_exposure == 0 (hazard avoided), OR
#         * no route at all, IF no hazard-free route exists any more
#           ("no feasible evacuation route" is the CORRECT answer here,
#           consistent with how S3 defines correctness).
#     Because ADAPTIVE_MOVEMENT_QUERY enforces hazard-avoidance as a hard
#     constraint (non-allowed nodes get cost 999999999 and are filtered out),
#     any route it does return is safe by construction; "correct" therefore
#     reduces to "found a route" OR "correctly found none". The
#     'adapted_but_unsafe' branch below is kept only as a sanity check and
#     should never actually trigger — if it does, there is a bug in the
#     hazard-avoidance constraint itself.
def run_dynamic_hazard_scenario(driver, csv_prefix: str = "route_evaluation_dynamic"):
    """Run scenario S2 and report the Route Adaptation Rate.

    Hazard timeline simulated (cumulative, same graph throughout):
        t0_no_hazard   -> baseline, no hazard
        t1_C1          -> C1 becomes hazardous
        t2_C1_E7       -> E7 also becomes hazardous
        t3_C1_E7_E5    -> E5 also becomes hazardous

    Adjust ROOM NAMES / hazard payloads below to match whichever corridors
    are actually on the shortest routes in your building graph -- the rooms
    used here (C1, E7, E5) are the same ones already used for the
    hazard_on_C1 / multi_hazard scenarios earlier in this file, chosen so
    the propagation is consistent with the rest of the evaluation.

    Output files:
        {csv_prefix}.csv               per-origin, per-step detail table
        {csv_prefix}_step_summary.csv  one row per step: Total Origins,
                                        HFR%, RSR%, etc. (thesis Table 4.4)
        {csv_prefix}_adaptations.csv   only the transitions requiring adaptation
        {csv_prefix}_rar_summary.csv   RAR% per transition + overall (Table 4.5)
    """

    steps = [
        ("t0_no_hazard", []),
        ("t1_C1", [build_hazard_query("C1")]),
        ("t2_C1_E7", [build_hazard_query("E7")]),
        ("t3_C1_E7_E5", [build_hazard_query("E5")]),
    ]

    # One clean graph for the whole timeline: hazards are added cumulatively
    # step by step, WITHOUT calling plain_graph() again in between. That is
    # what turns this into propagation rather than independent snapshots.
    plain_graph()

    prev_paths: dict = {}
    prev_dangerzone_ids: set = set()

    per_step_frames = []
    per_step_summaries = []
    adaptation_rows = []

    with driver.session() as session:
        # Degree is a property of the building graph, not of the hazard
        # state, so it is computed once here and reused for every step
        # instead of being recomputed from scratch four times.
        node_degrees = get_node_degrees(session)

        for step_name, hazard_queries in steps:
            for q in hazard_queries:
                session.run(q).consume()

            # Reuse the existing evaluate() for the human-readable per-origin
            # table (static vs adaptive, PW, HE, etc.) at this point in time.
            # n_timing_repeats=1: S2 measures adaptation correctness (RAR),
            # not latency -- the repeated/averaged timing lives in S0/S1/S3
            # via the default n_timing_repeats=5.
            df_step, summary_step = evaluate(
                session, n_timing_repeats=1, node_degrees=node_degrees
            )
            df_step.insert(0, "step", step_name)
            per_step_frames.append(df_step)

            summary_step = summary_step.copy()
            summary_step.insert(0, "step", step_name)
            per_step_summaries.append(summary_step)

            # Recompute just the pieces needed for the t-1 vs t comparison.
            dangerzone_records = list(session.run(DANGERZONES_QUERY))
            current_dangerzone_ids = set(get_place_ids(dangerzone_records))

            origins_records = list(session.run(BUILDING_QUERY))
            origins = get_place_names(origins_records)
            gathering_records = list(session.run(GATHERINGPOINTS_QUERY))
            target_ids = get_place_ids(gathering_records)
            safezone_records = list(session.run(SAFEZONES_QUERY))
            allowed_ids = get_place_ids(safezone_records)

            current_paths = {}
            for origin_name in origins:
                adaptive_params = {
                    "start_name": origin_name,
                    "target_ids": target_ids,
                    "allowed_ids": allowed_ids,
                }
                adaptive_records, _elapsed = timed_query(
                    session, ADAPTIVE_MOVEMENT_QUERY, adaptive_params
                )
                best_adaptive = pick_best_record(adaptive_records)
                current_paths[origin_name] = build_path_result(
                    best_adaptive, current_dangerzone_ids, 0.0
                )

            if step_name != "t0_no_hazard":
                newly_hazardous_ids = current_dangerzone_ids - prev_dangerzone_ids

                for origin_name in origins:
                    prev = prev_paths.get(origin_name)
                    curr = current_paths.get(origin_name)

                    if prev is None or not prev.found:
                        continue  # no previous route to invalidate at this origin

                    required = bool(set(prev.node_ids) & newly_hazardous_ids)
                    if not required:
                        continue

                    if curr.found and curr.hazard_exposure == 0:
                        outcome, correct = "adapted_safe_route", True
                    elif not curr.found:
                        outcome, correct = "no_feasible_route_reported", True
                    else:
                        outcome, correct = "adapted_but_unsafe", False

                    adaptation_rows.append(
                        {
                            "origin": origin_name,
                            "step": step_name,
                            "outcome": outcome,
                            "correct": correct,
                            "prev_weight": prev.weight,
                            "new_weight": curr.weight if curr.found else "N/A",
                        }
                    )

            prev_paths = current_paths
            prev_dangerzone_ids = current_dangerzone_ids

    dynamic_df = pd.concat(per_step_frames, ignore_index=True)
    dynamic_df.to_csv(f"{csv_prefix}.csv", index=False)
    print(f"\n[S2] Per-step per-origin table written to {csv_prefix}.csv")

    step_summary_df = pd.concat(per_step_summaries, ignore_index=True)
    step_summary_df.to_csv(f"{csv_prefix}_step_summary.csv", index=False)
    print(f"[S2] Per-step aggregate summary written to {csv_prefix}_step_summary.csv")
    print(step_summary_df.to_string(index=False))

    adaptation_df = pd.DataFrame(adaptation_rows)

    if adaptation_df.empty:
        print(
            "[S2] No origin required adaptation across the simulated timeline. "
            "Check that the hazard rooms (C1/E7/E5) actually lie on the "
            "previously-used adaptive routes for at least some origins."
        )
        return dynamic_df, adaptation_df

    adaptation_df.to_csv(f"{csv_prefix}_adaptations.csv", index=False)

    overall_required = len(adaptation_df)
    overall_correct = int(adaptation_df["correct"].sum())
    overall_rar = round(overall_correct / overall_required * 100, 2)

    per_step_summary = (
        adaptation_df.groupby("step")["correct"]
        .agg(required="count", correct="sum")
        .reset_index()
    )
    per_step_summary["RAR%"] = round(
        per_step_summary["correct"] / per_step_summary["required"] * 100, 2
    )
    per_step_summary.to_csv(f"{csv_prefix}_rar_summary.csv", index=False)

    print(f"[S2] Routes requiring adaptation (overall): {overall_required}")
    print(f"[S2] Correctly adapted (overall): {overall_correct}")
    print(f"[S2] Route Adaptation Rate (overall) = {overall_rar}%")
    print(per_step_summary.to_string(index=False))

    return dynamic_df, adaptation_df


# --------------------------------------------------------------------------
# Thesis tables — aggregate CSVs matching the tables proposed for the
# Evaluation chapter (X.4.1 .. X.7). Each function below produces exactly
# the columns discussed for the corresponding table, ready to be dropped
# into a LaTeX/Word table with minimal reformatting.
# --------------------------------------------------------------------------
def _numeric(series: pd.Series) -> pd.Series:
    """Coerce a column that may contain the string 'N/A' to numeric,
    turning 'N/A' into NaN so it's excluded from mean/max/etc."""
    return pd.to_numeric(series, errors="coerce")


def _scenario_metrics(df: pd.DataFrame) -> dict:
    """Aggregate metrics computed directly from a per-origin table (works
    for the S0/S1/S3 per-scenario df AND for a per-step slice of the S2
    dynamic df, since both share the same column names)."""
    total_origins = len(df)

    adaptive_found = df["Adaptive Success Rate"] == True  # noqa: E712
    valid_adaptive = int(adaptive_found.sum())

    adaptive_he = _numeric(df["Adaptive Hazard Exposure"])
    hazard_free_adaptive = int(((adaptive_he == 0) & adaptive_found).sum())

    rsr = round(hazard_free_adaptive / total_origins * 100, 2) if total_origins else 0.0
    hfr = (
        round(hazard_free_adaptive / valid_adaptive * 100, 2) if valid_adaptive else 0.0
    )

    # Trivial case: hazard sits on the origin room itself -> unsafe by
    # definition at step 0, no routing algorithm can change that. Mixing
    # these into the same RSR denominator as genuine reroute failures is
    # what produced the RSR discrepancy across tables noted in the draft.
    if "Origin In Hazard Zone" in df.columns:
        trivial_self_hazard = int(df["Origin In Hazard Zone"].fillna(False).sum())
    else:
        trivial_self_hazard = 0
    non_trivial_total = total_origins - trivial_self_hazard
    rsr_excl_trivial = (
        round(hazard_free_adaptive / non_trivial_total * 100, 2)
        if non_trivial_total > 0
        else "N/A"
    )

    if "Structural Class" in df.columns:
        spof_total = int((df["Structural Class"] == "SPOF (degree 1)").sum())
        spof_no_route = int(
            ((df["Structural Class"] == "SPOF (degree 1)") & (~adaptive_found)).sum()
        )
    else:
        spof_total, spof_no_route = 0, 0

    spco = _numeric(df["Safety Overhead (%)"])
    static_t = _numeric(df["Static Time (ms)"])
    adaptive_t = _numeric(df["Adaptive Time (ms)"])

    static_t_std = _numeric(df.get("Static Time Std (ms)", pd.Series(dtype=float)))
    adaptive_t_std = _numeric(df.get("Adaptive Time Std (ms)", pd.Series(dtype=float)))

    static_cost = _numeric(df["Static Cost"])
    adaptive_cost = _numeric(df["Adaptive Cost"])
    unchanged_route = int(
        (
            (static_cost == adaptive_cost) & static_cost.notna() & adaptive_cost.notna()
        ).sum()
    )

    return {
        "Total Origins": total_origins,
        "Valid Adaptive Routes": valid_adaptive,
        "RSR%": rsr,
        "RSR% (excl. trivial self-hazard)": rsr_excl_trivial,
        "Trivial Self-Hazard Origins": trivial_self_hazard,
        "SPOF Origins (degree 1)": spof_total,
        "SPOF Origins Without Route": spof_no_route,
        "HFR%": hfr,
        "SPCO mean %": round(spco.mean(), 2) if spco.notna().any() else "N/A",
        "SPCO median %": round(spco.median(), 2) if spco.notna().any() else "N/A",
        "SPCO max %": round(spco.max(), 2) if spco.notna().any() else "N/A",
        "Static Time mean (ms)": (
            round(static_t.mean(), 3) if static_t.notna().any() else "N/A"
        ),
        "Adaptive Time mean (ms)": (
            round(adaptive_t.mean(), 3) if adaptive_t.notna().any() else "N/A"
        ),
        "Adaptive Time max (ms)": (
            round(adaptive_t.max(), 3) if adaptive_t.notna().any() else "N/A"
        ),
        "Static Time mean-of-std (ms)": (
            round(static_t_std.mean(), 4) if static_t_std.notna().any() else "N/A"
        ),
        "Adaptive Time mean-of-std (ms)": (
            round(adaptive_t_std.mean(), 4) if adaptive_t_std.notna().any() else "N/A"
        ),
        "Adaptive Cost mean": (
            round(adaptive_cost.mean(), 2) if adaptive_cost.notna().any() else "N/A"
        ),
        "Origins with Static PW == Adaptive PW": unchanged_route,
    }


def generate_table_s0_summary(df_s0: pd.DataFrame, out="table_S0_summary.csv"):
    """Table 4.1 — No-Hazard Scenario (S0).

    Adds, on top of the usual HFR/RSR, the count of origins where the
    static and adaptive route coincide -> the concrete evidence that the
    framework does not alter routing unnecessarily when there is no hazard.
    """
    m = _scenario_metrics(df_s0)
    out_df = pd.DataFrame(
        [
            {
                "Number of Origins": m["Total Origins"],
                "Valid Adaptive Routes": m["Valid Adaptive Routes"],
                "Hazard-Free Route Rate (HFR%)": m["HFR%"],
                "Route Success Rate (RSR%)": m["RSR%"],
                "Origins with Static PW = Adaptive PW": m[
                    "Origins with Static PW == Adaptive PW"
                ],
                "Origins with Static PW = Adaptive PW (%)": (
                    round(
                        m["Origins with Static PW == Adaptive PW"]
                        / m["Total Origins"]
                        * 100,
                        2,
                    )
                    if m["Total Origins"]
                    else "N/A"
                ),
            }
        ]
    )
    out_df.to_csv(out, index=False)
    print(f"[Table S0] written to {out}")
    return out_df


def generate_table_s1_example(
    df_s1: pd.DataFrame, out="table_S1_example.csv", n_hazardous=1, n_safe=1
):
    """Table 4.2 — S1 illustrative example.

    Picks the origin(s) with the highest Safety Overhead (baseline crosses
    the hazard, adaptive pays the most to avoid it) and origin(s) where the
    baseline was already hazard-free (overhead = 0%), so the example shows
    both the "framework intervenes" and "framework leaves it alone" cases.
    """
    cols = [
        "origin",
        "Static Cost",
        "Adaptive Cost",
        "Safety Overhead (%)",
        "Static Hazard Exposure",
        "Adaptive Hazard Exposure",
        "Adaptive Success Rate",
    ]
    working = df_s1.copy()
    working["_spco_num"] = _numeric(working["Safety Overhead (%)"])

    hazardous = working[_numeric(working["Static Hazard Exposure"]) > 0].sort_values(
        "_spco_num", ascending=False
    )
    safe = working[_numeric(working["Static Hazard Exposure"]) == 0].sort_values(
        "_spco_num", ascending=True
    )

    example = pd.concat([hazardous.head(n_hazardous), safe.head(n_safe)])[cols]
    example.to_csv(out, index=False)
    print(f"[Table S1 example] written to {out}")
    return example


def generate_table_s1_distribution(
    df_s1: pd.DataFrame, out="table_S1_distribution.csv"
):
    """Table 4.3 — S1 distribution of the safety overhead, restricted to the
    origins whose STATIC route actually crossed the hazard (Static Hazard
    Exposure > 0). This is the "how much does avoidance cost on average"
    table, more informative than dumping all N origins.

    The "affected" set is further split into:
      - genuine reroute cases (static route crossed the hazard, but the
        origin room itself is NOT the hazard) -- this is what the
        adaptive framework is actually being tested on;
      - trivial self-hazard cases (the origin room IS the hazard) --
        unsafe by definition at step 0, kept separate so it doesn't
        silently drag down RSR/HFR the way it did before (this is the
        root cause of the "RSR 33% vs 85%" discrepancy noted earlier).
    """
    affected_all = df_s1[_numeric(df_s1["Static Hazard Exposure"]) > 0]
    if "Origin In Hazard Zone" in df_s1.columns:
        affected_genuine = affected_all[~affected_all["Origin In Hazard Zone"]]
        affected_trivial = affected_all[affected_all["Origin In Hazard Zone"]]
    else:
        affected_genuine, affected_trivial = affected_all, affected_all.iloc[0:0]

    m_all = _scenario_metrics(affected_all) if len(affected_all) else None
    m_genuine = _scenario_metrics(affected_genuine) if len(affected_genuine) else None

    out_df = pd.DataFrame(
        [
            {
                "N Origins with Static Route on Hazard (all)": len(affected_all),
                "N Origins (genuine reroute case)": len(affected_genuine),
                "N Origins (trivial: hazard = origin itself)": len(affected_trivial),
                "SPCO mean (%) [genuine only]": (
                    m_genuine["SPCO mean %"] if m_genuine else "N/A"
                ),
                "SPCO median (%) [genuine only]": (
                    m_genuine["SPCO median %"] if m_genuine else "N/A"
                ),
                "SPCO max (%) [genuine only]": (
                    m_genuine["SPCO max %"] if m_genuine else "N/A"
                ),
                "HFR% [genuine only]": m_genuine["HFR%"] if m_genuine else "N/A",
                "RSR% [genuine only, denom = genuine reroute cases]": (
                    m_genuine["RSR%"] if m_genuine else "N/A"
                ),
                "RSR% [all affected incl. trivial]": (
                    m_all["RSR%"] if m_all else "N/A"
                ),
            }
        ]
    )
    out_df.to_csv(out, index=False)
    print(f"[Table S1 distribution] written to {out}")
    return out_df


def generate_table_s2_step_evolution(
    dynamic_df: pd.DataFrame,
    adaptation_df: pd.DataFrame,
    out="table_S2_step_evolution.csv",
):
    """Table 4.4 — per-step evolution across the S2 dynamic-hazard timeline."""
    hazard_label = {
        "t0_no_hazard": "none",
        "t1_C1": "C1",
        "t2_C1_E7": "C1 + E7",
        "t3_C1_E7_E5": "C1 + E7 + E5",
    }

    required_per_step = (
        adaptation_df.groupby("step").size().to_dict()
        if not adaptation_df.empty
        else {}
    )

    rows = []
    for step_name, step_df in dynamic_df.groupby("step", sort=False):
        m = _scenario_metrics(step_df)
        rows.append(
            {
                "Step": step_name,
                "Active Hazard": hazard_label.get(step_name, step_name),
                "RSR%": m["RSR%"],
                "HFR%": m["HFR%"],
                "Mean Adaptive PW": m["Adaptive Cost mean"],
                "N Routes Recalculated": required_per_step.get(step_name, 0),
            }
        )

    # keep the original step order (t0, t1, t2, t3...) rather than whatever
    # groupby happens to return
    step_order = list(dict.fromkeys(dynamic_df["step"]))
    out_df = pd.DataFrame(rows).set_index("Step").loc[step_order].reset_index()
    out_df.to_csv(out, index=False)
    print(f"[Table S2 step evolution] written to {out}")
    return out_df


def generate_table_s2_rar_summary(
    adaptation_df: pd.DataFrame, out="table_S2_rar_summary.csv"
):
    """Table 4.5 — Route Adaptation Rate per transition + overall row."""
    if adaptation_df.empty:
        out_df = pd.DataFrame(
            [
                {
                    "Transition": "N/A",
                    "Required Adaptation": 0,
                    "Correctly Adapted": 0,
                    "RAR%": "N/A",
                }
            ]
        )
        out_df.to_csv(out, index=False)
        print(f"[Table S2 RAR] no adaptations recorded, placeholder written to {out}")
        return out_df

    per_step = (
        adaptation_df.groupby("step")["correct"]
        .agg(required="count", correct="sum")
        .reset_index()
        .rename(
            columns={
                "step": "Transition",
                "required": "Required Adaptation",
                "correct": "Correctly Adapted",
            }
        )
    )
    per_step["RAR%"] = round(
        per_step["Correctly Adapted"] / per_step["Required Adaptation"] * 100, 2
    )

    overall_required = len(adaptation_df)
    overall_correct = int(adaptation_df["correct"].sum())
    overall_row = pd.DataFrame(
        [
            {
                "Transition": "Overall",
                "Required Adaptation": overall_required,
                "Correctly Adapted": overall_correct,
                "RAR%": round(overall_correct / overall_required * 100, 2),
            }
        ]
    )

    out_df = pd.concat([per_step, overall_row], ignore_index=True)
    out_df.to_csv(out, index=False)
    print(f"[Table S2 RAR] written to {out}")
    return out_df


def generate_table_s2_adaptation_example(
    adaptation_df: pd.DataFrame, out="table_S2_adaptation_example.csv", n=2
):
    """Table 4.6 — 1-2 concrete adaptation examples, one per distinct
    outcome type when possible (e.g. one 'adapted_safe_route' and one
    'no_feasible_route_reported'), to show in the text what actually
    happens to a route when its corridor becomes hazardous."""
    if adaptation_df.empty:
        pd.DataFrame(
            columns=["origin", "step", "prev_weight", "new_weight", "outcome"]
        ).to_csv(out, index=False)
        print(
            f"[Table S2 example] no adaptations recorded, empty table written to {out}"
        )
        return pd.DataFrame()

    picked = adaptation_df.drop_duplicates(subset="outcome", keep="first").head(n)
    cols = ["origin", "step", "prev_weight", "new_weight", "outcome"]
    picked = picked[cols].rename(
        columns={
            "origin": "Origin",
            "step": "Step",
            "prev_weight": "PW Before",
            "new_weight": "PW After",
            "outcome": "Outcome",
        }
    )
    picked.to_csv(out, index=False)
    print(f"[Table S2 example] written to {out}")
    return picked


def _no_route_reason(row) -> str:
    """A specific, per-row reason instead of the same generic sentence
    repeated for every origin -- distinguishes the topologically-forced
    case (SPOF / origin itself hazardous) from a genuine multi-hazard
    encirclement, which is the distinction the thesis draft asked to make
    explicit rather than leaving ambiguous."""
    if row.get("Origin In Hazard Zone", False):
        return "Origin room itself is hazardous (trivial: unsafe from step 0)"
    if row.get("Structural Class") == "SPOF (degree 1)":
        return "SPOF: origin's single corridor is hazardous, no alternative exists"
    return "No path found among safe-only nodes (allowed_ids) — hazard encirclement"


def generate_table_s3_no_route(df_s3: pd.DataFrame, out="table_S3_no_route.csv"):
    """Table 4.7 — origins left without a feasible evacuation route once
    hazard has cut off all safe paths."""
    no_route = df_s3[df_s3["Adaptive Success Rate"] == False].copy()  # noqa: E712
    cols = [
        "origin",
        "Origin Degree",
        "Structural Class",
        "Origin In Hazard Zone",
        "Static Cost",
        "Static Hazard Exposure",
        "Adaptive Success Rate",
    ]
    cols = [c for c in cols if c in no_route.columns]
    no_route["Reason"] = no_route.apply(_no_route_reason, axis=1)
    no_route = no_route[cols + ["Reason"]].rename(
        columns={
            "origin": "Origin",
            "Static Cost": "Static PW",
            "Static Hazard Exposure": "Static HE",
            "Adaptive Success Rate": "Adaptive Success",
        }
    )
    no_route.to_csv(out, index=False)
    print(
        f"[Table S3 no-route] written to {out} ({len(no_route)} origins with no safe route)"
    )
    return no_route


def generate_table_structural_classes(
    scenario_dfs: dict, out="table_structural_classes.csv"
):
    """New table — topological degree classification of every origin
    (SPOF degree-1 / branch degree-2 / hub degree>=3), cross-referenced
    with which scenario(s) leave that origin without a route.

    This turns "why does S1/S3 fail exactly these origins and not others"
    from something explained only in prose into a table: SPOF origins are
    EXPECTED to fail whenever their single corridor is hazardous,
    independent of the routing algorithm -- it's a property of the
    building graph, not a limitation of the framework."""
    base_label = next(iter(scenario_dfs))
    base = scenario_dfs[base_label][
        ["origin", "Origin Degree", "Structural Class"]
    ].copy()

    for label, df in scenario_dfs.items():
        if "Adaptive Success Rate" not in df.columns:
            continue
        no_route = set(df.loc[~df["Adaptive Success Rate"], "origin"])
        base[f"No Route in {label}"] = base["origin"].isin(no_route)

    base = base.sort_values(["Origin Degree", "origin"]).reset_index(drop=True)
    base.to_csv(out, index=False)
    print(f"[Table structural classes] written to {out}")

    counts = base["Structural Class"].value_counts()
    print(f"[Table structural classes] distribution: {counts.to_dict()}")
    return base


def generate_table_cross_scenario_rsr(
    scenario_dfs: dict, out="table_cross_scenario_rsr.csv"
):
    """Table 4.8 — RSR% side by side across scenarios (the "degradation as
    hazard worsens" comparison). `scenario_dfs` maps a display label
    (e.g. "S0", "S1", "S2") to its per-origin DataFrame."""
    rows = []
    for label, df in scenario_dfs.items():
        m = _scenario_metrics(df)
        rows.append(
            {
                "Scenario": label,
                "N Origins": m["Total Origins"],
                "RSR% (all origins)": m["RSR%"],
                "RSR% (excl. trivial self-hazard)": m[
                    "RSR% (excl. trivial self-hazard)"
                ],
                "Trivial Self-Hazard Origins": m["Trivial Self-Hazard Origins"],
                "SPOF Origins Without Route": m["SPOF Origins Without Route"],
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)
    print(f"[Table cross-scenario RSR] written to {out}")
    print(
        "[Table cross-scenario RSR] both RSR definitions are reported explicitly "
        "so no caption needs to silently disambiguate the denominator."
    )
    return out_df


def generate_table_latency_summary(scenario_dfs: dict, out="table_latency_summary.csv"):
    """Table 4.9 — routing-query latency per scenario (explicitly scoped to
    Memgraph query time, NOT end-to-end/MQTT/Node-RED latency — say so in
    the surrounding text, it ties directly into the declared limitation)."""
    rows = []
    for label, df in scenario_dfs.items():
        m = _scenario_metrics(df)
        rows.append(
            {
                "Scenario": label,
                "Mean Static Time (ms)": m["Static Time mean (ms)"],
                "Static Time mean-of-per-origin-std (ms)": m[
                    "Static Time mean-of-std (ms)"
                ],
                "Mean Adaptive Time (ms)": m["Adaptive Time mean (ms)"],
                "Adaptive Time mean-of-per-origin-std (ms)": m[
                    "Adaptive Time mean-of-std (ms)"
                ],
                "Max Adaptive Time (ms)": m["Adaptive Time max (ms)"],
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)
    print(f"[Table latency summary] written to {out}")
    print(
        "[Table latency summary] each cell is now a mean over "
        "n_timing_repeats runs per origin (see evaluate(n_timing_repeats=...)), "
        "with the per-origin std reported alongside instead of a single "
        "unrepeated perf_counter() sample."
    )
    return out_df


def generate_table_master_summary(scenario_dfs: dict, out="table_master_summary.csv"):
    """Table 4.10 — one master row per scenario, the table most likely to
    also be referenced in the abstract/conclusions."""
    rows = []
    for label, df in scenario_dfs.items():
        m = _scenario_metrics(df)
        rows.append(
            {
                "Scenario": label,
                "N Origins": m["Total Origins"],
                "RSR% (all origins)": m["RSR%"],
                "RSR% (excl. trivial self-hazard)": m[
                    "RSR% (excl. trivial self-hazard)"
                ],
                "HFR%": m["HFR%"],
                "Mean SPCO (%)": m["SPCO mean %"],
                "Mean Adaptive Time (ms)": m["Adaptive Time mean (ms)"],
            }
        )
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out, index=False)
    print(f"[Table master summary] written to {out}")
    return out_df


def generate_all_thesis_tables(
    scenario_dfs: dict, dynamic_df: pd.DataFrame, adaptation_df: pd.DataFrame
):
    """Single entry point: call this once at the end of `_run_all_steps`
    with every per-origin DataFrame already computed, and it writes every
    CSV discussed for the Evaluation chapter tables (4.1 to 4.10).

    `scenario_dfs` must contain at least the keys "S0" and "S1" mapped to
    the corresponding per-origin df (the ones returned by evaluate() for
    the no-hazard / hazard-on-C1 runs). "S3" is optional -- its no-route
    table is only generated if present. The cross-scenario RSR comparison
    (Table 4.8) no longer includes S3: it compares S0, S1, and S2 (S2's
    snapshot being the last step of the dynamic hazard timeline).
    """
    generate_table_s0_summary(scenario_dfs["S0"])
    generate_table_s1_example(scenario_dfs["S1"])
    generate_table_s1_distribution(scenario_dfs["S1"])
    if "S1b" in scenario_dfs:
        # Same tables, but for the harder paired-hazard version of S1 (see
        # _run_all_steps): shows whether the framework still copes once the
        # "easy" escape hatches are removed too.
        generate_table_s1_example(
            scenario_dfs["S1b"], out="table_S1b_paired_example.csv"
        )
        generate_table_s1_distribution(
            scenario_dfs["S1b"], out="table_S1b_paired_distribution.csv"
        )
    generate_table_s2_step_evolution(dynamic_df, adaptation_df)
    generate_table_s2_rar_summary(adaptation_df)
    generate_table_s2_adaptation_example(adaptation_df)
    if "S3" in scenario_dfs:
        generate_table_s3_no_route(scenario_dfs["S3"])
    generate_table_structural_classes(scenario_dfs)

    # Cross-scenario RSR comparison (Table 4.8): S3 is no longer part of
    # this evaluation, so the comparison is S0 (no hazard) vs S1 (single
    # hazard on C1) vs S2 -- using the LAST step of the S2 dynamic timeline
    # (t3_C1_E7_E5, i.e. the fully-hazarded end state) as S2's snapshot,
    # since dynamic_df shares the exact same per-origin columns produced by
    # evaluate() and _scenario_metrics() works on it unchanged.
    last_step = dynamic_df["step"].iloc[-1]
    df_s2_final = dynamic_df[dynamic_df["step"] == last_step].drop(columns=["step"])
    rsr_scenario_dfs = {
        "S0": scenario_dfs["S0"],
        "S1": scenario_dfs["S1"],
        "S2": df_s2_final,
    }
    generate_table_cross_scenario_rsr(rsr_scenario_dfs)
    generate_table_latency_summary(scenario_dfs)
    # Master summary (Table 4.10) mirrors the cross-scenario RSR comparison:
    # S0, S1, S2 only (S2 = last step of the dynamic hazard timeline).
    # S1b/S3 are excluded here even though they're still in scenario_dfs
    # for the other tables above.
    generate_table_master_summary(rsr_scenario_dfs)


def main():
    driver = GraphDatabase.driver(MEMGRAPH_URI, auth=MEMGRAPH_AUTH)

    try:
        _run_all_steps(driver)
    finally:
        # Close the driver exactly once, after every step has used it.
        # (Previously each step closed the driver in its own `finally`
        # block, so the second and third steps tried to open a session
        # on an already-closed driver -> `DriverError("Driver closed")`.)
        driver.close()


def _run_all_steps(driver):
    # Step 1: create the plain graph via nodered, get data when no hazard is detected
    plain_graph()
    with driver.session() as session:
        df, summary_df = evaluate(session)
    suffix = "_no_hazards"
    create_df(df, summary_df, suffix)
    df_s0 = df

    # Step 2 (S1): an hazard is detected on C1, so a node is Hazard is connected
    # to the corresponding node. In the first floor quite all the rooms are
    # connected to C1, but TeamLab/AA1 have alternative paths to safely exit,
    # and LA1 has its own direct exit (E1) -- so this scenario mostly tests
    # the *degree-1* origins (cisco) whose only corridor is C1. See S1-bis
    # below for a harder test that also stresses degree-2 origins.
    plain_graph()
    with driver.session() as session:
        session.run(build_hazard_query("C1")).consume()
        df, summary_df = evaluate(session)
    suffix = "_hazard_on_C1"
    create_df(df, summary_df, suffix)
    df_s1 = df

    # Step 2b (S1-bis): a harder version of the same single-corridor test.
    # C1 alone leaves an escape hatch for every degree-2 origin that has a
    # direct exit next to it (LA1 -> E1). Adding that exit to the hazard
    # set forces a genuine reroute (or reveals genuine infeasibility) for
    # those origins too, instead of the framework simply never being
    # exercised on them. This directly implements suggestion (3) from the
    # earlier topology analysis (hazard on node PAIRS, not single nodes).
    plain_graph()
    with driver.session() as session:
        session.run(build_hazard_query("C1")).consume()
        session.run(build_hazard_query("E1")).consume()
        df, summary_df = evaluate(session)
    suffix = "_hazard_on_C1_E1_paired"
    create_df(df, summary_df, suffix)
    df_s1b = df

    # Step 3 (S3): multi hazard, first floor exits are dangerous. A fire in
    # AB3 and the meeting room, plus damage on the E5/E6/E7 exits, forces
    # people on the first floor toward the ground floor (or leaves some
    # origins with no feasible route at all).
    plain_graph()

    # NOTE: the neo4j/Memgraph driver's session.run() executes a single
    # Cypher statement per call. The original code concatenated four
    # MERGE/CREATE blocks (separated by ';') into one string and expected
    # a single run() to execute all of them, which does not work. Each
    # hazard is now created with its own run() call instead, built from
    # the single build_hazard_query() helper shared with S1/S1-bis/S2
    # (previously each scenario hand-duplicated this Cypher block).
    multi_hazard_rooms = ["E7", "E5", "E6", "SalaRiunioni", "AB3"]

    with driver.session() as session:
        for room in multi_hazard_rooms:
            session.run(build_hazard_query(room)).consume()
        df, summary_df = evaluate(session)
    suffix = "_multi_hazard"
    create_df(df, summary_df, suffix)
    df_s3 = df

    # Step 4 (S2): dynamic hazard propagation on a single continuous run,
    # tracking whether each origin's route needed adaptation and whether
    # the framework adapted it correctly (-> Route Adaptation Rate).
    dynamic_df, adaptation_df = run_dynamic_hazard_scenario(driver)

    # Step 5: generate every aggregate CSV discussed for the thesis
    # Evaluation chapter tables, built directly from the per-origin
    # DataFrames already computed above -- no need to re-read the
    # individual CSVs back from disk. S1-bis is included alongside S1 so
    # the cross-scenario / structural-class tables show the contrast
    # between the "easy" and "hard" versions of the same hazard.
    scenario_dfs = {"S0": df_s0, "S1": df_s1, "S1b": df_s1b, "S3": df_s3}
    generate_all_thesis_tables(scenario_dfs, dynamic_df, adaptation_df)


if __name__ == "__main__":
    main()
