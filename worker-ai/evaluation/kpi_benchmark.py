"""
kpi_benchmark.py
=================

Benchmark/collaudo KPI END-TO-END per HazardMapReduceManager, contro lo
stack REALE: LLM vero (ChatOpenAI via `create_agent`), tool reali di
`memgraph_custom_tools.get_memgraph_tools()`, e il vero server MCP
Memgraph configurato in quella funzione. Nessun componente e' mockato:
per lanciare lo script serve un server MCP Memgraph raggiungibile e le
credenziali LLM reali (lette da `agent_graph.AppSettings`, tipicamente
da .env), esattamente come per l'esecuzione normale del tuo agente.

Punto d'ingresso testato: `HazardMapReduceManager.analyze_data(data)`.
Se nel tuo progetto quel metodo si chiama ancora `process_data` (come nel
file che mi hai condiviso inizialmente), lo script lo rileva da solo e
usa quello come fallback — vedi `_resolve_entrypoint()`.

COSA VIENE MISURATO E COME (dato che qui non c'e' alcun doppio da cui
leggere lo stato interno, ogni KPI e' osservato dall'esterno):

  1. Latenza end-to-end: perf_counter() attorno alla chiamata reale.
  2. Overhead tool: ogni tool restituito da get_memgraph_tools() viene
     istrumentato (si avvolge il suo `.coroutine` reale) per sommare il
     tempo speso nelle chiamate reali al server MCP/Memgraph.
  3. Schema adherence: se la chiamata solleva una ValidationError
     Pydantic (o un errore il cui messaggio la richiama), la run viene
     marcata come non conforme allo schema al primo tentativo.
  4. Routing accuracy: euristica sul `danger_type` restituito (dato che
     con il grafo reale non intercettiamo l'edge routing interno):
     "fire" -> ci si aspetta un danger_type in {fire, smoke, heat};
     "earthquake" -> {earthquake}; "fire_earthquake" -> uno qualsiasi dei
     due; "normal" -> nessuna assessment ad alto rischio.
  5. Varianza del danger_score: calcolata a fine run (stdev) sui valori
     realmente restituiti dall'LLM per ciascuno scenario.
  6. Allucinazione topologica: PRIMA del loop, interroghiamo i tool
     reali (get_room_data/get_adjacent_rooms) per ottenere la topologia
     VERA della stanza di ogni scenario direttamente dal tuo grafo
     Memgraph; a ogni run confrontiamo le stanze citate nella
     justification con questa verita' di terra.
  7. API drop rate: qui NON viene iniettato alcun guasto finto (avrebbe
     poco senso contro un sistema reale) — viene semplicemente
     classificato e contato ogni fallimento di rete/API realmente
     osservato durante le run.

DA CONFIGURARE PRIMA DELL'USO
------------------------------
`SCENARIO_ROOMS` piu' sotto e `DEPARTMENT` usano segnaposto presi
dall'esempio che mi hai mostrato ("departmenta"/"ab1" ecc.). Se il tuo
department/room reale e' diverso, sostituiscili — altrimenti i tool reali
restituiranno "nessun dato trovato" per ogni run.

INPUT DELL'AGENTE
------------------
Formato REALE confermato via test manuale (non lo cambiamo, ci adeguiamo):

    {
        "room": "<department>:<room>",
        "sensor_data": [
            {"co2": ..., "temperature": ..., "humidity": ..., "tvoc": ...,
             "eco2": ..., "room": "<department>:<room>", "timestamp": "..."},
            ...  # tipicamente 5, uno per ciascuno degli ultimi 5 minuti,
                 # dal piu' vecchio al piu' recente
        ],
    }

Ogni elemento di `sensor_data` e' gia' il valore MASSIMO di quel minuto (se
in un minuto arrivano temperature 20,20,21,23,21, l'oggetto di quel minuto
porta solo 23). I sensori sismici (vibration_g/acceleration_g) non hanno
ancora un sensore reale collegato: restano sintetici, aggiunti alle stesse
letture. Le liste sono generate dinamicamente ad ogni run da una funzione
per scenario (`SCENARIO_GENERATORS`): normal, fire, earthquake,
fire_earthquake (evento combinato/improvviso tipo esplosione), cosi' ogni
run e' un caso leggermente diverso invece di un payload statico ripetuto.

USO
---
    python kpi_benchmark.py --n 5 --snapshots 5 --output kpi_evaluation.csv

Consiglio: parti con --n basso (3-5) per un primo smoke test reale,
dato che ogni run e' una vera chiamata LLM + vera query al grafo.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent_graph import HazardMapReduceManager  # noqa: E402
from src.memgraph_custom_tools import get_memgraph_tools  # noqa: E402

# ----------------------------------------------------------------------------
# 1. GENERAZIONE DINAMICA DELLE FINESTRE DI TELEMETRIA
# ----------------------------------------------------------------------------
# Formato REALE confermato via test manuale: `ainvoke`/`analyze_data` si
# aspetta un dict {"room": "<department>:<room>", "sensor_data": [...]}, dove
# ogni elemento di sensor_data e' la lettura GIA' aggregata (max) di un
# minuto: {"co2", "temperature", "humidity", "tvoc", "eco2", "room",
# "timestamp"}. I campi sismici (vibration_g/acceleration_g) non hanno ancora
# un sensore reale collegato: restano sintetici come prima, aggiunti alle
# stesse letture.
#
# Ogni scenario e' una funzione che restituisce un payload nuovo ad ogni
# chiamata (con rumore/trend casuali via `rng`), cosi' ogni run e' un caso
# di test leggermente diverso invece di un payload statico ripetuto.

DEPARTMENT = "departmenta"  # <-- DA CONFIGURARE: reparto reale nel grafo

# Stanza usata per ciascuno scenario (deve esistere davvero nel grafo).
SCENARIO_ROOMS: dict[str, str] = {
    "normal": "ab1",  # <-- DA CONFIGURARE
    "fire": "ab2",  # <-- DA CONFIGURARE
    "earthquake": "ab3",  # <-- DA CONFIGURARE
    "fire_earthquake": "ab2",  # <-- DA CONFIGURARE (es. esplosione)
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
    """Serie di n valori con andamento lineare start->end (evento che
    peggiora/migliora gradualmente minuto per minuto) piu' rumore."""
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
    """Serie con un salto brusco al minuto `onset_index` (evento improvviso,
    es. esplosione) invece di un'escalation graduale."""
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
    """Costruisce il payload REALE: {"room": "<department>:<room>",
    "sensor_data": [...]}, dal minuto piu' vecchio (indice 0) al piu'
    recente (indice n-1, ultimo elemento = adesso)."""
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
    # Escalation graduale: un incendio che si sviluppa minuto dopo minuto
    # (calore/CO2/VOC salgono, l'umidita' scende per il calore secco).
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
    # Escalation graduale della sismicita'; aria/gas restano al baseline.
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
    # Evento improvviso e simultaneo su tutti i sensori (es. esplosione):
    # baseline nei primi minuti, poi salto netto su aria/gas E vibrazione.
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

# Riconosce sia id "combinati" tipo "departmenta:ab1" sia id "nudi" tipo
# "ab1"/"R101" citati nella justification (case-insensitive: i nomi reali
# osservati sono minuscoli, es. "ab1", non "R101").
ROOM_ID_PATTERN = re.compile(r"\b([A-Za-z][\w-]*:[\w-]+|[A-Za-z]{1,4}\d{1,4})\b")

CSV_COLUMNS = [
    "run_id",
    "scenario_type",
    "latency_ms",
    "tool_overhead_ms",
    "schema_adherence",
    "routing_correct",
    "danger_score",
    "hallucination_detected",
    "api_failed",
]

# ----------------------------------------------------------------------------
# 2. STRUMENTAZIONE DEI TOOL REALI (overhead) — nessun doppio, solo un
#    wrapper che cronometra la coroutine reale di ciascun tool.
# ----------------------------------------------------------------------------

_tool_overhead_ms: ContextVar[list[float]] = ContextVar("tool_overhead_ms")


def instrument_tools(tools: list[Any]) -> list[Any]:
    """Avvolge il `.coroutine` reale di ogni tool per sommarne la durata
    nel contatore della run corrente (KPI 2), senza cambiarne il comportamento.
    """
    for t in tools:
        if getattr(t, "coroutine", None) is None:
            continue  # tool solo sincrono: non previsto qui, si ignora
        original: Callable[..., Awaitable[Any]] = t.coroutine

        async def wrapped(*args: Any, _original=original, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return await _original(*args, **kwargs)
            finally:
                bucket = _tool_overhead_ms.get(None)
                if bucket is not None:
                    bucket.append((time.perf_counter() - t0) * 1000)

        t.coroutine = wrapped
    return tools


async def fetch_ground_truth_room_ids(
    tools_by_name: dict[str, Any], department: str, room: str
) -> set[str]:
    """Interroga i tool REALI per ottenere la topologia vera della stanza,
    da usare come riferimento anti-allucinazione (KPI 6)."""
    known: set[str] = {room}
    try:
        room_data_tool = tools_by_name["get_room_data"]
        adjacent_tool = tools_by_name["get_adjacent_rooms"]
        room_raw = await room_data_tool.ainvoke(
            {"department": department, "room": room}
        )
        adjacent_raw = await adjacent_tool.ainvoke(
            {"department": department, "room": room, "depth": 2, "limit": 50}
        )
        known |= set(ROOM_ID_PATTERN.findall(str(room_raw)))
        known |= set(ROOM_ID_PATTERN.findall(str(adjacent_raw)))
    except Exception:
        # Se il fetch della verita' di terra fallisce (es. stanza non
        # configurata), il rilevamento di allucinazioni per questo
        # scenario sara' meno affidabile: viene segnalato a console.
        print(
            f"[WARN] impossibile recuperare la topologia reale per '{room}' in '{department}'"
        )
    return known


# ----------------------------------------------------------------------------
# 3. ENTRYPOINT DEL MANAGER — usa analyze_data, con fallback a process_data
# ----------------------------------------------------------------------------


def _resolve_entrypoint(
    manager: HazardMapReduceManager,
) -> Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]]:
    entrypoint = getattr(manager, "analyze_data", None)
    if entrypoint is not None:
        return entrypoint
    # Il file condiviso inizialmente espone questo metodo come `process_data`.
    entrypoint = getattr(manager, "process_data", None)
    if entrypoint is not None:
        print("[INFO] 'analyze_data' non trovato: uso 'process_data' come entrypoint.")
        return entrypoint
    raise AttributeError(
        "HazardMapReduceManager non espone ne' 'analyze_data' ne' 'process_data'."
    )


# ----------------------------------------------------------------------------
# 4. CLASSIFICAZIONE DEGLI ERRORI REALI (KPI 3 / KPI 7)
# ----------------------------------------------------------------------------


def classify_exception(exc: Exception) -> tuple[bool, bool]:
    """Ritorna (schema_adherence_ok, api_failed) a partire da un'eccezione
    realmente sollevata dallo stack (LLM, tool, rete)."""
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
    # Fallimento non classificato: lo trattiamo come un drop generico,
    # cosi' compare comunque nel tasso di fallimento invece di sparire.
    return True, True


# ----------------------------------------------------------------------------
# 5. ESECUZIONE DI UNA SINGOLA RUN REALE
# ----------------------------------------------------------------------------


async def run_single(
    entrypoint: Callable[[dict[str, Any]], Awaitable[list[dict[str, Any]]]],
    scenario_key: str,
    run_id: int,
    room: str,
    rng: random.Random,
    n_snapshots: int,
    known_room_ids: set[str],
) -> dict[str, Any]:
    token = _tool_overhead_ms.set([])

    row: dict[str, Any] = {
        "run_id": run_id,
        "scenario_type": scenario_key,
        "latency_ms": None,
        "tool_overhead_ms": 0.0,
        "schema_adherence": True,
        "routing_correct": False,
        "danger_score": None,
        "hallucination_detected": False,
        "api_failed": False,
    }

    # {"room": "<department>:<room>", "sensor_data": [...]}, generato ad hoc per questa run.
    window = SCENARIO_GENERATORS[scenario_key](rng, room, DEPARTMENT, n_snapshots)

    t0 = time.perf_counter()
    try:
        assessments = await entrypoint(window)
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        row["tool_overhead_ms"] = round(sum(_tool_overhead_ms.get([])), 2)
        row["danger_score"] = assessments[0]["danger_score"] if assessments else None

        # --- KPI 4: routing accuracy (euristica sul danger_type osservato) ---
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

        # --- KPI 6: allucinazione topologica contro la verita' di terra reale ---
        justification_text = " ".join(a.get("justification", "") for a in assessments)
        mentioned = set(ROOM_ID_PATTERN.findall(justification_text))
        row["hallucination_detected"] = bool(mentioned - known_room_ids)

    except Exception as exc:
        row["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        row["tool_overhead_ms"] = round(sum(_tool_overhead_ms.get([])), 2)
        schema_ok, api_failed = classify_exception(exc)
        row["schema_adherence"] = schema_ok
        row["api_failed"] = api_failed
        print(
            f"[run {run_id}] {scenario_key}: eccezione reale -> {type(exc).__name__}: {exc}"
        )
    finally:
        _tool_overhead_ms.reset(token)

    return row


# ----------------------------------------------------------------------------
# 6. HARNESS PRINCIPALE
# ----------------------------------------------------------------------------


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("\n=== Riepilogo KPI per scenario (stack reale) ===")
    for scenario_key in SCENARIO_GENERATORS:
        subset = [r for r in rows if r["scenario_type"] == scenario_key]
        if not subset:
            continue
        n = len(subset)
        latencies = [r["latency_ms"] for r in subset if r["latency_ms"] is not None]
        overheads = [r["tool_overhead_ms"] for r in subset]
        scores = [r["danger_score"] for r in subset if r["danger_score"] is not None]
        schema_rate = sum(r["schema_adherence"] for r in subset) / n
        routing_rate = sum(r["routing_correct"] for r in subset) / n
        halluc_rate = sum(r["hallucination_detected"] for r in subset) / n
        api_fail_rate = sum(r["api_failed"] for r in subset) / n
        score_stdev = statistics.pstdev(scores) if len(scores) >= 1 else float("nan")

        print(f"\n--- Scenario: {scenario_key} (n={n}) ---")
        print(
            f"  Latenza media:            {statistics.mean(latencies):.1f} ms"
            if latencies
            else "  Latenza media:            n/a"
        )
        print(f"  Overhead tool medio:      {statistics.mean(overheads):.1f} ms")
        print(f"  Schema adherence rate:    {schema_rate:.1%}")
        print(f"  Routing accuracy:         {routing_rate:.1%}")
        print(f"  Danger score stdev:       {score_stdev:.4f}")
        print(f"  Hallucination rate:       {halluc_rate:.1%}")
        print(f"  API drop rate osservato:  {api_fail_rate:.1%}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark KPI end-to-end (stack reale) per HazardMapReduceManager"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Numero di run per scenario (chiamate LLM reali)",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=5,
        help="Oggetti per finestra (minuti aggregati) inviati all'agente",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed opzionale per riprodurre le stesse finestre generate",
    )
    parser.add_argument("--output", type=str, default="kpi_evaluation.csv")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    manager = HazardMapReduceManager()
    tools = await get_memgraph_tools()  # connessione REALE al server MCP Memgraph
    tools = instrument_tools(tools)
    await manager.initialize_graph(tools=tools)
    entrypoint = _resolve_entrypoint(manager)

    tools_by_name = {t.name: t for t in tools}
    known_room_ids: dict[str, set[str]] = {}
    for scenario_key, room in SCENARIO_ROOMS.items():
        known_room_ids[scenario_key] = await fetch_ground_truth_room_ids(
            tools_by_name, DEPARTMENT, room
        )

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
                    scenario_key,
                    run_id,
                    room,
                    rng,
                    args.snapshots,
                    known_room_ids[scenario_key],
                )
                rows.append(row)
                writer.writerow(row)
                f.flush()  # scritta subito su disco, non bufferizzata fino alla fine
                print(
                    f"[run {run_id}] {scenario_key}: scritta su CSV (latenza {row['latency_ms']} ms)"
                )
                run_id += 1

    print(f"\nScritte {len(rows)} run in {output_path.resolve()}")
    print_summary(rows)


if __name__ == "__main__":
    asyncio.run(main())
