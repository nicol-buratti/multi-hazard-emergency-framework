import re
from typing import Any
import json
from neo4j import GraphDatabase
from pathlib import Path


def build_graph_injection_query(
    graph: dict[str, Any],
) -> tuple[str, dict[str, Any]]:

    places = graph.get("places", [])
    edges = graph.get("edges", [])

    # Replica il comportamento del nodo "Set Place Params" del flow Node-RED:
    # - le coordinate vengono trasformate in stringa lato client (JSONata $string),
    #   non dentro la query Cypher (toString() di Memgraph non accetta liste/mappe)
    # - l'attributo "danger", se null, viene forzato alla stringa "null" perché
    #   SET p += {...} in Cypher scarta silenziosamente le chiavi con valore null
    processed_places = []
    for place in places:
        place = dict(place)
        attributes = dict(place.get("attributes") or {})

        if "danger" in attributes and attributes["danger"] is None:
            attributes["danger"] = "null"

        place["attributes"] = attributes
        place["coordinates"] = json.dumps(place.get("coordinates"))
        processed_places.append(place)

    places = processed_places

    edges_by_type: dict[str, list[dict]] = {}

    for edge in edges:
        edge_type = edge.get("type", "CONNECTED_TO")

        # type può essere None
        if not isinstance(edge_type, str) or not edge_type:
            edge_type = "CONNECTED_TO"

        edge_type = re.sub(r"\s+", "_", edge_type.upper())

        # Il relationship type viene inserito direttamente
        # nella query, quindi deve essere un identificatore valido.
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", edge_type):
            edge_type = "CONNECTED_TO"

        edges_by_type.setdefault(edge_type, []).append(
            {
                "source": edge["source"],
                "target": edge["target"],
                "attributes": edge.get("attributes", {}),
            }
        )

    query_parts = []

    # Places
    query_parts.append("""
        UNWIND $places AS place
        MERGE (p:Place {id: place.id})
        SET p += coalesce(place.attributes, {}),
            p.name = place.name,
            p.coordinates = place.coordinates
    """)

    # Edges
    for edge_type in edges_by_type:
        query_parts.append(f"""
        WITH 1 AS _
        UNWIND $edges_{edge_type} AS edge
        MATCH (s:Place {{id: edge.source}})
        MATCH (t:Place {{id: edge.target}})
        MERGE (s)-[r:{edge_type}]->(t)
        SET r += coalesce(edge.attributes, {{}})
    """)

    query_parts.append("""
        RETURN true AS success
    """)

    query = "\n".join(query_parts)

    params = {
        "places": places,
    }

    for edge_type, edge_list in edges_by_type.items():
        params[f"edges_{edge_type}"] = edge_list

    return query, params


def plain_graph():
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir.parent / "node-red" / "place_graph.json"
    with open(file_path, "r", encoding="utf-8") as f:
        graph = json.load(f)

    # `MATCH (n)-[r]-(m) RETURN n,r,m` only *reads* matching patterns, it
    # never deletes anything, and it also skips nodes that have no
    # relationships. That's why hazard nodes from a previous run (e.g. the
    # C1 hazard) were still present in later steps: nothing was ever
    # actually removed from the graph. `DETACH DELETE` removes every node
    # (and, via DETACH, all of its relationships) in one pass.
    clear_query = """
MATCH (n)
DETACH DELETE n;
"""
    query, params = build_graph_injection_query(graph)

    print(query)

    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("", ""),
    )

    with driver.session(database="memgraph") as session:
        session.run(clear_query, {})
        session.run(query, params)

    driver.close()
