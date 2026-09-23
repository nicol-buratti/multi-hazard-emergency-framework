import logging

from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import List
from langchain_core.tools import tool, BaseTool

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


async def get_memgraph_tools() -> List[BaseTool]:
    """Connects to the Memgraph MCP server and returns a list of high-level LangChain/LangGraph tools.

    Returns:
        List[BaseTool]: A list of initialized LangChain tools ready to query the Memgraph database.
    """
    mcp_client = MultiServerMCPClient(
        {
            "memgraph": {
                "url": "http://localhost:8001/mcp",
                "transport": "streamable_http",
            },
        }
    )
    mcp_tools = await mcp_client.get_tools()
    query_tool = next(t for t in mcp_tools if t.name == "run_query")

    @tool
    async def get_room_data(department: str, room: str) -> str:
        """Retrieves the properties and metadata of a specific room from the graph database.

        Use this tool when you need basic information about a room, such as its exact name,
        dimensions, or primary function.

        Args:
            department (str): The name of the department where the room is located.
            room (str): The name or partial name of the room to search for.

        Returns:
            str: A JSON-like string containing the node properties of the requested room.
        """
        cypher = f"""
        MATCH (n:Place)
        WHERE toLower(n.department) = toLower('{department}')
        AND toLower(n.name) = toLower('{room}')
        RETURN n
        LIMIT 1
        """
        result = await query_tool.ainvoke({"query": cypher})
        return (
            result
            if result
            else f"No data found for room '{room}' in department '{department}'."
        )

    @tool
    async def get_adjacent_rooms(
        department: str, room: str, depth: int, limit: int
    ) -> str:
        """Explores the graph to find rooms connected to a specific starting room up to a maximum depth.

        Use this tool to understand spatial flow, horizontal topology, and room connectivity.

        Args:
            department (str): The name of the department where the starting room is located.
            room (str): The name of the starting room.
            depth (int): The maximum number of relationship hops to explore (e.g., 1 for direct neighbors, 2 for neighbors of neighbors).
            limit (int): The maximum number of node records to return.

        Returns:
            str: A stringified table or JSON representation of connected rooms, edges, and their traversal depth.
        """
        cypher = f"""
        MATCH (n:Place)
        WHERE toLower(n.department) = toLower('{department}')
        AND toLower(n.name) = toLower('{room}')

        MATCH p = (n)-[:CONNECTED_TO*1..{depth}]-(target)

        UNWIND range(0, length(p) - 1) AS i
        WITH
        relationships(p)[i] AS rel,
        nodes(p)[i] AS n_from,
        nodes(p)[i+1] AS n_to,
        i AS level

        ORDER BY level ASC

        WITH rel, collect({{n1: n_from, n2: n_to, depth: level}})[0] AS data

        RETURN
        data.n1.name AS n1,
        type(rel) AS edge,
        data.n2.name AS n2
        ORDER BY data.depth ASC, n1 ASC, n2 ASC
        LIMIT {limit}
        """
        result = await query_tool.ainvoke({"query": cypher})
        return (
            result
            if result
            else f"No connected rooms found for '{room}' within {limit} results."
        )

    @tool
    async def get_hvac_and_shaft_connections(room: str) -> str:
        """Finds non-adjacent rooms connected via shared building systems.

        Use this tool to trace potential hazards like smoke, gas, or leaks traveling through
        ventilation ducts (HVAC), service shafts, or elevator shafts.

        Args:
            room (str): The name of the room to check for system connections.

        Returns:
            str: A string detailing the connected rooms and the shared system infrastructure.
        """
        logger.info("\n[TEST TOOL CALLED] get_hvac_and_shaft_connections: %s\n", room)
        return "No real data available for HVAC and shaft connections at the moment."

    @tool
    async def get_vertical_topology(room: str) -> str:
        """Retrieves the 3D spatial relationships of a room.

        Use this tool to find out what is directly above (ceiling) or directly below (floor)
        the specified room across different building levels.

        Args:
            room (str): The name of the reference room.

        Returns:
            str: A string detailing the rooms located immediately above and below.
        """
        logger.info("\n[TEST TOOL CALLED] get_vertical_topology: %s\n", room)
        return "No real data available for vertical topology at the moment."

    @tool
    async def get_nearby_critical_hazards(room: str, radius_hops: int) -> str:
        """Searches for high-risk nodes within a specified radius of a given room.

        Use this tool for risk assessment to locate nearby hazards such as main electrical panels,
        oxygen cylinder storage, or chemical storage areas.

        Args:
            room (str): The name of the starting room.
            radius_hops (int): The search radius in graph hops (usually 1 or 2).

        Returns:
            str: A string listing nearby hazardous locations and their distance in hops.
        """
        logger.info("\n[TEST TOOL CALLED] get_nearby_critical_hazards: %s\n", room)
        return "No real data available for critical hazards at the moment."

    @tool
    async def get_structural_context(room: str) -> str:
        """Retrieves structural and architectural metadata for a specific room.

        Use this tool to identify the floor level, structural type (e.g., load-bearing wall,
        cantilevered slab), and proximity to structural vulnerabilities like seismic expansion joints.

        Args:
            room (str): The name of the room to inspect structurally.

        Returns:
            str: A string detailing the room's structural properties and context.
        """
        logger.info("\n[TEST TOOL CALLED] get_structural_context: %s\n", room)
        return "No real data available for structural context at the moment."

    return [
        get_room_data,
        get_adjacent_rooms,
        get_hvac_and_shaft_connections,
        get_vertical_topology,
        get_nearby_critical_hazards,
        get_structural_context,
    ]
