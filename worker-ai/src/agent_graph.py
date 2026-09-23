import logging
import operator
from typing import Annotated, Any, Literal, TypedDict, Union, get_args

from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field, ValidationError

from src.app_settings import AppSettings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)
ExpertType = Literal["fire", "earthquake"]
VALID_EXPERTS = get_args(ExpertType)


class ThreatAssessment(BaseModel):
    room: str = Field(
        description="The room identifier for which the assessment is made."
    )
    warning: Literal["none", "pre-alert"] = Field(
        description="Early warning indicator for suspicious data that may precede a danger"
    )
    danger: Literal["high", "medium", "low", "none"] = Field(
        description="Final assessed danger level based on all gathered data."
    )
    danger_type: Literal["fire", "smoke", "heat", "earthquake", "other", "none"] = (
        Field(description="Type of danger identified, if any.")
    )
    danger_score: float = Field(
        description="Severity score on a scale from 0.0 to 1.0.", ge=0.0, le=1.0
    )
    justification: str = Field(
        description="Brief justification for the assigned danger level."
    )


class EscalateAction(BaseModel):
    action: Literal["escalate"] = Field(
        description="Select this action if a specialized analysis is required."
    )
    required_experts: list[ExpertType] = Field(
        description="Experts required. Do not generate assessment."
    )


class AssessAction(BaseModel):
    action: Literal["assess"] = Field(
        description="Select this action if the conditions are safe and no experts are needed."
    )
    assessments: list[ThreatAssessment] = Field(
        description="List of safety assessments."
    )


TriageOutput = Union[EscalateAction, AssessAction]


class ExpertOutput(BaseModel):
    assessments: list[ThreatAssessment] = Field(
        description="List of threat assessments for the primary room and any affected neighboring rooms."
    )


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    data: dict[str, Any]
    required_experts: list[str]
    assessments: Annotated[list[dict[str, Any]], operator.add]


class ExpertState(TypedDict):
    messages: list[AnyMessage]
    data: dict[str, Any]


class HazardMapReduceManager:
    def __init__(self) -> None:
        # Pydantic parses .env and environment variables here
        settings = AppSettings()

        extra_body: dict[str, Any] | None = (
            {"models": settings.extra_llm_models} if settings.extra_llm_models else None
        )

        self.model: ChatOpenAI = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
            temperature=0.2,
            max_retries=2,
            extra_body=extra_body,
        )

        self.callbacks: list[Any] = []
        template_string: str = """IoT Context & Input Data:
{data}

Instructions:
Evaluate the reading values for safety/threat levels. Assess the primary room and determine if threat propagation requires assessing neighboring rooms."""
        self.prompt_template: PromptTemplate = PromptTemplate(
            input_variables=["data"], template=template_string
        )
        self.app: Any | None = None

    async def initialize_graph(
        self,
        tools: list[Any] | None = None,
        debug: bool = False,
        print_agent: bool = False,
    ) -> None:
        tools = tools or []

        triage_sys: SystemMessage = SystemMessage(
            content="""
You are the Hazard Assessment Triage, the entry point of an indoor telemetry monitoring system.
You receive multiple IoT telemetry snapshots for one room.

Your sole objective is to evaluate the input data and select the appropriate action based strictly on the provided structured output schema.

1. If the data exhibits anomalies, hazards, or complex patterns requiring specialized domain analysis, select the escalate action and populate the required experts using only the valid values defined in the schema.
Do not generate a threat assessment yourself.
2. If the conditions are safe, normal, or only require a basic warning without specialized analysis, select the assess action and provide the threat assessment directly.

Make your classification based entirely on the provided schema fields.
Do not assume the behavior of any downstream agents, and do not invent expert types or assessment values outside of the allowed definitions.
"""
        )
        fire_sys: SystemMessage = SystemMessage(
            content="""
You are the Fire Safety Expert, invoked only after the Triage agent has flagged
a possible fire-related anomaly in one room.

1. Analyze the telemetry for fire hazards: heat/temperature spikes, smoke
   density, and CO2 or other combustion gas readings.
2. Call the get_room_data tool for the primary room to ground your assessment
   in its known spatial properties, then call get_adjacent_rooms to identify
   neighboring rooms, ventilation paths, or connected structural nodes that
   fire or smoke could spread through. Query only as far (depth) as needed to
   answer whether propagation is likely — do not perform broad, speculative
   traversals.
3. Assess the primary room first. Only produce an additional assessment for a
   neighboring room when the topology and telemetry together indicate a
   credible propagation path (e.g. a connected room reachable via ventilation
   with no isolating barrier); do not assess rooms with no plausible link.
4. Base danger_score strictly on the evidence gathered: reserve 0.7-1.0 for
   confirmed high-heat/smoke combinations, 0.3-0.7 for a single ambiguous
   signal, and below 0.3 for weak or borderline readings. Keep justification
   short and tied to the specific values and graph facts you used — never state
   a room, value, or connection you did not actually observe or retrieve.
5. Tool failure handling (fallback): if a tool returns an error, times out,
   does not respond, or returns empty/unusable data, do NOT retry it. You are
   strictly forbidden from invoking the same tool more than once with the same
   parameters, whether the first call succeeded or failed. Never loop on a
   tool. Instead, proceed with the information you already have: base your
   assessment on the telemetry alone, do not assess any neighboring room whose
   connection you could not verify, and state briefly in the justification
   that topology/room data was unavailable. Do not invent any room, value, or
   connection to fill the gap, and treat a telemetry-only assessment with
   appropriate caution when choosing danger_score.
6. Once you have produced the ThreatAssessment(s) for every room you evaluated,
   stop: do not call further tools or continue reasoning.
        """
        )
        earthquake_sys: SystemMessage = SystemMessage(
            content="""
You are the Earthquake Safety Expert, invoked only after the Triage agent has
flagged a possible seismic anomaly in one room.

1. Analyze the telemetry for seismic activity: vibration levels, acceleration,
   and any indicators of structural shift.
2. Call the get_room_data tool for the primary room to ground your assessment
   in its known structural/material parameters, then call get_adjacent_rooms to
   map load-bearing dependencies and identify connected structural elements
   that damage could propagate through. Query only as far (depth) as needed to
   answer the propagation question — avoid broad, speculative traversals.
3. Assess the primary room first. Only produce an additional assessment for a
   neighboring or structurally connected room when the retrieved topology and
   material data indicate a credible damage-propagation path; do not assess
   rooms with no plausible structural link.
4. Base danger_score strictly on the evidence gathered: reserve 0.7-1.0 for
   confirmed high-magnitude vibration/acceleration combined with a vulnerable
   structural link, 0.3-0.7 for a single ambiguous signal, and below 0.3 for
   weak or borderline readings. Keep justification short and tied to the
   specific values and graph facts you used — never state a room, value, or
   structural dependency you did not actually observe or retrieve.
5. Tool failure handling (fallback): if a tool returns an error, times out,
   does not respond, or returns empty/unusable data, do NOT retry it. You are
   strictly forbidden from invoking the same tool more than once with the same
   parameters, whether the first call succeeded or failed. Never loop on a
   tool. Instead, proceed with the information you already have: base your
   assessment on the telemetry alone, do not assess any neighboring room whose
   structural link you could not verify, and state briefly in the justification
   that structural/topology data was unavailable. Do not invent any room,
   value, or structural dependency to fill the gap, and treat a telemetry-only
   assessment with appropriate caution when choosing danger_score.
6. Once you have produced the ThreatAssessment(s) for every room you evaluated,
   stop: do not call further tools or continue reasoning.
"""
        )

        triage_agent = create_agent(
            self.model,
            tools=tools,
            response_format=TriageOutput,
            system_prompt=triage_sys,
            name="Triage",
        )
        fire_agent = create_agent(
            self.model,
            tools=tools,
            response_format=ExpertOutput,
            system_prompt=fire_sys,
            name="Fire Agent",
        )
        earthquake_agent = create_agent(
            self.model,
            tools=tools,
            response_format=ExpertOutput,
            system_prompt=earthquake_sys,
            name="Earthquake Agent",
        )

        async def run_triage(state: GraphState) -> dict[str, Any]:
            invocation_result: dict[str, Any] = await triage_agent.ainvoke(
                {"messages": self.prompt_template.format(data=state["data"])}
            )
            triage_result: TriageOutput = invocation_result.get("structured_response")

            if triage_result.action == "escalate":
                return {
                    "required_experts": triage_result.required_experts,
                    "assessments": [],
                }

            return {
                "required_experts": [],
                "assessments": [a.model_dump() for a in triage_result.assessments],
            }

        def route_experts(state: GraphState) -> list[Send]:
            experts: list[str] = state.get("required_experts", [])
            if not experts:
                return [Send("safe_node", {})]

            sends: list[Send] = []
            input_prompt: HumanMessage = HumanMessage(
                content=self.prompt_template.format(data=state["data"])
            )

            for expert in experts:
                expert_lower = expert.lower()
                if expert_lower in VALID_EXPERTS:
                    sends.append(
                        Send(
                            f"{expert_lower}_node",
                            {
                                "messages": [input_prompt],
                                # "data": state["data"],
                            },
                        )
                    )
            return sends

        async def run_safe_node(state: GraphState) -> dict[str, Any]:
            return {}

        async def run_fire(state: ExpertState) -> dict[str, list[dict[str, Any]]]:
            invocation_result: dict[str, Any] = await fire_agent.ainvoke(state)
            expert_result: ExpertOutput = invocation_result.get("structured_response")
            return {"assessments": [a.model_dump() for a in expert_result.assessments]}

        async def run_earthquake(state: ExpertState) -> dict[str, list[dict[str, Any]]]:
            invocation_result: dict[str, Any] = await earthquake_agent.ainvoke(state)
            expert_result: ExpertOutput = invocation_result.get("structured_response")
            return {"assessments": [a.model_dump() for a in expert_result.assessments]}

        builder = StateGraph(GraphState)
        builder.add_node("triage", run_triage)
        builder.add_node("fire_node", run_fire)
        builder.add_node("earthquake_node", run_earthquake)
        builder.add_node("safe_node", run_safe_node)

        builder.add_edge(START, "triage")

        # Dynamically construct the allowed nodes list
        allowed_nodes = [f"{expert}_node" for expert in VALID_EXPERTS] + ["safe_node"]
        builder.add_conditional_edges("triage", route_experts, allowed_nodes)

        builder.add_edge("fire_node", END)
        builder.add_edge("earthquake_node", END)
        builder.add_edge("safe_node", END)

        self.app = builder.compile(debug=debug)

        if print_agent and self.app:
            logger.info("\n" + self.app.get_graph().draw_ascii())

    async def process_data(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        graph_result = await self.ainvoke(data)
        raw_assessments: list[dict[str, Any]] = graph_result.get("assessments", [])

        validated_assessments: list[dict[str, Any]] = []
        for assessment_data in raw_assessments:
            try:
                assessment = ThreatAssessment(**assessment_data)
                validated_assessments.append(assessment.model_dump())
            except ValidationError as e:
                logger.error(
                    f"Validation failed for assessment data {assessment_data}: {e}"
                )

        return validated_assessments

    async def ainvoke(self, data):
        if not self.app:
            await self.initialize_graph()

        thread_id: str = str(data.get("room", "default_room"))
        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "callbacks": self.callbacks,
        }

        initial_state: dict[str, Any] = {
            "messages": [],
            "data": data,
            "required_experts": [],
            "assessments": [],
        }

        graph_result: dict[str, Any] = await self.app.ainvoke(
            initial_state, config=config
        )
        return graph_result
