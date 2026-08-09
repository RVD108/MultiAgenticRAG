"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing & routing user queries, generating research plans
to answer user questions, conducting research, and formulating responses.
"""

from typing import Any, Literal, TypedDict, cast, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command
from main_graph.graph_states import AgentState, Router, GradeHallucinations, InputState
from utils.prompt import (
    ROUTER_SYSTEM_PROMPT,
    RESEARCH_PLAN_SYSTEM_PROMPT,
    MORE_INFO_SYSTEM_PROMPT,
    GENERAL_SYSTEM_PROMPT,
    CHECK_HALLUCINATIONS,
    RESPONSE_SYSTEM_PROMPT,
)
from subgraph.graph_builder import researcher_graph
from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import logging
from utils.utils import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").propagate = False
logging.getLogger("urllib3").propagate = False
logging.getLogger("httpx").propagate = False

GPT_4o_MINI = config["llm"]["gpt_4o_mini"]
GPT_4o = config["llm"]["gpt_4o"]
TEMPERATURE = config["llm"]["temperature"]


async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    Args:
        state (AgentState): Current agent state with conversation history.
        config (RunnableConfig): Runtime configuration.

    Returns:
        dict[str, Router]: Routing classification with type and logic.
    """
    model = ChatOpenAI(model=GPT_4o, temperature=TEMPERATURE, streaming=True)
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}] + state.messages
    logger.info("---ANALYZE AND ROUTE QUERY---")
    logger.info("MESSAGES: %s", state.messages)
    response = cast(Router, await model.with_structured_output(Router).ainvoke(messages))
    return {"router": response}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine next step based on query classification.

    Raises:
        ValueError: If unknown router type encountered.
    """
    _type = state.router["type"]
    if _type == "environmental":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type!r}")


async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for environmental queries."""

    class Plan(TypedDict):
        """Generate research plan."""
        steps: list[str]

    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    messages = [{"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}] + state.messages
    logger.info("---PLAN GENERATION---")
    response = cast(Plan, await model.with_structured_output(Plan).ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}


async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for clarifying information."""
    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(logic=state.router["logic"])
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan via the researcher sub-graph."""
    result = await researcher_graph.ainvoke({"question": state.steps[0]})
    docs = result["documents"]
    step = state.steps[0]
    logger.info("%d documents retrieved for step: %s", len(docs), step)
    return {"documents": result["documents"], "steps": state.steps[1:]}


def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Route back to research if steps remain, else generate final response."""
    if len(state.steps or []) > 0:
        return "conduct_research"
    return "respond"


async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general (non-environmental) query."""
    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = GENERAL_SYSTEM_PROMPT.format(logic=state.router["logic"])
    logger.info("---RESPONSE GENERATION---")
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


def _format_doc(doc: Document) -> str:
    """Format a single document as XML."""
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"
    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as an XML string for the LLM context window."""
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"<documents>\n{formatted}\n</documents>"


async def check_hallucinations(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Binary hallucination check — is the answer grounded in retrieved documents?"""
    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=state.documents, generation=state.messages[-1]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    logger.info("---CHECK HALLUCINATIONS---")
    response = cast(
        GradeHallucinations,
        await model.with_structured_output(GradeHallucinations).ainvoke(messages),
    )
    return {"hallucination": response}


def human_approval(state: AgentState):
    """Interrupt for human review if hallucination detected."""
    if state.hallucination.binary_score == "1":
        return "END"
    retry_generation = interrupt(
        {"question": "Is this correct?", "llm_output": state.messages[-1]}
    )
    if retry_generation == "y":
        return "respond"
    return "END"


async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate the final answer using retrieved context."""
    logger.info("--- RESPONSE GENERATION STEP ---")
    model = ChatOpenAI(model=GPT_4o, temperature=TEMPERATURE, streaming=True)
    context = format_docs(state.documents)
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
checkpointer = MemorySaver()

builder = StateGraph(AgentState, input=InputState)
builder.add_node(analyze_and_route_query)
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_node(create_research_plan)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(conduct_research)
builder.add_node("respond", respond)
builder.add_node(check_hallucinations)
builder.add_conditional_edges(
    "check_hallucinations", human_approval, {"END": END, "respond": "respond"}
)
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("respond", "check_hallucinations")

graph = builder.compile(checkpointer=checkpointer)
