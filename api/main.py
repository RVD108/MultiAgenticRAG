"""
FastAPI entrypoint for the Multi-Agentic RAG system.

Exposes the LangGraph pipeline via async REST endpoints with
Pydantic request/response schemas, background indexing support,
and auto-generated OpenAPI docs at /docs.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import logging
import time
import uuid

from main_graph.graph_builder import graph, InputState
from utils.utils import config, new_uuid
from langgraph.types import Command

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Multi-Agentic RAG API",
    description=(
        "Production-grade Multi-Agent RAG pipeline powered by LangGraph. "
        "Supports supervisor + sub-graph agents, hallucination detection, "
        "self-correcting retrieval loops, and Cohere re-ranking."
    ),
    version="1.0.0",
    contact={"name": "Shreyash Bhaskar Patil", "url": "https://github.com/RVD108"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (thread_id -> configurable dict)
_sessions: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user's question.")
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for multi-turn conversations. "
                    "If omitted, a new session is created.",
    )

    model_config = {"json_schema_extra": {"example": {"query": "What is Google's data center PUE efficiency?"}}}


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The agent's final answer.")
    session_id: str = Field(..., description="Session ID for follow-up queries.")
    latency_ms: float = Field(..., description="End-to-end response latency in milliseconds.")
    hallucination_score: str = Field(..., description="'1' = grounded, '0' = potential hallucination.")


class RetryRequest(BaseModel):
    session_id: str = Field(..., description="The session ID of the paused (hallucination-flagged) query.")
    confirm_retry: bool = Field(
        default=True,
        description="Set to true to retry generation, false to accept the current answer.",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


_start_time = time.time()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — confirms the API is running."""
    return HealthResponse(
        status="ok",
        version=app.version,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Submit a question to the Multi-Agent RAG pipeline.

    - Routes to the research sub-graph for environmental queries.
    - Falls back to a general LLM response for off-topic questions.
    - Runs hallucination detection on every research response.
    - Returns a `session_id` for multi-turn follow-up.
    """
    t0 = time.time()

    # Session management
    session_id = request.session_id or new_uuid()
    if session_id not in _sessions:
        _sessions[session_id] = {"configurable": {"thread_id": session_id}}

    thread = _sessions[session_id]
    input_state = InputState(messages=request.query)

    final_content = ""
    hallucination_score = "1"

    try:
        async for chunk, metadata in graph.astream(
            input=input_state, stream_mode="messages", config=thread
        ):
            if chunk.content:
                final_content += chunk.content

        # Check for hallucination interrupt
        graph_state = graph.get_state(thread)
        if graph_state and len(graph_state[-1]) > 0:
            interrupts = graph_state[-1][0].interrupts
            if interrupts:
                hallucination_score = "0"  # flagged — caller may use /retry

    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    latency_ms = round((time.time() - t0) * 1000, 2)
    logger.info("Query answered in %.1f ms (session=%s)", latency_ms, session_id)

    return QueryResponse(
        answer=final_content or "No answer generated.",
        session_id=session_id,
        latency_ms=latency_ms,
        hallucination_score=hallucination_score,
    )


@app.post("/retry", response_model=QueryResponse, tags=["RAG"])
async def retry_generation(request: RetryRequest):
    """
    Resume a paused session after a hallucination flag.

    When `/query` returns `hallucination_score='0'`, call this endpoint
    to either retry generation (`confirm_retry=true`) or accept the answer.
    """
    t0 = time.time()

    if request.session_id not in _sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{request.session_id}' not found.",
        )

    thread = _sessions[request.session_id]
    resume_value = "y" if request.confirm_retry else "n"
    final_content = ""

    try:
        async for chunk, metadata in graph.astream(
            Command(resume=resume_value), stream_mode="messages", config=thread
        ):
            if chunk.content:
                final_content += chunk.content
    except Exception as exc:
        logger.exception("Retry error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    latency_ms = round((time.time() - t0) * 1000, 2)
    return QueryResponse(
        answer=final_content or "No answer generated.",
        session_id=request.session_id,
        latency_ms=latency_ms,
        hallucination_score="1",
    )


@app.delete("/session/{session_id}", tags=["System"])
async def clear_session(session_id: str):
    """Delete a conversation session from memory."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del _sessions[session_id]
    return {"message": f"Session '{session_id}' cleared."}


@app.get("/sessions", tags=["System"])
async def list_sessions():
    """List all active session IDs."""
    return {"active_sessions": list(_sessions.keys()), "count": len(_sessions)}
