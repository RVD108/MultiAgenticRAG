# Multi-Agentic RAG with FastAPI

This repository contains an optimized and production-ready **Multi-Agentic Research RAG (Retrieval-Augmented Generation) System** built with **LangGraph** and exposed via a high-performance **FastAPI** REST API.

The system uses a supervisor agent to plan queries, orchestrate specialized parallel retriever subagents (using BM25 + similarity + MMR search over Chroma vector store), and apply Cohere re-ranking. It also contains self-correcting hallucination-check guardrails and human-in-the-loop triggers.

---

## Key Features

- **Supervisor Agent (LangGraph)**: Automatically routes incoming queries (environmental vs. general vs. clarification) and designs dynamic search plans.
- **Parallel Subgraph (Researcher)**: Expands queries and fetches documents in parallel (fan-out pattern).
- **Hybrid Retrieval & Re-ranking**: Combines Vector Similarity, MMR, and BM25 retrievers, compressed using Cohere Re-ranker.
- **FastAPI Endpoints**: Fully async endpoints exposing RAG queries, health status, and state recovery.
- **Hallucination Detection Guardrails**: Uses structured LLM evaluation to detect discrepancies, stopping responses for human validation.

---

## Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configuration & API Keys

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

Update options in `config.yaml` as needed (e.g. setting models, top_k values, etc.).

### 3. Loading Documents

To ingest a new document (e.g. Google's 2024 Environmental Report), place the PDF in the `retriever/` folder, set `load_documents: true` in `config.yaml`, and run:

```bash
python -m retriever.retriever
```

### 4. Running the FastAPI Server

Start the backend server using Uvicorn:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Access the interactive API documentation at:
**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** (Swagger UI)

---

## API Endpoints

- **`GET /health`**: Retrieve service status, version, and server uptime.
- **`POST /query`**: Ask a question. Returns `answer`, `session_id`, `latency_ms`, and `hallucination_score`.
- **`POST /retry`**: Resume a query thread flagged by the hallucination detection node.
- **`GET /sessions`**: List active sessions.
- **`DELETE /session/{session_id}`**: Clear a specific session.
