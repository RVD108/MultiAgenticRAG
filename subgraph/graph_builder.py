"""Researcher sub-graph.

Handles query expansion, parallel document retrieval from a Chroma vector store
using an ensemble of BM25 + similarity + MMR retrievers, and Cohere-based
contextual compression / re-ranking.
"""

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever, BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langgraph.types import Send

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import logging
from utils.utils import config

load_dotenv()

logger = logging.getLogger(__name__)

VECTORSTORE_COLLECTION = config["retriever"]["collection_name"]
VECTORSTORE_DIRECTORY = config["retriever"]["directory"]
TOP_K = config["retriever"]["top_k"]
TOP_K_COMPRESSION = config["retriever"]["top_k_compression"]
ENSEMBLE_WEIGHTS = config["retriever"]["ensemble_weights"]
COHERE_RERANK_MODEL = config["retriever"]["cohere_rerank_model"]


def _setup_vectorstore() -> Chroma:
    """Initialise and return the Chroma vector store."""
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=VECTORSTORE_COLLECTION,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIRECTORY,
    )


def _load_documents(vectorstore: Chroma) -> list[Document]:
    """Load all documents from the vector store as LangChain Document objects."""
    all_data = vectorstore.get(include=["documents", "metadatas"])
    documents: list[Document] = []
    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise ValueError(f"Expected metadata dict, got {type(meta)}")
        documents.append(Document(page_content=content, metadata=meta))
    return documents


def _build_retrievers(
    documents: list[Document], vectorstore: Chroma
) -> ContextualCompressionRetriever:
    """Build an ensemble retriever with Cohere re-ranking.

    Combines three base retrievers:
      - BM25 (keyword-based)
      - Chroma similarity search
      - Chroma MMR (maximal marginal relevance)

    Then applies Cohere re-ranking for contextual compression.
    """
    retriever_bm25 = BM25Retriever.from_documents(documents, search_kwargs={"k": TOP_K})
    retriever_vanilla = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    retriever_mmr = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": TOP_K}
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
        weights=ENSEMBLE_WEIGHTS,
    )
    compressor = CohereRerank(top_n=TOP_K_COMPRESSION, model=COHERE_RERANK_MODEL)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )


# Module-level initialisation (loaded once at import time for performance)
vectorstore = _setup_vectorstore()
documents = _load_documents(vectorstore)
compression_retriever = _build_retrievers(documents, vectorstore)


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Expand the research question into diverse sub-queries for retrieval."""

    class Response(TypedDict):
        queries: list[str]

    logger.info("---GENERATE QUERIES---")
    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.with_structured_output(Response).ainvoke(messages))
    queries = response["queries"]
    queries.append(state.question)  # always include the original question
    logger.info("Queries: %s", queries)
    return {"queries": queries}


async def retrieve_and_rerank_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve and re-rank documents for a single query."""
    logger.info("---RETRIEVING DOCUMENTS---")
    logger.info("Query: %s", state.query)
    response = compression_retriever.invoke(state.query)
    return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Fan-out: dispatch one retrieval task per generated query in parallel."""
    return [
        Send("retrieve_and_rerank_documents", QueryState(query=query))
        for query in state.queries
    ]


# ---------------------------------------------------------------------------
# Researcher sub-graph assembly
# ---------------------------------------------------------------------------
builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_and_rerank_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,
    path_map=["retrieve_and_rerank_documents"],
)
builder.add_edge("retrieve_and_rerank_documents", END)
researcher_graph = builder.compile()
