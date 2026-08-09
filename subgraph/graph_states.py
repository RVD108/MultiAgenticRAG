from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.documents import Document


@dataclass(kw_only=True)
class QueryState:
    """State for a single parallel retrieval task."""
    query: str


@dataclass(kw_only=True)
class ResearcherState:
    """State of the researcher sub-graph."""
    question: str
    queries: list[str] = field(default_factory=list)
    documents: Annotated[list[Document], lambda x, y: x + y] = field(default_factory=list)
