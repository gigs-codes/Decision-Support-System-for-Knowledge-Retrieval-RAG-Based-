"""Query and RAG response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000, description="Natural-language question")
    session_id: Optional[UUID] = Field(None, description="Optional session ID for multi-turn dialogue")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    score_threshold: float = Field(0.35, ge=0.0, le=1.0)
    filter_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata filters, e.g. {\"department\": \"engineering\"}",
    )
    include_sources: bool = True
    stream: bool = False


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score (cosine similarity)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CitedSource(BaseModel):
    document_id: str
    title: str
    excerpt: str
    score: float
    metadata: Dict[str, Any]


class RAGResponse(BaseModel):
    answer: str
    sources: List[CitedSource] = Field(default_factory=list)
    session_id: Optional[UUID] = None
    query_id: str
    model_used: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    chunks_retrieved: int
    chunks_used: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: Optional[float] = None   # aggregate relevance score
    follow_up_questions: List[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FeedbackRequest(BaseModel):
    query_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = Field(None, max_length=1000)
    helpful: bool
