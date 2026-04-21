"""Document domain models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class ChunkMetadata(BaseModel):
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    token_count: int


class DocumentMetadata(BaseModel):
    """Arbitrary key-value metadata stored alongside the document."""
    source: Optional[str] = None
    author: Optional[str] = None
    department: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    custom: Dict[str, Any] = Field(default_factory=dict)


# ── Request / Response schemas ────────────────────────────────────────────────

class DocumentIngestRequest(BaseModel):
    """Payload for ingesting a plain-text or JSON document."""
    title: str = Field(..., min_length=1, max_length=512)
    content: str = Field(..., min_length=10, description="Raw document text")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        return v.strip()


class DocumentResponse(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    status: DocumentStatus
    chunk_count: int = 0
    metadata: DocumentMetadata
    created_at: datetime
    indexed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    items: List[DocumentResponse]
    total: int
    page: int
    page_size: int


class DocumentChunk(BaseModel):
    """A single embedding-ready text chunk derived from a document."""
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    embedding: Optional[List[float]] = None
    chunk_meta: ChunkMetadata
    doc_meta: DocumentMetadata
