"""
/api/v1/documents — document ingestion & management endpoints.
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from app.core.config import settings
from app.core.logging import get_logger
from app.db.vector_store import VectorStoreManager
from app.models.document import (
    DocumentIngestRequest,
    DocumentListResponse,
    DocumentMetadata,
    DocumentResponse,
    DocumentStatus,
)
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService

router = APIRouter()
logger = get_logger(__name__)

# In-process document registry (use a DB in production)
_document_registry: dict[str, DocumentResponse] = {}


def get_pipeline(request: Request):
    return request.app.state.vector_store


@router.post(
    "/ingest",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a text document into the knowledge base",
)
async def ingest_document(
    body: DocumentIngestRequest,
    vs: VectorStoreManager = Depends(get_pipeline),
):
    doc_id = str(uuid.uuid4())
    record = DocumentResponse(
        id=uuid.UUID(doc_id),
        title=body.title,
        status=DocumentStatus.PROCESSING,
        metadata=body.metadata,
        created_at=datetime.utcnow(),
    )
    _document_registry[doc_id] = record

    try:
        embed_svc = EmbeddingService()
        retrieval_svc = RetrievalService(vs, embed_svc)

        chunks = retrieval_svc.chunk_document(
            document_id=doc_id,
            title=body.title,
            text=body.content,
            metadata=body.metadata,
        )
        indexed_count = await retrieval_svc.index_chunks(chunks)

        record.status = DocumentStatus.INDEXED
        record.chunk_count = indexed_count
        record.indexed_at = datetime.utcnow()
        logger.info("document_indexed", doc_id=doc_id, chunks=indexed_count, title=body.title)

    except Exception as exc:
        record.status = DocumentStatus.FAILED
        record.error_message = str(exc)
        logger.exception("document_ingest_failed", doc_id=doc_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc}",
        )

    return record


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a file (.txt, .md, .pdf, .json) into the knowledge base",
)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    author: Optional[str] = None,
    department: Optional[str] = None,
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    vs: VectorStoreManager = Depends(get_pipeline),
):
    if file.content_type not in settings.SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}",
        )

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > settings.MAX_DOCUMENT_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.MAX_DOCUMENT_SIZE_MB} MB limit",
        )

    # Decode text (PDF extraction would require a dedicated parser)
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode file as UTF-8 text")

    meta = DocumentMetadata(
        source=file.filename,
        author=author,
        department=department,
        tags=[t.strip() for t in tags.split(",")] if tags else [],
    )
    body = DocumentIngestRequest(
        title=title or file.filename or "Untitled",
        content=text,
        metadata=meta,
    )
    return await ingest_document(body, vs)


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all indexed documents",
)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[DocumentStatus] = None,
):
    docs = list(_document_registry.values())
    if status_filter:
        docs = [d for d in docs if d.status == status_filter]
    docs.sort(key=lambda d: d.created_at, reverse=True)
    start = (page - 1) * page_size
    return DocumentListResponse(
        items=docs[start : start + page_size],
        total=len(docs),
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a document by ID",
)
async def get_document(document_id: str):
    doc = _document_registry.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its embeddings",
)
async def delete_document(
    document_id: str,
    vs: VectorStoreManager = Depends(get_pipeline),
):
    if document_id not in _document_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    deleted = await vs.delete_document(document_id)
    del _document_registry[document_id]
    logger.info("document_deleted", doc_id=document_id, chunks_removed=deleted)
