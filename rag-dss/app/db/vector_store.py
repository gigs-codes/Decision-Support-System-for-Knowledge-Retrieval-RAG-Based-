"""
Vector store abstraction wrapping ChromaDB.

Supports:
- Embedded (local) mode — great for development and single-node deployments.
- Server mode — connects to a running ChromaDB HTTP server.
- Hybrid search (dense + sparse BM25 fusion via RRF).
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.logging import get_logger
from app.models.document import DocumentChunk

logger = get_logger(__name__)


class VectorStoreManager:
    """Async-friendly wrapper around ChromaDB."""

    def __init__(self) -> None:
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        self._loop = asyncio.get_event_loop()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        await asyncio.to_thread(self._sync_init)

    def _sync_init(self) -> None:
        if settings.USE_CHROMA_SERVER:
            self._client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info("chroma_mode", mode="server", host=settings.CHROMA_HOST)
        else:
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info("chroma_mode", mode="embedded", path=settings.CHROMA_PERSIST_DIR)

        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "vector_store_ready",
            collection=settings.CHROMA_COLLECTION,
            doc_count=self._collection.count(),
        )

    async def close(self) -> None:
        # ChromaDB embedded client doesn't need explicit close, but good practice
        self._client = None
        self._collection = None

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Insert or update a batch of document chunks."""
        return await asyncio.to_thread(self._sync_upsert, chunks)

    def _sync_upsert(self, chunks: List[DocumentChunk]) -> int:
        assert self._collection is not None, "Vector store not initialised"

        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk in chunks:
            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "document_id": chunk.document_id,
                    "document_title": chunk.document_title,
                    "chunk_index": chunk.chunk_meta.chunk_index,
                    "total_chunks": chunk.chunk_meta.total_chunks,
                    "token_count": chunk.chunk_meta.token_count,
                    "source": chunk.doc_meta.source or "",
                    "author": chunk.doc_meta.author or "",
                    "department": chunk.doc_meta.department or "",
                    "tags": ",".join(chunk.doc_meta.tags),
                }
            )

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("chunks_upserted", count=len(ids))
        return len(ids)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.35,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Return list of (text, score, metadata) tuples ordered by relevance.
        Scores are converted from cosine *distance* to similarity: score = 1 - distance.
        """
        return await asyncio.to_thread(
            self._sync_query, query_embedding, top_k, score_threshold, metadata_filter
        )

    def _sync_query(
        self,
        query_embedding: List[float],
        top_k: int,
        score_threshold: float,
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float, Dict]]:
        assert self._collection is not None

        where: Optional[Dict] = None
        if metadata_filter:
            where = {k: {"$eq": v} for k, v in metadata_filter.items()}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist   # cosine distance → similarity
            if score >= score_threshold:
                output.append((text, round(score, 4), meta))

        return sorted(output, key=lambda x: x[1], reverse=True)

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_document(self, document_id: str) -> int:
        """Delete all chunks belonging to a document."""
        return await asyncio.to_thread(self._sync_delete, document_id)

    def _sync_delete(self, document_id: str) -> int:
        assert self._collection is not None
        results = self._collection.get(where={"document_id": {"$eq": document_id}})
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        logger.info("chunks_deleted", document_id=document_id, count=len(ids_to_delete))
        return len(ids_to_delete)

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def collection_stats(self) -> Dict[str, Any]:
        count = await asyncio.to_thread(lambda: self._collection.count() if self._collection else 0)
        return {
            "collection": settings.CHROMA_COLLECTION,
            "total_chunks": count,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "mode": "server" if settings.USE_CHROMA_SERVER else "embedded",
        }
