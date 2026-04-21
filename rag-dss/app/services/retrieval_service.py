"""
Retrieval service: document chunking + ranked retrieval.

Chunking strategy:
  1. Split on paragraph/sentence boundaries (semantic chunking lite).
  2. Respect CHUNK_SIZE (tokens) and CHUNK_OVERLAP.
  3. Each chunk carries full provenance metadata.

Retrieval strategy:
  - Dense vector search (ChromaDB).
  - Optional BM25 sparse re-rank fused via Reciprocal Rank Fusion (RRF).
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import get_logger
from app.db.vector_store import VectorStoreManager
from app.models.document import ChunkMetadata, DocumentChunk, DocumentMetadata
from app.models.query import RetrievedChunk
from app.services.embedding_service import EmbeddingService

logger = get_logger(__name__)


class RetrievalService:
    def __init__(self, vector_store: VectorStoreManager, embedding_service: EmbeddingService) -> None:
        self._vs = vector_store
        self._embed = embedding_service

    # ── Chunking ──────────────────────────────────────────────────────────────

    def chunk_document(
        self,
        document_id: str,
        title: str,
        text: str,
        metadata: DocumentMetadata,
    ) -> List[DocumentChunk]:
        """Split raw text into overlapping, embedding-ready chunks."""
        paragraphs = self._split_paragraphs(text)
        chunks: List[DocumentChunk] = []
        buffer: List[str] = []
        buffer_tokens = 0
        chunk_index = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            if buffer_tokens + para_tokens > settings.CHUNK_SIZE and buffer:
                chunk_text = " ".join(buffer)
                chunks.append(
                    self._make_chunk(
                        document_id, title, chunk_text, chunk_index, metadata, text
                    )
                )
                chunk_index += 1
                # Keep overlap
                overlap_text = " ".join(buffer[-2:])  # last ~2 paras as overlap
                buffer = [overlap_text]
                buffer_tokens = self._estimate_tokens(overlap_text)

            buffer.append(para)
            buffer_tokens += para_tokens

        if buffer:
            chunks.append(
                self._make_chunk(
                    document_id, title, " ".join(buffer), chunk_index, metadata, text
                )
            )

        # Back-fill total_chunks
        total = len(chunks)
        for c in chunks:
            c.chunk_meta.total_chunks = total

        logger.debug("document_chunked", doc_id=document_id, chunks=total)
        return chunks

    def _make_chunk(
        self,
        doc_id: str,
        title: str,
        text: str,
        idx: int,
        meta: DocumentMetadata,
        full_text: str,
    ) -> DocumentChunk:
        start = full_text.find(text[:40])
        return DocumentChunk(
            chunk_id=f"{doc_id}::chunk_{idx}",
            document_id=doc_id,
            document_title=title,
            text=text.strip(),
            chunk_meta=ChunkMetadata(
                chunk_index=idx,
                total_chunks=0,  # filled later
                start_char=max(start, 0),
                end_char=max(start, 0) + len(text),
                token_count=self._estimate_tokens(text),
            ),
            doc_meta=meta,
        )

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        """Split on double newlines; fall back to sentence splits for very long paras."""
        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        result: List[str] = []
        for para in paras:
            if len(para) > 1500:
                # Split long paragraphs on sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)
                result.extend(s for s in sentences if s)
            else:
                result.append(para)
        return result

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(len(text) // 4, 1)

    # ── Embed & index ─────────────────────────────────────────────────────────

    async def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        texts = [c.text for c in chunks]
        embeddings = await self._embed.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return await self._vs.upsert_chunks(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Dense + optional BM25 hybrid retrieval with RRF fusion."""
        query_emb = await self._embed.embed_query(query)

        # Dense retrieval
        dense_results = await self._vs.query(
            query_embedding=query_emb,
            top_k=top_k * 2,  # over-fetch for re-rank headroom
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

        if not dense_results:
            return []

        if settings.HYBRID_SEARCH_ENABLED and len(dense_results) > 1:
            chunks = self._rrf_rerank(query, dense_results, top_k)
        else:
            chunks = dense_results[:top_k]

        return [
            RetrievedChunk(
                chunk_id=meta.get("chunk_id", f"chunk_{i}"),
                document_id=meta["document_id"],
                document_title=meta["document_title"],
                text=text,
                score=score,
                metadata={k: v for k, v in meta.items()
                          if k not in ("document_id", "document_title")},
            )
            for i, (text, score, meta) in enumerate(chunks)
        ]

    def _rrf_rerank(
        self,
        query: str,
        dense_results: List[Tuple[str, float, Dict]],
        top_k: int,
        k: int = 60,
    ) -> List[Tuple[str, float, Dict]]:
        """Reciprocal Rank Fusion of dense + BM25 sparse scores."""
        texts = [r[0] for r in dense_results]

        # BM25 sparse ranking
        tokenised = [t.lower().split() for t in texts]
        bm25 = BM25Okapi(tokenised)
        sparse_scores = bm25.get_scores(query.lower().split())

        dense_rank = {i: rank for rank, i in enumerate(range(len(texts)))}
        sparse_rank = {
            i: rank for rank, i in enumerate(
                sorted(range(len(texts)), key=lambda x: sparse_scores[x], reverse=True)
            )
        }

        alpha = settings.HYBRID_ALPHA
        rrf_scores = [
            alpha * (1 / (k + dense_rank.get(i, k)))
            + (1 - alpha) * (1 / (k + sparse_rank.get(i, k)))
            for i in range(len(texts))
        ]

        ranked = sorted(range(len(texts)), key=lambda i: rrf_scores[i], reverse=True)
        return [
            (dense_results[i][0], rrf_scores[i], dense_results[i][2])
            for i in ranked[:top_k]
        ]
