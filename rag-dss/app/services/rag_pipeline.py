"""
RAG Pipeline — orchestrates retrieval + generation + session management.

End-to-end flow:
  1.  Pre-process question (strip, normalise)
  2.  Load conversation history (if session provided)
  3.  Retrieve top-k chunks from vector store
  4.  Compute aggregate confidence score
  5.  Generate grounded answer via LLM
  6.  Persist conversation turn
  7.  Return structured RAGResponse
"""

import time
import uuid
from typing import Dict, List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.db.vector_store import VectorStoreManager
from app.models.query import (
    CitedSource,
    ConversationTurn,
    FeedbackRequest,
    QueryRequest,
    RAGResponse,
    RetrievedChunk,
)
from app.services.embedding_service import EmbeddingService
from app.services.generation_service import GenerationService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)

# In-process session store (swap for Redis in production)
_sessions: Dict[str, List[ConversationTurn]] = {}
_feedback_log: List[dict] = []


class RAGPipeline:
    def __init__(self, vector_store: VectorStoreManager) -> None:
        self._embed = EmbeddingService()
        self._retriever = RetrievalService(vector_store, self._embed)
        self._generator = GenerationService()

    # ── Main query entrypoint ─────────────────────────────────────────────────

    async def query(self, request: QueryRequest) -> RAGResponse:
        query_id = str(uuid.uuid4())
        total_start = time.perf_counter()

        # 1. Resolve session history
        session_id = request.session_id
        history: List[ConversationTurn] = []
        if session_id:
            history = _sessions.get(str(session_id), [])

        # 2. Retrieve
        retrieval_start = time.perf_counter()
        chunks = await self._retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            metadata_filter=request.filter_metadata or None,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        logger.info(
            "retrieval_complete",
            query_id=query_id,
            chunks_found=len(chunks),
            latency_ms=round(retrieval_ms, 1),
        )

        if not chunks:
            return self._no_context_response(
                request, query_id, session_id, retrieval_ms
            )

        # 3. Generate
        gen_start = time.perf_counter()
        answer, follow_ups, gen_ms = await self._generator.generate(
            question=request.question,
            chunks=chunks,
            history=history[-settings.MAX_HISTORY_TURNS * 2 :],
        )
        gen_ms = (time.perf_counter() - gen_start) * 1000

        # 4. Confidence = weighted average relevance of used chunks
        confidence = round(
            sum(c.score for c in chunks) / len(chunks), 3
        ) if chunks else 0.0

        # 5. Persist conversation
        if session_id:
            sid = str(session_id)
            _sessions.setdefault(sid, [])
            _sessions[sid].append(
                ConversationTurn(role="user", content=request.question)
            )
            _sessions[sid].append(
                ConversationTurn(role="assistant", content=answer)
            )
            # Trim to window
            if len(_sessions[sid]) > settings.MAX_HISTORY_TURNS * 2:
                _sessions[sid] = _sessions[sid][-settings.MAX_HISTORY_TURNS * 2 :]

        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "query_complete",
            query_id=query_id,
            total_ms=round(total_ms, 1),
            confidence=confidence,
            provider=settings.LLM_PROVIDER,
        )

        sources = (
            [
                CitedSource(
                    document_id=c.document_id,
                    title=c.document_title,
                    excerpt=c.text[:300] + ("…" if len(c.text) > 300 else ""),
                    score=c.score,
                    metadata=c.metadata,
                )
                for c in chunks
            ]
            if request.include_sources
            else []
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            session_id=session_id,
            query_id=query_id,
            model_used=self._model_label(),
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=round(gen_ms, 2),
            total_latency_ms=round(total_ms, 2),
            chunks_retrieved=len(chunks),
            chunks_used=len(chunks),
            confidence_score=confidence,
            follow_up_questions=follow_ups,
        )

    # ── Session management ────────────────────────────────────────────────────

    def get_session(self, session_id: str) -> List[ConversationTurn]:
        return _sessions.get(session_id, [])

    def create_session(self) -> str:
        sid = str(uuid.uuid4())
        _sessions[sid] = []
        return sid

    def clear_session(self, session_id: str) -> bool:
        if session_id in _sessions:
            del _sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[dict]:
        return [
            {"session_id": k, "turn_count": len(v) // 2}
            for k, v in _sessions.items()
        ]

    # ── Feedback ──────────────────────────────────────────────────────────────

    def record_feedback(self, feedback: FeedbackRequest) -> None:
        _feedback_log.append(feedback.model_dump())
        logger.info("feedback_recorded", query_id=feedback.query_id, rating=feedback.rating)

    def get_feedback_stats(self) -> dict:
        if not _feedback_log:
            return {"total": 0}
        ratings = [f["rating"] for f in _feedback_log]
        return {
            "total": len(_feedback_log),
            "avg_rating": round(sum(ratings) / len(ratings), 2),
            "helpful_pct": round(
                sum(1 for f in _feedback_log if f["helpful"]) / len(_feedback_log) * 100, 1
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _model_label(self) -> str:
        mapping = {
            "openai": settings.OPENAI_MODEL,
            "anthropic": settings.ANTHROPIC_MODEL,
            "local": settings.LOCAL_LLM_MODEL,
        }
        return mapping.get(settings.LLM_PROVIDER, "unknown")

    def _no_context_response(
        self,
        request: QueryRequest,
        query_id: str,
        session_id,
        retrieval_ms: float,
    ) -> RAGResponse:
        return RAGResponse(
            answer=(
                "I couldn't find relevant information in the knowledge base to answer "
                "your question. Please try rephrasing or upload relevant documents first."
            ),
            sources=[],
            session_id=session_id,
            query_id=query_id,
            model_used="none",
            retrieval_latency_ms=round(retrieval_ms, 2),
            generation_latency_ms=0.0,
            total_latency_ms=round(retrieval_ms, 2),
            chunks_retrieved=0,
            chunks_used=0,
            confidence_score=0.0,
            follow_up_questions=[],
        )
