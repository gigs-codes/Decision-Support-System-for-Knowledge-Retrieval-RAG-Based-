"""
Embedding service.

Uses sentence-transformers (local, no API key required) by default.
The model is loaded once and kept in memory (thread-safe for inference).
"""

import asyncio
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_MODEL_INSTANCE: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL)
        _MODEL_INSTANCE = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("embedding_model_loaded", dimension=settings.EMBEDDING_DIMENSION)
    return _MODEL_INSTANCE


class EmbeddingService:
    """Async wrapper around sentence-transformers for batch text embedding."""

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return normalised embeddings for a list of texts."""
        return await asyncio.to_thread(self._sync_embed, texts)

    def _sync_embed(self, texts: List[str]) -> List[List[float]]:
        model = _get_model()
        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=settings.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,   # unit vectors for cosine similarity
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        results = await self.embed_texts([text])
        return results[0]

    async def similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity between two pre-normalised embeddings."""
        vec_a = np.array(a)
        vec_b = np.array(b)
        return float(np.dot(vec_a, vec_b))   # already unit vectors
