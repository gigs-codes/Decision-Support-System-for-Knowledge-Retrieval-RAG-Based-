"""Health and readiness endpoints."""

from fastapi import APIRouter, Request

from app.core.config import settings

router = APIRouter()


@router.get("/health", tags=["Health"], summary="Liveness probe")
async def health():
    return {"status": "ok", "version": settings.VERSION}


@router.get("/health/ready", tags=["Health"], summary="Readiness probe — checks vector store")
async def readiness(request: Request):
    try:
        vs = request.app.state.vector_store
        stats = await vs.collection_stats()
        return {"status": "ready", "vector_store": stats}
    except Exception as exc:
        return {"status": "not_ready", "error": str(exc)}


@router.get("/health/config", tags=["Health"], summary="Non-sensitive config summary")
async def config_summary():
    return {
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": {
            "openai": settings.OPENAI_MODEL,
            "anthropic": settings.ANTHROPIC_MODEL,
            "local": settings.LOCAL_LLM_MODEL,
        }.get(settings.LLM_PROVIDER),
        "embedding_model": settings.EMBEDDING_MODEL,
        "retrieval": {
            "top_k": settings.RETRIEVAL_TOP_K,
            "score_threshold": settings.RETRIEVAL_SCORE_THRESHOLD,
            "hybrid_search": settings.HYBRID_SEARCH_ENABLED,
        },
        "chunking": {
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        },
    }
