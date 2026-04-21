"""
/api/v1/query — RAG query endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.core.logging import get_logger
from app.models.query import FeedbackRequest, QueryRequest, RAGResponse
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()
logger = get_logger(__name__)

_pipeline_cache: dict[str, RAGPipeline] = {}


def get_rag_pipeline(request: Request) -> RAGPipeline:
    vs = request.app.state.vector_store
    # Cheap singleton per process
    key = id(vs)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = RAGPipeline(vs)
    return _pipeline_cache[key]


@router.post(
    "/",
    response_model=RAGResponse,
    summary="Submit a question to the RAG decision-support engine",
    description=(
        "Retrieves the most relevant knowledge-base chunks for the question, "
        "then generates a grounded, cited answer. Supports multi-turn sessions."
    ),
)
async def query(
    body: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> RAGResponse:
    try:
        return await pipeline.query(body)
    except Exception as exc:
        logger.exception("query_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {exc}",
        )


@router.post(
    "/feedback",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Submit user feedback on a query response",
)
async def submit_feedback(
    body: FeedbackRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    pipeline.record_feedback(body)


@router.get(
    "/feedback/stats",
    summary="Aggregate feedback statistics",
)
async def feedback_stats(pipeline: RAGPipeline = Depends(get_rag_pipeline)):
    return pipeline.get_feedback_stats()
