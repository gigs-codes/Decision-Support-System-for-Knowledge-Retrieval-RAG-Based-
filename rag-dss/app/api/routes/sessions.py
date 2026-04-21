"""
/api/v1/sessions — multi-turn conversation session management.
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request

from app.models.query import ConversationTurn
from app.services.rag_pipeline import RAGPipeline

router = APIRouter()

_pipeline_cache: dict = {}


def get_pipeline(request: Request) -> RAGPipeline:
    vs = request.app.state.vector_store
    key = id(vs)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = RAGPipeline(vs)
    return _pipeline_cache[key]


@router.post("/", summary="Create a new conversation session")
async def create_session(pipeline: RAGPipeline = Depends(get_pipeline)):
    sid = pipeline.create_session()
    return {"session_id": sid}


@router.get("/", summary="List all active sessions")
async def list_sessions(pipeline: RAGPipeline = Depends(get_pipeline)):
    return {"sessions": pipeline.list_sessions()}


@router.get("/{session_id}", response_model=List[ConversationTurn], summary="Get session history")
async def get_session(session_id: str, pipeline: RAGPipeline = Depends(get_pipeline)):
    history = pipeline.get_session(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return history


@router.delete("/{session_id}", status_code=204, summary="Clear a session")
async def delete_session(session_id: str, pipeline: RAGPipeline = Depends(get_pipeline)):
    if not pipeline.clear_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
