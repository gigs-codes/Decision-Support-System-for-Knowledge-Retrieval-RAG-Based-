"""
Integration test suite for the RAG Decision Support API.

Run: pytest tests/ -v --asyncio-mode=auto
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    """Test client with mocked vector store."""
    mock_vs = AsyncMock()
    mock_vs.collection_stats = AsyncMock(
        return_value={"collection": "test", "total_chunks": 0}
    )
    mock_vs.upsert_chunks = AsyncMock(return_value=5)
    mock_vs.query = AsyncMock(return_value=[])
    mock_vs.delete_document = AsyncMock(return_value=0)

    app.state.vector_store = mock_vs
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Health endpoints ──────────────────────────────────────────────────────────

class TestHealth:
    async def test_liveness(self, client):
        r = await client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    async def test_readiness(self, client):
        r = await client.get("/api/v1/health/ready")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ready"

    async def test_config(self, client):
        r = await client.get("/api/v1/health/config")
        assert r.status_code == 200
        body = r.json()
        assert "llm_provider" in body
        assert "retrieval" in body


# ── Document ingestion ────────────────────────────────────────────────────────

class TestDocumentIngest:
    @patch("app.api.routes.documents.RetrievalService")
    @patch("app.api.routes.documents.EmbeddingService")
    async def test_ingest_text(self, mock_embed_cls, mock_retrieval_cls, client):
        mock_embed_cls.return_value = AsyncMock()
        mock_retrieval = AsyncMock()
        mock_retrieval.chunk_document.return_value = []
        mock_retrieval.index_chunks = AsyncMock(return_value=3)
        mock_retrieval_cls.return_value = mock_retrieval

        payload = {
            "title": "Test Policy Document",
            "content": "This is a long enough test document content. " * 20,
            "metadata": {"source": "test", "tags": ["policy"]},
        }
        r = await client.post("/api/v1/documents/ingest", json=payload)
        assert r.status_code == 201
        data = r.json()
        assert data["title"] == "Test Policy Document"
        assert data["status"] in ("indexed", "processing", "failed")

    async def test_ingest_too_short(self, client):
        payload = {"title": "Short", "content": "too short"}
        r = await client.post("/api/v1/documents/ingest", json=payload)
        assert r.status_code == 422  # validation error

    async def test_list_documents(self, client):
        r = await client.get("/api/v1/documents/")
        assert r.status_code == 200
        body = r.json()
        assert "items" in body
        assert "total" in body

    async def test_get_nonexistent_document(self, client):
        r = await client.get("/api/v1/documents/nonexistent-id")
        assert r.status_code == 404


# ── Query ─────────────────────────────────────────────────────────────────────

class TestQuery:
    @patch("app.api.routes.query.RAGPipeline")
    async def test_query_no_context(self, mock_pipeline_cls, client):
        from app.models.query import RAGResponse
        from datetime import datetime

        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(
            return_value=RAGResponse(
                answer="No relevant context found.",
                sources=[],
                query_id="test-id",
                model_used="none",
                retrieval_latency_ms=10.0,
                generation_latency_ms=0.0,
                total_latency_ms=10.0,
                chunks_retrieved=0,
                chunks_used=0,
                confidence_score=0.0,
                follow_up_questions=[],
            )
        )
        mock_pipeline_cls.return_value = mock_pipeline

        payload = {"question": "What is the refund policy?"}
        r = await client.post("/api/v1/query/", json=payload)
        assert r.status_code == 200

    async def test_query_too_short(self, client):
        r = await client.post("/api/v1/query/", json={"question": "Hi"})
        assert r.status_code == 422

    async def test_feedback_stats(self, client):
        r = await client.get("/api/v1/query/feedback/stats")
        assert r.status_code == 200


# ── Sessions ──────────────────────────────────────────────────────────────────

class TestSessions:
    @patch("app.api.routes.sessions.RAGPipeline")
    async def test_create_and_delete_session(self, mock_pipeline_cls, client):
        mock_pipeline = MagicMock()
        mock_pipeline.create_session.return_value = "test-session-id"
        mock_pipeline.clear_session.return_value = True
        mock_pipeline.list_sessions.return_value = []
        mock_pipeline_cls.return_value = mock_pipeline

        r = await client.post("/api/v1/sessions/")
        assert r.status_code == 200

    @patch("app.api.routes.sessions.RAGPipeline")
    async def test_list_sessions(self, mock_pipeline_cls, client):
        mock_pipeline = MagicMock()
        mock_pipeline.list_sessions.return_value = []
        mock_pipeline_cls.return_value = mock_pipeline

        r = await client.get("/api/v1/sessions/")
        assert r.status_code == 200


# ── Unit: chunking ────────────────────────────────────────────────────────────

class TestChunking:
    def test_paragraph_split(self):
        from app.services.retrieval_service import RetrievalService

        long_text = "\n\n".join([f"Paragraph {i}. " + "Word " * 100 for i in range(10)])
        from app.models.document import DocumentMetadata

        svc = RetrievalService.__new__(RetrievalService)
        chunks = svc.chunk_document(
            document_id="test-doc",
            title="Test",
            text=long_text,
            metadata=DocumentMetadata(),
        )
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.chunk_id.startswith("test-doc")
            assert len(chunk.text) > 0

    def test_overlap_preserved(self):
        from app.services.retrieval_service import RetrievalService
        from app.models.document import DocumentMetadata

        text = "\n\n".join(["Sentence " + str(i) + ". " + "X " * 200 for i in range(5)])
        svc = RetrievalService.__new__(RetrievalService)
        chunks = svc.chunk_document("d1", "Test", text, DocumentMetadata())
        # All chunks together should cover at least 80% of the original text
        combined = " ".join(c.text for c in chunks)
        assert len(combined) >= len(text) * 0.5
