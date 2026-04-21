# RAG Decision Support System

> **Production-grade knowledge retrieval API powered by Retrieval-Augmented Generation.**  
> Upload documents → ask questions → get cited, grounded answers.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

The **RAG Decision Support System** is a web-based knowledge retrieval prototype that demonstrates how Retrieval-Augmented Generation can ground AI responses in verified document sources — eliminating hallucination by design.

Users ingest documents (policy manuals, technical guides, research papers) into a semantic vector database, then query them in natural language. The system retrieves the most relevant passages and feeds them as context to an LLM, producing cited, transparent answers.

```
User Question ──► Embedding ──► Vector Search ──► Top-K Chunks
                                                        │
                                         ┌──────────────┘
                                         ▼
                              LLM (with cited context) ──► Grounded Answer + Sources
```

---

## Key Features

| Feature | Description |
|---|---|
| **REST API** | Full OpenAPI spec with FastAPI — auto-generated docs at `/api/docs` |
| **Hybrid Search** | Dense vector (sentence-transformers) + BM25 sparse, fused via Reciprocal Rank Fusion |
| **Multi-Provider LLM** | OpenAI, Anthropic Claude, or local Ollama — switchable via env var |
| **Multi-turn Sessions** | Conversation memory with configurable history window |
| **Semantic Chunking** | Paragraph-aware, overlap-preserving document splitting |
| **File Ingestion** | REST upload `.txt`, `.md`, `.json`, `.csv` files |
| **Metadata Filtering** | Query scoped to department, author, or custom tags |
| **Rate Limiting** | Sliding-window per-IP limiter with `Retry-After` headers |
| **Feedback Loop** | Per-query thumbs up/down; aggregate stats endpoint |
| **Dashboard UI** | Standalone HTML frontend — zero framework dependencies |
| **Docker-ready** | Multi-stage Dockerfile + Compose for one-command deploy |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI App                          │
│                                                             │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐  │
│  │  /documents  │  │    /query      │  │   /sessions    │  │
│  │  (ingest,    │  │  (RAG query,   │  │  (multi-turn   │  │
│  │  upload,     │  │   feedback,    │  │   history)     │  │
│  │  delete)     │  │   streaming)   │  │                │  │
│  └──────┬───────┘  └───────┬────────┘  └────────────────┘  │
│         │                  │                                 │
│  ┌──────▼──────────────────▼──────────────────────────────┐ │
│  │                    RAG Pipeline                         │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌────────────┐  │ │
│  │  │   Embedding   │  │   Retrieval   │  │ Generation │  │ │
│  │  │   Service     │  │   Service     │  │  Service   │  │ │
│  │  │ (all-MiniLM)  │  │ (dense+BM25)  │  │(OpenAI/   │  │ │
│  │  └───────┬───────┘  └───────┬───────┘  │Anthropic/ │  │ │
│  │          │                  │           │  Ollama)  │  │ │
│  └──────────┼──────────────────┼───────────┴────────────┘ │ │
│             │                  │                            │ │
│  ┌──────────▼──────────────────▼──────────────────────┐    │ │
│  │           ChromaDB Vector Store                     │    │ │
│  │  (embedded PersistentClient or HTTP server mode)    │    │ │
│  └─────────────────────────────────────────────────────┘    │ │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- An LLM API key (OpenAI, Anthropic) — or a local [Ollama](https://ollama.com) instance

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/rag-decision-support.git
cd rag-decision-support

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key
```

Minimal `.env` for OpenAI:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

For local Ollama (no API key needed):
```env
LLM_PROVIDER=local
LOCAL_LLM_MODEL=mistral
```

### 3. Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Seed sample data

```bash
python scripts/ingest_sample_data.py
```

This ingests three sample documents (Remote Work Policy, API Integration Guide, Data Governance Framework).

### 5. Query

```bash
curl -s -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the remote work expense reimbursement policy?"}' \
  | python -m json.tool
```

### 6. Open the dashboard

Open `frontend/index.html` directly in your browser (no server needed).  
Set the API base URL to `http://localhost:8000` in the top bar.

---

## Docker

```bash
# Quick start (embedded ChromaDB)
docker compose up api

# With ChromaDB server
docker compose --profile server up

# Development (hot-reload)
docker compose --profile dev up api-dev
```

---

## API Reference

Interactive docs: **http://localhost:8000/api/docs**

### Core endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/ingest` | Ingest plain-text document |
| `POST` | `/api/v1/documents/upload` | Upload a file (txt/md/json/csv) |
| `GET`  | `/api/v1/documents/` | List all indexed documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete document + embeddings |
| `POST` | `/api/v1/query/` | Submit a RAG query |
| `POST` | `/api/v1/query/feedback` | Submit answer feedback |
| `POST` | `/api/v1/sessions/` | Create a conversation session |
| `GET`  | `/api/v1/sessions/{id}` | Retrieve session history |
| `GET`  | `/api/v1/health` | Liveness probe |
| `GET`  | `/api/v1/health/ready` | Readiness probe |

### Example: query with session

```python
import httpx

BASE = "http://localhost:8000"

# Create a session for multi-turn dialogue
session = httpx.post(f"{BASE}/api/v1/sessions/").json()
session_id = session["session_id"]

# First turn
r = httpx.post(f"{BASE}/api/v1/query/", json={
    "question": "What are the working hours for remote employees?",
    "session_id": session_id,
    "top_k": 5,
    "include_sources": True,
})
data = r.json()
print(data["answer"])
print("Sources:", [s["title"] for s in data["sources"]])
print("Follow-ups:", data["follow_up_questions"])

# Second turn — context is maintained
r2 = httpx.post(f"{BASE}/api/v1/query/", json={
    "question": "What about the equipment policy?",
    "session_id": session_id,
})
print(r2.json()["answer"])
```

### Query payload options

```json
{
  "question": "What is the GDPR data deletion requirement?",
  "session_id": "optional-uuid",
  "top_k": 5,
  "score_threshold": 0.35,
  "filter_metadata": { "department": "Legal" },
  "include_sources": true
}
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `local` |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `RETRIEVAL_TOP_K` | `5` | Default chunks to retrieve |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.35` | Min cosine similarity |
| `HYBRID_SEARCH_ENABLED` | `true` | Enable BM25 + dense fusion |
| `HYBRID_ALPHA` | `0.7` | Dense weight in hybrid fusion |
| `CHUNK_SIZE` | `512` | Token budget per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap tokens between chunks |
| `USE_CHROMA_SERVER` | `false` | `false` = embedded, `true` = HTTP |
| `REQUIRE_API_KEY` | `false` | Enforce `X-API-Key` header |
| `RATE_LIMIT_RPM` | `60` | Requests/minute per IP |
| `SESSION_TTL_MINUTES` | `60` | Session expiry |

---

## Testing

```bash
pytest tests/ -v
```

With coverage report:
```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

---

## Project Structure

```
rag-decision-support/
├── app/
│   ├── main.py                   # FastAPI app factory, middleware, lifespan
│   ├── api/routes/
│   │   ├── documents.py          # Ingest, upload, list, delete
│   │   ├── query.py              # RAG query + feedback
│   │   ├── sessions.py           # Multi-turn session management
│   │   └── health.py             # Liveness, readiness, config
│   ├── core/
│   │   ├── config.py             # Pydantic-settings configuration
│   │   ├── logging.py            # Structured JSON logging (structlog)
│   │   └── middleware.py         # API key auth + rate limiter
│   ├── db/
│   │   └── vector_store.py       # ChromaDB async wrapper
│   ├── models/
│   │   ├── document.py           # Document domain models
│   │   └── query.py              # Query / response models
│   └── services/
│       ├── embedding_service.py  # Sentence-transformers wrapper
│       ├── retrieval_service.py  # Chunking + hybrid retrieval
│       ├── generation_service.py # Multi-provider LLM client
│       └── rag_pipeline.py       # End-to-end RAG orchestration
├── frontend/
│   └── index.html                # Standalone dashboard (no build step)
├── tests/
│   └── test_api.py               # API integration + unit tests
├── scripts/
│   └── ingest_sample_data.py     # Sample data seeder
├── Dockerfile                    # Multi-stage production image
├── docker-compose.yml            # Dev + prod compose config
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Production Deployment

For a multi-replica production deployment, replace the following in-process stores with persistent equivalents:

| Component | Dev (current) | Production recommendation |
|---|---|---|
| Session store | In-process dict | **Redis** |
| Document registry | In-process dict | **PostgreSQL** |
| Vector store | Embedded ChromaDB | **ChromaDB server** or **Qdrant** |
| Rate limiter | In-process | **Redis** sliding window |
| Embedding cache | None | **Redis** or file cache |

---

## License

MIT — see [LICENSE](LICENSE).
