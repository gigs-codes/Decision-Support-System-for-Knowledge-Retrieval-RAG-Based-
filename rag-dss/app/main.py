"""
RAG-Based Decision Support System
Entry point for the FastAPI application.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import documents, query, sessions, health
from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.middleware import APIKeyMiddleware, RateLimitMiddleware
from app.db.vector_store import VectorStoreManager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager — startup & shutdown hooks."""
    setup_logging(settings.LOG_LEVEL)
    logger.info("🚀 Starting RAG Decision Support System v%s", settings.VERSION)

    # Initialize vector store
    vs_manager = VectorStoreManager()
    await vs_manager.initialize()
    app.state.vector_store = vs_manager

    logger.info("✅ Vector store initialised — collection: %s", settings.CHROMA_COLLECTION)
    yield

    # Graceful shutdown
    logger.info("🛑 Shutting down RAG Decision Support System")
    await vs_manager.close()


app = FastAPI(
    title="RAG Decision Support System",
    description=(
        "A production-grade knowledge retrieval and decision-support API powered by "
        "Retrieval-Augmented Generation (RAG). Upload documents, query the knowledge base, "
        "and receive grounded, cited AI responses."
    ),
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── Middleware stack (order matters — outermost applied last) ─────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.ENABLE_RATE_LIMIT:
    app.add_middleware(RateLimitMiddleware, max_requests=settings.RATE_LIMIT_RPM)

if settings.REQUIRE_API_KEY:
    app.add_middleware(APIKeyMiddleware, api_key=settings.API_KEY)


# ── Request timing ────────────────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{(time.perf_counter() - start) * 1000:.2f}ms"
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])
