"""
Custom ASGI middleware.

- APIKeyMiddleware  : validates X-API-Key header on every non-health request.
- RateLimitMiddleware: sliding-window in-process rate limiter (per IP).
"""

import time
from collections import defaultdict, deque
from typing import Callable, Deque

from fastapi import status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.logging import get_logger

logger = get_logger(__name__)

_EXEMPT_PATHS = {"/api/v1/health", "/api/docs", "/api/redoc", "/api/openapi.json", "/"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests that don't carry the correct X-API-Key header."""

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != self._api_key:
            logger.warning("invalid_api_key", path=request.url.path, ip=_get_ip(request))
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter: max `max_requests` per 60-second window per IP.
    State is in-process (use Redis in a multi-replica deployment).
    """

    def __init__(self, app: ASGIApp, max_requests: int = 60) -> None:
        super().__init__(app)
        self._max_requests = max_requests
        self._window = 60  # seconds
        self._buckets: dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        ip = _get_ip(request)
        now = time.monotonic()
        bucket = self._buckets[ip]

        # Evict timestamps outside the window
        while bucket and bucket[0] < now - self._window:
            bucket.popleft()

        if len(bucket) >= self._max_requests:
            retry_after = int(self._window - (now - bucket[0])) + 1
            logger.warning("rate_limit_hit", ip=ip, count=len(bucket))
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded. Retry after {retry_after}s.",
                    "retry_after": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._max_requests)
        response.headers["X-RateLimit-Remaining"] = str(self._max_requests - len(bucket))
        return response


def _get_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
