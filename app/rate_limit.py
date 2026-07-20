"""
Basit bellek içi rate limiter.
"""

import time
from collections import defaultdict
from typing import Optional

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class RateLimitMiddleware:
    def __init__(self, app: ASGIApp, limit: int = 100, window: int = 60):
        self.app = app
        self.limit = limit
        self.window = window
        self.hits: dict[str, list[float]] = defaultdict(list)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope)
        client_ip = self._get_client_ip(request)

        path = scope.get("path", "")
        if request.method != "POST" or not (path.startswith("/faces/recognize") or path.startswith("/faces/enroll")):
            await self.app(scope, receive, send)
            return

        # Debug: her limited istekte state'i kontrol et
        now = time.time()
        window_start = now - self.window
        recent = [t for t in self.hits.get(client_ip, []) if t > window_start]

        if len(recent) >= self.limit:
            response = JSONResponse(
                status_code=429,
                content={"detail": "Çok fazla istek gönderdiniz. Lütfen biraz bekleyin."},
                headers={
                    "Retry-After": str(self.window),
                    "X-RateLimit-Limit": str(self.limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
            await response(scope, receive, send)
            return

        recent.append(now)
        self.hits[client_ip] = recent

        response = await self.app(scope, receive, send)
        return response

    def _get_client_ip(self, request: Request) -> str:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "unknown"
