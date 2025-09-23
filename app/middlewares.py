import time, structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from .logging_config import logger, new_request_id, safe_hash_bytes

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        req_id = request.headers.get("x-request-id") or new_request_id()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=req_id)

        start = time.perf_counter()
        try:
            body = await request.body()
            payload_size = len(body) if body else 0
            payload_sig  = safe_hash_bytes(body[:4096]) if body else None

            response: Response = await call_next(request)
            dur_ms = (time.perf_counter() - start) * 1000

            logger.info("http_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(dur_ms, 2),
                request_id=req_id,
                user_agent=request.headers.get("user-agent"),
                client_ip=request.client.host if request.client else None,
                payload_size=payload_size,
                payload_sig=payload_sig,
            )
            response.headers["x-request-id"] = req_id
            return response
        except Exception:
            dur_ms = (time.perf_counter() - start) * 1000
            logger.exception("http_error",
                path=request.url.path,
                duration_ms=round(dur_ms, 2),
                request_id=req_id,
            )
            raise
