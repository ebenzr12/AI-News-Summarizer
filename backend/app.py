"""
Starlette ASGI web application for AI News Summarizer.
Exposes versioned REST API endpoints (/api/v1/*) and serves the responsive frontend SPA.
"""

import json
import logging
import time
import uuid
from typing import Optional
import anyio
from pydantic import ValidationError as PydanticValidationError
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from config.config import CORS_ALLOW_ORIGINS, FRONTEND_DIR
from backend.errors import (
    AppError,
    InputValidationError,
    handle_app_error,
    handle_language_error,
    handle_model_not_ready,
    handle_pydantic_error,
    handle_summarization_error,
    handle_translation_error,
    handle_unexpected_error,
    handle_value_error,
)
from backend.schemas import (
    DetectLanguageRequest,
    SamplesResponse,
    SummarizeRequest,
    TranslateRequest,
)
from backend.services import SummarizerService
from src.language_detector import LanguageDetectionError
from src.summarizer import ModelNotReadyError, SummarizationError
from src.translator import TranslationError

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID, timing header, and baseline security headers."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)

        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


async def _parse_json_body(request: Request) -> dict:
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise InputValidationError(f"Body JSON tidak valid: {exc}") from exc
    if not isinstance(data, dict):
        raise InputValidationError("Body JSON harus berupa objek JSON.")
    return data


def create_app(service: Optional[SummarizerService] = None) -> Starlette:
    """
    Application factory allowing clean dependency injection in tests and production.
    """
    app_service = service or SummarizerService()

    async def health(request: Request) -> JSONResponse:
        res = await anyio.to_thread.run_sync(app_service.get_health)
        return JSONResponse(res.model_dump())

    async def samples(request: Request) -> JSONResponse:
        items = app_service.list_samples()
        payload = SamplesResponse(samples=items)
        return JSONResponse(payload.model_dump())

    async def detect_language_endpoint(request: Request) -> JSONResponse:
        body = await _parse_json_body(request)
        req_model = DetectLanguageRequest(**body)
        res = await anyio.to_thread.run_sync(
            app_service.detect_language_and_stats, req_model
        )
        return JSONResponse(res.model_dump())

    async def summarize_endpoint(request: Request) -> JSONResponse:
        body = await _parse_json_body(request)
        req_model = SummarizeRequest(**body)
        res = await anyio.to_thread.run_sync(app_service.summarize, req_model)
        return JSONResponse(res.model_dump())

    async def translate_endpoint(request: Request) -> JSONResponse:
        body = await _parse_json_body(request)
        req_model = TranslateRequest(**body)
        res = await anyio.to_thread.run_sync(app_service.translate, req_model)
        return JSONResponse(res.model_dump())

    async def serve_index(request: Request) -> FileResponse:
        index_path = FRONTEND_DIR / "index.html"
        if not index_path.exists():
            raise AppError(
                "Frontend index.html tidak ditemukan.",
                code="FRONTEND_MISSING",
                status_code=404,
            )
        return FileResponse(str(index_path))

    routes = [
        Route("/", endpoint=serve_index, methods=["GET"]),
        Route("/api/v1/health", endpoint=health, methods=["GET"]),
        Route("/api/v1/samples", endpoint=samples, methods=["GET"]),
        Route("/api/v1/detect-language", endpoint=detect_language_endpoint, methods=["POST"]),
        Route("/api/v1/summarize", endpoint=summarize_endpoint, methods=["POST"]),
        Route("/api/v1/translate", endpoint=translate_endpoint, methods=["POST"]),
    ]

    if FRONTEND_DIR.exists():
        routes.append(
            Mount("/static", app=StaticFiles(directory=str(FRONTEND_DIR)), name="static")
        )

    exception_handlers = {
        AppError: handle_app_error,
        PydanticValidationError: handle_pydantic_error,
        LanguageDetectionError: handle_language_error,
        ValueError: handle_value_error,
        ModelNotReadyError: handle_model_not_ready,
        TranslationError: handle_translation_error,
        SummarizationError: handle_summarization_error,
        Exception: handle_unexpected_error,
    }

    app = Starlette(routes=routes, exception_handlers=exception_handlers)
    app.state.service = app_service

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOW_ORIGINS,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    return app


app = create_app()
