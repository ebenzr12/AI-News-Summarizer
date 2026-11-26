"""
Structured API error hierarchy and Starlette exception handlers.
"""

import logging
from typing import Any, Optional
from starlette.requests import Request
from starlette.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from src.language_detector import LanguageDetectionError
from src.summarizer import ModelNotReadyError, SummarizationError
from src.translator import TranslationError

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base application error mapped to a structured JSON HTTP response."""

    def __init__(
        self,
        message: str,
        code: str = "APP_ERROR",
        status_code: int = 400,
        details: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class InputValidationError(AppError):
    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=422,
            details=details,
        )


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Optional[Any] = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    payload = {
        "error": {
            "code": code,
            "message": message,
            "details": details,
        },
        "request_id": request_id,
    }
    return JSONResponse(status_code=status_code, content=payload)


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    return _error_response(
        request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def handle_pydantic_error(request: Request, exc: PydanticValidationError) -> JSONResponse:
    return _error_response(
        request,
        status_code=422,
        code="VALIDATION_ERROR",
        message="Input tidak valid. Periksa kembali field permintaan Anda.",
        details=exc.errors(),
    )


async def handle_language_error(request: Request, exc: LanguageDetectionError) -> JSONResponse:
    return _error_response(
        request,
        status_code=422,
        code="UNSUPPORTED_LANGUAGE",
        message=str(exc),
    )


async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    return _error_response(
        request,
        status_code=422,
        code="INVALID_INPUT",
        message=str(exc),
    )


async def handle_model_not_ready(request: Request, exc: ModelNotReadyError) -> JSONResponse:
    return _error_response(
        request,
        status_code=503,
        code="MODEL_NOT_READY",
        message=str(exc),
    )


async def handle_translation_error(request: Request, exc: TranslationError) -> JSONResponse:
    return _error_response(
        request,
        status_code=503,
        code="TRANSLATION_UNAVAILABLE",
        message=str(exc),
    )


async def handle_summarization_error(request: Request, exc: SummarizationError) -> JSONResponse:
    return _error_response(
        request,
        status_code=502,
        code="SUMMARIZATION_FAILED",
        message=str(exc),
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server exception", exc_info=exc)
    return _error_response(
        request,
        status_code=500,
        code="INTERNAL_SERVER_ERROR",
        message="Terjadi kesalahan internal pada server.",
    )
