"""Error envelope ``{"error": ErrorDetail}`` used for every non-2xx response."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from contracts.api import ErrorDetail, PipelineStage


# Body of every error response. Deliberately no docstring: Pydantic would emit it as a
# schema "description" that components.schemas.ErrorResponse in api/openapi.yaml lacks.
class ErrorResponse(BaseModel):
    error: ErrorDetail


class ApiError(Exception):
    """Raised by route handlers; rendered as ``{"error": {...}}`` with ``status_code``."""

    def __init__(self, status_code: int, code: str, message: str, stage: PipelineStage | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = ErrorDetail(code=code, message=message, stage=stage)


def not_found(message: str = "Unknown or expired job.") -> ApiError:
    """404 with code ``not_found``."""
    return ApiError(404, "not_found", message)


_STATUS_CODES = {404: "not_found", 405: "method_not_allowed", 413: "too_large", 415: "unsupported_media_type"}


def error_response(status_code: int, detail: ErrorDetail, headers: dict[str, str] | None = None) -> JSONResponse:
    """Render an ErrorDetail in the spec envelope."""
    body = ErrorResponse(error=detail).model_dump(mode="json")
    return JSONResponse(body, status_code=status_code, headers=headers)


async def _api_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(exc.status_code, exc.detail)


async def _http_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    code = _STATUS_CODES.get(exc.status_code, "http_error")
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return error_response(exc.status_code, ErrorDetail(code=code, message=message), dict(exc.headers or {}))


def _format_loc(loc: tuple[object, ...]) -> str:
    return ".".join(str(p) for p in loc if p != "body") or "body"


def _is_settings_error(error: Mapping[str, Any]) -> bool:
    """An error inside the ``settings`` object (not a missing ``settings`` key)."""
    loc = tuple(error.get("loc", ()))
    return loc[:2] == ("body", "settings") and (len(loc) > 2 or error.get("type") != "missing")


async def _validation_error(_: Request, exc: Exception) -> JSONResponse:
    """Path-parameter errors (bad UUID) are 404; body errors are 422 invalid_settings/invalid_request."""
    assert isinstance(exc, RequestValidationError)
    errors = list(exc.errors())
    if any(e.get("loc", ("",))[0] == "path" for e in errors):
        return error_response(404, ErrorDetail(code="not_found", message="Unknown or expired job."))
    in_settings = bool(errors) and all(_is_settings_error(e) for e in errors)
    code = "invalid_settings" if in_settings else "invalid_request"
    message = "; ".join(f"{_format_loc(tuple(e.get('loc', ())))}: {e.get('msg', 'invalid')}" for e in errors[:5])
    return error_response(422, ErrorDetail(code=code, message=message or "Invalid request."))


async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    return error_response(500, ErrorDetail(code="internal_error", message=f"{type(exc).__name__}: {exc}"))


def install_error_handlers(app: FastAPI) -> None:
    """Replace FastAPI's ``{"detail": ...}`` bodies with the spec's ``{"error": {...}}`` shape."""
    app.add_exception_handler(ApiError, _api_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled)
