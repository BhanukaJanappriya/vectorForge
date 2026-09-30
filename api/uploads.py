"""Streaming multipart upload handling for ``POST /convert``.

The request body is parsed incrementally (python-multipart) and the ``file`` part is
written straight to disk; the transfer is aborted with 413 as soon as it exceeds the size
limit, so an oversized upload is never buffered in memory. The file type is decided by
magic bytes, never by the file name or the client's Content-Type.
"""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from PIL import Image
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import Request

from api.errors import ApiError
from contracts.api import PipelineStage

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"
EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg"}
PIL_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG"}
MAX_FIELD_BYTES = 64 * 1024
"""Limit for non-file form fields (the settings JSON)."""
MAX_PARTS = 16
MAX_FILENAME_CHARS = 200


def sniff_media_type(head: bytes) -> str | None:
    """Media type from the first bytes of a file (PNG or JPEG), else None."""
    if head.startswith(PNG_MAGIC):
        return "image/png"
    if head.startswith(JPEG_MAGIC):
        return "image/jpeg"
    return None


def safe_filename(raw: str | None) -> str:
    """Base name of a client-supplied file name, without path parts or control characters."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:MAX_FILENAME_CHARS] or "upload"


def _upload_error(status: int, code: str, message: str) -> ApiError:
    return ApiError(status, code, message, PipelineStage.UPLOAD)


@dataclass
class ParsedUpload:
    """Result of streaming a multipart body."""

    path: Path | None = None
    filename: str | None = None
    size: int = 0
    settings_raw: str | None = None


@dataclass
class _Part:
    name: str = ""
    filename: str | None = None
    is_file_target: bool = False
    data: bytearray = field(default_factory=bytearray)


class _MultipartSink:
    """python-multipart callbacks writing the ``file`` part to ``dest``."""

    def __init__(self, dest: Path, max_bytes: int) -> None:
        self.dest = dest
        self.max_bytes = max_bytes
        self.result = ParsedUpload()
        self.parts = 0
        self._part = _Part()
        self._header_name = b""
        self._header_value = b""
        self._disposition = b""
        self._fh: BinaryIO | None = None

    # python-multipart callback protocol ------------------------------------

    def on_part_begin(self) -> None:
        self.parts += 1
        if self.parts > MAX_PARTS:
            raise _upload_error(422, "invalid_request", f"Too many form fields (max {MAX_PARTS}).")
        self._part = _Part()
        self._disposition = b""

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        if self._header_name.strip().lower() == b"content-disposition":
            self._disposition = self._header_value
        self._header_name = b""
        self._header_value = b""

    def on_headers_finished(self) -> None:
        _, options = parse_options_header(self._disposition)
        name = options.get(b"name")
        if name is None:
            raise _upload_error(422, "invalid_request", "A multipart part has no Content-Disposition name.")
        self._part.name = name.decode("utf-8", "replace")
        raw_filename = options.get(b"filename")
        self._part.filename = raw_filename.decode("utf-8", "replace") if raw_filename is not None else None
        if self._part.name == "file":
            if self.result.path is not None:
                raise _upload_error(422, "invalid_request", "Only one file may be uploaded.")
            self._part.is_file_target = True
            self.result.path = self.dest
            self.result.filename = safe_filename(self._part.filename)
            self._fh = open(self.dest, "wb")  # noqa: SIM115 - closed in on_part_end / close()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._part.is_file_target:
            assert self._fh is not None
            self.result.size += len(chunk)
            if self.result.size > self.max_bytes:
                raise _upload_error(413, "too_large", f"File exceeds the {self.max_bytes}-byte upload limit.")
            self._fh.write(chunk)
            return
        if len(self._part.data) + len(chunk) > MAX_FIELD_BYTES:
            raise _upload_error(422, "invalid_request", f"Form field {self._part.name!r} is too large.")
        self._part.data.extend(chunk)

    def on_part_end(self) -> None:
        if self._part.is_file_target:
            self.close()
        elif self._part.name == "settings":
            self.result.settings_raw = self._part.data.decode("utf-8", "replace")

    def close(self) -> None:
        """Close the output file (idempotent)."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def callbacks(self) -> dict[str, object]:
        """Callback mapping for MultipartParser."""
        return {
            "on_part_begin": self.on_part_begin,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
        }


async def receive_upload(request: Request, dest: Path, max_bytes: int) -> ParsedUpload:
    """Stream a ``multipart/form-data`` body; the ``file`` part is written to ``dest``.

    Raises ApiError: 415 if the body is not multipart, 413 once the file exceeds
    ``max_bytes`` (checked against Content-Length first, then while streaming),
    422 for malformed bodies or a missing file.
    """
    content_type, params = parse_options_header(request.headers.get("content-type", ""))
    if content_type != b"multipart/form-data":
        raise _upload_error(415, "unsupported_media_type", "Request body must be multipart/form-data.")
    boundary = params.get(b"boundary")
    if not boundary:
        raise _upload_error(422, "invalid_request", "Missing multipart boundary.")
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes + MAX_PARTS * MAX_FIELD_BYTES:
        raise _upload_error(413, "too_large", f"Request exceeds the {max_bytes}-byte upload limit.")

    sink = _MultipartSink(dest, max_bytes)
    parser = MultipartParser(boundary, sink.callbacks())  # type: ignore[arg-type]
    try:
        async for chunk in request.stream():
            if chunk:
                parser.write(chunk)
        parser.finalize()
    except ApiError:
        raise
    except Exception as exc:  # python-multipart raises several parse error types
        raise _upload_error(422, "invalid_request", f"Malformed multipart body: {exc}") from exc
    finally:
        sink.close()
    if sink.result.path is None:
        raise _upload_error(422, "invalid_request", "Missing 'file' form field.")
    return sink.result


def check_image(path: Path, max_pixels: int) -> str:
    """Validate an uploaded file by content and return its media type.

    Raises ApiError: 415 if it is not PNG/JPEG by magic bytes, 413 if it has more than
    ``max_pixels`` pixels, 422 if it cannot be decoded.
    """
    with open(path, "rb") as fh:
        head = fh.read(len(PNG_MAGIC))
    if not head:
        raise _upload_error(422, "invalid_image", "The uploaded file is empty.")
    media_type = sniff_media_type(head)
    if media_type is None:
        raise _upload_error(415, "unsupported_media_type", "Only PNG and JPEG images are accepted.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                width, height = img.size
                if width * height > max_pixels:
                    raise _upload_error(
                        413, "too_large", f"Image has {width}x{height} pixels; the limit is {max_pixels} pixels."
                    )
                if img.format != PIL_FORMATS[media_type]:
                    raise _upload_error(422, "invalid_image", "The file content does not match its image signature.")
                img.verify()
    except ApiError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise _upload_error(413, "too_large", f"Image is too large: {exc}") from exc
    except Exception as exc:  # PIL raises many types for corrupt data
        raise _upload_error(422, "invalid_image", f"The image could not be decoded: {exc}") from exc
    return media_type


def discard(path: Path) -> None:
    """Best-effort removal of a partial upload."""
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)
