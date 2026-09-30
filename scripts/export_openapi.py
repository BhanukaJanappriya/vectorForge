"""Generate api/openapi.yaml from contracts.api models.

Usage: python scripts/export_openapi.py [--check]
--check exits non-zero if the committed file is out of date (used in CI/tests).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic.json_schema import models_json_schema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts.api import (  # noqa: E402
    API_PREFIX,
    MAX_UPLOAD_BYTES,
    ConfigResponse,
    ErrorDetail,
    FileKind,
    HealthResponse,
    JobResponse,
    RerunRequest,
)
from contracts.schemas import SCHEMA_VERSION, Settings  # noqa: E402

OUT = ROOT / "api" / "openapi.yaml"
REF = "#/components/schemas/{model}"


def _ref(name: str) -> dict[str, str]:
    return {"$ref": REF.format(model=name)}


def _json(name: str) -> dict[str, Any]:
    return {"application/json": {"schema": _ref(name)}}


def _error(description: str) -> dict[str, Any]:
    return {"description": description, "content": _json("ErrorResponse")}


def build_spec() -> dict[str, Any]:
    """Return the OpenAPI 3.1 document as a dict."""
    _, defs = models_json_schema(
        [
            (m, "serialization")
            for m in (JobResponse, ConfigResponse, HealthResponse, ErrorDetail, Settings, RerunRequest)
        ],
        ref_template=REF,
    )
    schemas: dict[str, Any] = defs["$defs"]
    schemas["ErrorResponse"] = {
        "type": "object",
        "required": ["error"],
        "properties": {"error": _ref("ErrorDetail")},
        "title": "ErrorResponse",
    }
    job_id = {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "format": "uuid"}}
    kinds = [k.value for k in FileKind]

    paths: dict[str, Any] = {
        f"{API_PREFIX}/health": {
            "get": {
                "operationId": "getHealth",
                "summary": "Liveness + available external tools",
                "responses": {"200": {"description": "OK", "content": _json("HealthResponse")}},
            }
        },
        f"{API_PREFIX}/config": {
            "get": {
                "operationId": "getConfig",
                "summary": "Default settings and upload limits (drives the UI form)",
                "responses": {"200": {"description": "OK", "content": _json("ConfigResponse")}},
            }
        },
        f"{API_PREFIX}/convert": {
            "post": {
                "operationId": "convert",
                "summary": "Upload an image and start a conversion job",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "required": ["file"],
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "format": "binary",
                                        "description": f"PNG or JPEG, <= {MAX_UPLOAD_BYTES} bytes",
                                    },
                                    "settings": {
                                        "type": "string",
                                        "contentMediaType": "application/json",
                                        "description": "JSON-encoded Settings; omitted fields use defaults",
                                    },
                                },
                            },
                            "encoding": {"settings": {"contentType": "application/json"}},
                        }
                    },
                },
                "responses": {
                    "202": {"description": "Job accepted", "content": _json("JobResponse")},
                    "413": _error("File too large"),
                    "415": _error("Unsupported media type"),
                    "422": _error("Invalid settings or undecodable image"),
                },
            }
        },
        f"{API_PREFIX}/jobs/{{job_id}}/rerun": {
            "parameters": [job_id],
            "post": {
                "operationId": "rerunJob",
                "summary": "Start a new job from the same upload with new settings (palette edit/merge)",
                "requestBody": {"required": True, "content": _json("RerunRequest")},
                "responses": {
                    "202": {"description": "New job accepted", "content": _json("JobResponse")},
                    "404": _error("Unknown or expired job"),
                    "422": _error("Invalid settings"),
                },
            },
        },
        f"{API_PREFIX}/jobs/{{job_id}}": {
            "parameters": [job_id],
            "get": {
                "operationId": "getJob",
                "summary": "Poll job status/result",
                "responses": {
                    "200": {"description": "OK", "content": _json("JobResponse")},
                    "404": _error("Unknown job"),
                },
            },
            "delete": {
                "operationId": "deleteJob",
                "summary": "Delete a job and its files",
                "responses": {"204": {"description": "Deleted"}, "404": _error("Unknown job")},
            },
        },
        f"{API_PREFIX}/jobs/{{job_id}}/files/{{kind}}": {
            "parameters": [
                job_id,
                {"name": "kind", "in": "path", "required": True, "schema": {"type": "string", "enum": kinds}},
                {
                    "name": "download",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "boolean", "default": False},
                    "description": "Send Content-Disposition: attachment",
                },
            ],
            "get": {
                "operationId": "getJobFile",
                "summary": "Download an input/output file",
                "responses": {
                    "200": {
                        "description": "File contents",
                        "content": {
                            "image/svg+xml": {"schema": {"type": "string"}},
                            "image/png": {"schema": {"type": "string", "format": "binary"}},
                            "image/jpeg": {"schema": {"type": "string", "format": "binary"}},
                            "application/postscript": {"schema": {"type": "string", "format": "binary"}},
                            "application/illustrator": {"schema": {"type": "string", "format": "binary"}},
                        },
                    },
                    "404": _error("Unknown job, or file not produced / not ready"),
                },
            },
        },
    }
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "VectorForge API",
            "version": SCHEMA_VERSION,
            "description": (
                "Raster (PNG/JPG) to vector (SVG/AI/EPS) conversion. "
                "Generated from contracts/api.py -- do not edit by hand."
            ),
        },
        "paths": paths,
        "components": {"schemas": dict(sorted(schemas.items()))},
    }


def render() -> str:
    return yaml.safe_dump(build_spec(), sort_keys=False, allow_unicode=True, width=100)


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT} is out of date; run python scripts/export_openapi.py", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
