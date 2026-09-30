"""OpenAPI document of the app, normalized to the conventions of ``api/openapi.yaml``.

FastAPI generates the document from the routes; :func:`normalize` then removes the
framework artefacts the committed spec does not use, so ``app.openapi()`` can be diffed
against the file (tests/test_api.py). The normalizations are purely representational:

* parameter schemas lose FastAPI's auto-generated ``title`` and the copy of the
  parameter ``description`` that FastAPI also writes into the schema;
* parameters shared by every operation of a path are hoisted to the path level;
* FastAPI's implicit ``422 HTTPValidationError`` response is dropped from operations that
  do not declare a 422 (their validation errors are rendered as 404 by api.errors), along
  with the then-unused ``HTTPValidationError`` / ``ValidationError`` schemas;
* ``default: null`` entries, which FastAPI drops when serializing the document with
  ``exclude_none=True``, are restored from Pydantic's own JSON schema of the same models.
"""

from __future__ import annotations

import copy
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaMode, models_json_schema

TITLE = "VectorForge API"
VERSION = "1.0.0"
DESCRIPTION = (
    "Raster (PNG/JPG) to vector (SVG/AI/EPS) conversion. Generated from contracts/api.py -- do not edit by hand."
)
HTTP_METHODS = ("get", "put", "post", "delete", "patch", "options", "head", "trace")
_FASTAPI_VALIDATION_SCHEMAS = ("HTTPValidationError", "ValidationError")
_HTTP_VALIDATION_REF = "#/components/schemas/HTTPValidationError"


def _is_implicit_422(response: dict[str, Any]) -> bool:
    content = response.get("content", {}).get("application/json", {})
    return bool(content.get("schema", {}).get("$ref") == _HTTP_VALIDATION_REF)


def normalize(spec: dict[str, Any]) -> dict[str, Any]:
    """Apply the normalizations listed in the module docstring (returns a new dict)."""
    spec = copy.deepcopy(spec)
    for path_item in spec.get("paths", {}).values():
        operations = [path_item[m] for m in HTTP_METHODS if m in path_item]
        for op in operations:
            for param in op.get("parameters", []):
                schema = param.get("schema", {})
                schema.pop("title", None)
                if "description" in param and schema.get("description") == param["description"]:
                    del schema["description"]
            responses = op.get("responses", {})
            if "422" in responses and _is_implicit_422(responses["422"]):
                del responses["422"]
        first = operations[0].get("parameters", []) if operations else []
        shared = [p for p in first if all(p in op.get("parameters", []) for op in operations)]
        if shared:
            path_item["parameters"] = shared
            for op in operations:
                rest = [p for p in op.get("parameters", []) if p not in shared]
                if rest:
                    op["parameters"] = rest
                else:
                    op.pop("parameters", None)
    schemas = spec.get("components", {}).get("schemas", {})
    text = repr(spec.get("paths", {}))
    for name in _FASTAPI_VALIDATION_SCHEMAS:
        if name in schemas and f"#/components/schemas/{name}" not in text:
            del schemas[name]
    return spec


def _route_models(app: FastAPI) -> list[tuple[type[BaseModel], JsonSchemaMode]]:
    """Pydantic models used by the routes, with the JSON-schema mode FastAPI uses for them."""
    found: dict[type[BaseModel], JsonSchemaMode] = {}

    def add(model: Any, mode: JsonSchemaMode) -> None:
        if isinstance(model, type) and issubclass(model, BaseModel):
            found.setdefault(model, mode)

    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        add(route.response_model, "serialization")
        for response in route.responses.values():
            add(response.get("model"), "serialization")
        if route.body_field is not None:
            add(route.body_field.field_info.annotation, "validation")
    return list(found.items())


def restore_null_defaults(spec: dict[str, Any], models: list[tuple[type[BaseModel], JsonSchemaMode]]) -> None:
    """Re-add property-level ``default: null`` that FastAPI's exclude_none serialization removed."""
    if not models:
        return
    _, defs = models_json_schema(models, ref_template="#/components/schemas/{model}")
    reference: dict[str, Any] = defs.get("$defs", {})
    for name, schema in spec.get("components", {}).get("schemas", {}).items():
        ref_props = reference.get(name, {}).get("properties", {})
        for prop_name, prop in schema.get("properties", {}).items():
            ref_prop = ref_props.get(prop_name, {})
            if "default" in ref_prop and ref_prop["default"] is None and "default" not in prop:
                prop["default"] = None


def build_openapi(app: FastAPI) -> dict[str, Any]:
    """Generate (once) and cache the normalized OpenAPI document of ``app``."""
    if app.openapi_schema is None:
        raw = get_openapi(
            title=TITLE,
            version=VERSION,
            openapi_version="3.1.0",
            description=DESCRIPTION,
            routes=app.routes,
        )
        spec = normalize(raw)
        restore_null_defaults(spec, _route_models(app))
        app.openapi_schema = spec
    return app.openapi_schema
