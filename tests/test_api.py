"""Tests for the FastAPI app (api/). Pipeline stages are injected fakes, so these tests do not
depend on vectorize/assemble/export existing. Stage fakes are module-level so they pickle
into the ProcessPoolExecutor test."""

from __future__ import annotations

import io
import json
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from api import main as api_main
from api.config import AppConfig
from api.main import SVG_CSP, create_app, detect_capabilities
from api.openapi import normalize
from api.store import JOB_FILE, JobGoneError, JobStore, StoredJob, canonical_job_id, utcnow
from api.uploads import check_image, safe_filename, sniff_media_type
from api.worker import JobManager, execute_job
from contracts.api import (
    API_PREFIX,
    JOB_TTL_SECONDS,
    MAX_UPLOAD_BYTES,
    ErrorDetail,
    FileKind,
    JobResponse,
    JobStatus,
    PipelineStage,
)
from contracts.fixtures import (
    make_image_class,
    make_line_map,
    make_palette,
    make_preprocess_result,
    make_vector_document,
)
from contracts.schemas import (
    ExportBundle,
    ImageInput,
    ImageMode,
    OutputFormat,
    QualityReport,
    Settings,
    SourceFormat,
    StageError,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "api" / "openapi.yaml"
SAMPLES = ROOT / "samples"
POLL_TIMEOUT_S = 30.0


# ============================================================================ stage fakes


def fake_load_image(path: Path) -> ImageInput:
    with Image.open(path) as img:
        width, height = img.size
        fmt = img.format
    return ImageInput(
        path=path,
        width=width,
        height=height,
        has_alpha=False,
        mode=ImageMode.RGB,
        source_format=SourceFormat.PNG if fmt == "PNG" else SourceFormat.JPEG,
        file_size_bytes=path.stat().st_size,
    )


def fake_preprocess(image: ImageInput, settings: Settings) -> Any:
    return make_preprocess_result(64)


def fake_classify(pre: Any, settings: Settings) -> Any:
    return make_image_class()


def fake_quantize(pre: Any, cls: Any, settings: Settings) -> Any:
    if settings.palette_override:
        base = make_palette(pre)
        return base
    return make_palette(pre)


def fake_extract_lines(pre: Any, cls: Any, settings: Settings) -> Any:
    return make_line_map(pre)


def fake_vectorize(pre: Any, cls: Any, palette: Any, lines: Any, settings: Settings) -> Any:
    return make_vector_document(64, settings)


def fake_assemble(doc: Any, settings: Settings) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {doc.width} {doc.height}"><rect/></svg>'


def fake_export(svg: str, doc: Any, settings: Settings, out_dir: Path) -> ExportBundle:
    svg_path = out_dir / "output.svg"
    svg_path.write_text(svg, encoding="utf-8")
    png = out_dir / "preview.png"
    Image.new("RGBA", (doc.width, doc.height), (255, 0, 0, 255)).save(png)
    ai = eps = None
    if OutputFormat.AI in settings.output_formats:
        ai = out_dir / "output.ai"
        ai.write_bytes(b"%PDF-1.5 fake ai")
    if OutputFormat.EPS in settings.output_formats:
        eps = out_dir / "output.eps"
        eps.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0 fake")
    return ExportBundle(svg_path=svg_path, preview_png_path=png, ai_path=ai, eps_path=eps, warnings=["fake export"])


def fake_export_outside(svg: str, doc: Any, settings: Settings, out_dir: Path) -> ExportBundle:
    elsewhere = out_dir.parent.parent / f"elsewhere-{uuid.uuid4().hex}"
    elsewhere.mkdir()
    return fake_export(svg, doc, settings, elsewhere)


def fake_evaluate(pre: Any, palette: Any, lines: Any, doc: Any, bundle: ExportBundle, elapsed: float) -> QualityReport:
    return QualityReport(
        ssim=0.99,
        mean_delta_e=0.1,
        max_delta_e=0.2,
        gap_ratio=0.0,
        node_count=9,
        file_size_bytes=bundle.svg_path.stat().st_size,
        processing_time_s=elapsed,
        checks=[],
    )


def failing_vectorize(*_: Any) -> Any:
    raise StageError("vectorize", "no paths could be traced")


GATE = threading.Event()
"""Blocks gated_vectorize until set (thread executor only)."""


def gated_vectorize(pre: Any, cls: Any, palette: Any, lines: Any, settings: Settings) -> Any:
    GATE.wait(POLL_TIMEOUT_S)
    return fake_vectorize(pre, cls, palette, lines, settings)


FAKES: dict[str, Any] = {
    "load_image": fake_load_image,
    "preprocess": fake_preprocess,
    "classify": fake_classify,
    "quantize": fake_quantize,
    "extract_lines": fake_extract_lines,
    "vectorize": fake_vectorize,
    "assemble_svg": fake_assemble,
    "export": fake_export,
    "evaluate": fake_evaluate,
}


# ============================================================================ helpers


def png_bytes(size: tuple[int, int] = (32, 24), noise: bool = False) -> bytes:
    if noise:
        arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        img = Image.fromarray(arr)
    else:
        img = Image.new("RGB", size, (200, 30, 30))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def jpeg_bytes(size: tuple[int, int] = (40, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(buf, "JPEG")
    return buf.getvalue()


def make_config(tmp_path: Path, **kw: Any) -> AppConfig:
    base: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "executor": "thread",
        "max_workers": 2,
        "cleanup_interval_seconds": 3600.0,
    }
    base.update(kw)
    return AppConfig(**base)


@pytest.fixture
def app_factory(tmp_path: Path) -> Iterator[Any]:
    clients: list[TestClient] = []

    def build(stages: dict[str, Any] | None = None, **config_kw: Any) -> TestClient:
        app = create_app(make_config(tmp_path, **config_kw), stages=stages if stages is not None else FAKES)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield build
    GATE.set()
    for client in clients:
        client.__exit__(None, None, None)
    GATE.clear()


@pytest.fixture
def client(app_factory: Any) -> TestClient:
    return app_factory()


def convert(
    client: TestClient,
    content: bytes | None = None,
    filename: str = "logo.png",
    settings: Any = None,
    **kw: Any,
) -> Any:
    files: dict[str, Any] = {"file": (filename, content if content is not None else png_bytes(), "image/png")}
    data = {"settings": settings if isinstance(settings, str) else json.dumps(settings)} if settings is not None else {}
    return client.post(f"{API_PREFIX}/convert", files=files, data=data, **kw)


def wait_for(client: TestClient, job_id: str, statuses: set[str] | None = None) -> dict[str, Any]:
    statuses = statuses or {"succeeded", "failed"}
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        body = client.get(f"{API_PREFIX}/jobs/{job_id}").json()
        if body.get("status") in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not reach {statuses}: {body}")


def wait_idle(client: TestClient) -> None:
    manager: JobManager = client.app.state.manager  # type: ignore[attr-defined]
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while manager.active() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert manager.active() == 0


def as_job(body: dict[str, Any]) -> JobResponse:
    """Validate a served JobResponse. QualityReport.passed is a computed (output-only) field that
    the contract model forbids as input, so it is checked and dropped first."""
    body = json.loads(json.dumps(body))
    quality = (body.get("result") or {}).get("quality")
    if quality is not None:
        assert quality.pop("passed") is all(c["passed"] for c in quality["checks"])
    return JobResponse.model_validate(body)


def assert_error(resp: Any, status: int, code: str | None = None) -> dict[str, Any]:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}, body
    detail = ErrorDetail.model_validate(body["error"])
    if code is not None:
        assert detail.code == code, body
    return body["error"]


# ============================================================================ OpenAPI diff


def _diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for key in sorted(set(a) | set(b), key=str):
            if key not in a:
                out.append(f"missing in app: {path}/{key}")
            elif key not in b:
                out.append(f"extra in app: {path}/{key}")
            else:
                out.extend(_diff(a[key], b[key], f"{path}/{key}"))
        return out
    return [] if a == b else [f"differs at {path}: app={a!r} spec={b!r}"]


def test_openapi_matches_committed_spec(client: TestClient) -> None:
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    app_spec = client.app.openapi()  # type: ignore[attr-defined]
    assert set(app_spec["paths"]) == set(spec["paths"])
    for path, item in spec["paths"].items():
        ops = {k for k in item if k != "parameters"}
        assert {k for k in app_spec["paths"][path] if k != "parameters"} == ops, path
        for method in ops:
            assert _diff(app_spec["paths"][path][method], item[method], f"{path} {method}") == []
        assert _diff(app_spec["paths"][path].get("parameters"), item.get("parameters"), path) == []
    assert set(app_spec["components"]["schemas"]) == set(spec["components"]["schemas"])
    assert _diff(app_spec["components"], spec["components"], "components") == []
    assert _diff(app_spec, spec) == []
    served = client.get(f"{API_PREFIX}/openapi.json").json()
    assert served == app_spec


def test_normalize_keeps_operation_specific_parameters() -> None:
    shared = {"name": "job_id", "in": "path", "required": True, "schema": {"type": "string", "title": "Job Id"}}
    extra = {"name": "q", "in": "query", "required": False, "schema": {"type": "string"}}
    spec = {"paths": {"/x": {"get": {"parameters": [shared, extra]}, "delete": {"parameters": [shared]}}}}
    out = normalize(spec)
    assert out["paths"]["/x"]["parameters"] == [{**shared, "schema": {"type": "string"}}]
    assert out["paths"]["/x"]["get"]["parameters"] == [extra]
    assert "parameters" not in out["paths"]["/x"]["delete"]


# ============================================================================ health / config


def test_health(client: TestClient) -> None:
    resp = client.get(f"{API_PREFIX}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["schema_version"] == "1.0.0" and body["version"]
    assert set(body["capabilities"]) == {"inkscape", "potrace", "vtracer", "cairosvg", "ghostscript"}
    assert all(isinstance(v, bool) for v in body["capabilities"].values())


def test_capabilities_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    detect_capabilities.cache_clear()
    monkeypatch.setattr(api_main.shutil, "which", lambda name: f"/usr/bin/{name}")
    try:
        caps = detect_capabilities()
        assert caps["inkscape"] and caps["potrace"] and caps["ghostscript"]
    finally:
        detect_capabilities.cache_clear()
    assert api_main._importable("json") is True
    assert api_main._importable("vf_module_that_does_not_exist") is False


def test_config(app_factory: Any) -> None:
    client = app_factory(max_upload_bytes=12345, job_ttl_seconds=99)
    body = client.get(f"{API_PREFIX}/config").json()
    assert Settings.model_validate(body["defaults"]) == Settings()
    assert body["limits"] == {
        "max_upload_bytes": 12345,
        "job_ttl_seconds": 99,
        "max_pixels": 36_000_000,
        "accepted_media_types": ["image/png", "image/jpeg"],
    }


# ============================================================================ convert -> poll -> download


def test_convert_poll_download(client: TestClient) -> None:
    resp = convert(client, png_bytes((32, 24)), filename="C:\\fake\\dir\\My Logo.png")
    assert resp.status_code == 202, resp.text
    job = as_job(resp.json())
    assert job.status == JobStatus.QUEUED and job.stage == PipelineStage.UPLOAD and job.progress == 0
    assert job.filename == "My Logo.png"
    assert job.settings == Settings()
    assert job.expires_at > job.created_at
    assert canonical_job_id(job.job_id)

    done = as_job(wait_for(client, job.job_id))
    assert done.status == JobStatus.SUCCEEDED, done.error
    assert done.stage == PipelineStage.DONE and done.progress == 1.0
    assert done.expires_at >= done.updated_at + timedelta(seconds=JOB_TTL_SECONDS) - timedelta(seconds=1)
    result = done.result
    assert result is not None
    assert (result.width, result.height) == (32, 24)
    assert result.layer_count == 3
    assert result.palette_hex == ["#ffffff", "#dc322f", "#000000"]
    assert result.warnings == ["fake export"]
    assert result.quality.ssim == 0.99
    kinds = [f.kind for f in result.files]
    assert kinds == [FileKind.ORIGINAL, FileKind.SVG, FileKind.PNG, FileKind.AI, FileKind.EPS]

    expected = {
        "original": "image/png",
        "svg": "image/svg+xml",
        "png": "image/png",
        "ai": "application/illustrator",
        "eps": "application/postscript",
    }
    for f in result.files:
        assert f.url == f"{API_PREFIX}/jobs/{job.job_id}/files/{f.kind.value}"
        r = client.get(f.url)
        assert r.status_code == 200, (f.kind, r.text)
        assert r.headers["content-type"].split(";")[0] == expected[f.kind.value] == f.media_type
        assert int(r.headers["content-length"]) == f.size_bytes == len(r.content)
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["content-disposition"].startswith("inline")
        if f.kind == FileKind.SVG:
            assert r.headers["content-security-policy"] == SVG_CSP
            assert r.text.startswith("<svg")
        else:
            assert "content-security-policy" not in r.headers

    r = client.get(f"{API_PREFIX}/jobs/{job.job_id}/files/svg", params={"download": "true"})
    assert r.headers["content-disposition"] == "attachment; filename*=utf-8''My%20Logo.svg"  # RFC 5987
    r = client.get(f"{API_PREFIX}/jobs/{job.job_id}/files/png", params={"download": True})
    assert "My%20Logo_preview.png" in r.headers["content-disposition"]
    r = client.get(f"{API_PREFIX}/jobs/{job.job_id}/files/original", params={"download": True})
    assert r.content == png_bytes((32, 24))
    assert "My%20Logo.png" in r.headers["content-disposition"]


def test_convert_jpeg_with_misleading_name_and_settings(client: TestClient) -> None:
    settings = {"output_formats": ["png"], "max_colors": 4, "detail_level": "high"}
    resp = convert(client, jpeg_bytes(), filename="photo.png", settings=settings)
    assert resp.status_code == 202, resp.text
    body = wait_for(client, resp.json()["job_id"])
    assert body["status"] == "succeeded"
    assert body["settings"]["output_formats"] == ["svg", "png"]
    files = {f["kind"]: f for f in body["result"]["files"]}
    assert set(files) == {"original", "svg", "png"}
    assert files["original"]["media_type"] == "image/jpeg"
    orig = client.get(files["original"]["url"], params={"download": 1})
    assert orig.headers["content-type"] == "image/jpeg"
    assert 'filename="photo.jpg"' in orig.headers["content-disposition"]
    assert_error(client.get(f"{API_PREFIX}/jobs/{body['job_id']}/files/eps"), 404, "not_found")
    assert_error(client.get(f"{API_PREFIX}/jobs/{body['job_id']}/files/ai"), 404, "not_found")


def test_settings_as_file_part_and_empty_settings(client: TestClient) -> None:
    files = {
        "file": ("a.png", png_bytes(), "application/octet-stream"),
        "settings": ("s.json", json.dumps({"smoothing": 10}).encode(), "application/json"),
    }
    resp = client.post(f"{API_PREFIX}/convert", files=files)
    assert resp.status_code == 202, resp.text
    assert resp.json()["settings"]["smoothing"] == 10
    resp = convert(client, settings="   ")
    assert resp.status_code == 202 and resp.json()["settings"] == Settings().model_dump(mode="json")
    wait_idle(client)


# ============================================================================ upload errors


def _jobs_on_disk(client: TestClient) -> list[Path]:
    jobs_dir: Path = client.app.state.config.jobs_dir  # type: ignore[attr-defined]
    return list(jobs_dir.iterdir()) if jobs_dir.exists() else []


def test_413_streaming_limit(app_factory: Any) -> None:
    client = app_factory(max_upload_bytes=1000)
    content = png_bytes((64, 64), noise=True)
    assert len(content) > 1000
    err = assert_error(convert(client, content), 413, "too_large")
    assert err["stage"] == "upload"
    assert _jobs_on_disk(client) == []
    assert convert(client, png_bytes((8, 8))).status_code == 202


def test_413_content_length_precheck(app_factory: Any) -> None:
    client = app_factory(max_upload_bytes=10)
    body = b"\x89PNG\r\n\x1a\n" + b"\0" * (1100 * 1024)
    assert_error(convert(client, body), 413, "too_large")
    assert _jobs_on_disk(client) == []


def test_413_default_limit_is_contract_value(client: TestClient) -> None:
    assert client.app.state.config.max_upload_bytes == MAX_UPLOAD_BYTES  # type: ignore[attr-defined]


def test_413_too_many_pixels(app_factory: Any) -> None:
    client = app_factory(max_pixels=100)
    assert_error(convert(client, png_bytes((20, 20))), 413, "too_large")


@pytest.mark.parametrize(
    "content",
    [b"GIF89a" + b"\0" * 64, b"%PDF-1.4 not an image", b"<svg xmlns='http://www.w3.org/2000/svg'/>"],
)
def test_415_by_magic_bytes_not_extension(client: TestClient, content: bytes) -> None:
    err = assert_error(convert(client, content, filename="image.png"), 415, "unsupported_media_type")
    assert err["stage"] == "upload"
    assert _jobs_on_disk(client) == []


def test_415_non_multipart_body(client: TestClient) -> None:
    assert_error(client.post(f"{API_PREFIX}/convert", json={"file": "x"}), 415, "unsupported_media_type")


@pytest.mark.parametrize(
    ("settings", "code"),
    [
        ("{not json", "invalid_settings"),
        ({"smoothing": 500}, "invalid_settings"),
        ({"unknown_field": 1}, "invalid_settings"),
        ({"palette_override": ["red"]}, "invalid_settings"),
    ],
)
def test_422_invalid_settings(client: TestClient, settings: Any, code: str) -> None:
    err = assert_error(convert(client, settings=settings), 422, code)
    assert err["message"].startswith("Invalid settings")
    assert _jobs_on_disk(client) == []


@pytest.mark.parametrize(
    "content",
    [
        b"\x89PNG\r\n\x1a\n" + b"garbage" * 20,
        b"\xff\xd8\xff" + b"garbage" * 20,
        b"",
        png_bytes((64, 64), noise=True)[:120],
    ],
)
def test_422_undecodable_image(client: TestClient, content: bytes) -> None:
    assert_error(convert(client, content), 422, "invalid_image")
    assert _jobs_on_disk(client) == []


def test_422_malformed_multipart(client: TestClient) -> None:
    url = f"{API_PREFIX}/convert"
    assert_error(client.post(url, files={"settings": (None, "{}")}), 422, "invalid_request")  # no file part
    no_boundary = {"content-type": "multipart/form-data"}
    assert_error(client.post(url, content=b"x", headers=no_boundary), 422, "invalid_request")
    two_files = [("file", ("a.png", png_bytes(), "image/png")), ("file", ("b.png", png_bytes(), "image/png"))]
    assert_error(client.post(url, files=two_files), 422, "invalid_request")
    many = {f"f{i}": "x" for i in range(20)}
    assert_error(client.post(url, files={"file": ("a.png", png_bytes())}, data=many), 422, "invalid_request")
    huge = {"settings": "x" * (70 * 1024)}
    assert_error(client.post(url, files={"file": ("a.png", png_bytes())}, data=huge), 422, "invalid_request")
    bad_headers = {"content-type": "multipart/form-data; boundary=zzz"}
    body = b"--zzz\r\nContent-Type: text/plain\r\n\r\nhello\r\n--zzz--\r\n"
    assert_error(client.post(url, content=body, headers=bad_headers), 422, "invalid_request")
    broken = b'--zzz\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n\r\n\x89PNG\r\n--zzz\x00\x00'
    assert client.post(url, content=broken, headers=bad_headers).status_code in (415, 422)
    assert _jobs_on_disk(client) == []


# ============================================================================ jobs


def test_404_unknown_and_invalid_job_ids(client: TestClient) -> None:
    unknown = str(uuid.uuid4())
    for job_id in (unknown, "not-a-uuid", unknown.upper(), "..%2F..%2Fetc"):
        assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}"), 404, "not_found")
        assert_error(client.delete(f"{API_PREFIX}/jobs/{job_id}"), 404, "not_found")
        assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}/files/svg"), 404, "not_found")
        assert_error(client.post(f"{API_PREFIX}/jobs/{job_id}/rerun", json={"settings": {}}), 404, "not_found")


def test_404_unknown_route_and_405(client: TestClient) -> None:
    assert_error(client.get(f"{API_PREFIX}/nope"), 404, "not_found")
    assert_error(client.put(f"{API_PREFIX}/convert"), 405, "method_not_allowed")


def test_file_kind_whitelist(client: TestClient) -> None:
    job_id = convert(client).json()["job_id"]
    wait_for(client, job_id)
    for kind in ("exe", "job.json", "upload.png", "SVG", "%2E%2E"):
        assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}/files/{kind}"), 404, "not_found")


def test_running_job_progress_and_files_not_ready(app_factory: Any) -> None:
    GATE.clear()
    client = app_factory(stages={**FAKES, "vectorize": gated_vectorize})
    job_id = convert(client).json()["job_id"]
    body = wait_for(client, job_id, {"running"})
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while body["stage"] != "vectorize" and time.monotonic() < deadline:
        time.sleep(0.02)
        body = client.get(f"{API_PREFIX}/jobs/{job_id}").json()
    assert body["stage"] == "vectorize" and 0 < body["progress"] < 1
    assert body["result"] is None and body["error"] is None
    assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}/files/svg"), 404, "not_found")
    assert client.get(f"{API_PREFIX}/jobs/{job_id}/files/original").status_code == 200
    GATE.set()
    assert wait_for(client, job_id)["status"] == "succeeded"


def test_failed_job_reports_stage_error(app_factory: Any) -> None:
    client = app_factory(stages={**FAKES, "vectorize": failing_vectorize})
    job_id = convert(client).json()["job_id"]
    body = wait_for(client, job_id)
    assert body["status"] == "failed"
    assert body["error"]["code"] == "stage_failed"
    assert body["error"]["stage"] == "vectorize"
    assert "no paths could be traced" in body["error"]["message"]
    assert body["stage"] == "vectorize" and body["result"] is None
    assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}/files/svg"), 404, "not_found")


def test_export_outside_job_dir_fails(app_factory: Any) -> None:
    client = app_factory(stages={**FAKES, "export": fake_export_outside})
    body = wait_for(client, convert(client).json()["job_id"])
    assert body["status"] == "failed" and body["error"]["stage"] == "export"


def test_delete_job(client: TestClient) -> None:
    job_id = convert(client).json()["job_id"]
    wait_for(client, job_id)
    job_dir = client.app.state.store.job_dir(job_id)  # type: ignore[attr-defined]
    assert job_dir.is_dir()
    resp = client.delete(f"{API_PREFIX}/jobs/{job_id}")
    assert resp.status_code == 204 and resp.content == b""
    assert not job_dir.exists()
    assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}"), 404, "not_found")
    assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}/files/original"), 404, "not_found")
    assert_error(client.delete(f"{API_PREFIX}/jobs/{job_id}"), 404, "not_found")


def test_delete_while_running_stops_worker(app_factory: Any) -> None:
    GATE.clear()
    client = app_factory(stages={**FAKES, "vectorize": gated_vectorize})
    job_id = convert(client).json()["job_id"]
    wait_for(client, job_id, {"running"})
    assert client.delete(f"{API_PREFIX}/jobs/{job_id}").status_code == 204
    GATE.set()
    wait_idle(client)
    assert not client.app.state.store.job_dir(job_id).exists()  # type: ignore[attr-defined]
    assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}"), 404)


# ============================================================================ rerun


def test_rerun_with_new_settings(client: TestClient) -> None:
    source = convert(client, filename="logo.png").json()
    wait_for(client, source["job_id"])
    new_settings = {"palette_override": ["#ff0000", "#00ff00", "#ff0000"], "output_formats": ["svg"]}
    resp = client.post(f"{API_PREFIX}/jobs/{source['job_id']}/rerun", json={"settings": new_settings})
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job["job_id"] != source["job_id"]
    assert job["source_job_id"] == source["job_id"]
    assert job["filename"] == "logo.png"
    assert job["settings"]["palette_override"] == ["#ff0000", "#00ff00"]
    done = wait_for(client, job["job_id"])
    assert done["status"] == "succeeded"
    assert {f["kind"] for f in done["result"]["files"]} == {"original", "svg"}
    # the re-run owns a copy of the upload: deleting the source does not break it
    assert client.delete(f"{API_PREFIX}/jobs/{source['job_id']}").status_code == 204
    assert client.get(f"{API_PREFIX}/jobs/{job['job_id']}/files/original").content == png_bytes()


def test_rerun_validation(client: TestClient) -> None:
    job_id = convert(client).json()["job_id"]
    url = f"{API_PREFIX}/jobs/{job_id}/rerun"
    assert_error(client.post(url, json={"settings": {"smoothing": -1}}), 422, "invalid_settings")
    assert_error(client.post(url, json={"settings": {"mode": "photo"}}), 422, "invalid_settings")
    assert_error(client.post(url, json={}), 422, "invalid_request")
    assert_error(client.post(url, content=b"not json", headers={"content-type": "application/json"}), 422)
    wait_idle(client)


def test_rerun_source_vanishes_during_copy(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    job_id = convert(client).json()["job_id"]
    wait_for(client, job_id)

    def gone(*_: Any) -> None:
        raise FileNotFoundError("deleted concurrently")

    monkeypatch.setattr(api_main.shutil, "copyfile", gone)
    before = len(_jobs_on_disk(client))
    assert_error(client.post(f"{API_PREFIX}/jobs/{job_id}/rerun", json={"settings": {}}), 404, "not_found")
    assert len(_jobs_on_disk(client)) == before


# ============================================================================ expiry / cleanup


def test_sweeper_deletes_expired_jobs(client: TestClient) -> None:
    store: JobStore = client.app.state.store  # type: ignore[attr-defined]
    job_id = convert(client).json()["job_id"]
    body = as_job(wait_for(client, job_id))
    assert store.sweep(body.expires_at - timedelta(seconds=1)) == []
    assert store.sweep(body.expires_at + timedelta(seconds=1)) == [job_id]
    assert not store.job_dir(job_id).exists()
    assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}"), 404)


def test_periodic_cleanup_task_removes_expired_job(app_factory: Any) -> None:
    client = app_factory(job_ttl_seconds=1, cleanup_interval_seconds=0.05)
    job_id = convert(client).json()["job_id"]
    body = wait_for(client, job_id)
    assert body["status"] == "succeeded"
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while client.get(f"{API_PREFIX}/jobs/{job_id}").status_code == 200 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert_error(client.get(f"{API_PREFIX}/jobs/{job_id}"), 404)
    assert not client.app.state.store.job_dir(job_id).exists()  # type: ignore[attr-defined]


def test_cleanup_loop_survives_errors(app_factory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    client = app_factory(cleanup_interval_seconds=0.02)
    store: JobStore = client.app.state.store  # type: ignore[attr-defined]
    calls: list[int] = []

    def broken_sweep(now: Any = None) -> list[str]:
        calls.append(1)
        raise OSError("disk hiccup")

    monkeypatch.setattr(store, "sweep", broken_sweep)
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while len(calls) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert len(calls) >= 2
    assert client.get(f"{API_PREFIX}/health").status_code == 200


def _stored(store: JobStore, status: JobStatus = JobStatus.QUEUED) -> StoredJob:
    job_id, _ = store.create_dir()
    now = utcnow()
    job = JobResponse(
        job_id=job_id,
        status=status,
        stage=PipelineStage.QUANTIZE,
        progress=0.2,
        created_at=now,
        updated_at=now,
        expires_at=now + store.ttl,
        filename="x.png",
        settings=Settings(),
    )
    stored = StoredJob(job=job, upload_name="upload.png", upload_media_type="image/png")
    store.save(stored, create=True)
    return stored


def test_startup_marks_interrupted_jobs_failed(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = JobStore(config.jobs_dir, config.job_ttl_seconds)
    running = _stored(store, JobStatus.RUNNING)
    done = _stored(store, JobStatus.SUCCEEDED)
    with TestClient(create_app(config, stages=FAKES)) as client:
        body = client.get(f"{API_PREFIX}/jobs/{running.job.job_id}").json()
        assert body["status"] == "failed" and body["error"]["code"] == "interrupted"
        assert body["error"]["stage"] == "quantize"
        assert client.get(f"{API_PREFIX}/jobs/{done.job.job_id}").json()["status"] == "succeeded"


# ============================================================================ store


def test_store_edge_cases(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "missing" / "jobs", 60)
    assert list(store.ids()) == []
    assert store.load("nope") is None
    assert store.delete("nope") is False
    assert store.delete(str(uuid.uuid4())) is False
    stored = _stored(store)
    job_id = stored.job.job_id
    (store.job_dir(job_id) / JOB_FILE).write_text("{corrupt", encoding="utf-8")
    assert store.load(job_id) is None
    assert store.sweep() == []  # orphan/corrupt dirs are only swept after the TTL
    assert store.sweep(utcnow() + timedelta(seconds=61)) == [job_id]
    (store.jobs_dir / "not-a-job").mkdir()
    assert list(store.ids()) == []


def test_store_save_after_delete_raises_job_gone(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs", 60)
    stored = _stored(store)
    assert store.delete(stored.job.job_id) is True
    with pytest.raises(JobGoneError):
        store.save(stored)
    with pytest.raises(JobGoneError):
        store.save(stored, create=True)
    assert canonical_job_id(str(uuid.uuid4()))
    assert canonical_job_id(None) is None  # type: ignore[arg-type]


# ============================================================================ worker / manager


def test_execute_job_edge_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    store = JobStore(data_dir / "jobs", 60)
    assert execute_job(str(data_dir), str(uuid.uuid4()), 60, FAKES) is None
    done = _stored(store, JobStatus.SUCCEEDED)
    assert execute_job(str(data_dir), done.job.job_id, 60, FAKES) == JobStatus.SUCCEEDED

    queued = _stored(store)
    (store.job_dir(queued.job.job_id) / "upload.png").write_bytes(png_bytes())

    def explode(*_: Any) -> Any:
        raise ValueError("result assembly bug")

    monkeypatch.setattr("api.worker.build_result", explode)
    assert execute_job(str(data_dir), queued.job.job_id, 60, FAKES) == JobStatus.FAILED
    failed = store.load(queued.job.job_id)
    assert failed is not None and failed.job.error is not None
    assert failed.job.error.code == "internal_error" and "result assembly bug" in failed.job.error.message


def _manager(tmp_path: Path) -> tuple[JobManager, JobStore]:
    config = make_config(tmp_path)
    store = JobStore(config.jobs_dir, config.job_ttl_seconds)
    return JobManager(config, store, FAKES), store


def test_manager_records_worker_crash(tmp_path: Path) -> None:
    manager, store = _manager(tmp_path)
    stored = _stored(store, JobStatus.RUNNING)
    future: Future[Any] = Future()
    future.set_exception(BrokenProcessPool("killed by OOM"))
    manager.start()
    manager._on_done(stored.job.job_id, future)
    assert manager._executor is None
    reloaded = store.load(stored.job.job_id)
    assert reloaded is not None and reloaded.job.status == JobStatus.FAILED
    assert reloaded.job.error is not None and reloaded.job.error.code == "worker_crashed"

    ok: Future[Any] = Future()
    ok.set_result(JobStatus.SUCCEEDED)
    manager._on_done(stored.job.job_id, ok)
    cancelled: Future[Any] = Future()
    cancelled.cancel()
    manager._on_done(stored.job.job_id, cancelled)
    crashed_again: Future[Any] = Future()
    crashed_again.set_exception(RuntimeError("x"))
    manager._on_done(stored.job.job_id, crashed_again)  # already terminal: unchanged
    manager._on_done(str(uuid.uuid4()), crashed_again)  # unknown job: ignored
    gone = _stored(store, JobStatus.RUNNING)
    store.delete(gone.job.job_id)
    manager._on_done(gone.job.job_id, crashed_again)
    manager.shutdown()


class _BrokenOnce:
    def __init__(self) -> None:
        self.shut = False

    def submit(self, *_: Any) -> Any:
        raise BrokenProcessPool("pool died")

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self.shut = True


def test_manager_recreates_broken_pool(tmp_path: Path) -> None:
    manager, store = _manager(tmp_path)
    stored = _stored(store)
    (store.job_dir(stored.job.job_id) / "upload.png").write_bytes(png_bytes())
    broken = _BrokenOnce()
    manager._executor = broken  # type: ignore[assignment]
    manager.submit(stored.job.job_id)
    assert broken.shut
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while manager.active() and time.monotonic() < deadline:
        time.sleep(0.02)
    manager.shutdown()
    reloaded = store.load(stored.job.job_id)
    assert reloaded is not None and reloaded.job.status == JobStatus.SUCCEEDED


def test_dispatch_failure_marks_job_failed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    manager: JobManager = client.app.state.manager  # type: ignore[attr-defined]

    def refuse(job_id: str) -> None:
        raise RuntimeError("executor shut down")

    monkeypatch.setattr(manager, "submit", refuse)
    job_id = convert(client).json()["job_id"]
    body = client.get(f"{API_PREFIX}/jobs/{job_id}").json()
    assert body["status"] == "failed" and body["error"]["code"] == "internal_error"
    manager.mark_failed_now(str(uuid.uuid4()), ErrorDetail(code="x", message="y"))


def test_manager_cancel_queued_job(tmp_path: Path) -> None:
    config = make_config(tmp_path, max_workers=1)
    store = JobStore(config.jobs_dir, config.job_ttl_seconds)
    GATE.clear()
    manager = JobManager(config, store, {**FAKES, "vectorize": gated_vectorize})
    first, second = _stored(store), _stored(store)
    for s in (first, second):
        (store.job_dir(s.job.job_id) / "upload.png").write_bytes(png_bytes())
        manager.submit(s.job.job_id)
    manager.cancel(second.job.job_id)
    manager.cancel(str(uuid.uuid4()))
    GATE.set()
    manager.shutdown()
    GATE.clear()
    reloaded = store.load(second.job.job_id)
    assert reloaded is not None and reloaded.job.status == JobStatus.QUEUED


def test_process_pool_executor_end_to_end(app_factory: Any) -> None:
    """The production executor: spawn-started worker processes running the (picklable) fakes."""
    client = app_factory(executor="process", max_workers=1)
    job_id = convert(client).json()["job_id"]
    body = wait_for(client, job_id)
    assert body["status"] == "succeeded", body["error"]
    assert client.get(f"{API_PREFIX}/jobs/{job_id}/files/svg").status_code == 200


# ============================================================================ config / uploads units


def test_config_from_env(tmp_path: Path) -> None:
    cfg = AppConfig.from_env(
        {
            "DATA_DIR": str(tmp_path),
            "MAX_WORKERS": "3",
            "WORKER_EXECUTOR": "Thread",
            "JOB_TTL_SECONDS": "120",
            "CLEANUP_INTERVAL_SECONDS": "5",
            "MAX_UPLOAD_BYTES": "",
        }
    )
    assert cfg.data_dir == tmp_path.resolve() and cfg.jobs_dir == tmp_path.resolve() / "jobs"
    assert (cfg.max_workers, cfg.executor, cfg.job_ttl_seconds) == (3, "thread", 120)
    assert cfg.cleanup_interval_seconds == 5.0 and cfg.max_upload_bytes == MAX_UPLOAD_BYTES
    assert AppConfig.from_env({}).executor == "process"
    with pytest.raises(ValueError, match="MAX_WORKERS"):
        AppConfig.from_env({"MAX_WORKERS": "0"})
    with pytest.raises(ValueError, match="integer"):
        AppConfig.from_env({"JOB_TTL_SECONDS": "soon"})
    with pytest.raises(ValueError, match="WORKER_EXECUTOR"):
        AppConfig.from_env({"WORKER_EXECUTOR": "celery"})


def test_upload_helpers(tmp_path: Path) -> None:
    assert sniff_media_type(png_bytes()) == "image/png"
    assert sniff_media_type(jpeg_bytes()) == "image/jpeg"
    assert sniff_media_type(b"GIF89a") is None
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("a\x00b\nc.png") == "abc.png"
    assert safe_filename(None) == "upload" and safe_filename("C:\\x\\") == "upload"
    assert len(safe_filename("a" * 500)) == 200
    # PNG signature followed by a JPEG body: Pillow disagrees with the magic bytes
    mismatch = tmp_path / "m.bin"
    mismatch.write_bytes(b"\xff\xd8\xff" + png_bytes()[3:])
    with pytest.raises(Exception, match="decoded|signature"):
        check_image(mismatch, 10_000)


@pytest.mark.parametrize("sample", sorted(p.name for p in SAMPLES.glob("[0-9][0-9]_*.*") if p.suffix != ".json"))
def test_all_samples_upload_poll_download(client: TestClient, sample: str) -> None:
    content = (SAMPLES / sample).read_bytes()
    resp = convert(client, content, filename=sample)
    assert resp.status_code == 202, resp.text
    body = wait_for(client, resp.json()["job_id"])
    assert body["status"] == "succeeded"
    for f in body["result"]["files"]:
        assert client.get(f["url"]).status_code == 200
