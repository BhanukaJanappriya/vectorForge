"""Tests for pipeline.export (output.svg, preview.png, output.ai, output.eps).

Locally (no native Cairo, no Inkscape) the fallback routes run: resvg for preview.png, the direct
pikepdf writer for .ai and pycairo for .eps. The CairoSVG / Inkscape route logic is exercised with
fakes, and the real tools are tested by skip-guarded tests that run in the Docker image.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import fitz
import numpy as np
import pikepdf
import pytest
from lxml import etree
from PIL import Image

from contracts.fixtures import make_vector_document
from contracts.schemas import (
    ExportBundle,
    ExportToolUnavailableError,
    OutputFormat,
    Settings,
    StageError,
    VectorDocument,
    VectorLayer,
)
from pipeline import export as export_mod
from pipeline.assemble import INKSCAPE_NS, SVG_NS, assemble_svg
from pipeline.export import (
    PX_TO_PT,
    export,
    find_inkscape,
    parse_path,
    pdf_ocg_names,
    render_png,
    run_inkscape,
    write_cairo_eps,
    write_direct_pdf,
)
from tests.test_assemble import metadata, rect_path, render, rich_document, ssim, synthetic_09

ALL = Settings()


def _cairosvg_available() -> bool:
    try:
        import cairosvg  # noqa: F401
    except (ImportError, OSError):
        return False
    return True


def _no_cairosvg(svg: str, width: int, height: int) -> bytes:
    raise ExportToolUnavailableError("cairosvg unavailable (OSError)")


@pytest.fixture
def fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the local fallback routes, whatever is installed on the machine."""
    monkeypatch.setattr(export_mod, "find_inkscape", lambda: None)
    monkeypatch.setattr(export_mod, "_render_cairosvg", _no_cairosvg)


def run_export(doc: VectorDocument, out_dir: Path, settings: Settings = ALL) -> ExportBundle:
    return export(assemble_svg(doc, settings), doc, settings, out_dir)


def render_pdf(path: Path, width: int, height: int) -> np.ndarray:
    """Rasterize page 1 of a PDF with PyMuPDF at 96 dpi (px space) -> (H, W, 4)."""
    with fitz.open(path) as pdf:
        pix = pdf[0].get_pixmap(matrix=fitz.Matrix(1 / PX_TO_PT, 1 / PX_TO_PT), alpha=True)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    assert img.shape[:2] == (height, width)
    return img


def load_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGBA"))


def transparent_doc() -> VectorDocument:
    """No background layer: most of the canvas must stay transparent."""
    layer = VectorLayer(
        id="color_1_E53935",
        name="color_1_#E53935",
        role="fill",
        color_hex="#e53935",
        paths=[rect_path(20, 10, 80, 50, 1) + rect_path(40, 20, 60, 40, 1)],
        z_order=0,
    )
    return VectorDocument(width=121, height=67, layers=[layer], metadata=metadata())


# --------------------------------------------------------------------------------------
# Fallback routes (what runs locally)
# --------------------------------------------------------------------------------------


def test_fallback_export_writes_everything_and_reports_routes(tmp_path: Path, fallbacks: None) -> None:
    doc = make_vector_document()
    bundle = run_export(doc, tmp_path)
    assert bundle.svg_path == tmp_path / "output.svg"
    assert bundle.preview_png_path == tmp_path / "preview.png"
    assert bundle.ai_path == tmp_path / "output.ai"
    assert bundle.eps_path == tmp_path / "output.eps"
    assert bundle.warnings == [
        "cairosvg unavailable (OSError): preview.png rendered with resvg",
        "inkscape unavailable: .ai written by direct PDF writer (3 OCG layers)",
        "inkscape unavailable: .eps written by pycairo PostScript surface (EPS has no layers)",
    ]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["output.ai", "output.eps", "output.svg", "preview.png"]


def test_output_svg_is_the_given_svg(tmp_path: Path, fallbacks: None) -> None:
    doc = make_vector_document()
    svg = assemble_svg(doc, ALL)
    bundle = export(svg, doc, ALL, tmp_path / "nested" / "dir")
    assert bundle.svg_path.read_text(encoding="utf-8") == svg


@pytest.mark.parametrize(
    ("formats", "ai", "eps"),
    [
        ([OutputFormat.SVG], False, False),
        ([OutputFormat.PNG], False, False),
        ([OutputFormat.AI], True, False),
        ([OutputFormat.EPS, OutputFormat.PNG], False, True),
    ],
)
def test_only_requested_formats_plus_preview(
    tmp_path: Path, fallbacks: None, formats: list[OutputFormat], ai: bool, eps: bool
) -> None:
    settings = Settings(output_formats=formats)
    bundle = run_export(make_vector_document(settings=settings), tmp_path, settings)
    assert bundle.preview_png_path.is_file()
    assert bundle.svg_path.is_file()
    assert (bundle.ai_path is not None) == ai == (tmp_path / "output.ai").exists()
    assert (bundle.eps_path is not None) == eps == (tmp_path / "output.eps").exists()


@pytest.mark.parametrize("make_doc", [make_vector_document, rich_document, transparent_doc])
def test_preview_matches_input_size_and_render(tmp_path: Path, fallbacks: None, make_doc) -> None:  # type: ignore[no-untyped-def]
    doc = make_doc()
    bundle = run_export(doc, tmp_path)
    with Image.open(bundle.preview_png_path) as img:
        assert img.size == (doc.width, doc.height)
        assert img.mode == "RGBA"
    preview = load_png(bundle.preview_png_path)
    assert np.array_equal(preview, render(bundle.svg_path.read_text(encoding="utf-8"), doc.width, doc.height))


def test_preview_keeps_transparency(tmp_path: Path, fallbacks: None) -> None:
    bundle = run_export(transparent_doc(), tmp_path)
    alpha = load_png(bundle.preview_png_path)[..., 3]
    assert alpha[0, 0] == 0  # no background fill added
    assert alpha[30, 50] == 0  # evenodd hole
    assert alpha[15, 25] == 255


def test_preview_written_even_if_png_not_requested(tmp_path: Path, fallbacks: None) -> None:
    settings = Settings(output_formats=[OutputFormat.SVG])
    bundle = run_export(make_vector_document(), tmp_path, settings)
    assert bundle.preview_png_path.stat().st_size > 0


@pytest.mark.parametrize("make_doc", [make_vector_document, rich_document, transparent_doc])
def test_ai_has_one_ocg_per_layer_named_like_the_layer(tmp_path: Path, fallbacks: None, make_doc) -> None:  # type: ignore[no-untyped-def]
    doc = make_doc()
    bundle = run_export(doc, tmp_path)
    assert bundle.ai_path is not None
    assert pdf_ocg_names(bundle.ai_path) == [layer.name for layer in doc.layers]
    with pikepdf.open(bundle.ai_path) as pdf:
        assert pdf.pdf_version >= "1.5"
        d = pdf.Root.OCProperties.D
        assert [str(o.Name) for o in d.Order] == [layer.name for layer in reversed(doc.layers)]
        assert len(d.ON) == len(doc.layers)
        page = pdf.pages[0]
        assert [float(v) for v in page.MediaBox] == [0, 0, doc.width * PX_TO_PT, doc.height * PX_TO_PT]
        ops = [str(op) for _, op in pikepdf.parse_content_stream(page)]
        assert ops.count("BDC") == ops.count("EMC") == len(doc.layers)


def test_ai_starts_with_pdf_header(tmp_path: Path, fallbacks: None) -> None:
    bundle = run_export(make_vector_document(), tmp_path)
    assert bundle.ai_path is not None
    assert bundle.ai_path.read_bytes().startswith(b"%PDF-1.")


@pytest.mark.parametrize("make_doc", [make_vector_document, rich_document, transparent_doc])
def test_ai_rerenders_like_preview(tmp_path: Path, fallbacks: None, make_doc) -> None:  # type: ignore[no-untyped-def]
    doc = make_doc()
    bundle = run_export(doc, tmp_path)
    assert bundle.ai_path is not None
    score = ssim(render_pdf(bundle.ai_path, doc.width, doc.height), load_png(bundle.preview_png_path))
    assert score >= 0.98, score


def test_ai_rerenders_like_preview_on_09_like_document(tmp_path: Path, fallbacks: None) -> None:
    doc = synthetic_09(shapes=60)
    bundle = run_export(doc, tmp_path)
    assert bundle.ai_path is not None
    score = ssim(render_pdf(bundle.ai_path, doc.width, doc.height), load_png(bundle.preview_png_path))
    assert score >= 0.98, score


def test_translucent_layer_uses_group_opacity(tmp_path: Path) -> None:
    square = rect_path(10, 10, 60, 60, 0)
    overlapping = rect_path(40, 10, 90, 60, 0)
    layer = VectorLayer(
        id="a", name="a", role="fill", color_hex="#0000ff", paths=[square, overlapping], z_order=0, opacity=0.5
    )
    doc = VectorDocument(width=100, height=70, layers=[layer], metadata=metadata())
    write_direct_pdf(doc, tmp_path / "o.pdf")
    img = render_pdf(tmp_path / "o.pdf", 100, 70)
    # group opacity: the overlap of the two paths is NOT darker than the single-coverage parts
    assert abs(int(img[30, 50, 3]) - int(img[30, 20, 3])) <= 2
    assert 110 <= img[30, 20, 3] <= 145


def test_eps_is_valid_encapsulated_postscript(tmp_path: Path, fallbacks: None) -> None:
    doc = rich_document()
    bundle = run_export(doc, tmp_path)
    assert bundle.eps_path is not None
    data = bundle.eps_path.read_bytes()
    assert data.startswith(b"%!PS-Adobe-3.0 EPSF-3.0")
    assert b"%%BoundingBox:" in data and b"%%EOF" in data


def test_eps_renders_with_ghostscript(tmp_path: Path, fallbacks: None) -> None:
    gs = shutil.which("gs") or shutil.which("gswin64c")
    if gs is None:
        pytest.skip("ghostscript not installed here; runs in Docker")
    doc = make_vector_document()
    bundle = run_export(doc, tmp_path)
    out = tmp_path / "eps.png"
    subprocess.run(
        [
            gs,
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-dEPSCrop",
            "-sDEVICE=pngalpha",
            "-r96",
            f"-sOutputFile={out}",
            str(bundle.eps_path),
        ],
        check=True,
    )
    with Image.open(out) as img:
        assert abs(img.size[0] - doc.width) <= 1 and abs(img.size[1] - doc.height) <= 1


def test_eps_skipped_with_warning_when_pycairo_missing(
    tmp_path: Path, fallbacks: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "cairo", None)  # makes "import cairo" raise ImportError
    bundle = run_export(make_vector_document(), tmp_path)
    assert bundle.eps_path is None
    assert not (tmp_path / "output.eps").exists()
    assert "inkscape unavailable; pycairo unavailable: EPS skipped" in bundle.warnings


def test_write_cairo_eps_raises_without_pycairo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cairo", None)
    with pytest.raises(ExportToolUnavailableError):
        write_cairo_eps(make_vector_document(), tmp_path / "x.eps")


def test_translucent_layer_in_eps(tmp_path: Path) -> None:
    doc = rich_document()
    assert any(layer.opacity < 1 for layer in doc.layers)
    write_cairo_eps(doc, tmp_path / "x.eps")
    assert (tmp_path / "x.eps").stat().st_size > 0


def test_empty_document_exports(tmp_path: Path, fallbacks: None) -> None:
    doc = VectorDocument(width=30, height=20, layers=[], metadata=metadata())
    bundle = run_export(doc, tmp_path)
    assert bundle.ai_path is not None and pdf_ocg_names(bundle.ai_path) == []
    assert load_png(bundle.preview_png_path).shape == (20, 30, 4)


def test_ai_direct_writer_ocg_mismatch_raises(tmp_path: Path, fallbacks: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_mod, "pdf_ocg_names", lambda path: ["wrong"])
    with pytest.raises(StageError, match="direct PDF writer produced OCGs"):
        run_export(make_vector_document(), tmp_path)


def test_ai_write_failure_is_a_stage_error(tmp_path: Path, fallbacks: None, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(doc: VectorDocument, path: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(export_mod, "write_direct_pdf", boom)
    with pytest.raises(StageError, match="cannot write output.ai: disk full"):
        run_export(make_vector_document(), tmp_path)


def test_pdf_ocg_names_without_ocproperties(tmp_path: Path) -> None:
    pdf = pikepdf.new()
    pdf.add_blank_page()
    pdf.save(tmp_path / "plain.pdf")
    assert pdf_ocg_names(tmp_path / "plain.pdf") == []


# --------------------------------------------------------------------------------------
# preview renderer selection
# --------------------------------------------------------------------------------------


def test_render_png_prefers_cairosvg(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def svg2png(bytestring: bytes, output_width: int, output_height: int) -> bytes:
        calls.append((output_width, output_height))
        buf = io.BytesIO()
        Image.new("RGBA", (output_width, output_height), (1, 2, 3, 0)).save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setitem(sys.modules, "cairosvg", types.SimpleNamespace(svg2png=svg2png))
    png, route, warnings = render_png(assemble_svg(make_vector_document(), ALL), 64, 64)
    assert (route, warnings, calls) == ("cairosvg", [], [(64, 64)])
    with Image.open(io.BytesIO(png)) as image:
        assert (image.size, image.mode) == ((64, 64), "RGBA")


def test_render_png_converts_non_rgba_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def rgb(svg: str, width: int, height: int) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (width, height), (9, 8, 7)).save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(export_mod, "_render_cairosvg", rgb)
    png, _, _ = render_png("<svg/>", 5, 4)
    with Image.open(io.BytesIO(png)) as image:
        assert image.mode == "RGBA" and image.getpixel((0, 0)) == (9, 8, 7, 255)


def test_render_png_rejects_non_png(monkeypatch: pytest.MonkeyPatch) -> None:
    def jpeg(svg: str, width: int, height: int) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (width, height)).save(buf, format="JPEG")
        return buf.getvalue()

    monkeypatch.setattr(export_mod, "_render_cairosvg", jpeg)
    with pytest.raises(StageError, match="expected PNG"):
        render_png("<svg/>", 5, 4)


def test_render_png_falls_back_when_cairo_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def svg2png(**_: object) -> bytes:
        raise RuntimeError("cairo exploded")

    monkeypatch.setitem(sys.modules, "cairosvg", types.SimpleNamespace(svg2png=svg2png))
    _, route, warnings = render_png(assemble_svg(make_vector_document(), ALL), 64, 64)
    assert route == "resvg"
    assert warnings == ["cairosvg failed: cairo exploded: preview.png rendered with resvg"]


def test_render_png_no_renderer_is_a_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cairosvg", None)
    monkeypatch.setitem(sys.modules, "resvg_py", None)
    with pytest.raises(StageError, match="cannot render preview.png: cairosvg unavailable.*resvg_py unavailable"):
        render_png(assemble_svg(make_vector_document(), ALL), 64, 64)


def test_render_png_wrong_size_is_a_stage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def small(svg: str, width: int, height: int) -> bytes:
        buf = io.BytesIO()
        Image.new("RGBA", (width - 1, height), 0).save(buf, format="PNG")
        return buf.getvalue()

    monkeypatch.setattr(export_mod, "_render_cairosvg", small)
    with pytest.raises(StageError, match="expected"):
        render_png("<svg/>", 10, 10)


def test_cairosvg_route_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not _cairosvg_available():
        pytest.skip("CairoSVG / native Cairo unavailable here; runs in Docker")
    monkeypatch.setattr(export_mod, "find_inkscape", lambda: None)
    doc = rich_document()
    bundle = run_export(doc, tmp_path)
    assert not any("preview.png" in w for w in bundle.warnings)
    preview = load_png(bundle.preview_png_path)
    assert preview.shape == (doc.height, doc.width, 4)
    assert ssim(preview, render(bundle.svg_path.read_text(encoding="utf-8"), doc.width, doc.height)) >= 0.98


# --------------------------------------------------------------------------------------
# path parsing
# --------------------------------------------------------------------------------------


def test_parse_path_expands_implicit_commands() -> None:
    assert parse_path("M1 2 3 4L5 6 7 8C1 1 2 2 3 3Z M0 0") == [
        ("M", (1.0, 2.0)),
        ("L", (3.0, 4.0)),
        ("L", (5.0, 6.0)),
        ("L", (7.0, 8.0)),
        ("C", (1.0, 1.0, 2.0, 2.0, 3.0, 3.0)),
        ("Z", ()),
        ("M", (0.0, 0.0)),
    ]


def test_parse_path_scientific_and_signs() -> None:
    assert parse_path("M1e2-.5L-1.5+2") == [("M", (100.0, -0.5)), ("L", (-1.5, 2.0))]


@pytest.mark.parametrize("d", ["M1 2L3", "M1 2Z 5 5", "M1 2C1 2 3 4 5", "M1 L2 3"])
def test_parse_path_rejects_malformed(d: str) -> None:
    with pytest.raises(StageError):
        parse_path(d)


# --------------------------------------------------------------------------------------
# Inkscape route logic (fake CLI) and detection
# --------------------------------------------------------------------------------------

FAKE_INKSCAPE = textwrap.dedent(
    """
    import os, sys, time
    from pathlib import Path
    import pikepdf
    from lxml import etree

    args = sys.argv[1:]
    mode = os.environ["FAKE_INKSCAPE_MODE"]
    opts = dict(a[2:].split("=", 1) for a in args if a.startswith("--") and "=" in a)
    if mode == "fail":
        sys.stderr.write("fake inkscape: boom\\n")
        sys.exit(3)
    if mode == "sleep":
        time.sleep(10)
    out = Path(opts["export-filename"])
    if mode == "empty":
        sys.exit(0)
    if opts["export-type"] == "eps":
        out.write_bytes(b"%!PS-Adobe-3.0 EPSF-3.0\\n%%BoundingBox: 0 0 1 1\\n%%EOF\\n")
        sys.exit(0)
    labels = [g.get("{http://www.inkscape.org/namespaces/inkscape}label")
              for g in etree.parse(args[-1]).getroot().iter("{http://www.w3.org/2000/svg}g")]
    pdf = pikepdf.new()
    pdf.add_blank_page()
    if mode == "layers":
        ocgs = [pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.OCG, Name=pikepdf.String(n))) for n in labels]
        pdf.Root.OCProperties = pikepdf.Dictionary(OCGs=pikepdf.Array(ocgs), D=pikepdf.Dictionary())
    pdf.save(out)
    """
)


@pytest.fixture
def fake_inkscape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    script = tmp_path / "fake_inkscape.py"
    script.write_text(FAKE_INKSCAPE, encoding="utf-8")
    monkeypatch.setattr(export_mod, "find_inkscape", lambda: [sys.executable, str(script)])
    monkeypatch.setattr(export_mod, "_render_cairosvg", _no_cairosvg)

    def use(mode: str) -> None:
        monkeypatch.setenv("FAKE_INKSCAPE_MODE", mode)

    return use


def test_inkscape_route_used_when_layers_survive(tmp_path: Path, fake_inkscape) -> None:  # type: ignore[no-untyped-def]
    fake_inkscape("layers")
    doc = make_vector_document()
    bundle = run_export(doc, tmp_path / "out")
    assert bundle.warnings == ["cairosvg unavailable (OSError): preview.png rendered with resvg"]
    assert bundle.ai_path is not None and pdf_ocg_names(bundle.ai_path) == [la.name for la in doc.layers]
    with pikepdf.open(bundle.ai_path) as pdf:  # the fake's (blank) page, not the direct writer's
        assert "/Properties" not in pdf.pages[0].Resources
    assert bundle.eps_path is not None and bundle.eps_path.read_bytes().startswith(b"%!PS-Adobe-3.0 EPSF")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "output.ai",
        "output.eps",
        "output.svg",
        "preview.png",
    ]


def test_inkscape_pdf_without_layers_falls_back_to_direct_writer(tmp_path: Path, fake_inkscape) -> None:  # type: ignore[no-untyped-def]
    fake_inkscape("nolayers")
    doc = make_vector_document()
    bundle = run_export(doc, tmp_path)
    assert "inkscape PDF lost the layers (0/3 OCGs): .ai written by direct PDF writer (3 OCG layers)" in bundle.warnings
    assert bundle.ai_path is not None and pdf_ocg_names(bundle.ai_path) == [la.name for la in doc.layers]
    assert not any(".eps" in w for w in bundle.warnings)  # EPS came from (fake) Inkscape


@pytest.mark.parametrize(
    ("mode", "needle"), [("fail", "exit 3): fake inkscape: boom"), ("empty", "exit 0): no output")]
)
def test_inkscape_failure_falls_back(tmp_path: Path, fake_inkscape, mode: str, needle: str) -> None:  # type: ignore[no-untyped-def]
    fake_inkscape(mode)
    bundle = run_export(make_vector_document(), tmp_path)
    ai_warning = next(w for w in bundle.warnings if ".ai" in w)
    eps_warning = next(w for w in bundle.warnings if ".eps" in w)
    assert ai_warning.startswith("inkscape failed") and needle in ai_warning
    assert ai_warning.endswith(".ai written by direct PDF writer (3 OCG layers)")
    assert eps_warning.endswith(".eps written by pycairo PostScript surface (EPS has no layers)")
    assert bundle.ai_path is not None and bundle.eps_path is not None


def test_run_inkscape_timeout_is_stage_error(tmp_path: Path, fake_inkscape, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    fake_inkscape("sleep")
    monkeypatch.setattr(export_mod, "INKSCAPE_TIMEOUT_S", 0.5)
    svg = tmp_path / "a.svg"
    svg.write_text(assemble_svg(make_vector_document(), ALL), encoding="utf-8")
    with pytest.raises(StageError, match="timed out"):
        run_inkscape(svg, tmp_path / "a.pdf", "pdf")


def test_run_inkscape_missing_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(export_mod, "find_inkscape", lambda: None)
    with pytest.raises(ExportToolUnavailableError):
        run_inkscape(tmp_path / "a.svg", tmp_path / "a.pdf", "pdf")
    monkeypatch.setattr(export_mod, "find_inkscape", lambda: [str(tmp_path / "does-not-exist.exe")])
    with pytest.raises(StageError, match="inkscape pdf export failed"):
        run_inkscape(tmp_path / "a.svg", tmp_path / "a.pdf", "pdf")


def test_find_inkscape_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "my-inkscape"
    monkeypatch.setattr(export_mod.shutil, "which", lambda name: str(exe) if name == str(exe) else None)
    monkeypatch.setenv(export_mod.INKSCAPE_ENV, str(exe))
    assert find_inkscape() == [str(exe)]
    monkeypatch.delenv(export_mod.INKSCAPE_ENV)
    monkeypatch.setattr(export_mod, "_WINDOWS_INKSCAPE", (tmp_path / "missing.exe",))
    assert find_inkscape() is None
    exe.write_text("")
    monkeypatch.setattr(export_mod, "_WINDOWS_INKSCAPE", (exe,))
    assert find_inkscape() == [str(exe)]


# --------------------------------------------------------------------------------------
# Real Inkscape (Docker image); skipped when Inkscape is absent
# --------------------------------------------------------------------------------------


def _require_inkscape() -> list[str]:
    command = find_inkscape()
    if command is None:
        pytest.skip("Inkscape not installed here; runs in the Docker image")
    return command


def _inkscape_layer_labels(command: list[str], src: Path, tmp_path: Path) -> list[str]:
    """Open ``src`` in Inkscape, save it as Inkscape SVG and return the layer labels."""
    out = tmp_path / f"{src.stem}_{src.suffix[1:]}_roundtrip.svg"
    subprocess.run(
        [*command, "--export-type=svg", f"--export-filename={out}", str(src)],
        check=True,
        capture_output=True,
        timeout=120,
    )
    root = etree.parse(str(out)).getroot()
    return [
        g.get(f"{{{INKSCAPE_NS}}}label")
        for g in root.iter(f"{{{SVG_NS}}}g")
        if g.get(f"{{{INKSCAPE_NS}}}groupmode") == "layer"
    ]


def test_real_inkscape_export_keeps_layers(tmp_path: Path) -> None:
    _require_inkscape()
    doc = rich_document()
    bundle = run_export(doc, tmp_path)
    assert bundle.ai_path is not None and bundle.eps_path is not None
    assert pdf_ocg_names(bundle.ai_path) == [la.name for la in doc.layers] or sorted(
        pdf_ocg_names(bundle.ai_path)
    ) == sorted(la.name for la in doc.layers)
    assert not any("inkscape unavailable" in w or "inkscape failed" in w for w in bundle.warnings), bundle.warnings


def test_real_inkscape_opens_svg_with_layers(tmp_path: Path) -> None:
    command = _require_inkscape()
    doc = rich_document()
    bundle = run_export(doc, tmp_path)
    assert _inkscape_layer_labels(command, bundle.svg_path, tmp_path) == [la.name for la in doc.layers]


def test_real_inkscape_opens_ai_with_layers(tmp_path: Path) -> None:
    command = _require_inkscape()
    doc = make_vector_document()
    bundle = run_export(doc, tmp_path)
    assert bundle.ai_path is not None
    pdf_copy = tmp_path / "output_ai.pdf"  # Inkscape picks its importer by extension
    shutil.copy(bundle.ai_path, pdf_copy)
    labels = _inkscape_layer_labels(command, pdf_copy, tmp_path)
    assert sorted(labels) == sorted(la.name for la in doc.layers), labels


# --------------------------------------------------------------------------------------
# Performance
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_assemble_plus_export_time_on_09_like_document(tmp_path: Path, fallbacks: None) -> None:
    doc = synthetic_09()
    run_export(doc, tmp_path / "warmup")
    start = time.perf_counter()
    svg = assemble_svg(doc, ALL)
    t_assemble = time.perf_counter() - start
    export(svg, doc, ALL, tmp_path / "timed")
    elapsed = time.perf_counter() - start
    print(f"\n09-like: assemble {t_assemble:.3f} s, assemble+export {elapsed:.3f} s (fallback routes, no Inkscape)")
    assert elapsed <= 1.5


@pytest.mark.slow
def test_assemble_plus_export_time_on_real_09(tmp_path: Path, fallbacks: None) -> None:
    from tests.test_assemble import _real_09_document

    doc = _real_09_document()
    run_export(doc, tmp_path / "warmup")
    start = time.perf_counter()
    export(assemble_svg(doc, ALL), doc, ALL, tmp_path / "timed")
    elapsed = time.perf_counter() - start
    print(f"\nreal 09: assemble+export {elapsed:.3f} s")
    assert elapsed <= 1.5
