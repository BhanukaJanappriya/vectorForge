"""Export: write output.svg, preview.png (always), output.ai and output.eps.

Stage entrypoint: :func:`export` (see ``contracts/stages.py``).

About ``output.ai``
-------------------
The native Adobe Illustrator format is proprietary and undocumented. Since Illustrator 9,
``.ai`` files are PDF documents, and Illustrator opens a plain PDF saved with the ``.ai``
extension directly. VectorForge's ``output.ai`` is therefore a **PDF 1.5 file with the .ai
extension**. It has one Optional Content Group (OCG, a "PDF layer") per VectorLayer, named
exactly like the layer (e.g. ``color_1_#E53935``) and ordered top layer first. Any PDF viewer
(Acrobat, Preview, pdf.js) can open it by renaming it to ``.pdf``. It contains no Illustrator
private data (no ``AIPrivateData``), so Illustrator opens it as a PDF import.

Routes (chosen at run time; importing this module never requires Cairo or Inkscape)
-----------------------------------------------------------------------------------
* ``preview.png``: primary **CairoSVG**, rendering the final optimized SVG. Fallback **resvg**
  (``resvg_py``), which also renders the real SVG. Either way the PNG is exactly the source
  ``width x height``, RGBA, with a transparent background.
* ``output.ai``: primary **Inkscape CLI** (SVG -> PDF). Its output is accepted only if pikepdf
  finds one OCG per layer with matching names. Otherwise, or if Inkscape is missing or fails,
  the **direct PDF writer** is used. It maps the VectorDocument's absolute ``M/L/C/Z`` paths 1:1
  onto PDF ``m/l/c/h`` operators with pikepdf and wraps each layer in ``/OC ... BDC ... EMC``
  marked content.
* ``output.eps``: primary **Inkscape CLI**. Fallback **pycairo** ``PSSurface`` with
  ``set_eps(True)``. EPS has no layer concept.

Every fallback and every skipped format is recorded in ``ExportBundle.warnings``. Nothing
degrades silently. The route taken for every artefact is also logged at INFO level.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pikepdf
from PIL import Image

from contracts.schemas import (
    ExportBundle,
    ExportToolUnavailableError,
    OutputFormat,
    Settings,
    StageError,
    VectorDocument,
    VectorLayer,
)

log = logging.getLogger(__name__)

_STAGE = "export"
PX_TO_PT = 0.75
"""CSS px (1/96 in) to PDF/PostScript points (1/72 in)."""
INKSCAPE_ENV = "VECTORFORGE_INKSCAPE"
"""Environment variable that may point at the Inkscape executable."""
INKSCAPE_TIMEOUT_S = 120.0
_WINDOWS_INKSCAPE = (Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Inkscape" / "bin" / "inkscape.exe",)

_TOKEN_RE = re.compile(r"[MLCZ]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "C": 6, "Z": 0}
_PDF_OP = {"M": "m", "L": "l", "C": "c"}

PathOp = tuple[str, tuple[float, ...]]
"""One drawing operation: ``(command, coordinates)`` with command in M/L/C/Z."""


# --------------------------------------------------------------------------------------
# Path parsing (VectorDocument paths are absolute M/L/C/Z only)
# --------------------------------------------------------------------------------------


def parse_path(d: str) -> list[PathOp]:
    """Parse absolute ``M/L/C/Z`` path data into operations, expanding implicit repeats.

    Extra coordinate pairs after ``M`` are implicit ``L`` commands, as in SVG.

    Raises:
        StageError: on malformed path data (wrong number of coordinates).
    """
    ops: list[PathOp] = []
    tokens = _TOKEN_RE.findall(d)
    i, cmd = 0, ""
    while i < len(tokens):
        tok = tokens[i]
        if tok in _ARITY:
            cmd = tok
            i += 1
            if cmd == "Z":
                ops.append(("Z", ()))
                continue
        elif cmd in ("", "Z"):
            raise StageError(_STAGE, f"path data has coordinates without a command: {d[:60]!r}")
        n = _ARITY[cmd]
        if i + n > len(tokens) or any(t in _ARITY for t in tokens[i : i + n]):
            raise StageError(_STAGE, f"{cmd} needs {n} coordinates in path {d[:60]!r}")
        ops.append((cmd, tuple(float(t) for t in tokens[i : i + n])))
        i += n
        if cmd == "M":
            cmd = "L"
    return ops


def _hex_to_unit_rgb(color_hex: str) -> tuple[float, float, float]:
    """``#rrggbb`` -> RGB floats in [0, 1]."""
    value = int(color_hex[1:], 16)
    return ((value >> 16) & 255) / 255.0, ((value >> 8) & 255) / 255.0, (value & 255) / 255.0


def _num(value: float) -> str:
    """Compact PDF number with 2 decimals (matches output.svg precision)."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


# --------------------------------------------------------------------------------------
# Inkscape detection
# --------------------------------------------------------------------------------------


def find_inkscape() -> list[str] | None:
    """Command prefix that runs the Inkscape CLI, or None if Inkscape is not installed.

    Looks at ``$VECTORFORGE_INKSCAPE``, then ``inkscape`` on PATH, then the default Windows
    install location. Detection runs at call time, never at import time.
    """
    configured = os.environ.get(INKSCAPE_ENV)
    candidates = [configured] if configured else []
    candidates.append("inkscape")
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return [found]
    for path in _WINDOWS_INKSCAPE:
        if path.is_file():
            return [str(path)]
    return None


def run_inkscape(svg_path: Path, out_path: Path, export_type: str) -> None:
    """Convert ``svg_path`` to ``out_path`` (``pdf`` or ``eps``) with the Inkscape 1.x CLI.

    Raises:
        ExportToolUnavailableError: if Inkscape is not installed.
        StageError: if Inkscape fails, times out, or writes no file.
    """
    command = find_inkscape()
    if command is None:
        raise ExportToolUnavailableError("inkscape unavailable")
    out_path.unlink(missing_ok=True)
    args = [
        *command,
        f"--export-type={export_type}",
        f"--export-filename={out_path}",
        "--export-area-page",
        str(svg_path),
    ]
    if export_type == "pdf":
        args.insert(-1, "--export-pdf-version=1.5")
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=INKSCAPE_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StageError(_STAGE, f"inkscape {export_type} export failed: {exc}") from exc
    if proc.returncode != 0 or not out_path.is_file() or out_path.stat().st_size == 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or ["no output"]
        raise StageError(_STAGE, f"inkscape {export_type} export failed (exit {proc.returncode}): {tail[0][:200]}")


# --------------------------------------------------------------------------------------
# preview.png
# --------------------------------------------------------------------------------------


def _render_cairosvg(svg: str, width: int, height: int) -> bytes:
    """Render with CairoSVG. Raises ExportToolUnavailableError if Cairo cannot be loaded."""
    try:
        import cairosvg  # noqa: PLC0415 - imported lazily: needs the native Cairo library
    except (ImportError, OSError) as exc:
        raise ExportToolUnavailableError(f"cairosvg unavailable ({type(exc).__name__})") from exc
    data = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=width, output_height=height)
    return bytes(data)


def _render_resvg(svg: str, width: int, height: int) -> bytes:
    """Render with resvg (``resvg_py``)."""
    try:
        import resvg_py  # noqa: PLC0415 - optional local fallback
    except ImportError as exc:
        raise ExportToolUnavailableError("resvg_py unavailable") from exc
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height))


def _as_rgba_png(data: bytes, route: str, width: int, height: int) -> bytes:
    """Validate rendered PNG bytes; pass them through untouched when already RGBA.

    Only the PNG header is read (no decode/re-encode, which costs ~0.4 s at 2000x2000).
    """
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PNG":
            raise StageError(_STAGE, f"{route} returned {image.format}, expected PNG")
        if image.size != (width, height):
            raise StageError(_STAGE, f"{route} rendered {image.size}, expected {(width, height)}")
        if image.mode == "RGBA":
            return data
        buf = io.BytesIO()
        image.convert("RGBA").save(buf, format="PNG", compress_level=1)
        return buf.getvalue()


def render_png(svg: str, width: int, height: int) -> tuple[bytes, str, list[str]]:
    """Rasterize ``svg`` to an RGBA PNG of exactly ``width x height`` with transparency.

    Returns:
        ``(png_bytes, route, warnings)`` where route is ``"cairosvg"`` or ``"resvg"``.

    Raises:
        StageError: if no renderer works or the result has the wrong size.
    """
    warnings: list[str] = []
    errors: list[str] = []
    for route, renderer in (("cairosvg", _render_cairosvg), ("resvg", _render_resvg)):
        try:
            data = renderer(svg, width, height)
        except ExportToolUnavailableError as exc:
            errors.append(str(exc))
            continue
        except Exception as exc:  # renderer bugs must not hide behind the fallback silently
            errors.append(f"{route} failed: {exc}")
            continue
        png = _as_rgba_png(data, route, width, height)
        if errors:
            warnings.append(f"{'; '.join(errors)}: preview.png rendered with {route}")
        return png, route, warnings
    raise StageError(_STAGE, "cannot render preview.png: " + "; ".join(errors))


# --------------------------------------------------------------------------------------
# Direct PDF writer (.ai) with one OCG per layer
# --------------------------------------------------------------------------------------


def _pdf_path_ops(ops: list[PathOp]) -> Iterator[str]:
    """PDF path-construction operators for parsed SVG operations."""
    for cmd, coords in ops:
        if cmd == "Z":
            yield "h"
        else:
            yield " ".join(_num(c) for c in coords) + " " + _PDF_OP[cmd]


def _layer_content(layer: VectorLayer) -> str:
    """Content-stream operators painting one layer (without OCG/opacity wrapping)."""
    color = " ".join(f"{c:.4f}".rstrip("0").rstrip(".") for c in _hex_to_unit_rgb(layer.color_hex))
    parts: list[str] = []
    if layer.is_stroke:
        parts.append(f"{color} RG {_num(float(layer.stroke_width or 1.0))} w 1 J 1 j")
        paint = "S"
    else:
        parts.append(f"{color} rg")
        paint = "f*" if layer.fill_rule == "evenodd" else "f"
    for d in layer.paths:
        ops = parse_path(d)
        if not ops:
            continue
        parts.extend(_pdf_path_ops(ops))
        parts.append(paint)
    return "\n".join(parts) + "\n"


def write_direct_pdf(doc: VectorDocument, path: Path) -> None:
    """Write ``doc`` as a PDF with one named Optional Content Group per layer.

    The page is ``width*0.75 x height*0.75`` pt. A single ``cm`` maps SVG px (y down) onto PDF
    user space (y up), so path coordinates are written exactly as in the SVG. Layers with
    opacity < 1 are drawn through a transparency-group Form XObject so the opacity applies to
    the layer as a whole, like SVG group opacity.
    """
    pdf = pikepdf.new()
    page_w, page_h = doc.width * PX_TO_PT, doc.height * PX_TO_PT
    properties = pikepdf.Dictionary()
    ext_gstates = pikepdf.Dictionary()
    xobjects = pikepdf.Dictionary()
    ocgs: list[pikepdf.Object] = []
    content = [f"{_num(PX_TO_PT)} 0 0 {_num(-PX_TO_PT)} 0 {_num(page_h)} cm\n"]
    for i, layer in enumerate(doc.layers):
        ocg = pdf.make_indirect(pikepdf.Dictionary(Type=pikepdf.Name.OCG, Name=pikepdf.String(layer.name)))
        ocgs.append(ocg)
        properties[f"/oc{i}"] = ocg
        body = _layer_content(layer)
        content.append(f"/OC /oc{i} BDC\nq\n")
        if layer.opacity < 1.0:
            form = pikepdf.Stream(pdf, body.encode("ascii"))
            form.Type = pikepdf.Name.XObject
            form.Subtype = pikepdf.Name.Form
            form.BBox = pikepdf.Array([0, 0, doc.width, doc.height])
            form.Group = pikepdf.Dictionary(S=pikepdf.Name.Transparency)
            xobjects[f"/Fx{i}"] = pdf.make_indirect(form)
            ext_gstates[f"/gs{i}"] = pikepdf.Dictionary(
                Type=pikepdf.Name.ExtGState, ca=float(layer.opacity), CA=float(layer.opacity)
            )
            content.append(f"/gs{i} gs /Fx{i} Do\n")
        else:
            content.append(body)
        content.append("Q\nEMC\n")

    resources = pikepdf.Dictionary(Properties=properties)
    if len(ext_gstates):
        resources.ExtGState = ext_gstates
        resources.XObject = xobjects
    page = pikepdf.Dictionary(
        Type=pikepdf.Name.Page,
        MediaBox=pikepdf.Array([0, 0, page_w, page_h]),
        Resources=resources,
        Contents=pdf.make_stream("".join(content).encode("ascii")),
    )
    pdf.pages.append(pikepdf.Page(page))
    if ocgs:
        pdf.Root.OCProperties = pikepdf.Dictionary(
            OCGs=pikepdf.Array(ocgs),
            D=pikepdf.Dictionary(
                Name=pikepdf.String("VectorForge layers"),
                Order=pikepdf.Array(list(reversed(ocgs))),  # top layer first, as layer panels show it
                ON=pikepdf.Array(ocgs),
                OFF=pikepdf.Array([]),
            ),
        )
    pdf.docinfo[pikepdf.Name.Title] = pikepdf.String(Path(doc.metadata.source_filename).stem)
    pdf.docinfo[pikepdf.Name.Creator] = pikepdf.String(doc.metadata.generator)
    pdf.docinfo[pikepdf.Name.Producer] = pikepdf.String(f"{doc.metadata.generator} direct PDF writer (pikepdf)")
    pdf.save(path, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.generate, min_version="1.5")


def pdf_ocg_names(path: Path) -> list[str]:
    """Names of the Optional Content Groups declared in a PDF, in ``/OCGs`` order ([] if none)."""
    with pikepdf.open(path) as pdf:
        props = pdf.Root.get("/OCProperties")
        if props is None or "/OCGs" not in props:
            return []
        return [str(ocg.get("/Name", "")) for ocg in props.OCGs]


def _export_ai(svg_path: Path, doc: VectorDocument, ai_path: Path) -> list[str]:
    """Write ``ai_path`` via Inkscape if it keeps the layers, else with the direct writer."""
    wanted = [layer.name for layer in doc.layers]
    reason: str
    try:
        with tempfile.TemporaryDirectory(dir=ai_path.parent) as tmp:
            tmp_pdf = Path(tmp) / "inkscape.pdf"
            run_inkscape(svg_path, tmp_pdf, "pdf")
            got = pdf_ocg_names(tmp_pdf)
            if sorted(got) == sorted(wanted):
                shutil.move(str(tmp_pdf), ai_path)
                log.info("output.ai: inkscape route (%d OCG layers)", len(got))
                return []
            reason = f"inkscape PDF lost the layers ({len(got)}/{len(wanted)} OCGs)"
    except ExportToolUnavailableError as exc:
        reason = str(exc)
    except (StageError, pikepdf.PdfError) as exc:
        reason = f"inkscape failed ({exc})"
    write_direct_pdf(doc, ai_path)
    got = pdf_ocg_names(ai_path)
    if got != wanted:
        raise StageError(_STAGE, f"direct PDF writer produced OCGs {got}, expected {wanted}")
    log.info("output.ai: direct PDF writer (%s)", reason)
    return [f"{reason}: .ai written by direct PDF writer ({len(got)} OCG layers)"]


# --------------------------------------------------------------------------------------
# EPS
# --------------------------------------------------------------------------------------


def write_cairo_eps(doc: VectorDocument, path: Path) -> None:
    """Write ``doc`` as EPS with pycairo's PostScript surface.

    Raises:
        ExportToolUnavailableError: if pycairo is not installed.
    """
    try:
        import cairo  # noqa: PLC0415 - optional: pycairo ships its own Cairo build
    except ImportError as exc:
        raise ExportToolUnavailableError("pycairo unavailable") from exc
    surface = cairo.PSSurface(str(path), doc.width * PX_TO_PT, doc.height * PX_TO_PT)
    surface.set_eps(True)
    ctx = cairo.Context(surface)
    ctx.scale(PX_TO_PT, PX_TO_PT)
    for layer in doc.layers:
        if layer.opacity < 1.0:
            ctx.push_group()
        ctx.set_source_rgb(*_hex_to_unit_rgb(layer.color_hex))
        if layer.is_stroke:
            ctx.set_line_width(float(layer.stroke_width or 1.0))
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        else:
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD if layer.fill_rule == "evenodd" else cairo.FILL_RULE_WINDING)
        for d in layer.paths:
            for cmd, c in parse_path(d):
                if cmd == "M":
                    ctx.move_to(*c)
                elif cmd == "L":
                    ctx.line_to(*c)
                elif cmd == "C":
                    ctx.curve_to(*c)
                else:
                    ctx.close_path()
            if layer.is_stroke:
                ctx.stroke()
            else:
                ctx.fill()
        if layer.opacity < 1.0:
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(layer.opacity)
    surface.finish()


def _export_eps(svg_path: Path, doc: VectorDocument, eps_path: Path) -> tuple[Path | None, list[str]]:
    """Write ``eps_path`` via Inkscape, else pycairo. Returns (path or None if skipped, warnings)."""
    try:
        run_inkscape(svg_path, eps_path, "eps")
        log.info("output.eps: inkscape route")
        return eps_path, []
    except ExportToolUnavailableError as exc:
        reason = str(exc)
    except StageError as exc:
        reason = f"inkscape failed ({exc})"
    try:
        write_cairo_eps(doc, eps_path)
    except ExportToolUnavailableError as exc:
        eps_path.unlink(missing_ok=True)
        log.warning("output.eps skipped: %s; %s", reason, exc)
        return None, [f"{reason}; {exc}: EPS skipped"]
    log.info("output.eps: pycairo PostScript surface (%s)", reason)
    return eps_path, [f"{reason}: .eps written by pycairo PostScript surface (EPS has no layers)"]


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------


def export(svg: str, doc: VectorDocument, settings: Settings, out_dir: Path) -> ExportBundle:
    """Write output.svg, preview.png (always) and the requested .ai/.eps into ``out_dir``.

    Only formats in ``settings.output_formats`` are written, plus ``preview.png``, which QA
    always needs. See the module docstring for the routes and their fallbacks.

    Raises:
        StageError: if a mandatory artefact (SVG, preview, requested .ai) cannot be produced.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    formats = set(settings.output_formats)

    svg_path = out_dir / "output.svg"
    svg_path.write_text(svg, encoding="utf-8")

    png, route, render_warnings = render_png(svg, doc.width, doc.height)
    warnings.extend(render_warnings)
    preview_path = out_dir / "preview.png"
    preview_path.write_bytes(png)
    log.info("preview.png: %s route (%dx%d)", route, doc.width, doc.height)

    ai_path: Path | None = None
    if OutputFormat.AI in formats:
        ai_path = out_dir / "output.ai"
        try:
            warnings.extend(_export_ai(svg_path, doc, ai_path))
        except (pikepdf.PdfError, OSError, ValueError) as exc:
            raise StageError(_STAGE, f"cannot write output.ai: {exc}") from exc

    eps_path: Path | None = None
    if OutputFormat.EPS in formats:
        eps_path, eps_warnings = _export_eps(svg_path, doc, out_dir / "output.eps")
        warnings.extend(eps_warnings)

    return ExportBundle(
        svg_path=svg_path, preview_png_path=preview_path, ai_path=ai_path, eps_path=eps_path, warnings=warnings
    )


__all__: list[str] = [
    "PX_TO_PT",
    "export",
    "find_inkscape",
    "parse_path",
    "pdf_ocg_names",
    "render_png",
    "run_inkscape",
    "write_cairo_eps",
    "write_direct_pdf",
]
