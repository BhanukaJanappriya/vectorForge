"""Independent structural and rendering validation of an EPS file.

The checks read the PostScript text only (no stage code), so they hold for every export route
(Inkscape, pycairo, any future writer). Ghostscript, when installed, adds an independent render.

    report = validate_eps(Path("output.eps"), expected_size=(512, 512), source_has_alpha=False)
    report.passed        # False if any check failed (warnings do not fail)
    report.to_dict()     # JSON friendly

This is a technical check of the file. It says nothing about marketplace acceptance, licensing or
originality of the artwork.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from contracts.schemas import QualityThresholds

PX_TO_PT = 0.75
"""Pixel to point factor used by the exporters (96 dpi source space, 72 dpi PostScript)."""
BBOX_TOLERANCE_PT = 2.0
"""Cairo rounds the BoundingBox outwards to whole points."""
GS_TIMEOUT_S = 60
DOS_EPS_MAGIC = b"\xc5\xd0\xd3\xc6"

Status = Literal["pass", "warn", "fail", "skip"]

_PATH_OPS = re.compile(r"(?<![\w/%.-])(?:m|l|c|moveto|lineto|curveto|rmoveto|rlineto|rcurveto)(?![\w*])")
_PAINT_OPS = re.compile(r"(?<![\w/%.-])(?:f\*?|S|fill|eofill|stroke)(?![\w*])")
_RASTER_OPS = re.compile(
    r"(?<![\w/%.-])(?:image|imagemask|colorimage|cairo_image|cairo_imagemask|cairo_ascii85_file)(?![\w])"
    r"|%%BeginBinary|<~"
)
_RECTFILL = re.compile(r"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+rectfill")
_BBOX = re.compile(r"^%%BoundingBox:\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class EpsCheck:
    """Outcome of one validation check."""

    name: str
    status: Status
    detail: str


@dataclass(frozen=True)
class EpsReport:
    """All checks for one EPS file plus the measurements they were derived from."""

    path: str
    size_bytes: int
    bbox: tuple[int, int, int, int] | None
    path_ops: int
    paint_ops: int
    checks: list[EpsCheck] = field(default_factory=list)
    render_ssim: float | None = None

    @property
    def failed(self) -> list[EpsCheck]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[EpsCheck]:
        return [c for c in self.checks if c.status == "warn"]

    @property
    def passed(self) -> bool:
        """True when no check failed. Warnings and skipped checks do not fail the file."""
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def find_ghostscript() -> str | None:
    """Return the Ghostscript executable on PATH (``gs``, or ``gswin64c`` on Windows), if any."""
    for name in ("gs", "gswin64c", "gswin32c"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _split_body(text: str) -> str:
    """The page content: everything after the prolog/setup comments, so macro definitions are excluded."""
    for marker in ("%%EndPageSetup", "%%EndSetup", "%%EndProlog"):
        idx = text.find(marker)
        if idx >= 0:
            return text[idx + len(marker) :]
    return text


def _check_structure(raw: bytes, text: str) -> tuple[list[EpsCheck], tuple[int, int, int, int] | None]:
    checks: list[EpsCheck] = []
    first = text.split("\n", 1)[0].strip()
    if raw.startswith(DOS_EPS_MAGIC):
        checks.append(EpsCheck("header", "fail", "DOS EPS binary wrapper (preview header); plain EPS expected"))
    elif first.startswith("%!PS-Adobe-") and "EPSF-" in first:
        checks.append(EpsCheck("header", "pass", first))
    else:
        checks.append(EpsCheck("header", "fail", f"first line is {first[:40]!r}, expected '%!PS-Adobe-3.0 EPSF-3.0'"))

    head = text[: text.find("%%EndComments")] if "%%EndComments" in text else text[:4096]
    match = _BBOX.search(head)
    bbox: tuple[int, int, int, int] | None = None
    if match is None:
        detail = "no %%BoundingBox in header comments"
        if "%%BoundingBox: (atend)" in text:
            detail = "%%BoundingBox deferred with (atend); importers need it in the header"
        checks.append(EpsCheck("bounding_box", "fail", detail))
    else:
        llx, lly, urx, ury = (int(g) for g in match.groups())
        if urx > llx and ury > lly:
            bbox = (llx, lly, urx, ury)
            checks.append(EpsCheck("bounding_box", "pass", f"{llx} {lly} {urx} {ury}"))
        else:
            checks.append(EpsCheck("bounding_box", "fail", f"empty or inverted box {llx} {lly} {urx} {ury}"))

    tail = text.rstrip()[-200:]
    ok_end = tail.endswith("%%EOF")
    checks.append(
        EpsCheck("terminator", "pass" if ok_end else "fail", "%%EOF" if ok_end else "missing %%EOF (truncated?)")
    )

    begins, ends = text.count("%%BeginProlog"), text.count("%%EndProlog")
    balanced = begins == ends
    checks.append(EpsCheck("prolog_balanced", "pass" if balanced else "fail", f"{begins} begin / {ends} end"))
    gsaves = len(re.findall(r"(?<![\w/%.-])q(?![\w])", _split_body(text)))
    grestores = len(re.findall(r"(?<![\w/%.-])Q(?![\w])", _split_body(text)))
    if gsaves or grestores:
        same = gsaves == grestores
        checks.append(EpsCheck("graphics_state", "pass" if same else "fail", f"{gsaves} save / {grestores} restore"))
    return checks, bbox


def _check_canvas(bbox: tuple[int, int, int, int], expected_size: tuple[int, int], px_to_pt: float) -> EpsCheck:
    llx, lly, urx, ury = bbox
    page_w, page_h = expected_size[0] * px_to_pt, expected_size[1] * px_to_pt
    bw, bh = urx - llx, ury - lly
    tol = BBOX_TOLERANCE_PT
    if llx < -tol or lly < -tol or urx > page_w + tol or ury > page_h + tol:
        return EpsCheck(
            "canvas_size", "fail", f"BoundingBox {bw}x{bh}pt extends beyond the {page_w:.0f}x{page_h:.0f}pt canvas"
        )
    if abs(bw - page_w) <= tol and abs(bh - page_h) <= tol:
        return EpsCheck("canvas_size", "pass", f"{bw}x{bh}pt matches the source canvas")
    return EpsCheck(
        "canvas_size",
        "warn",
        f"BoundingBox {bw}x{bh}pt is cropped to the artwork; source canvas is {page_w:.0f}x{page_h:.0f}pt",
    )


def _check_background(body: str, bbox: tuple[int, int, int, int], source_has_alpha: bool) -> EpsCheck:
    """An opaque rectangle covering the whole page is a background fill that transparent sources must not get."""
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    for match in _RECTFILL.finditer(body[:4000]):
        _, _, w, h = (float(g) for g in match.groups())
        if abs(w - bw) <= BBOX_TOLERANCE_PT and abs(h - bh) <= BBOX_TOLERANCE_PT:
            if source_has_alpha:
                return EpsCheck(
                    "background_fill", "warn", "full-page opaque rectangle although the source has transparency"
                )
            return EpsCheck("background_fill", "pass", "full-page rectangle present; source is opaque")
    return EpsCheck("background_fill", "pass", "no full-page background rectangle")


def _check_content(body: str) -> tuple[list[EpsCheck], int, int]:
    path_ops = len(_PATH_OPS.findall(body))
    paint_ops = len(_PAINT_OPS.findall(body))
    checks: list[EpsCheck] = []
    if path_ops and paint_ops:
        checks.append(EpsCheck("vector_content", "pass", f"{path_ops} path operators, {paint_ops} paint operators"))
    else:
        checks.append(EpsCheck("vector_content", "fail", f"{path_ops} path operators, {paint_ops} paint operators"))
    raster = _RASTER_OPS.search(body)
    if raster:
        checks.append(EpsCheck("no_raster", "fail", f"raster operator in page content: {raster.group(0)!r}"))
    else:
        checks.append(EpsCheck("no_raster", "pass", "no image operators in page content"))
    return checks, path_ops, paint_ops


def render_eps(eps_path: Path, dpi: float = 72.0) -> tuple[np.ndarray | None, str]:
    """Render ``eps_path`` cropped to its BoundingBox with Ghostscript.

    Returns ``(rgba_array_or_None, message)``. ``None`` with a message means Ghostscript is missing or failed.
    """
    gs = find_ghostscript()
    if gs is None:
        return None, "ghostscript not found on PATH"
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "render.png"
        cmd = [
            gs, "-q", "-dBATCH", "-dNOPAUSE", "-dSAFER", "-dEPSCrop", "-sDEVICE=pngalpha",
            "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4", f"-r{dpi:g}", f"-sOutputFile={out}", str(eps_path),
        ]  # fmt: skip
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=GS_TIMEOUT_S, check=False)
        except subprocess.TimeoutExpired:
            return None, f"ghostscript timed out after {GS_TIMEOUT_S}s"
        if proc.returncode != 0 or not out.is_file():
            return None, f"ghostscript exit {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:200]}"
        image = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, "ghostscript produced an unreadable PNG"
    if image.ndim == 3 and image.shape[2] == 3:
        image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
    return image, "ok"


def _flatten_on_white(bgra: np.ndarray) -> np.ndarray:
    alpha = bgra[..., 3:4].astype(np.float32) / 255.0
    flat: np.ndarray = (bgra[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return flat


def render_on_canvas(
    eps_path: Path, bbox: tuple[int, int, int, int], size_px: tuple[int, int], px_to_pt: float = PX_TO_PT
) -> np.ndarray | None:
    """Render the EPS at source scale and place it at its BoundingBox offset on a white ``size_px`` canvas.

    Returns BGR ``uint8`` of shape (h, w, 3), or None if Ghostscript is unavailable or failed.
    """
    rendered, _ = render_eps(eps_path, dpi=72.0 / px_to_pt)
    if rendered is None:
        return None
    width, height = size_px
    canvas = np.full((height, width, 3), 255, np.uint8)
    flat = _flatten_on_white(rendered)
    x0 = int(round(bbox[0] / px_to_pt))
    y0 = int(round(height - bbox[3] / px_to_pt))
    ys, xs = max(y0, 0), max(x0, 0)
    ye, xe = min(y0 + flat.shape[0], height), min(x0 + flat.shape[1], width)
    if ye > ys and xe > xs:
        canvas[ys:ye, xs:xe] = flat[ys - y0 : ye - y0, xs - x0 : xe - x0]
    return canvas


def render_similarity(
    eps_path: Path, bbox: tuple[int, int, int, int], reference_path: Path, px_to_pt: float = PX_TO_PT
) -> float | None:
    """SSIM between the Ghostscript render of the EPS and the source image, both flattened on white."""
    ref = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
    if ref is None:
        return None
    if ref.ndim == 2:
        ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGRA)
    elif ref.shape[2] == 3:
        ref = np.dstack([ref, np.full(ref.shape[:2], 255, np.uint8)])
    height, width = ref.shape[:2]
    canvas = render_on_canvas(eps_path, bbox, (width, height), px_to_pt)
    if canvas is None:
        return None
    gray_a = cv2.cvtColor(_flatten_on_white(ref), cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(gray_a, gray_b, data_range=255))  # type: ignore[no-untyped-call]


def _check_render(eps_path: Path, bbox: tuple[int, int, int, int] | None) -> EpsCheck:
    rendered, message = render_eps(eps_path)
    if rendered is None:
        status: Status = "skip" if message.startswith("ghostscript not found") else "fail"
        return EpsCheck("render", status, message)
    expected_w, expected_h = (bbox[2] - bbox[0], bbox[3] - bbox[1]) if bbox else rendered.shape[1::-1]
    h, w = rendered.shape[:2]
    if abs(w - expected_w) > 1 or abs(h - expected_h) > 1:
        return EpsCheck("render", "fail", f"rendered {w}x{h}px but BoundingBox is {expected_w}x{expected_h}pt")
    inked = float(np.mean(rendered[..., 3] > 0))
    if inked == 0.0:
        return EpsCheck("render", "fail", "render is blank (nothing was painted inside the BoundingBox)")
    return EpsCheck("render", "pass", f"ghostscript rendered {w}x{h}px, {inked:.1%} of pixels inked")


def validate_eps(
    path: Path,
    *,
    expected_size: tuple[int, int] | None = None,
    source_has_alpha: bool = False,
    reference_path: Path | None = None,
    render: bool = True,
    px_to_pt: float = PX_TO_PT,
) -> EpsReport:
    """Run every EPS check on ``path``.

    Args:
        path: EPS file to inspect.
        expected_size: source canvas ``(width, height)`` in px; enables the canvas size check.
        source_has_alpha: the source has transparency; an opaque full-page fill then warns.
        reference_path: source image; enables the render similarity check (needs Ghostscript).
        render: run the Ghostscript render check when Ghostscript is installed.
        px_to_pt: pixel to point scale used by the exporter.
    """
    raw = path.read_bytes()
    text = raw.decode("latin-1")
    checks: list[EpsCheck] = []
    if not raw:
        return EpsReport(str(path), 0, None, 0, 0, [EpsCheck("header", "fail", "file is empty")])

    structure, bbox = _check_structure(raw, text)
    checks.extend(structure)
    body = _split_body(text)
    content, path_ops, paint_ops = _check_content(body)
    checks.extend(content)
    if bbox is not None:
        if expected_size is not None:
            checks.append(_check_canvas(bbox, expected_size, px_to_pt))
        checks.append(_check_background(body, bbox, source_has_alpha))

    ssim_value: float | None = None
    if render:
        checks.append(_check_render(path, bbox))
        if reference_path is not None and bbox is not None and checks[-1].status == "pass":
            ssim_value = render_similarity(path, bbox, reference_path, px_to_pt)
            if ssim_value is not None:
                floor = min(QualityThresholds.SSIM_MIN.values())
                checks.append(
                    EpsCheck(
                        "render_similarity",
                        "pass" if ssim_value >= floor else "fail",
                        f"SSIM {ssim_value:.4f} against the source (minimum {floor})",
                    )
                )
    return EpsReport(str(path), len(raw), bbox, path_ops, paint_ops, checks, ssim_value)


def format_report(report: EpsReport) -> str:
    """Human readable multi-line summary."""
    lines = [f"{report.path}  {report.size_bytes / 1024:.1f} KB  bbox={report.bbox}"]
    lines += [f"  [{c.status:>4}] {c.name}: {c.detail}" for c in report.checks]
    lines.append(f"  => {'PASS' if report.passed else 'FAIL'} ({len(report.warnings)} warnings)")
    return "\n".join(lines)
