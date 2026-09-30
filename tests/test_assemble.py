"""Tests for pipeline.assemble (SVG assembly + optimization)."""

from __future__ import annotations

import io
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import pytest
from lxml import etree
from PIL import Image
from skimage.metrics import structural_similarity

from contracts.fixtures import make_vector_document
from contracts.schemas import (
    SCHEMA_VERSION,
    DetailLevel,
    DocumentMetadata,
    ImageClassLabel,
    Settings,
    StageError,
    VectorDocument,
    VectorLayer,
    layer_name,
)
from pipeline import assemble
from pipeline.assemble import (
    INKSCAPE_NS,
    SVG_NS,
    VF_NS,
    assemble_svg,
    build_svg,
    format_number,
    round_path,
    svg_problems,
)

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------------------
# Synthetic documents (shared with tests/test_export.py)
# --------------------------------------------------------------------------------------

PALETTE_09 = ["#ffffff", "#dc322f", "#268bd2", "#fac81e", "#3ca050", "#14285a", "#f08228"]


def ellipse_path(cx: float, cy: float, rx: float, ry: float, segments: int = 12, decimals: int = 3) -> str:
    """Closed cubic-Bezier ellipse as absolute M/C/Z path data."""
    step = 2 * math.pi / segments
    k = 4 / 3 * math.tan(step / 4)
    parts = [f"M{cx + rx:.{decimals}f} {cy:.{decimals}f}"]
    for i in range(segments):
        a0, a1 = i * step, (i + 1) * step
        p0 = (cx + rx * math.cos(a0), cy + ry * math.sin(a0))
        p3 = (cx + rx * math.cos(a1), cy + ry * math.sin(a1))
        c1 = (p0[0] - k * rx * math.sin(a0), p0[1] + k * ry * math.cos(a0))
        c2 = (p3[0] + k * rx * math.sin(a1), p3[1] - k * ry * math.cos(a1))
        parts.append("C" + " ".join(f"{v:.{decimals}f}" for v in (*c1, *c2, *p3)))
    return "".join(parts) + "Z"


def rect_path(x0: float, y0: float, x1: float, y1: float, decimals: int = 3) -> str:
    """Closed rectangle as absolute M/L/Z path data."""
    f = f".{decimals}f"
    return f"M{x0:{f}} {y0:{f}}L{x1:{f}} {y0:{f}}L{x1:{f}} {y1:{f}}L{x0:{f}} {y1:{f}}Z"


def metadata(settings: Settings | None = None, palette: list[str] | None = None) -> DocumentMetadata:
    return DocumentMetadata(
        source_filename="synthetic.png",
        image_class=ImageClassLabel.FLAT_COLOR,
        settings=settings or Settings(),
        palette_hex=palette or [],
    )


def synthetic_09(seed: int = 1337, shapes: int = 140, segments: int = 16) -> VectorDocument:
    """2000x2000 flat-color document shaped like sample 09 (ellipses + rectangles, 7 colors).

    Coordinates carry 3 decimals like a typical tracer, and each shape is stacked with a
    0.5 px dilation (so lower layers extend under upper ones).
    """
    rng = random.Random(seed)
    per_color: dict[int, list[str]] = {i: [] for i in range(1, 7)}
    for _ in range(shapes):
        x, y = rng.uniform(0, 1700), rng.uniform(0, 1700)
        s = rng.uniform(60, 300)
        color = rng.randint(1, 6)
        if rng.random() < 0.5:
            r = s / 2 + 0.5
            per_color[color].append(ellipse_path(x + s / 2, y + s / 2, r, r, segments))
        else:
            per_color[color].append(rect_path(x - 0.5, y - 0.5, x + s + 0.5, y + s * 0.6 + 0.5))
    layers = [
        VectorLayer(
            id=layer_name("background", 1, PALETTE_09[0])[0],
            name=layer_name("background", 1, PALETTE_09[0])[1],
            role="background",
            color_hex=PALETTE_09[0],
            paths=[rect_path(0, 0, 2000, 2000)],
            z_order=0,
            palette_index=0,
        )
    ]
    for color, paths in per_color.items():
        if not paths:
            continue
        lid, name = layer_name("fill", len(layers) + 1, PALETTE_09[color])
        layers.append(
            VectorLayer(
                id=lid,
                name=name,
                role="fill",
                color_hex=PALETTE_09[color],
                paths=paths,
                z_order=len(layers),
                palette_index=color,
            )
        )
    return VectorDocument(width=2000, height=2000, layers=layers, metadata=metadata(palette=PALETTE_09))


def rich_document(width: int = 320, height: int = 200, n_layers: int = 40) -> VectorDocument:
    """Many layers, evenodd holes, nonzero fill, strokes (open + closed) and a translucent layer."""
    rng = random.Random(7)
    layers: list[VectorLayer] = []
    colors = [f"#{rng.randrange(1 << 24):06x}" for _ in range(n_layers)]
    for i, color in enumerate(colors):
        z = len(layers)
        if i == 0:
            role, paths = "background", [rect_path(0, 0, width, height, 0)]
        elif i % 10 == 3:  # ring with an evenodd hole
            role = "fill"
            cx, cy = rng.uniform(40, width - 40), rng.uniform(40, height - 40)
            paths = [rect_path(cx - 30, cy - 30, cx + 30, cy + 30) + rect_path(cx - 12, cy - 12, cx + 12, cy + 12)]
        else:
            role = "fill"
            paths = [
                ellipse_path(rng.uniform(0, width), rng.uniform(0, height), rng.uniform(3, 30), rng.uniform(3, 30), 8)
                for _ in range(rng.randint(1, 4))
            ]
        is_stroke = i % 10 == 7
        if is_stroke:
            role = "line"
            y = rng.uniform(10, height - 10)
            paths = [f"M5.123 {y:.3f}L{width - 5.456:.3f} {y + 7.891:.3f}", "M20 20L60 20L60 60Z"]
        lid, name = layer_name(role, z + 1, color)  # type: ignore[arg-type]
        layers.append(
            VectorLayer(
                id=lid,
                name=name,
                role=role,  # type: ignore[arg-type]
                color_hex=color,
                paths=paths,
                z_order=z,
                is_stroke=is_stroke,
                stroke_width=2.5 if is_stroke else None,
                fill_rule="nonzero" if i % 10 == 5 else "evenodd",
                opacity=0.6 if i % 10 == 9 else 1.0,
            )
        )
    return VectorDocument(width=width, height=height, layers=layers, metadata=metadata(palette=colors))


# --------------------------------------------------------------------------------------
# Helpers: SVG path decoding (any absolute/relative/shorthand form) and rendering
# --------------------------------------------------------------------------------------

_TOKEN = re.compile(r"[MmLlHhVvCcSsZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Z": 0}


def absolute_segments(d: str) -> list[tuple[float | str, ...]]:
    """Decode path data (M/L/H/V/C/S/Z, absolute or relative) into absolute M/L/C/Z segments."""
    tokens = _TOKEN.findall(d)
    out: list[tuple[float | str, ...]] = []
    x = y = sx = sy = 0.0
    last_ctrl: tuple[float, float] | None = None
    i, cmd = 0, ""
    while i < len(tokens):
        if tokens[i].upper() in _ARITY:
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                out.append(("Z",))
                x, y, last_ctrl = sx, sy, None
                continue
        big, rel = cmd.upper(), cmd.islower()
        v = [float(t) for t in tokens[i : i + _ARITY[big]]]
        i += _ARITY[big]
        ox, oy = (x, y) if rel else (0.0, 0.0)
        if big == "M":
            x, y = ox + v[0], oy + v[1]
            sx, sy, last_ctrl = x, y, None
            out.append(("M", x, y))
            cmd = "l" if rel else "L"
        elif big in "LHV":
            if big == "H":
                x = ox + v[0]
            elif big == "V":
                y = oy + v[0]
            else:
                x, y = ox + v[0], oy + v[1]
            out.append(("L", x, y))
            last_ctrl = None
        elif big == "C":
            c1, c2, p = (ox + v[0], oy + v[1]), (ox + v[2], oy + v[3]), (ox + v[4], oy + v[5])
            out.append(("C", *c1, *c2, *p))
            last_ctrl, (x, y) = c2, p
        else:  # S
            c1 = (2 * x - last_ctrl[0], 2 * y - last_ctrl[1]) if last_ctrl else (x, y)
            c2, p = (ox + v[0], oy + v[1]), (ox + v[2], oy + v[3])
            out.append(("C", *c1, *c2, *p))
            last_ctrl, (x, y) = c2, p
    return out


def _straighten(segments: list[tuple[float | str, ...]], tol: float = 1e-9) -> list[tuple[float | str, ...]]:
    """Replace cubics whose control points lie ON the chord segment by lines.

    That substitution traces exactly the same point set (scour performs it). Cubics whose
    collinear control points overshoot the endpoints are left alone, so they would be reported.
    """
    out: list[tuple[float | str, ...]] = []
    x = y = sx = sy = 0.0
    for seg in segments:
        if seg[0] == "C":
            x1, y1, x2, y2, ex, ey = (float(v) for v in seg[1:])
            dx, dy = ex - x, ey - y
            length2 = dx * dx + dy * dy
            on_chord = length2 > 0 and all(
                abs((px - x) * dy - (py - y) * dx) <= tol * math.sqrt(length2)
                and -tol <= ((px - x) * dx + (py - y) * dy) / length2 <= 1 + tol
                for px, py in ((x1, y1), (x2, y2))
            )
            out.append(("L", ex, ey) if on_chord else seg)
            x, y = ex, ey
        elif seg[0] == "Z":
            out.append(seg)
            x, y = sx, sy
        else:
            out.append(seg)
            x, y = float(seg[1]), float(seg[2])
            if seg[0] == "M":
                sx, sy = x, y
    return out


def same_geometry(a: str, b: str, tol: float = 1e-6) -> bool:
    sa, sb = _straighten(absolute_segments(a)), absolute_segments(b)
    if [s[0] for s in sa] != [s[0] for s in sb]:
        return False
    return all(
        abs(float(p) - float(q)) <= tol for s, t in zip(sa, sb, strict=True) for p, q in zip(s[1:], t[1:], strict=True)
    )


def render(svg: str, width: int, height: int) -> np.ndarray:
    """Rasterize with resvg (always available locally) -> (H, W, 4) uint8."""
    import resvg_py

    data = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height))
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))


def over_white_gray(rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:4].astype(np.float64) / 255.0
    rgb = rgba[..., :3].astype(np.float64) * a + 255.0 * (1 - a)
    return rgb @ np.array([0.299, 0.587, 0.114])


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(over_white_gray(a), over_white_gray(b), data_range=255.0))


def parse(svg: str) -> etree._Element:
    return etree.fromstring(svg.encode("utf-8"), etree.XMLParser(resolve_entities=False, no_network=True))


def layer_groups(root: etree._Element) -> list[etree._Element]:
    return [g for g in root.iter(f"{{{SVG_NS}}}g") if g.get(f"{{{INKSCAPE_NS}}}groupmode") == "layer"]


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_svg() -> str:
    return assemble_svg(make_vector_document(), Settings())


def test_root_is_svg11_with_viewbox_and_size(fixture_svg: str) -> None:
    root = parse(fixture_svg)
    assert root.tag == f"{{{SVG_NS}}}svg"
    assert root.get("version") == "1.1"
    assert root.get("viewBox") == "0 0 64 64"
    assert (root.get("width"), root.get("height")) == ("64", "64")
    assert fixture_svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')


def test_one_named_inkscape_layer_per_vector_layer_in_z_order(fixture_svg: str) -> None:
    doc = make_vector_document()
    groups = layer_groups(parse(fixture_svg))
    assert [(g.get("id"), g.get(f"{{{INKSCAPE_NS}}}label")) for g in groups] == [(la.id, la.name) for la in doc.layers]
    for g, layer in zip(groups, doc.layers, strict=True):
        assert g.getparent().tag == f"{{{SVG_NS}}}svg"  # top-level groups -> real layers
        assert len(g.findall(f"{{{SVG_NS}}}path")) == len(layer.paths)


def test_paint_lives_on_groups(fixture_svg: str) -> None:
    fill_bg, fill_red, stroke = layer_groups(parse(fixture_svg))
    assert fill_red.get("fill") == "#dc322f"
    assert fill_red.get("fill-rule") == "evenodd"
    assert fill_bg.get("fill") in ("#fff", "#ffffff")
    assert stroke.get("fill") == "none"
    assert stroke.get("stroke") in ("#000", "#000000")
    assert float(stroke.get("stroke-width", "1")) == 1.0  # scour drops the default value 1
    assert stroke.get("stroke-linejoin") == "round"
    assert stroke.get("stroke-linecap") == "round"
    for g in (fill_bg, fill_red, stroke):
        assert all(p.get("fill") is None and p.get("stroke") is None for p in g)


def test_nonzero_stroke_width_and_opacity_are_written() -> None:
    doc = rich_document()
    groups = {g.get("id"): g for g in layer_groups(parse(assemble_svg(doc, Settings())))}
    for layer in doc.layers:
        g = groups[layer.id]
        if layer.is_stroke:
            assert float(g.get("stroke-width")) == layer.stroke_width
        else:
            assert g.get("fill-rule", "nonzero") == layer.fill_rule
        assert float(g.get("opacity", "1")) == pytest.approx(layer.opacity)


def test_metadata_block(fixture_svg: str) -> None:
    root = parse(fixture_svg)
    meta = root.find(f"{{{SVG_NS}}}metadata")
    assert meta is not None
    info = meta.find(f"{{{VF_NS}}}document")
    assert info.get("generator") == "VectorForge"
    assert info.get("schema-version") == SCHEMA_VERSION
    assert info.get("source") == "fixture.png"
    assert info.get("image-class") == "flat_color"
    settings_json = info.find(f"{{{VF_NS}}}settings").text
    assert Settings.model_validate(json.loads(settings_json)) == Settings()


def test_metadata_records_given_settings() -> None:
    settings = Settings(detail_level=DetailLevel.HIGH, max_colors=5, palette_override=["#112233"])
    info = parse(assemble_svg(make_vector_document(), settings)).find(f".//{{{VF_NS}}}settings")
    assert Settings.model_validate_json(info.text) == settings


def test_no_scripts_foreign_objects_or_external_refs() -> None:
    svg = assemble_svg(rich_document(), Settings())
    root = parse(svg)
    locals_ = {etree.QName(el).localname for el in root.iter() if isinstance(el.tag, str)}
    assert not locals_ & {"script", "foreignObject", "image", "use", "a", "style"}
    for el in root.iter():
        for name, value in el.attrib.items():
            assert "href" not in name and not etree.QName(name).localname.startswith("on")
            assert "url(" not in value
    assert "<!ENTITY" not in svg and "<!DOCTYPE" not in svg


def test_svg_problems_clean_for_assembled_output(fixture_svg: str) -> None:
    assert svg_problems(fixture_svg, make_vector_document()) == []


def test_eval_validator_accepts_output(tmp_path: Path, fixture_svg: str) -> None:
    from eval.evaluate import svg_problems as eval_svg_problems

    path = tmp_path / "output.svg"
    path.write_text(fixture_svg, encoding="utf-8")
    assert eval_svg_problems(path, make_vector_document()) == []


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda s: s.replace("</svg>", ""), "not well-formed"),
        (lambda s: s.replace('xmlns="http://www.w3.org/2000/svg"', 'xmlns="urn:x"'), "SVG namespace"),
        (lambda s: s.replace('viewBox="0 0 64 64"', 'viewBox="0 0 1 1"'), "viewBox"),
        (lambda s: s.replace('width="64"', 'width="65"'), "width/height"),
        (lambda s: s.replace('id="color_2_DC322F"', 'id="other"'), "layer groups"),
        (lambda s: s.replace("</svg>", "<script>alert(1)</script></svg>"), "forbidden element <script>"),
        (lambda s: s.replace("</svg>", '<foreignObject width="1" height="1"/></svg>'), "foreignObject"),
        (lambda s: s.replace("<path ", '<path onclick="x()" ', 1), "forbidden attribute onclick"),
        (lambda s: s.replace("<path ", '<path fill="url(http://evil/x.svg#p)" ', 1), "forbidden attribute fill"),
        (
            lambda s: s.replace(
                "</svg>", '<image xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="x.png"/></svg>'
            ),
            "forbidden",
        ),
    ],
)
def test_svg_problems_detects_defects(fixture_svg: str, mutate, needle: str) -> None:  # type: ignore[no-untyped-def]
    problems = svg_problems(mutate(fixture_svg), make_vector_document())
    assert any(needle in p for p in problems), problems


def test_internal_url_refs_allowed(fixture_svg: str) -> None:
    svg = fixture_svg.replace("<path ", '<path clip-path="url(#c)" ', 1)
    assert svg_problems(svg, make_vector_document()) == []


def test_assemble_raises_stage_error_when_optimizer_breaks_document(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assemble, "optimize_svg", lambda svg, significant_digits: svg.replace("layer", "sublayer"))
    with pytest.raises(StageError, match="optimized SVG is invalid"):
        assemble_svg(make_vector_document(), Settings())


def test_missing_root_tag_after_optimization_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(assemble, "optimize_svg", lambda svg, significant_digits: "<not-svg/>")
    with pytest.raises(StageError, match="no <svg> root"):
        assemble_svg(make_vector_document(), Settings())


def test_root_size_not_in_scientific_notation() -> None:
    svg = assemble_svg(synthetic_09(shapes=6), Settings())
    root = parse(svg)
    assert root.get("viewBox") == "0 0 2000 2000"
    assert (root.get("width"), root.get("height")) == ("2000", "2000")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.0, "1"),
        (1.5, "1.5"),
        (1.234, "1.23"),
        (1.235001, "1.24"),
        (-0.001, "0"),
        (0.0, "0"),
        (-2.5, "-2.5"),
        (100, "100"),
    ],
)
def test_format_number(value: float, expected: str) -> None:
    assert format_number(value) == expected


def test_round_path_keeps_commands_and_reports_max() -> None:
    d, largest = round_path("M1.23456 -2.005L1e2 3.999C0 0 1.1 2.2 -1500.129 .5Z")
    assert d == "M1.23 -2L100 4C0 0 1.1 2.2 -1500.13 .5Z".replace(" .5", " 0.5")
    assert largest == pytest.approx(1500.129)


def test_coordinates_have_at_most_two_decimals() -> None:
    svg = assemble_svg(rich_document(), Settings())
    for p in parse(svg).iter(f"{{{SVG_NS}}}path"):
        for num in _TOKEN.findall(p.get("d")):
            if num.upper() not in _ARITY:
                mantissa = num.lower().split("e")[0]
                assert len(mantissa.split(".")[1]) <= 2 if "." in mantissa else True, num


@pytest.mark.parametrize("make_doc", [make_vector_document, rich_document, lambda: synthetic_09(shapes=30)])
def test_optimization_preserves_rounded_geometry_exactly(make_doc) -> None:  # type: ignore[no-untyped-def]
    doc = make_doc()
    optimized = [p.get("d") for p in parse(assemble_svg(doc, Settings())).iter(f"{{{SVG_NS}}}path")]
    originals = [round_path(d)[0] for layer in doc.layers for d in layer.paths]
    assert len(optimized) == len(originals)
    mismatched = [(a, b) for a, b in zip(originals, optimized, strict=True) if not same_geometry(a, b)]
    assert not mismatched, mismatched[:2]


def test_rounding_error_bounded() -> None:
    doc = rich_document()
    for layer in doc.layers:
        for d in layer.paths:
            raw, rounded = absolute_segments(d), absolute_segments(round_path(d)[0])
            for s, t in zip(raw, rounded, strict=True):
                assert all(abs(float(p) - float(q)) <= 0.005 + 1e-9 for p, q in zip(s[1:], t[1:], strict=True))


@pytest.mark.parametrize("make_doc", [make_vector_document, rich_document])
def test_optimized_renders_identically_to_unoptimized(make_doc) -> None:  # type: ignore[no-untyped-def]
    doc = make_doc()
    opt = render(assemble_svg(doc, Settings()), doc.width, doc.height)
    raw = render(build_svg(doc, Settings(), optimize=False), doc.width, doc.height)
    assert ssim(opt, raw) >= 0.999
    assert np.abs(opt[..., 3].astype(int) - raw[..., 3].astype(int)).max() <= 8


def test_evenodd_hole_and_nonzero_render_correctly() -> None:
    left = rect_path(10, 10, 90, 90, 0) + rect_path(30, 30, 70, 70, 0)
    right = rect_path(110, 10, 190, 90, 0) + rect_path(130, 30, 170, 70, 0)  # same winding -> filled under nonzero
    layers = [
        VectorLayer(id="a", name="a", role="fill", color_hex="#ff0000", paths=[left], z_order=0, fill_rule="evenodd"),
        VectorLayer(id="b", name="b", role="fill", color_hex="#0000ff", paths=[right], z_order=1, fill_rule="nonzero"),
    ]
    doc = VectorDocument(width=200, height=100, layers=layers, metadata=metadata())
    img = render(assemble_svg(doc, Settings()), 200, 100)
    assert tuple(img[20, 20]) == (255, 0, 0, 255)
    assert img[50, 50, 3] == 0  # evenodd hole is transparent
    assert img[5, 5, 3] == 0  # outside everything stays transparent (no background added)
    assert tuple(img[50, 150]) == (0, 0, 255, 255)  # nonzero: inner square with same winding stays filled


def test_empty_document_is_valid() -> None:
    doc = VectorDocument(width=10, height=20, layers=[], metadata=metadata())
    svg = assemble_svg(doc, Settings())
    assert svg_problems(svg, doc) == []
    assert render(svg, 10, 20)[..., 3].max() == 0


def test_many_layers_all_present() -> None:
    doc = rich_document(n_layers=60)
    groups = layer_groups(parse(assemble_svg(doc, Settings())))
    assert [g.get(f"{{{INKSCAPE_NS}}}label") for g in groups] == [la.name for la in doc.layers]


def test_optimized_is_at_least_20_percent_smaller_on_09_like_document() -> None:
    doc = synthetic_09()
    raw = build_svg(doc, Settings(), optimize=False)
    opt = assemble_svg(doc, Settings())
    assert len(opt.encode()) <= 0.8 * len(raw.encode()), (len(opt), len(raw))


def test_renders_with_cairosvg_when_available() -> None:
    try:
        import cairosvg
    except (ImportError, OSError) as exc:
        pytest.skip(f"CairoSVG / native Cairo unavailable here ({type(exc).__name__}); runs in Docker")
    doc = rich_document()
    svg = assemble_svg(doc, Settings())
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=doc.width, output_height=doc.height)
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    assert img.shape == (doc.height, doc.width, 4)
    assert ssim(img, render(svg, doc.width, doc.height)) >= 0.98


def _real_09_document() -> VectorDocument:
    """Run the real upstream stages on samples/09 (skips while pipeline.vectorize is missing)."""
    try:
        from pipeline.vectorize import vectorize
    except ImportError:
        pytest.skip("pipeline.vectorize not available yet")
    from pipeline.classify import classify
    from pipeline.lines import extract_lines
    from pipeline.preprocess import load_image, preprocess
    from pipeline.quantize import quantize

    settings = Settings()
    image = load_image(REPO / "samples" / "09_large_2000.png")
    pre = preprocess(image, settings)
    cls = classify(pre, settings)
    palette = quantize(pre, cls, settings)
    lines = extract_lines(pre, cls, settings) if cls.label != ImageClassLabel.FLAT_COLOR else None
    return vectorize(pre, cls, palette, lines, settings)


@pytest.mark.slow
def test_real_09_smaller_and_identical_render() -> None:
    doc = _real_09_document()
    raw = build_svg(doc, Settings(), optimize=False)
    opt = assemble_svg(doc, Settings())
    print(f"\n09 real: raw {len(raw.encode())} B -> optimized {len(opt.encode())} B")
    assert len(opt.encode()) <= 0.8 * len(raw.encode())
    assert ssim(render(opt, doc.width, doc.height), render(raw, doc.width, doc.height)) >= 0.999


@pytest.mark.slow
def test_assemble_time_on_09_like_document() -> None:
    doc = synthetic_09()
    assemble_svg(doc, Settings())  # warm-up (imports, regex compilation)
    start = time.perf_counter()
    assemble_svg(doc, Settings())
    elapsed = time.perf_counter() - start
    print(f"\nassemble 09-like: {elapsed:.3f} s")
    assert elapsed <= 1.0
