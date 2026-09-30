"""Metric self-tests for eval/evaluate.py and eval/raster.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from contracts.fixtures import (
    make_line_map,
    make_palette,
    make_preprocess_result,
    make_vector_document,
)
from contracts.schemas import ExportBundle, MetricName, QualityThresholds, VectorLayer
from eval import evaluate as ev
from eval.oracle import oracle_svg
from eval.raster import (
    choose_supersampling,
    document_node_count,
    fill_coverage,
    node_count,
    parse_path,
    render_document,
    stroke_coverage,
)

# Sharma, Wu & Dalal (2005), "The CIEDE2000 color-difference formula: implementation notes,
# supplementary test data, and mathematical observations", Table 1: (LAB1, LAB2, dE00).
SHARMA_2005 = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize(("lab1", "lab2", "expected"), SHARMA_2005)
def test_ciede2000_matches_sharma_2005(lab1: tuple, lab2: tuple, expected: float) -> None:
    assert float(ev.delta_e(np.array(lab1), np.array(lab2))) == pytest.approx(expected, abs=1e-4)
    # Symmetric.
    assert float(ev.delta_e(np.array(lab2), np.array(lab1))) == pytest.approx(expected, abs=1e-4)


def test_ciede2000_vectorized_matches_table() -> None:
    a = np.array([p[0] for p in SHARMA_2005])
    b = np.array([p[1] for p in SHARMA_2005])
    np.testing.assert_allclose(ev.delta_e(a, b), [p[2] for p in SHARMA_2005], atol=1e-4)


def test_identical_images_give_ssim_one_and_zero_delta_e() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (48, 64), dtype=np.uint8)
    assert ev.ssim_score(img, None, img, None) == pytest.approx(1.0, abs=1e-12)
    assert ev.ssim_score(img, alpha, img, alpha) == pytest.approx(1.0, abs=1e-12)
    lab = ev.rgb_to_lab(img.reshape(-1, 3))
    assert float(ev.delta_e(lab, lab).max()) == 0.0


def test_ssim_detects_difference_and_tiny_images() -> None:
    img = np.zeros((32, 32, 3), np.uint8)
    img[8:24, 8:24] = 255
    assert ev.ssim_score(img, None, np.zeros_like(img), None) < 0.5
    tiny = np.full((2, 2, 3), 10, np.uint8)
    assert ev.ssim_score(tiny, None, tiny, None) == 1.0
    assert ev.ssim_score(tiny, None, tiny + 100, None) < 1.0
    # 4x4 image: window shrinks to 3.
    small = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    assert ev.ssim_score(small, None, small, None) == pytest.approx(1.0)


def test_gray_over_white_matches_skimage_composite() -> None:
    from skimage.color import rgb2gray

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (10, 12, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (10, 12), dtype=np.uint8)
    expected = rgb2gray(ev.composite_over_white(rgb, alpha).astype(np.float64))
    np.testing.assert_allclose(ev.gray_over_white(rgb, alpha), expected, atol=1e-5)
    np.testing.assert_allclose(ev.composite_over_white(rgb, None), rgb / 255.0, atol=1e-6)


def test_transparent_pixels_composite_to_white() -> None:
    rgb = np.zeros((4, 4, 3), np.uint8)
    alpha = np.zeros((4, 4), np.uint8)
    assert np.allclose(ev.gray_over_white(rgb, alpha), 1.0)


def test_median_lab_subsamples_large_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ev, "MEDIAN_SAMPLE_CAP", 10)
    pixels = np.tile(np.array([[200, 10, 10]], np.uint8), (1000, 1))
    np.testing.assert_allclose(ev.median_lab(pixels), ev.rgb_to_lab(pixels[:1])[0])


# ---------------------------------------------------------------------------- gap / IoU


def test_gap_ratio_and_alpha_iou() -> None:
    src = np.zeros((20, 20), np.uint8)
    src[2:18, 2:18] = 255
    prev = src.copy()
    assert ev.gap_ratio(src, prev) == 0.0
    assert ev.alpha_iou(src, prev) == 1.0
    prev[10, 2:18] = 0  # 1-px transparent seam through the shape
    interior = ev.interior_mask(src, src.shape)
    assert interior.sum() == 14 * 14  # 1-px erosion of the 16x16 square
    assert ev.gap_ratio(src, prev) == pytest.approx(14 / 196)
    assert ev.gap_mask(src, prev).sum() == 14
    assert ev.alpha_iou(src, prev) == pytest.approx((256 - 16) / 256)
    # Silhouette anti-aliasing (outer ring) is ignored.
    ring = src.copy()
    ring[2, :] = 0
    assert ev.gap_ratio(src, ring) == 0.0


def test_gap_ratio_opaque_source_keeps_image_border() -> None:
    prev = np.full((10, 10), 255, np.uint8)
    prev[0, :] = 0  # transparent top row: the image border is NOT eroded for opaque sources
    assert ev.gap_ratio(None, prev) == pytest.approx(0.1)
    assert ev.gap_ratio(np.zeros((5, 5), np.uint8), np.zeros((5, 5), np.uint8)) == 0.0
    assert ev.alpha_iou(np.zeros((5, 5), np.uint8), np.zeros((5, 5), np.uint8)) == 1.0


# ---------------------------------------------------------------------------- checks


def test_build_checks_thresholds_and_comparators() -> None:
    checks = ev.build_checks(
        ssim=0.95,
        ssim_min=0.90,
        mean_de=1.0,
        max_de=2.0,
        gap=0.0,
        iou=None,
        time_s=1.0,
        time_budget_s=2.0,
        svg_ok=True,
    )
    names = [c.name for c in checks]
    assert names == [
        MetricName.SSIM,
        MetricName.MEAN_DELTA_E,
        MetricName.MAX_DELTA_E,
        MetricName.GAP_RATIO,
        MetricName.PROCESSING_TIME,
        MetricName.SVG_VALID,
    ]
    assert all(c.passed for c in checks)
    by = {c.name: c for c in checks}
    assert by[MetricName.MAX_DELTA_E].threshold == QualityThresholds.MAX_DELTA_E
    assert by[MetricName.MEAN_DELTA_E].threshold == QualityThresholds.MAX_MEAN_DELTA_E
    assert by[MetricName.GAP_RATIO].threshold == QualityThresholds.MAX_GAP_RATIO
    bad = ev.build_checks(
        ssim=0.5,
        ssim_min=0.9,
        mean_de=2.0,
        max_de=3.0,
        gap=0.001,
        iou=0.5,
        time_s=3.0,
        time_budget_s=2.0,
        svg_ok=False,
    )
    assert not any(c.passed for c in bad)
    assert MetricName.ALPHA_IOU in [c.name for c in bad]


# ---------------------------------------------------------------------------- raster


def test_node_count() -> None:
    assert node_count("M0 0L10 0L10 10Z") == 3
    assert node_count("M0 0C1 1 2 2 3 3C4 4 5 5 6 6Z") == 3
    assert node_count("M0 0L1 1 2 2 3 3") == 4  # implicit repeats
    assert node_count("M0 0 1 1Z M5 5L6 6Z") == 4
    assert document_node_count(make_vector_document()) == 4 + 4 + 2
    with pytest.raises(ValueError):
        node_count("L0 0")


def test_parse_path_subpaths_and_curves() -> None:
    subs = parse_path("M0 0L4 0L4 4Z M10 10L12 12")
    assert [s.closed for s in subs] == [True, False]
    assert subs[0].points.shape == (3, 2)
    curve = parse_path("M0 0C0 10 10 10 10 0")[0]
    assert curve.points.shape[0] > 3
    np.testing.assert_allclose(curve.points[-1], [10, 0])
    after_close = parse_path("M0 0L2 0L2 2ZL0 2L0 0Z")
    assert len(after_close) == 2 and after_close[1].points[0].tolist() == [0.0, 0.0]
    assert parse_path("M1 1") == []


def test_fill_coverage_exact_rectangles_and_rules() -> None:
    square = parse_path("M2 3L7 3L7 9L2 9Z")
    cov = fill_coverage(square, 10, 12)
    expected = np.zeros((12, 10), np.float32)
    expected[3:9, 2:7] = 1
    np.testing.assert_array_equal(cov, expected)
    ring = parse_path("M0 0L10 0L10 10L0 10Z M3 3L7 3L7 7L3 7Z")
    eo = fill_coverage(ring, 10, 10, "evenodd")
    assert eo[5, 5] == 0 and eo[1, 1] == 1
    nz = fill_coverage(ring, 10, 10, "nonzero")  # same winding direction -> hole filled
    assert nz[5, 5] == 1
    half = fill_coverage(parse_path("M0 0L1.5 0L1.5 1L0 1Z"), 2, 1, ss=4)
    np.testing.assert_allclose(half, [[1.0, 0.5]])
    assert fill_coverage([], 3, 3).sum() == 0
    assert fill_coverage(parse_path("M0 0L3 0"), 3, 3).sum() == 0  # degenerate: horizontal only
    assert fill_coverage(parse_path("M0 50L3 50L3 60Z"), 3, 3).sum() == 0  # off-canvas


def test_stroke_coverage_draws_line() -> None:
    cov = stroke_coverage(parse_path("M0 5.5L20 5.5"), 20, 10, stroke_width=1.0, ss=1)
    assert cov[5].sum() >= 19 and cov[0].sum() == 0
    cov2 = stroke_coverage(parse_path("M2 2L8 2L8 8Z"), 10, 10, stroke_width=2.0, ss=2)
    assert 0 < cov2.max() <= 1


def test_render_fixture_document() -> None:
    doc = make_vector_document(64)
    assert choose_supersampling(doc) == 3  # it has a stroke
    rgba = render_document(doc)
    assert rgba.shape == (64, 64, 4)
    assert (rgba[..., 3] == 255).all()
    assert tuple(rgba[32, 32, :3]) == (220, 50, 47)
    assert tuple(rgba[2, 2, :3]) == (255, 255, 255)
    assert rgba[8, 32, :3].max() < 80  # the black frame line


def test_render_opacity_and_transparency() -> None:
    doc = make_vector_document(16)
    layer = VectorLayer(
        id="color_1_000000",
        name="color_1_#000000",
        role="fill",
        color_hex="#000000",
        paths=["M0 0L8 0L8 8L0 8Z"],
        z_order=0,
        opacity=0.5,
    )
    doc = doc.model_copy(update={"layers": [layer]})
    assert choose_supersampling(doc) == 1
    rgba = render_document(doc)
    assert rgba[0, 0, 3] == 128 and rgba[12, 12, 3] == 0
    empty = doc.model_copy(update={"layers": [layer.model_copy(update={"paths": ["M40 40L50 40L50 50Z"]})]})
    assert render_document(empty)[..., 3].max() == 0


# ---------------------------------------------------------------------------- evaluate()


def _bundle(tmp_path: Path, svg: str, preview: np.ndarray) -> ExportBundle:
    svg_path = tmp_path / "output.svg"
    svg_path.write_text(svg, encoding="utf-8")
    png = tmp_path / "preview.png"
    Image.fromarray(preview).save(png)
    return ExportBundle(svg_path=svg_path, preview_png_path=png)


def test_evaluate_identity_on_contract_fixtures(tmp_path: Path) -> None:
    pre = make_preprocess_result(64)
    palette = make_palette(pre)
    doc = make_vector_document(64)
    bundle = _bundle(tmp_path, oracle_svg(doc), np.asarray(pre.image))
    report = ev.evaluate(pre, palette, make_line_map(pre), doc, bundle, 0.1)
    assert report.ssim == pytest.approx(1.0, abs=1e-9)
    assert report.mean_delta_e == pytest.approx(0.0, abs=1e-6)
    assert report.max_delta_e == pytest.approx(0.0, abs=1e-6)
    assert report.gap_ratio == 0.0 and report.alpha_iou is None
    assert report.node_count == 10
    assert report.file_size_bytes == bundle.svg_path.stat().st_size
    assert report.passed, [c for c in report.checks if not c.passed]


def test_evaluate_alpha_source_and_regions(tmp_path: Path) -> None:
    pre = make_preprocess_result(64, has_alpha=True)
    palette = make_palette(pre)
    doc = make_vector_document(64)
    preview = np.dstack([np.asarray(pre.image), np.asarray(pre.alpha)])
    report, regions = ev.evaluate_detailed(pre, palette, None, doc, _bundle(tmp_path, oracle_svg(doc), preview), 0.1)
    assert report.alpha_iou == 1.0 and report.passed
    # Without a LineMap the line layer falls back to its palette region.
    assert {r.layer_id for r in regions} == {layer.id for layer in doc.layers}


def test_evaluate_flags_wrong_time_and_invalid_svg(tmp_path: Path) -> None:
    pre = make_preprocess_result(64)
    doc = make_vector_document(64)
    bundle = _bundle(tmp_path, "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1 1'/>", np.asarray(pre.image))
    report = ev.evaluate(pre, make_palette(pre), None, doc, bundle, 99.0)
    failed = {c.name for c in report.checks if not c.passed}
    assert failed == {MetricName.PROCESSING_TIME, MetricName.SVG_VALID}


def test_layer_palette_index_fallbacks() -> None:
    pre = make_preprocess_result(64)
    palette = make_palette(pre)
    layer = make_vector_document(64).layers[1]
    assert ev._layer_palette_index(layer.model_copy(update={"palette_index": None}), palette) == 1
    near = layer.model_copy(update={"palette_index": None, "color_hex": "#dd3330"})
    assert ev._layer_palette_index(near, palette) == 1


def test_uncovered_region_scores_missing_layer() -> None:
    pre = make_preprocess_result(64)
    palette = make_palette(pre)
    doc = make_vector_document(64)
    doc = doc.model_copy(update={"layers": [doc.layers[0], doc.layers[2]]})  # red layer dropped
    rendered = np.full((64, 64, 3), 255, np.uint8)
    regions = ev.region_delta_es(pre, palette, None, doc, preview_rgb=rendered)
    uncovered = [r for r in regions if r.role == "uncovered"]
    assert [r.layer_id for r in uncovered] == ["uncovered:#dc322f"]
    assert uncovered[0].delta_e > 30
    # Without a preview the missing layer is invisible (contract-literal definition).
    assert all(r.role != "uncovered" for r in ev.region_delta_es(pre, palette, None, doc))


def test_source_rgba_reads_original_file(tmp_path: Path) -> None:
    pre = make_preprocess_result(32)
    original = np.full((32, 32, 3), 7, np.uint8)
    path = tmp_path / "src.png"
    Image.fromarray(original).save(path)
    pre = pre.model_copy(update={"source": pre.source.model_copy(update={"path": path})})
    rgb, alpha = ev.source_rgba(pre)
    assert alpha is None and (rgb == 7).all()
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    pre2 = pre.model_copy(update={"source": pre.source.model_copy(update={"path": broken})})
    rgb2, _ = ev.source_rgba(pre2)
    np.testing.assert_array_equal(rgb2, np.asarray(pre.image))


def test_load_preview_resizes(tmp_path: Path) -> None:
    path = tmp_path / "p.png"
    Image.fromarray(np.zeros((10, 20, 3), np.uint8)).save(path)
    out = ev.load_preview(path, 40, 20)
    assert out.shape == (20, 40, 4) and (out[..., 3] == 255).all()


# ---------------------------------------------------------------------------- SVG validity


@pytest.mark.parametrize(
    ("svg", "problem"),
    [
        ("<svg", "well-formed"),
        ("<html xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'/>", "root element"),
        ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'/>", "viewBox"),
        ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='a b c d'/>", "viewBox"),
        ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'/>", "layer ids missing"),
    ],
)
def test_svg_problems_detects(tmp_path: Path, svg: str, problem: str) -> None:
    path = tmp_path / "x.svg"
    path.write_text(svg, encoding="utf-8")
    problems = ev.svg_problems(path, make_vector_document(64))
    assert any(problem in p for p in problems), problems


def test_svg_problems_no_drawables(tmp_path: Path) -> None:
    doc = make_vector_document(64)
    groups = "".join(f"<g id='{layer.id}'/>" for layer in doc.layers)
    path = tmp_path / "x.svg"
    path.write_text(f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0,0,64,64'>{groups}</svg>", encoding="utf-8")
    assert ev.svg_problems(path, doc) == ["no drawable elements"]
    path.write_text(oracle_svg(doc), encoding="utf-8")
    assert ev.svg_problems(path, doc) == []


def test_delta_e_uses_original_pixels_not_preprocessed(tmp_path: Path) -> None:
    """Preprocess upscaled x2 and blurred black to gray: Delta-E must compare with the original."""
    from contracts.fixtures import make_image_input
    from contracts.schemas import DenoiseParams, PreprocessResult

    original = np.full((16, 16, 3), 255, np.uint8)
    original[:, :8] = 0  # left half black
    path = tmp_path / "orig.png"
    Image.fromarray(original).save(path)
    processed = np.full((32, 32, 3), 255, np.uint8)
    processed[:, :16] = 64  # the "blurred" preprocess output
    pre = PreprocessResult(
        source=make_image_input(16, path=path),
        image=processed,
        alpha=None,
        scale_factor=2.0,
        denoise=DenoiseParams(method="none", strength=0.0),
    )
    np.testing.assert_array_equal(ev.processing_space_source(pre)[:, :16], 0)
    pal = make_palette(make_preprocess_result(32))
    labels = np.zeros((32, 32), np.int32)
    labels[:, :16] = 2  # palette color 2 is black
    counts = np.bincount(labels.ravel(), minlength=3)
    colors = [c.model_copy(update={"pixel_count": int(counts[c.index])}) for c in pal.colors]
    palette = pal.model_copy(update={"colors": colors, "label_map": labels})
    doc = make_vector_document(16)
    black = doc.layers[2].model_copy(update={"is_stroke": False, "stroke_width": None, "role": "fill"})
    gray = black.model_copy(update={"color_hex": "#404040"})
    de_black = {
        r.layer_id: r.delta_e
        for r in ev.region_delta_es(pre, palette, None, doc.model_copy(update={"layers": [black]}))
    }
    de_gray = {
        r.layer_id: r.delta_e for r in ev.region_delta_es(pre, palette, None, doc.model_copy(update={"layers": [gray]}))
    }
    assert de_black[black.id] == pytest.approx(0.0, abs=1e-6)
    assert de_gray[gray.id] > 15  # #404040 vs black is Delta-E ~17.6


def test_alpha_iou_check_uses_contract_threshold() -> None:
    checks = ev.build_checks(
        ssim=1.0, ssim_min=0.9, mean_de=0, max_de=0, gap=0, iou=0.975, time_s=0, time_budget_s=2, svg_ok=True
    )
    alpha = next(c for c in checks if c.name == MetricName.ALPHA_IOU)
    assert alpha.threshold == QualityThresholds.MIN_ALPHA_IOU and not alpha.passed


def test_p95_pixel_delta_e_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    img = np.full((40, 40, 3), 255, np.uint8)
    img[5:35, 5:35] = (220, 50, 47)
    assert ev.pixel_delta_e_p95(img, None, img, None) == 0.0
    merged = img.copy()
    merged[5:35, 5:12] = (250, 200, 30)  # a minority region rendered in the wrong color
    assert ev.pixel_delta_e_p95(merged, None, img, None) > 30
    alpha = np.zeros((40, 40), np.uint8)
    assert ev.pixel_delta_e_p95(img, alpha, img, alpha) == 0.0  # no interior pixels
    monkeypatch.setattr(ev, "P95_SAMPLE_CAP", 100)
    assert ev.pixel_delta_e_p95(merged, np.full((40, 40), 255, np.uint8), img, None) > 30


def _two_color_line_art() -> tuple:
    """White canvas with a black and a blue 2-px stroke; one LineMap covering both strokes."""
    from contracts.schemas import LineMap, Palette, PaletteColor, rgb_to_hex

    pre = make_preprocess_result(64)
    image = np.full((64, 64, 3), 255, np.uint8)
    image[10:12, 5:60] = (0, 0, 0)
    image[40:42, 5:60] = (38, 139, 210)
    pre = pre.model_copy(update={"image": image})
    labels = np.zeros((64, 64), np.int32)
    labels[10:12, 5:60] = 1
    labels[40:42, 5:60] = 2
    rgbs = [(255, 255, 255), (0, 0, 0), (38, 139, 210)]
    counts = np.bincount(labels.ravel(), minlength=3)
    lab = ev.rgb_to_lab(np.array(rgbs, np.uint8))
    colors = [
        PaletteColor(index=i, rgb=c, lab=tuple(lab[i]), hex=rgb_to_hex(c), pixel_count=int(counts[i]))
        for i, c in enumerate(rgbs)
    ]
    palette = Palette(colors=colors, label_map=labels)
    mask = labels > 0
    skeleton = np.zeros_like(mask)
    skeleton[10, 5:60] = skeleton[40, 5:60] = True
    line_map = LineMap(
        mask=mask,
        skeleton=skeleton,
        width_map=np.where(skeleton, 2.0, 0.0).astype(np.float32),
        median_stroke_width=2.0,
        color_rgb=(0, 0, 0),
    )
    doc = make_vector_document(64)
    bg = doc.layers[0]

    def line(z: int, hex_: str, index: int | None) -> VectorLayer:
        return VectorLayer(
            id=f"line_{z + 1}_{hex_[1:].upper()}",
            name=f"line_{z + 1}_#{hex_[1:].upper()}",
            role="line",
            color_hex=hex_,
            paths=["M5 11L60 11"],
            z_order=z,
            is_stroke=True,
            stroke_width=2.0,
            palette_index=index,
        )

    return pre, palette, line_map, doc, bg, line


def test_multi_color_line_layers_use_own_palette_region() -> None:
    pre, palette, line_map, doc, bg, line = _two_color_line_art()
    layers = [bg, line(1, "#000000", 1), line(2, "#268bd2", 2)]
    regions = ev.region_delta_es(pre, palette, line_map, doc.model_copy(update={"layers": layers}))
    by_id = {r.layer_id: r for r in regions}
    assert by_id["line_2_000000"].delta_e == pytest.approx(0.0, abs=1e-6)
    assert by_id["line_3_268BD2"].delta_e == pytest.approx(0.0, abs=1e-6)
    assert by_id["line_3_268BD2"].pixels == 110  # blue stroke only, not the whole LineMap
    # Wrong color on a line layer still fails.
    wrong = [bg, line(1, "#000000", 1), line(2, "#dc322f", 2)]
    regions = ev.region_delta_es(pre, palette, line_map, doc.model_copy(update={"layers": wrong}))
    assert max(r.delta_e for r in regions) > QualityThresholds.MAX_DELTA_E


def test_line_layer_region_fallbacks() -> None:
    pre, palette, line_map, doc, bg, line = _two_color_line_art()
    # palette_index points at a region the LineMap does not cover -> palette region alone.
    regions = ev.region_delta_es(pre, palette, line_map, doc.model_copy(update={"layers": [line(1, "#ffffff", 0)]}))
    assert regions[0].pixels == 64 * 64 - 220 and regions[0].delta_e == pytest.approx(0.0, abs=1e-6)
    # palette_index None -> whole LineMap (legacy behaviour): median of 110 black + 110 blue pixels.
    regions = ev.region_delta_es(pre, palette, line_map, doc.model_copy(update={"layers": [line(1, "#000000", None)]}))
    assert regions[0].pixels == 220 and regions[0].palette_index is None
