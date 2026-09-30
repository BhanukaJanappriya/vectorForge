"""Oracle stubs and corrupted variants (eval/oracle.py, eval/corrupt.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from contracts.schemas import ImageClassLabel, ImageMode, MetricName, SourceFormat
from eval import corrupt
from eval.evaluate import evaluate, svg_problems
from eval.oracle import (
    _border_background,
    label_rectangles,
    oracle_classify,
    oracle_document,
    oracle_export,
    oracle_load_image,
    oracle_preprocess,
    oracle_quantize,
    oracle_svg,
)
from eval.raster import render_document

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
RED, BLUE, YELLOW = (220, 50, 47), (38, 139, 210), (250, 200, 30)


def make_sample(root: Path, name: str = "90_synthetic", alpha: bool = False) -> Path:
    """A small anti-aliased flat-color sample with ground truth, like samples/generate.py makes."""
    ss, w, h = 4, 160, 120
    mode, bg = ("RGBA", (0, 0, 0, 0)) if alpha else ("RGB", (255, 255, 255))
    img = Image.new(mode, (w * ss, h * ss), bg)
    d = ImageDraw.Draw(img)
    d.ellipse((10 * ss, 10 * ss, 90 * ss, 90 * ss), fill=RED)
    d.polygon([(100 * ss, 10 * ss), (150 * ss, 100 * ss), (60 * ss, 100 * ss)], fill=BLUE)
    d.rectangle((20 * ss, 95 * ss, 140 * ss, 115 * ss), fill=YELLOW)
    img = img.resize((w, h), Image.Resampling.BOX)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.png"
    img.save(path)
    palette = ["#dc322f", "#268bd2", "#fac81e"] if alpha else ["#ffffff", "#dc322f", "#268bd2", "#fac81e"]
    truth = {
        "file": path.name,
        "width": w,
        "height": h,
        "has_alpha": alpha,
        "expected_class": "flat_color",
        "palette_hex": palette,
        "stroke_widths_px": None,
        "median_stroke_width_px": None,
        "notes": "synthetic test sample",
    }
    path.with_suffix(".json").write_text(json.dumps(truth), encoding="utf-8")
    return path


def test_oracle_load_and_preprocess(tmp_path: Path) -> None:
    path = make_sample(tmp_path)
    info = oracle_load_image(path)
    assert (info.width, info.height, info.has_alpha, info.mode) == (160, 120, False, ImageMode.RGB)
    assert info.source_format == SourceFormat.PNG
    pre = oracle_preprocess(info)
    assert pre.alpha is None and pre.scale_factor == 1.0 and pre.image.shape == (120, 160, 3)
    assert pre.background_lab is None  # contract default
    apath = make_sample(tmp_path, "91_alpha", alpha=True)
    ainfo = oracle_load_image(apath)
    assert ainfo.has_alpha and ainfo.mode == ImageMode.RGBA
    assert oracle_preprocess(ainfo).alpha is not None
    jpg = tmp_path / "x.jpg"
    Image.open(path).save(jpg, quality=90)
    assert oracle_load_image(jpg).source_format == SourceFormat.JPEG
    # RGBA file whose alpha is fully opaque -> has_alpha False.
    opaque = tmp_path / "opaque_rgba.png"
    Image.open(path).convert("RGBA").save(opaque)
    assert not oracle_load_image(opaque).has_alpha


def test_oracle_classify() -> None:
    cls = oracle_classify("line_art")
    assert cls.label == ImageClassLabel.LINE_ART and cls.confidence == 1.0


def test_oracle_quantize_ground_truth_palette(tmp_path: Path) -> None:
    pre = oracle_preprocess(oracle_load_image(make_sample(tmp_path)))
    palette = oracle_quantize(pre, ["#ffffff", "#dc322f", "#268bd2", "#fac81e", "#000000"])
    assert [c.hex for c in palette.colors] == ["#ffffff", "#dc322f", "#268bd2", "#fac81e"]  # unused black dropped
    assert palette.background_index == 0
    assert int((np.asarray(palette.label_map) >= 0).sum()) == 160 * 120


def test_oracle_quantize_alpha_has_no_background(tmp_path: Path) -> None:
    pre = oracle_preprocess(oracle_load_image(make_sample(tmp_path, alpha=True)))
    palette = oracle_quantize(pre, ["#dc322f", "#268bd2", "#fac81e"])
    assert palette.background_index is None
    labels = np.asarray(palette.label_map)
    assert (labels[~pre.opaque_mask] == -1).all() and (labels[pre.opaque_mask] >= 0).all()


def test_oracle_quantize_kmeans_uses_region_medians(tmp_path: Path) -> None:
    pre = oracle_preprocess(oracle_load_image(make_sample(tmp_path)))
    palette = oracle_quantize(pre, None, n_auto=4)
    hexes = {c.hex for c in palette.colors}
    assert {"#ffffff", "#dc322f", "#268bd2", "#fac81e"} <= hexes


def test_kmeans_subsamples_many_colors() -> None:
    from eval.oracle import _kmeans_palette

    rng = np.random.default_rng(0)
    lab = rng.uniform(0, 100, (60_000, 3))
    centers = _kmeans_palette(lab, np.ones(60_000), 5, iterations=3)
    assert centers.shape == (5, 3)


def test_border_background() -> None:
    labels = np.zeros((6, 6), np.int32)
    assert _border_background(labels) == 0
    labels[:, :2] = 1
    labels[:, 2:4] = 2  # three-way split of the border: no color reaches 50%
    assert _border_background(labels) is None
    assert _border_background(np.full((3, 3), -1, np.int32)) is None


def test_label_rectangles_reconstruct_labels() -> None:
    rng = np.random.default_rng(3)
    labels = rng.integers(-1, 3, (17, 23)).astype(np.int32)
    labels[5:12, 4:15] = 2
    x0, y0, x1, y1, lab = label_rectangles(labels)
    rebuilt = np.full_like(labels, -1)
    for a, b, c, d, v in zip(x0, y0, x1, y1, lab, strict=True):
        assert (rebuilt[b:d, a:c] == -1).all()  # rectangles never overlap
        rebuilt[b:d, a:c] = v
    np.testing.assert_array_equal(rebuilt, labels)
    assert len(lab) < int((labels >= 0).sum())  # merging happened


def test_oracle_document_renders_label_image_exactly(tmp_path: Path) -> None:
    pre = oracle_preprocess(oracle_load_image(make_sample(tmp_path)))
    palette = oracle_quantize(pre, ["#ffffff", "#dc322f", "#268bd2", "#fac81e"])
    doc = oracle_document(pre, palette, ImageClassLabel.FLAT_COLOR)
    assert doc.layers[0].role == "background" and doc.layers[0].paths == ["M0 0L160 0L160 120L0 120Z"]
    assert [layer.name for layer in doc.layers][0] == "color_1_#FFFFFF"
    lut = np.array([c.rgb for c in palette.colors], np.uint8)
    rgba = render_document(doc)
    np.testing.assert_array_equal(rgba[..., :3], lut[np.asarray(palette.label_map)])
    assert (rgba[..., 3] == 255).all()
    flat = oracle_document(pre, palette, ImageClassLabel.FLAT_COLOR, stack_background=False)
    np.testing.assert_array_equal(render_document(flat), rgba)
    assert flat.layers[0].role == "background" and len(flat.layers[0].paths[0]) > 30


def test_oracle_document_scales_to_source_space(tmp_path: Path) -> None:
    pre = oracle_preprocess(oracle_load_image(make_sample(tmp_path)))
    palette = oracle_quantize(pre, ["#ffffff", "#dc322f", "#268bd2", "#fac81e"])
    half = pre.model_copy(update={"scale_factor": 2.0})
    doc = oracle_document(half, palette, ImageClassLabel.FLAT_COLOR)
    assert "." in doc.layers[1].paths[0]  # coordinates halved -> fractional


def test_oracle_svg_and_export_are_valid(tmp_path: Path) -> None:
    pre = oracle_preprocess(oracle_load_image(make_sample(tmp_path, alpha=True)))
    palette = oracle_quantize(pre, ["#dc322f", "#268bd2", "#fac81e"])
    doc = oracle_document(pre, palette, ImageClassLabel.FLAT_COLOR)
    stroke = doc.layers[-1].model_copy(
        update={
            "id": "line_9_000000",
            "name": "line_9_#000000",
            "z_order": 99,
            "is_stroke": True,
            "stroke_width": 1.5,
            "opacity": 0.5,
            "role": "line",
        }
    )
    doc = doc.model_copy(update={"layers": [*doc.layers, stroke]})
    svg = oracle_svg(doc)
    assert 'inkscape:label="line_9_#000000"' in svg and 'stroke-width="1.5"' in svg and 'opacity="0.5"' in svg
    bundle = oracle_export(svg, doc, tmp_path / "out")
    assert svg_problems(bundle.svg_path, doc) == []
    with Image.open(bundle.preview_png_path) as img:
        assert img.mode == "RGBA" and img.size == (160, 120)


def test_oracle_passes_all_checks_on_real_sample(tmp_path: Path) -> None:
    path = SAMPLES / "01_logo_4color.png"
    truth = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    pre = oracle_preprocess(oracle_load_image(path))
    palette = oracle_quantize(pre, truth["palette_hex"])
    doc = oracle_document(pre, palette, ImageClassLabel(truth["expected_class"]))
    report = evaluate(pre, palette, None, doc, oracle_export(oracle_svg(doc), doc, tmp_path), 0.5)
    assert report.passed, [c for c in report.checks if not c.passed]
    assert report.ssim > 0.99 and report.max_delta_e < 0.5


# ---------------------------------------------------------------------------- corruptions


@pytest.mark.parametrize("name", ["shift_colors", "missing_layer", "seam"])
def test_corruption_fails_on_expected_check(tmp_path: Path, name: str) -> None:
    sample = make_sample(tmp_path / "samples")
    result = corrupt.run_corruption(sample, name, tmp_path / "out")
    assert result.baseline_failed == []
    assert result.ok, result
    assert set(result.flipped) >= set(corrupt.EXPECTED[name])
    assert MetricName.GAP_RATIO.value not in result.flipped or name == "seam"


def test_blur_corruption_on_edge_dense_sample(tmp_path: Path) -> None:
    result = corrupt.run_corruption(SAMPLES / "04_text.png", "blur", tmp_path)
    assert result.ok and result.flipped == [MetricName.SSIM.value]


def test_seam_on_transparent_sample_and_default_plan(tmp_path: Path) -> None:
    make_sample(tmp_path / "s", "92_alpha", alpha=True)
    results = corrupt.run_plan(tmp_path / "s", tmp_path / "o", plan=[("92_alpha", "seam")])
    assert results[0].ok and results[0].to_json()["corruption"] == "seam"
    with pytest.raises(FileNotFoundError):
        corrupt.run_plan(tmp_path / "s", tmp_path / "o", plan=[("nope", "seam")])


def test_corruption_helpers() -> None:
    labels = np.zeros((4, 4), np.int32)
    labels[:, 2:] = 1
    seamed = corrupt.seam_labels(labels)
    assert (seamed[:, 1] == -1).all() and (seamed[:, [0, 2, 3]] >= 0).all()
    from contracts.fixtures import make_vector_document

    doc = make_vector_document(32)
    shifted = corrupt.shift_colors(doc)
    assert all(a.color_hex != b.color_hex for a, b in zip(doc.layers, shifted.layers, strict=True))
    assert len(corrupt.drop_layer(doc).layers) == 2
    assert [layer.id for layer in corrupt.drop_layer(doc, "color_1_FFFFFF").layers][0] == "color_2_DC322F"


def test_unknown_corruption_raises(tmp_path: Path) -> None:
    pre = oracle_preprocess(oracle_load_image(make_sample(tmp_path)))
    palette = oracle_quantize(pre, ["#ffffff", "#dc322f"])
    with pytest.raises(ValueError, match="unknown corruption"):
        corrupt.corrupt_and_evaluate(pre, palette, ImageClassLabel.FLAT_COLOR, "nope", tmp_path / "o")
