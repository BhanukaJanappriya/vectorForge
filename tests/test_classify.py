"""Tests for pipeline/classify.py: sample accuracy, forced modes, determinism, synthetic
images for each class, transparency handling and the 150 ms budget."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from contracts.fixtures import make_image_input, make_preprocess_result
from contracts.schemas import (
    DenoiseParams,
    ImageClassLabel,
    PreprocessResult,
    ProcessingMode,
    Settings,
)
from pipeline.classify import classify, extract_features, score
from pipeline.preprocess import load_image, preprocess

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
MANIFEST = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SAMPLES.glob("*.json"))]


def _pre(rgb: np.ndarray, alpha: np.ndarray | None = None, scale: float = 1.0) -> PreprocessResult:
    """Wrap a synthetic processing-space image in a valid PreprocessResult."""
    h, w = rgb.shape[:2]
    src_w, src_h = round(w / scale), round(h / scale)
    source = make_image_input(size=src_w, has_alpha=alpha is not None).model_copy(
        update={"width": src_w, "height": src_h}
    )
    return PreprocessResult(
        source=source, image=rgb, alpha=alpha, scale_factor=scale, denoise=DenoiseParams(method="none", strength=0)
    )


def _canvas(h: int = 480, w: int = 640, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    return np.full((h, w, 3), color, np.uint8)


def _line_drawing() -> np.ndarray:
    img = _canvas()
    cv2.circle(img, (320, 240), 180, (0, 0, 0), 4, lineType=cv2.LINE_AA)
    cv2.line(img, (60, 400), (600, 60), (0, 0, 0), 3, lineType=cv2.LINE_AA)
    cv2.ellipse(img, (320, 240), (220, 90), 20, 0, 300, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    return img


def _flat_shapes() -> np.ndarray:
    img = _canvas()
    cv2.circle(img, (200, 200), 120, (220, 50, 47), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (350, 100), (600, 400), (38, 139, 210), -1)
    cv2.fillPoly(img, [np.array([[100, 450], [300, 300], [330, 460]])], (250, 200, 30), lineType=cv2.LINE_AA)
    return img


def _outlined_shapes() -> np.ndarray:
    img = _flat_shapes()
    cv2.circle(img, (200, 200), 120, (0, 0, 0), 5, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (350, 100), (600, 400), (0, 0, 0), 5)
    cv2.polylines(img, [np.array([[100, 450], [300, 300], [330, 460]])], True, (0, 0, 0), 5, lineType=cv2.LINE_AA)
    return img


def _gradient() -> np.ndarray:
    x = np.linspace(0, 1, 640, dtype=np.float32)[None, :, None]
    y = np.linspace(0, 1, 480, dtype=np.float32)[:, None, None]
    a = np.array([240, 80, 40], np.float32)
    b = np.array([30, 170, 200], np.float32)
    img = a * (1 - x) + b * x
    img = img * (1 - 0.3 * y) + 0.3 * y * 255
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_results() -> dict[str, PreprocessResult]:
    return {e["file"]: preprocess(load_image(SAMPLES / e["file"]), Settings()) for e in MANIFEST}


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_samples_classified_correctly(entry: dict, sample_results: dict[str, PreprocessResult]) -> None:
    result = classify(sample_results[entry["file"]], Settings())
    assert result.label.value == entry["expected_class"], result.features
    assert not result.forced
    assert result.confidence >= 0.8
    assert result.confidence == pytest.approx(result.features[f"p_{result.label.value}"], abs=1e-4)


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
@pytest.mark.parametrize("detail", ["low", "high"])
def test_samples_stable_across_detail_levels(entry: dict, detail: str) -> None:
    settings = Settings(detail_level=detail)
    pre = preprocess(load_image(SAMPLES / entry["file"]), settings)
    assert classify(pre, settings).label.value == entry["expected_class"]


def test_probabilities_sum_to_one(sample_results: dict[str, PreprocessResult]) -> None:
    for pre in sample_results.values():
        feats = classify(pre, Settings()).features
        total = feats["p_line_art"] + feats["p_flat_color"] + feats["p_mixed"]
        assert total == pytest.approx(1.0, abs=1e-3)


def test_classify_deterministic(sample_results: dict[str, PreprocessResult]) -> None:
    pre = sample_results["10_mixed_scene.png"]
    assert classify(pre, Settings()) == classify(pre, Settings())


_TIMING_SCRIPT = """
import json, sys, time
from pathlib import Path
import numpy as np
from contracts.schemas import DenoiseParams, PreprocessResult, Settings
from contracts.fixtures import make_image_input
from pipeline.classify import classify
from pipeline.preprocess import load_image, preprocess
if sys.argv[1] == "noise":
    img = np.random.default_rng(3).integers(0, 256, (1200, 1700, 3), dtype=np.uint8)
    src = make_image_input(1700).model_copy(update={"width": 1700, "height": 1200})
    pre = PreprocessResult(source=src, image=img, alpha=None, scale_factor=1.0,
                           denoise=DenoiseParams(method="none", strength=0))
else:
    pre = preprocess(load_image(Path(sys.argv[1])), Settings())
classify(pre, Settings())  # warm-up (OpenCV thread pool, page-in)
times = []
for _ in range(5):
    start = time.perf_counter()
    classify(pre, Settings())
    times.append(time.perf_counter() - start)
print(json.dumps(times))
"""


def _classify_seconds(target: str) -> float:
    """Best-of-5 classify time measured in a fresh interpreter.

    A fresh process keeps the measurement independent of the memory state of the test
    session (on a memory-starved machine, a large pytest process pays page faults on every
    allocation, which inflates timings several-fold for reasons unrelated to classify).
    """
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c", _TIMING_SCRIPT, target], cwd=root, capture_output=True, text=True, check=True
    )
    return min(json.loads(out.stdout.strip().splitlines()[-1]))


@pytest.mark.slow
def test_classify_time_budget_large_sample() -> None:
    best = _classify_seconds(str(SAMPLES / "09_large_2000.png"))
    assert best <= 0.15, best


@pytest.mark.slow
def test_classify_time_budget_2mp_noise() -> None:
    best = _classify_seconds("noise")
    assert best <= 0.15, best


# --------------------------------------------------------------------------------------
# Forced modes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "label"),
    [
        (ProcessingMode.LINE_ART, ImageClassLabel.LINE_ART),
        (ProcessingMode.FLAT_COLOR, ImageClassLabel.FLAT_COLOR),
        (ProcessingMode.MIXED, ImageClassLabel.MIXED),
    ],
)
def test_forced_mode(mode: ProcessingMode, label: ImageClassLabel) -> None:
    result = classify(make_preprocess_result(), Settings(mode=mode))
    assert result.label is label
    assert result.forced is True
    assert result.confidence == 1.0


# --------------------------------------------------------------------------------------
# Synthetic images (built in the test)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "label"),
    [
        (_line_drawing, ImageClassLabel.LINE_ART),
        (_flat_shapes, ImageClassLabel.FLAT_COLOR),
        (_outlined_shapes, ImageClassLabel.MIXED),
        (_gradient, ImageClassLabel.MIXED),
    ],
    ids=["line_drawing", "flat_shapes", "outlined_shapes", "gradient"],
)
def test_synthetic_classes(builder: object, label: ImageClassLabel) -> None:
    result = classify(_pre(builder()), Settings())  # type: ignore[operator]
    assert result.label is label, result.features


def test_upscaled_input_uses_source_resolution() -> None:
    img = _line_drawing()
    up = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_NEAREST_EXACT)
    direct = extract_features(_pre(img))
    scaled = extract_features(_pre(up, scale=2.0))
    assert classify(_pre(up, scale=2.0), Settings()).label is ImageClassLabel.LINE_ART
    assert scaled["ink_ratio"] == pytest.approx(direct["ink_ratio"], rel=0.25)


def test_line_art_on_transparent_background() -> None:
    img = _line_drawing()
    alpha = np.where(np.all(img > 250, axis=2), 0, 255).astype(np.uint8)
    img[alpha == 0] = 0  # RGB under transparency is black, like the strokes
    result = classify(_pre(img, alpha), Settings())
    assert result.label is ImageClassLabel.LINE_ART, result.features
    assert result.features["transparent_bg"] == 1.0


def test_flat_logo_on_transparent_background() -> None:
    img = _flat_shapes()
    alpha = np.where(np.all(img > 250, axis=2), 0, 255).astype(np.uint8)
    result = classify(_pre(img, alpha), Settings())
    assert result.label is ImageClassLabel.FLAT_COLOR, result.features


def test_fully_transparent_image() -> None:
    result = classify(_pre(_canvas(64, 64), np.zeros((64, 64), np.uint8)), Settings())
    assert result.label is ImageClassLabel.FLAT_COLOR
    assert result.confidence == 0.5
    assert result.features["opaque_fraction"] == 0.0


def test_uniform_image_has_no_ink() -> None:
    feats = extract_features(_pre(_canvas(64, 64, (10, 200, 30))))
    assert feats["ink_ratio"] == 0.0 and feats["ink_extent"] == 0.0 and feats["ink_components"] == 0.0
    assert classify(_pre(_canvas(64, 64)), Settings()).label is ImageClassLabel.FLAT_COLOR


def test_fixture_preprocess_result() -> None:
    result = classify(make_preprocess_result(128, has_alpha=True), Settings())
    assert result.label in set(ImageClassLabel)
    assert 0.0 <= result.confidence <= 1.0


def test_features_reported() -> None:
    feats = classify(_pre(_flat_shapes()), Settings()).features
    for key in (
        "n_colors", "colors_95", "gradient_area", "smooth_ratio", "edge_density", "dark_ratio",
        "bg_share", "fill_ratio", "thin_ink_ratio", "ink_extent", "outline_ratio",
    ):
        assert key in feats


def test_score_extremes() -> None:
    base = {
        "gradient_area": 0.0, "colors_95": 3.0, "thin_ink_ratio": 1.0, "edge_per_ink": 0.5,
        "fill_ratio": 0.0, "ink_extent": 0.6, "ink_ratio": 0.05, "outline_fraction": 0.0, "outline_ratio": 0.0,
    }
    assert max(score(base).items(), key=lambda kv: kv[1])[0] is ImageClassLabel.LINE_ART
    gradient = {**base, "gradient_area": 0.5}
    assert score(gradient)[ImageClassLabel.MIXED] > 0.99
    many_colors = {**base, "colors_95": 60.0, "fill_ratio": 0.5}
    assert score(many_colors)[ImageClassLabel.MIXED] > 0.99
    assert score({"opaque_fraction": 0.0})[ImageClassLabel.FLAT_COLOR] == 0.5


# --------------------------------------------------------------------------------------
# Interaction with background removal
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_samples_classified_correctly_after_background_removal(entry: dict) -> None:
    settings = Settings(remove_background=True)
    pre = preprocess(load_image(SAMPLES / entry["file"]), settings)
    assert classify(pre, settings).label.value == entry["expected_class"]


def test_background_removed_without_background_lab_uses_transparency() -> None:
    img = _line_drawing()
    alpha = np.where(np.all(img > 250, axis=2), 0, 255).astype(np.uint8)
    pre = _pre(img, alpha).model_copy(update={"background_removed": True})
    assert pre.background_lab is None
    assert classify(pre, Settings()).features["transparent_bg"] == 1.0


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_samples_robust_to_heavy_jpeg_compression(entry: dict, tmp_path: Path) -> None:
    """Re-encoding at JPEG q=30 (4:2:0) must not change the class (ringing is not a gradient)."""
    with Image.open(SAMPLES / entry["file"]) as img:
        if img.mode == "RGBA":
            pytest.skip("JPEG has no alpha")
        path = tmp_path / "recompressed.jpg"
        img.convert("RGB").save(path, format="JPEG", quality=30)
    pre = preprocess(load_image(path), Settings())
    assert classify(pre, Settings()).label.value == entry["expected_class"]


def test_gradient_area_small_image_is_zero() -> None:
    feats = extract_features(_pre(_gradient()[:16, :16].copy()))
    assert feats["gradient_area"] == 0.0


def test_removed_background_is_composited_back(tmp_path: Path) -> None:
    """With background_lab, enclosed regions of the removed color stay 'background' (line art)."""
    settings = Settings(remove_background=True)
    pre = preprocess(load_image(SAMPLES / "02_lineart_black.png"), settings)
    assert pre.background_removed and pre.background_lab is not None
    result = classify(pre, settings)
    assert result.label is ImageClassLabel.LINE_ART
    assert result.features["transparent_bg"] == 0.0
