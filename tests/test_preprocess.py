"""Tests for pipeline/preprocess.py: load_image validation, mode normalization, denoise,
rescaling, transparency handling and background removal."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms

import pipeline.preprocess as pp
from contracts.schemas import (
    DetailLevel,
    ImageInput,
    ImageMode,
    InvalidImageError,
    PreprocessResult,
    Settings,
    SourceFormat,
    StageError,
)
from pipeline.preprocess import (
    estimate_noise,
    fill_transparent_rgb,
    load_image,
    preprocess,
    remove_background,
    sniff_format,
    target_scale,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
MANIFEST = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SAMPLES.glob("*.json"))]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _save(img: Image.Image, path: Path, fmt: str = "PNG", **kwargs: object) -> Path:
    img.save(path, format=fmt, **kwargs)
    return path


def _logo(size: tuple[int, int] = (600, 600)) -> Image.Image:
    """White canvas with a red disc and a blue square (flat art, opaque)."""
    w, h = size
    arr = np.full((h, w, 3), 255, np.uint8)
    cv2.circle(arr, (w // 3, h // 2), min(w, h) // 5, (220, 50, 47), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(arr, (w // 2, h // 4), (w // 2 + w // 5, h // 4 + h // 5), (38, 139, 210), -1)
    return Image.fromarray(arr)


def _run(path: Path, **settings: object) -> PreprocessResult:
    return preprocess(load_image(path), Settings(**settings))


def lab_bin_count(rgb: np.ndarray) -> int:
    """Unique colors after rounding to Delta-E-2 LAB bins."""
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return len(np.unique(np.floor(lab / 2.0).astype(np.int32).reshape(-1, 3), axis=0))


def edge_sharpness(rgb: np.ndarray, edges: np.ndarray) -> float:
    """Mean Sobel gradient magnitude at the given (source Canny) edge pixels."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    return float(mag[edges].mean())


# --------------------------------------------------------------------------------------
# load_image
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["file"] for e in MANIFEST])
def test_load_image_samples(entry: dict) -> None:
    info = load_image(SAMPLES / entry["file"])
    assert (info.width, info.height) == (entry["width"], entry["height"])
    assert info.has_alpha == entry["has_alpha"]
    expected_fmt = SourceFormat.JPEG if entry["file"].endswith(".jpg") else SourceFormat.PNG
    assert info.source_format is expected_fmt
    assert info.file_size_bytes == (SAMPLES / entry["file"]).stat().st_size


def test_sniff_format() -> None:
    assert sniff_format(pp.PNG_MAGIC + b"rest") is SourceFormat.PNG
    assert sniff_format(b"\xff\xd8\xff\xe0rest") is SourceFormat.JPEG
    assert sniff_format(b"GIF89a") is None
    assert sniff_format(b"") is None


def test_format_detected_by_magic_not_extension(tmp_path: Path) -> None:
    png_named_jpg = _save(_logo((64, 64)), tmp_path / "actually_png.jpg")
    assert load_image(png_named_jpg).source_format is SourceFormat.PNG
    jpg_named_png = _save(_logo((64, 64)), tmp_path / "actually_jpg.png", "JPEG")
    assert load_image(jpg_named_png).source_format is SourceFormat.JPEG


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("text.png", b"hello, I am not an image at all"),
        ("gif.png", None),
        ("empty.png", b""),
        ("garbage_png.png", pp.PNG_MAGIC + b"\x00" * 200),
        ("garbage_jpg.jpg", b"\xff\xd8\xff\xe0" + b"\x13" * 200),
    ],
)
def test_invalid_files_rejected(tmp_path: Path, name: str, payload: bytes | None) -> None:
    path = tmp_path / name
    if payload is None:
        Image.new("RGB", (8, 8)).save(path, format="GIF")
    else:
        path.write_bytes(payload)
    with pytest.raises(InvalidImageError):
        load_image(path)


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidImageError):
        load_image(tmp_path / "nope.png")
    with pytest.raises(InvalidImageError):
        pp.decode_image(tmp_path / "nope.png")


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_truncated_files_rejected(tmp_path: Path, fmt: str) -> None:
    buf = io.BytesIO()
    _logo((256, 256)).save(buf, format=fmt)
    data = buf.getvalue()
    path = tmp_path / f"truncated.{fmt.lower()}"
    path.write_bytes(data[: len(data) // 2])
    with pytest.raises(InvalidImageError):
        load_image(path)


def test_oversized_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _save(_logo((64, 64)), tmp_path / "big.png")
    monkeypatch.setattr(pp, "MAX_PIXELS", 64 * 64 - 1)
    with pytest.raises(InvalidImageError, match="pixel limit"):
        load_image(path)


def test_has_alpha_only_when_some_pixel_transparent(tmp_path: Path) -> None:
    opaque = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
    info = load_image(_save(opaque, tmp_path / "opaque_rgba.png"))
    assert info.mode is ImageMode.RGBA
    assert info.has_alpha is False
    pre = preprocess(info, Settings())
    assert pre.alpha is None
    opaque.putpixel((3, 3), (10, 20, 30, 254))
    assert load_image(_save(opaque, tmp_path / "one_pixel.png")).has_alpha is True


# --------------------------------------------------------------------------------------
# EXIF orientation
# --------------------------------------------------------------------------------------


def _marked(w: int = 60, h: int = 30) -> Image.Image:
    """Landscape image: gray with a red block at the top-left corner."""
    arr = np.full((h, w, 3), 128, np.uint8)
    arr[: h // 2, : w // 3] = (255, 0, 0)
    return Image.fromarray(arr)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_exif_rotation_applied(tmp_path: Path, fmt: str) -> None:
    img = _marked()
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW to display
    path = _save(img, tmp_path / f"rotated.{fmt.lower()}", fmt, exif=exif.tobytes())
    info = load_image(path)
    assert (info.width, info.height) == (30, 60)
    pre = preprocess(info, Settings())
    rgb = np.asarray(pre.image)
    expected = np.asarray(img.transpose(Image.Transpose.ROTATE_270))
    expected = cv2.resize(expected, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
    # red block moved to the top-right corner
    assert rgb[2, -2, 0] > 200 and rgb[2, -2, 1] < 60
    assert rgb[-2, 2, 0] < 160
    assert np.abs(rgb.astype(int) - expected.astype(int)).mean() < 6


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_orientations_match_pillow(orientation: int) -> None:
    arr = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    out = pp._orient(arr, orientation)
    op = pp._ORIENTATION_OPS.get(orientation)
    expected = arr if op is None else np.asarray(Image.fromarray(arr).transpose(op))
    np.testing.assert_array_equal(out, expected)


# --------------------------------------------------------------------------------------
# Mode normalization (every Pillow mode we accept)
# --------------------------------------------------------------------------------------


def test_mode_l(tmp_path: Path) -> None:
    arr = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (32, 1))
    pre = _run(_save(Image.fromarray(arr, "L"), tmp_path / "gray.png"))
    assert pre.source.mode is ImageMode.L and pre.alpha is None
    rgb = np.asarray(pre.image)
    assert rgb.shape == (64, 128, 3)  # upscaled 2x (smaller side < 500)
    assert np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])


def test_mode_1_bilevel(tmp_path: Path) -> None:
    img = Image.new("1", (40, 40), 1)
    img.paste(0, (10, 10, 30, 30))
    pre = _run(_save(img, tmp_path / "bilevel.png"))
    assert pre.source.mode is ImageMode.L
    assert set(np.unique(np.asarray(pre.image))) <= set(range(256))
    assert np.asarray(pre.image)[40, 40].tolist() == [0, 0, 0]


def test_mode_la(tmp_path: Path) -> None:
    arr = np.zeros((40, 40, 2), np.uint8)
    arr[..., 0] = 90
    arr[10:30, 10:30, 1] = 255
    pre = _run(_save(Image.fromarray(arr, "LA"), tmp_path / "la.png"))
    assert pre.source.mode is ImageMode.LA and pre.source.has_alpha
    assert pre.alpha is not None
    assert pre.alpha[40, 40] == 255 and pre.alpha[0, 0] == 0
    assert np.all(np.asarray(pre.image) == 90)


def test_mode_p_with_trns(tmp_path: Path) -> None:
    img = Image.new("P", (40, 40), 0)
    img.putpalette([255, 255, 255, 220, 50, 47] + [0] * 762)
    img.paste(1, (10, 10, 30, 30))
    path = _save(img, tmp_path / "p_trns.png", transparency=0)
    info = load_image(path)
    assert info.mode is ImageMode.P and info.has_alpha
    pre = preprocess(info, Settings())
    assert pre.alpha is not None and pre.alpha[0, 0] == 0 and pre.alpha[40, 40] == 255
    # transparent pixels take the nearest opaque color (red), not the palette's white
    assert np.asarray(pre.image)[0, 0].tolist() == [220, 50, 47]


def test_mode_p_opaque(tmp_path: Path) -> None:
    img = _logo((64, 64)).convert("P", palette=Image.Palette.ADAPTIVE, colors=8)
    pre = _run(_save(img, tmp_path / "p.png"))
    assert pre.source.mode is ImageMode.P and pre.alpha is None


def test_mode_pa_and_rgba_premultiplied() -> None:
    pa = Image.new("PA", (8, 8))
    rgb, alpha = pp._normalize(pa)
    assert rgb.shape == (8, 8, 3) and alpha is not None
    rgba_pm = Image.new("RGBa", (8, 8), (10, 10, 10, 128))
    rgb, alpha = pp._normalize(rgba_pm)
    assert alpha is not None and int(alpha[0, 0]) == 128


def test_mode_rgb_with_trns_and_rgbx_and_l_trns() -> None:
    rgb_img = Image.new("RGB", (8, 8), (1, 2, 3))
    rgb_img.info["transparency"] = (1, 2, 3)
    _, alpha = pp._normalize(rgb_img)
    assert alpha is not None and int(alpha.max()) == 0
    l_img = Image.new("L", (8, 8), 7)
    l_img.info["transparency"] = 7
    _, alpha = pp._normalize(l_img)
    assert alpha is not None and int(alpha.max()) == 0
    rgbx = Image.new("RGBX", (8, 8), (5, 6, 7, 0))
    rgb, alpha = pp._normalize(rgbx)
    assert alpha is None and rgb[0, 0].tolist() == [5, 6, 7]
    ycc = Image.new("RGB", (8, 8), (200, 100, 50)).convert("YCbCr")
    rgb, _ = pp._normalize(ycc)
    assert np.abs(rgb[0, 0].astype(int) - [200, 100, 50]).max() <= 3


def test_mode_rgba_sample_alpha_preserved_exactly() -> None:
    path = SAMPLES / "07_transparent_logo.png"
    pre = _run(path)
    src = np.asarray(Image.open(path))
    assert pre.scale_factor == 1.0
    assert pre.alpha is not None
    np.testing.assert_array_equal(pre.alpha, src[..., 3])
    opaque = src[..., 3] == 255
    np.testing.assert_array_equal(np.asarray(pre.image)[opaque], src[..., :3][opaque])


def test_mode_cmyk_jpeg(tmp_path: Path) -> None:
    img = Image.new("CMYK", (64, 64), (0, 255, 255, 0))  # pure red in naive CMYK
    path = _save(img, tmp_path / "cmyk.jpg", "JPEG", quality=95)
    info = load_image(path)
    assert info.mode is ImageMode.CMYK
    rgb = np.asarray(preprocess(info, Settings()).image)
    r, g, b = rgb[64, 64].astype(int)
    assert r > 200 and g < 60 and b < 60


def test_mode_16bit_gray(tmp_path: Path) -> None:
    arr = np.full((40, 40), 65535, np.uint16)
    arr[:, :20] = 257 * 100
    path = _save(Image.fromarray(arr), tmp_path / "gray16.png")
    info = load_image(path)
    assert info.mode is ImageMode.L
    rgb = np.asarray(preprocess(info, Settings()).image)
    assert rgb[0, 0].tolist() == [100, 100, 100]
    assert rgb[0, -1].tolist() == [255, 255, 255]


@pytest.mark.parametrize(
    ("mode", "values", "expected"),
    [("I", [0, 65535], [0, 255]), ("I", [0, 200], [0, 200]), ("F", [0.0, 1.0], [0, 255]), ("F", [0.0, 300.0], [0, 1])],
)
def test_high_bit_depth_modes(mode: str, values: list[float], expected: list[int]) -> None:
    arr = np.array([values], dtype=np.float32 if mode == "F" else np.int32)
    out = np.asarray(pp._to_uint8_gray(Image.fromarray(arr, mode)))
    assert out.tolist() == [expected]


def test_unsupported_mode_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _save(_logo((16, 16)), tmp_path / "x.png")
    monkeypatch.setattr(pp, "_MODE_MAP", {"L": ImageMode.L})
    with pytest.raises(InvalidImageError, match="unsupported pixel mode"):
        load_image(path)


def test_icc_profile_srgb_roundtrip(tmp_path: Path) -> None:
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    img = _logo((64, 64))
    pre = _run(_save(img, tmp_path / "icc.png", icc_profile=icc))
    src = np.asarray(img.resize((128, 128), Image.Resampling.NEAREST)).astype(int)
    assert np.abs(np.asarray(pre.image).astype(int) - src).mean() < 3


def test_icc_profile_invalid_falls_back(tmp_path: Path) -> None:
    img = _logo((32, 32))
    pre = _run(_save(img, tmp_path / "bad_icc.png", icc_profile=b"definitely not an icc profile"))
    assert np.asarray(pre.image)[0, 0].tolist() == [255, 255, 255]


def test_icc_profile_mismatched_colorspace_falls_back() -> None:
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    out = pp._apply_icc(Image.new("CMYK", (4, 4), (0, 0, 0, 0)), icc)
    assert out.mode == "RGB"


# --------------------------------------------------------------------------------------
# Transparency fill
# --------------------------------------------------------------------------------------


def test_fill_transparent_rgb_nearest_opaque() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (24, 31, 3), dtype=np.uint8)
    alpha = np.zeros((24, 31), np.uint8)
    alpha[5:9, 4:7] = 255
    alpha[15:20, 20:28] = 200
    out = fill_transparent_rgb(rgb, alpha)
    opaque_yx = np.argwhere(alpha > 0)
    np.testing.assert_array_equal(out[alpha > 0], rgb[alpha > 0])
    for y, x in np.argwhere(alpha == 0)[::7]:
        d = np.hypot(opaque_yx[:, 0] - y, opaque_yx[:, 1] - x)
        candidates = {tuple(rgb[p[0], p[1]]) for p in opaque_yx[d <= d.min() + 1.0]}
        assert tuple(out[y, x]) in candidates


def test_fill_transparent_rgb_noop_cases() -> None:
    rgb = np.zeros((4, 4, 3), np.uint8)
    assert fill_transparent_rgb(rgb, None) is rgb
    assert fill_transparent_rgb(rgb, np.full((4, 4), 255, np.uint8)) is rgb
    assert fill_transparent_rgb(rgb, np.zeros((4, 4), np.uint8)) is rgb


def test_transparent_rgb_filled_in_preprocess(tmp_path: Path) -> None:
    arr = np.zeros((40, 40, 4), np.uint8)  # transparent black everywhere
    arr[10:30, 10:30] = (38, 139, 210, 255)
    pre = _run(_save(Image.fromarray(arr, "RGBA"), tmp_path / "t.png"))
    assert np.all(np.asarray(pre.image) == (38, 139, 210))  # no black halo anywhere


# --------------------------------------------------------------------------------------
# Rescaling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("size", "scale"),
    [((512, 512), 1.0), ((600, 400), 2.0), ((499, 1000), 2.0), ((3000, 1000), 2048 / 3000), ((400, 3000), 2048 / 3000)],
)
def test_target_scale(size: tuple[int, int], scale: float) -> None:
    assert target_scale(*size) == pytest.approx(scale)


def test_upscale_small_image(tmp_path: Path) -> None:
    pre = _run(_save(_logo((120, 90)), tmp_path / "small.png"))
    assert pre.scale_factor == 2.0
    assert (pre.width, pre.height) == (240, 180)


def test_downscale_large_image(tmp_path: Path) -> None:
    pre = _run(_save(_logo((3000, 600)), tmp_path / "wide.png"))
    assert max(pre.width, pre.height) <= pp.MAX_LONG_SIDE
    assert pre.scale_factor == pytest.approx(2048 / 3000)
    assert pre.denoise.extra["scale_factor"] == pytest.approx(pre.scale_factor)


def test_downscale_with_alpha(tmp_path: Path) -> None:
    arr = np.zeros((300, 2500, 4), np.uint8)
    arr[50:250, 100:2400] = (220, 50, 47, 255)
    pre = _run(_save(Image.fromarray(arr, "RGBA"), tmp_path / "wide_alpha.png"))
    assert pre.alpha is not None and pre.alpha.shape == pre.image.shape[:2]
    assert pre.scale_factor < 1.0


def test_upscale_with_alpha(tmp_path: Path) -> None:
    arr = np.zeros((50, 50, 4), np.uint8)
    arr[10:40, 10:40] = (220, 50, 47, 255)
    pre = _run(_save(Image.fromarray(arr, "RGBA"), tmp_path / "small_alpha.png"))
    assert pre.alpha is not None and pre.alpha.shape == (100, 100)
    assert pre.alpha[50, 50] == 255 and pre.alpha[2, 2] == 0


# --------------------------------------------------------------------------------------
# Denoise
# --------------------------------------------------------------------------------------


def test_clean_png_untouched() -> None:
    path = SAMPLES / "01_logo_4color.png"
    pre = _run(path)
    assert pre.denoise.method == "none" and pre.denoise.strength == 0.0
    np.testing.assert_array_equal(pre.image, np.asarray(Image.open(path).convert("RGB")))


def test_jpeg_artifacts_reduced_without_blurring_edges() -> None:
    path = SAMPLES / "06_jpeg_artifacts.jpg"
    src = np.asarray(Image.open(path).convert("RGB"))
    pre = _run(path)
    out = np.asarray(pre.image)
    assert pre.denoise.method == "bilateral" and pre.denoise.jpeg_deblock
    before, after = lab_bin_count(src), lab_bin_count(out)
    assert after <= 0.5 * before, (before, after)
    edges = cv2.Canny(cv2.cvtColor(src, cv2.COLOR_RGB2GRAY), 50, 150) > 0
    assert edge_sharpness(out, edges) >= 0.9 * edge_sharpness(src, edges)


@pytest.mark.parametrize("detail", list(DetailLevel))
def test_jpeg_reduction_every_detail_level(detail: DetailLevel) -> None:
    path = SAMPLES / "06_jpeg_artifacts.jpg"
    src = np.asarray(Image.open(path).convert("RGB"))
    out = np.asarray(_run(path, detail_level=detail).image)
    assert lab_bin_count(out) <= 0.5 * lab_bin_count(src)


def test_denoise_strength_scales_with_detail() -> None:
    path = SAMPLES / "06_jpeg_artifacts.jpg"
    strengths = [_run(path, detail_level=d).denoise.strength for d in (DetailLevel.LOW, DetailLevel.HIGH)]
    assert strengths[0] > strengths[1]


def test_noisy_png_gets_light_bilateral(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    arr = np.asarray(_logo((520, 520))).astype(np.int16) + rng.normal(0, 4, (520, 520, 3)).astype(np.int16)
    path = _save(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), tmp_path / "noisy.png")
    pre = _run(path)
    assert pre.denoise.method == "bilateral" and not pre.denoise.jpeg_deblock
    assert pre.denoise.extra["noise_sigma"] > pp.PNG_NOISE_THRESHOLD
    assert estimate_noise(np.asarray(pre.image)) < pre.denoise.extra["noise_sigma"]


def test_estimate_noise_edge_cases() -> None:
    assert estimate_noise(np.zeros((2, 2, 3), np.uint8)) == 0.0
    ramp = np.tile((np.arange(12) * 20).astype(np.uint8), (5, 1))
    assert estimate_noise(np.dstack([ramp] * 3)) == 0.0  # only strong gradients, no smooth pixels


# --------------------------------------------------------------------------------------
# Background removal
# --------------------------------------------------------------------------------------


def test_background_removed_from_white_canvas(tmp_path: Path) -> None:
    path = _save(_logo((600, 600)), tmp_path / "logo.png")
    pre = _run(path, remove_background=True)
    assert pre.background_removed
    assert pre.alpha is not None
    assert pre.alpha[0, 0] == 0 and pre.alpha[-1, -1] == 0
    assert pre.alpha[300, 200] == 255  # red disc
    assert pre.denoise.extra["bg_fraction"] > 0.5
    assert not pre.source.has_alpha


def test_background_keeps_enclosed_holes() -> None:
    rgb = np.full((60, 60, 3), 255, np.uint8)
    cv2.rectangle(rgb, (10, 10), (50, 50), (20, 40, 90), 6)  # ring enclosing a white hole
    alpha, info = remove_background(rgb, None, 10.0)
    assert alpha is not None
    assert alpha[0, 0] == 0 and alpha[30, 30] == 255
    assert info["bg_fraction"] > 0


def test_background_not_removed_when_disabled(tmp_path: Path) -> None:
    pre = _run(_save(_logo((600, 600)), tmp_path / "logo.png"))
    assert not pre.background_removed and pre.alpha is None


def test_background_skipped_for_transparent_border() -> None:
    path = SAMPLES / "07_transparent_logo.png"
    pre = _run(path, remove_background=True)
    assert not pre.background_removed
    np.testing.assert_array_equal(pre.alpha, np.asarray(Image.open(path))[..., 3])
    assert pre.denoise.extra["bg_skipped_transparent_border"] == 1.0


def test_background_not_removed_when_border_not_uniform() -> None:
    rng = np.random.default_rng(2)
    rgb = rng.integers(0, 256, (50, 50, 3), dtype=np.uint8)
    alpha, info = remove_background(rgb, None, 10.0)
    assert alpha is None and info["bg_border_share"] < pp.BG_MIN_BORDER_SHARE


def test_background_removal_with_opaque_alpha_border() -> None:
    rgb = np.full((40, 40, 3), 250, np.uint8)
    rgb[10:30, 10:30] = (220, 50, 47)
    alpha = np.full((40, 40), 255, np.uint8)
    alpha[15:20, 15:20] = 0
    new_alpha, _ = remove_background(rgb, alpha, 10.0)
    assert new_alpha is not None
    assert new_alpha[0, 0] == 0 and new_alpha[25, 25] == 255 and new_alpha[16, 16] == 0


def test_background_removal_fully_transparent_input() -> None:
    rgb = np.zeros((10, 10, 3), np.uint8)
    alpha = np.zeros((10, 10), np.uint8)
    alpha[0, :] = 255  # opaque border share < 0.5 -> treated as transparent-bordered
    assert remove_background(rgb, alpha, 10.0)[0] is None


def test_background_antialiased_fringe_gets_partial_alpha() -> None:
    rgb = np.full((80, 80, 3), 255, np.uint8)
    cv2.circle(rgb, (40, 40), 25, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    alpha, _ = remove_background(rgb, None, 10.0)
    assert alpha is not None
    partial = (alpha > 0) & (alpha < 255)
    assert partial.any()


def test_remove_background_jpeg_sample_matches_ground_truth() -> None:
    pre = _run(SAMPLES / "06_jpeg_artifacts.jpg", remove_background=True)
    assert pre.background_removed
    rgb = np.asarray(pre.image)[pre.alpha == 255]
    # no white pixels remain opaque (white = background of this sample)
    assert np.mean(np.all(rgb > 245, axis=1)) < 0.02


# --------------------------------------------------------------------------------------
# Contract consistency & misc
# --------------------------------------------------------------------------------------


def test_result_arrays_readonly_and_typed() -> None:
    pre = _run(SAMPLES / "04_text.png")
    assert pre.image.dtype == np.uint8 and not pre.image.flags.writeable
    assert pre.scale_factor == 2.0 and (pre.width, pre.height) == (1600, 600)


def test_source_has_alpha_corrected_when_inconsistent(tmp_path: Path) -> None:
    path = _save(_logo((64, 64)), tmp_path / "opaque.png")
    info = load_image(path)
    lying = info.model_copy(update={"has_alpha": True})
    pre = preprocess(lying, Settings())
    assert pre.alpha is None and pre.source.has_alpha is False


def test_size_mismatch_raises_stage_error(tmp_path: Path) -> None:
    path = _save(_logo((64, 64)), tmp_path / "x.png")
    bad = ImageInput(
        path=path, width=10, height=10, has_alpha=False, mode=ImageMode.RGB,
        source_format=SourceFormat.PNG, file_size_bytes=10,
    )
    with pytest.raises(StageError):
        preprocess(bad, Settings())


_TIMING_SCRIPT = """
import json, sys, time
from pathlib import Path
from contracts.schemas import Settings
from pipeline.preprocess import load_image, preprocess
info = load_image(Path(sys.argv[1]))
times = []
for _ in range(3):
    start = time.perf_counter()
    preprocess(info, Settings())
    times.append(time.perf_counter() - start)
print(json.dumps(times))
"""


@pytest.mark.slow
def test_preprocess_large_sample_time_budget() -> None:
    """Best-of-3 preprocess time on the 2000x2000 sample, measured in a fresh interpreter
    so the result does not depend on the memory state of the test session."""
    root = Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "-c", _TIMING_SCRIPT, str(SAMPLES / "09_large_2000.png")],
        cwd=root, capture_output=True, text=True, check=True,
    )
    best = min(json.loads(out.stdout.strip().splitlines()[-1]))
    assert best <= 1.5, best


def test_background_lab_recorded_whether_or_not_removed(tmp_path: Path) -> None:
    path = _save(_logo((600, 600)), tmp_path / "logo.png")
    for remove in (True, False):
        pre = _run(path, remove_background=remove)
        assert pre.background_lab is not None
        lum, a, b = pre.background_lab
        assert lum == pytest.approx(100.0, abs=0.5) and abs(a) < 1 and abs(b) < 1
        assert "bg_l" not in pre.denoise.extra
        assert pre.background_removed is remove


def test_background_lab_none_without_uniform_border(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    noisy = Image.fromarray(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8))
    assert _run(_save(noisy, tmp_path / "noise.png")).background_lab is None
    assert _run(SAMPLES / "07_transparent_logo.png").background_lab is None


# --------------------------------------------------------------------------------------
# Upscaling must not invent colors
# --------------------------------------------------------------------------------------


def _color_counts(rgb: np.ndarray) -> dict[str, int]:
    colors, counts = np.unique(rgb.reshape(-1, 3), axis=0, return_counts=True)
    return {"#{:02x}{:02x}{:02x}".format(*c): int(n) for c, n in zip(colors, counts, strict=True)}


@pytest.mark.parametrize("name", ["04_text.png", "08_thin_lines.png", "03_cartoon_outlined.png"])
def test_upscale_preserves_ground_truth_palette(name: str) -> None:
    """Every ground-truth color keeps (at least) its source pixel share after the 2x upscale."""
    entry = json.loads((SAMPLES / name).with_suffix(".json").read_text(encoding="utf-8"))
    src = np.asarray(Image.open(SAMPLES / name).convert("RGB"))
    pre = _run(SAMPLES / name)
    assert pre.scale_factor == 2.0
    before, after = _color_counts(src), _color_counts(np.asarray(pre.image))
    n_src, n_out = src.shape[0] * src.shape[1], pre.width * pre.height
    for color in entry["palette_hex"]:
        share_src = before[color] / n_src
        share_out = after.get(color, 0) / n_out
        assert share_out >= 0.95 * share_src, (color, before[color], after.get(color, 0))
        assert after.get(color, 0) >= 400, color  # substantial, not a handful of stray pixels


@pytest.mark.parametrize("name", ["04_text.png", "08_thin_lines.png", "10_mixed_scene.png"])
def test_upscale_creates_no_new_colors(name: str) -> None:
    src = np.asarray(Image.open(SAMPLES / name).convert("RGB"))
    pre = _run(SAMPLES / name)
    if pre.denoise.method != "none":  # 10 is denoised first; compare against the denoised colors
        assert set(_color_counts(np.asarray(pre.image))) <= set(
            _color_counts(pp.iterated_bilateral(src, int(pre.denoise.extra["iterations"]), pre.denoise.strength))
        )
    else:
        assert set(_color_counts(np.asarray(pre.image))) <= set(_color_counts(src))


def test_upscale_is_exact_pixel_duplication(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    arr = rng.integers(0, 256, (6, 8, 4), dtype=np.uint8).repeat(5, axis=0).repeat(5, axis=1)  # flat blocks
    arr[..., 3] = np.where(arr[..., 3] > 128, 255, arr[..., 3])
    arr[..., 3][arr[..., 3] == 0] = 1  # no alpha-0 pixels -> no transparent RGB refill
    pre = _run(_save(Image.fromarray(arr, "RGBA"), tmp_path / "rand.png"))
    dup = arr.repeat(2, axis=0).repeat(2, axis=1)
    assert pre.denoise.method == "none"
    np.testing.assert_array_equal(pre.alpha, dup[..., 3])
    np.testing.assert_array_equal(pre.image, dup[..., :3])
