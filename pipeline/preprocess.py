"""Stage 0/1: image loading, validation and preprocessing.

``load_image`` validates an uploaded file (magic bytes, pixel count, decodability) and
describes it. ``preprocess`` decodes it to sRGB, splits alpha, removes JPEG artifacts with an
edge-preserving (iterated bilateral) filter, optionally removes a uniform border-connected
background, and rescales into *processing space* (see ``contracts/schemas.py``).

Processing order (chosen so expensive filters never run on more than ~4 MP):

1. decode + ICC -> sRGB + EXIF orientation -> ``rgb`` (H, W, 3) uint8, ``alpha`` or None
2. fill RGB under fully transparent pixels from the nearest opaque pixel
3. downscale (INTER_AREA) if the long side exceeds ``MAX_LONG_SIDE``
4. denoise: JPEG -> iterated bilateral; noisy PNG -> light bilateral; clean PNG -> untouched
5. background detection (-> ``background_lab``) and removal (``settings.remove_background``)
6. 2x upscale (nearest-neighbour, so no new colors) when the smaller side is below ``UPSCALE_BELOW``
7. re-fill RGB under transparent pixels (resampling blends across alpha borders)
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, UnidentifiedImageError

from contracts.api import MAX_PIXELS
from contracts.schemas import (
    DenoiseParams,
    DetailLevel,
    ImageInput,
    ImageMode,
    InvalidImageError,
    PreprocessResult,
    Settings,
    SourceFormat,
    StageError,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

MAX_LONG_SIDE = 2048
"""Processing-space long side cap (time budget)."""
UPSCALE_BELOW = 500
"""Images whose smaller side is below this are upscaled 2x (subject to MAX_LONG_SIDE)."""

JPEG_BILATERAL: dict[DetailLevel, tuple[int, float]] = {
    DetailLevel.LOW: (4, 60.0),
    DetailLevel.MEDIUM: (3, 50.0),
    DetailLevel.HIGH: (3, 40.0),
}
"""detail_level -> (iterations, sigmaColor) for JPEG deblocking. Lower detail = stronger smoothing."""
BILATERAL_DIAMETER = 9
BILATERAL_SIGMA_SPACE = 5.0

PNG_NOISE_THRESHOLD = 0.6
"""Estimated noise sigma (8-bit levels) above which a PNG gets a light bilateral pass."""
PNG_NOISE_GAIN: dict[DetailLevel, float] = {DetailLevel.LOW: 5.0, DetailLevel.MEDIUM: 4.0, DetailLevel.HIGH: 3.0}

BG_TOLERANCE: dict[DetailLevel, float] = {DetailLevel.LOW: 14.0, DetailLevel.MEDIUM: 10.0, DetailLevel.HIGH: 7.0}
"""CIE76 Delta-E flood-fill tolerance for background removal."""
BG_MIN_BORDER_SHARE = 0.5
"""A background exists only if at least this share of border pixels matches one color."""

_EXIF_ORIENTATION_TAG = 0x0112
_ORIENTATION_OPS: dict[int, Image.Transpose] = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
_MODE_MAP: dict[str, ImageMode] = {
    "1": ImageMode.L,
    "L": ImageMode.L,
    "I": ImageMode.L,
    "I;16": ImageMode.L,
    "I;16L": ImageMode.L,
    "I;16B": ImageMode.L,
    "I;16N": ImageMode.L,
    "F": ImageMode.L,
    "LA": ImageMode.LA,
    "La": ImageMode.LA,
    "P": ImageMode.P,
    "PA": ImageMode.P,
    "RGB": ImageMode.RGB,
    "RGBX": ImageMode.RGB,
    "YCbCr": ImageMode.RGB,
    "LAB": ImageMode.RGB,
    "HSV": ImageMode.RGB,
    "RGBA": ImageMode.RGBA,
    "RGBa": ImageMode.RGBA,
    "CMYK": ImageMode.CMYK,
}
_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


@dataclass(frozen=True)
class DecodedImage:
    """Decoded, oriented, sRGB pixels of a source file."""

    rgb: np.ndarray
    """(H, W, 3) uint8 sRGB."""
    alpha: np.ndarray | None
    """(H, W) uint8, or None if the file has no alpha channel / transparency."""
    raw_mode: str
    source_format: SourceFormat


# --------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------


def sniff_format(header: bytes) -> SourceFormat | None:
    """Return the format identified by the file's magic bytes, or None if unsupported."""
    if header.startswith(PNG_MAGIC):
        return SourceFormat.PNG
    if header.startswith(JPEG_MAGIC):
        return SourceFormat.JPEG
    return None


def _to_uint8_gray(img: Image.Image) -> Image.Image:
    """Convert high-bit-depth single-channel modes (I;16*, I, F) to 8-bit L."""
    arr = np.nan_to_num(np.asarray(img).astype(np.float64))
    if img.mode == "F" and float(arr.max(initial=0.0)) <= 1.0:
        arr = arr * 255.0  # normalized float data
    elif img.mode.startswith("I;16") or float(arr.max(initial=0.0)) > 255.0:
        arr = arr / 257.0  # 16-bit range -> 8-bit
    return Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "L")


def _apply_icc(img: Image.Image, icc: bytes | None) -> Image.Image:
    """Convert an L/RGB/CMYK image to sRGB RGB, honouring an embedded ICC profile when possible."""
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            out = ImageCms.profileToProfile(img, src, _SRGB_PROFILE, outputMode="RGB")
            if out is not None:
                return out
        except (ImageCms.PyCMSError, OSError, ValueError, TypeError):
            pass  # malformed or mismatched profile: fall back to Pillow's naive conversion
    return img.convert("RGB")


def _normalize(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalize any Pillow mode to (sRGB uint8 RGB, uint8 alpha or None)."""
    icc = img.info.get("icc_profile")
    mode = img.mode
    alpha: Image.Image | None = None
    if mode in ("I", "F") or mode.startswith("I;16"):
        base = _to_uint8_gray(img)
    elif mode == "1":
        base = img.convert("L")
    elif mode in ("P", "PA"):
        has_transparency = mode == "PA" or "transparency" in img.info or img.palette.mode == "RGBA"
        converted = img.convert("RGBA" if has_transparency else "RGB")
        if has_transparency:
            alpha = converted.getchannel("A")
            converted = converted.convert("RGB")
        base, icc = converted, None
    elif mode in ("LA", "La", "RGBA", "RGBa"):
        target = "LA" if mode.startswith("L") else "RGBA"
        converted = img.convert(target) if mode != target else img
        alpha = converted.getchannel("A")
        base = converted.convert(target[:-1])
    elif mode == "RGBX":
        base = img.convert("RGB")
    elif mode in ("L", "RGB", "CMYK"):
        base = img
        if mode == "L" and "transparency" in img.info:
            alpha = img.convert("LA").getchannel("A")
        elif mode == "RGB" and "transparency" in img.info:
            alpha = img.convert("RGBA").getchannel("A")
    else:
        base = img.convert("RGB")
    rgb = _apply_icc(base, icc if isinstance(icc, bytes) else None)
    rgb_arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
    alpha_arr = None if alpha is None else np.ascontiguousarray(np.asarray(alpha, dtype=np.uint8))
    return rgb_arr, alpha_arr


def _orient(arr: np.ndarray, orientation: int) -> np.ndarray:
    """Apply an EXIF orientation (1..8) to an (H, W[, C]) array."""
    op = _ORIENTATION_OPS.get(orientation)
    if op is None:
        return arr
    return np.ascontiguousarray(np.asarray(Image.fromarray(arr).transpose(op)))


def decode_image(path: Path) -> DecodedImage:
    """Validate and fully decode ``path``. Raises InvalidImageError on any problem."""
    path = Path(path)
    try:
        with path.open("rb") as fh:
            header = fh.read(16)
    except OSError as exc:
        raise InvalidImageError(f"cannot read {path.name}: {exc}") from exc
    fmt = sniff_format(header)
    if fmt is None:
        raise InvalidImageError(f"{path.name}: not a PNG or JPEG file (magic bytes {header[:8].hex()})")
    try:
        with Image.open(path) as img:
            if img.format not in ("PNG", "JPEG", "MPO"):
                raise InvalidImageError(f"{path.name}: decoded as {img.format}, expected PNG/JPEG")
            width, height = img.size
            if width <= 0 or height <= 0:
                raise InvalidImageError(f"{path.name}: empty image")
            if width * height > MAX_PIXELS:
                raise InvalidImageError(f"{path.name}: {width}x{height} exceeds the {MAX_PIXELS} pixel limit")
            if img.mode not in _MODE_MAP:
                raise InvalidImageError(f"{path.name}: unsupported pixel mode {img.mode}")
            img.load()
            raw_mode = img.mode
            orientation = int(img.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
            rgb, alpha = _normalize(img)
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError(f"{path.name}: corrupt or truncated image ({exc})") from exc
    rgb = _orient(rgb, orientation)
    alpha = None if alpha is None else _orient(alpha, orientation)
    return DecodedImage(rgb=rgb, alpha=alpha, raw_mode=raw_mode, source_format=fmt)


def load_image(path: Path) -> ImageInput:
    """Validate (magic bytes, size, decodability) and describe an uploaded file.

    Dimensions are reported after EXIF orientation. ``has_alpha`` is True only if some pixel
    has alpha < 255. Raises InvalidImageError.
    """
    path = Path(path)
    if not path.is_file():
        raise InvalidImageError(f"file not found: {path}")
    size = path.stat().st_size
    if size == 0:
        raise InvalidImageError(f"{path.name}: file is empty")
    decoded = decode_image(path)
    h, w = decoded.rgb.shape[:2]
    return ImageInput(
        path=path,
        width=w,
        height=h,
        has_alpha=decoded.alpha is not None and int(decoded.alpha.min()) < 255,
        mode=_MODE_MAP[decoded.raw_mode],
        source_format=decoded.source_format,
        file_size_bytes=size,
    )


# --------------------------------------------------------------------------------------
# Pixel operations
# --------------------------------------------------------------------------------------


def fill_transparent_rgb(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Return a copy of ``rgb`` whose fully transparent pixels take the nearest opaque pixel's color.

    Prevents downstream filters/tracers from seeing arbitrary (often black) RGB under alpha=0,
    which would invent edges along alpha borders. Images with no or only transparent pixels
    are returned unchanged.
    """
    if alpha is None:
        return rgb
    transparent = alpha == 0
    if not transparent.any() or transparent.all():
        return rgb
    # Labels index the zero (opaque) pixels of the source in raster order.
    _, labels = cv2.distanceTransformWithLabels(
        transparent.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5, labelType=cv2.DIST_LABEL_PIXEL
    )
    opaque_flat = np.flatnonzero(~transparent.ravel())
    out = rgb.copy()
    src_idx = opaque_flat[labels[transparent] - 1]
    out[transparent] = rgb.reshape(-1, 3)[src_idx]
    return out


def estimate_noise(rgb: np.ndarray) -> float:
    """Estimate Gaussian noise sigma (8-bit levels) on non-edge pixels (Immerkaer's method).

    Flat vector-style art returns ~0; photographic/noisy sources return > 1.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if min(gray.shape) < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    lap = np.abs(cv2.filter2D(gray, -1, kernel))[1:-1, 1:-1]
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)[1:-1, 1:-1]
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)[1:-1, 1:-1]
    smooth = (np.abs(gx) + np.abs(gy)) < 48.0
    if not smooth.any():
        return 0.0
    return float(math.sqrt(math.pi / 2.0) / 6.0 * lap[smooth].mean())


def iterated_bilateral(rgb: np.ndarray, iterations: int, sigma_color: float) -> np.ndarray:
    """Edge-preserving smoothing: ``iterations`` passes of a 9-px bilateral filter."""
    out = rgb
    for _ in range(iterations):
        out = cv2.bilateralFilter(out, BILATERAL_DIAMETER, sigma_color, BILATERAL_SIGMA_SPACE)
    return out


def _denoise(
    rgb: np.ndarray, fmt: SourceFormat, detail: DetailLevel
) -> tuple[np.ndarray, DenoiseParams]:
    """Pick and apply a denoiser; return the result and a record of what was done."""
    noise = estimate_noise(rgb)
    base_extra = {"noise_sigma": round(noise, 4)}
    if fmt is SourceFormat.JPEG:
        iterations, sigma_color = JPEG_BILATERAL[detail]
        out = iterated_bilateral(rgb, iterations, sigma_color)
        return out, DenoiseParams(
            method="bilateral",
            strength=sigma_color,
            jpeg_deblock=True,
            extra={
                **base_extra,
                "iterations": float(iterations),
                "diameter": float(BILATERAL_DIAMETER),
                "sigma_space": BILATERAL_SIGMA_SPACE,
            },
        )
    if noise > PNG_NOISE_THRESHOLD:
        sigma_color = float(np.clip(PNG_NOISE_GAIN[detail] * noise, 8.0, 40.0))
        iterations = 2 if noise > 3.0 else 1
        out = iterated_bilateral(rgb, iterations, sigma_color)
        return out, DenoiseParams(
            method="bilateral",
            strength=sigma_color,
            extra={
                **base_extra,
                "iterations": float(iterations),
                "diameter": float(BILATERAL_DIAMETER),
                "sigma_space": BILATERAL_SIGMA_SPACE,
            },
        )
    return rgb, DenoiseParams(method="none", strength=0.0, extra=base_extra)


def _border_mask(shape: tuple[int, int], width: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:width, :] = mask[-width:, :] = True
    mask[:, :width] = mask[:, -width:] = True
    return mask


def _to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> float32 CIELAB (L in 0..100)."""
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def detect_background(
    rgb: np.ndarray, alpha: np.ndarray | None, tolerance: float
) -> tuple[tuple[float, float, float] | None, dict[str, float]]:
    """Detect a uniform border background color.

    The candidate is the median CIELAB color of the most common coarse (4 Delta-E) bin among
    opaque border pixels. It counts as a background only if at least ``BG_MIN_BORDER_SHARE``
    of those border pixels are within ``tolerance`` (CIE76) of it. Returns (LAB or None, info);
    None when the border is mostly transparent or not uniform.
    """
    h, w = rgb.shape[:2]
    border = _border_mask((h, w))
    info: dict[str, float] = {}
    if alpha is not None:
        if float(np.mean(alpha[border] < 128)) > 0.5:
            info["bg_skipped_transparent_border"] = 1.0
            return None, info
        border &= alpha >= 128
    seeds = _to_lab(rgb[border][None, :, :])[0]
    coarse = np.floor(seeds / 4.0).astype(np.int32)
    _, inverse, counts = np.unique(coarse, axis=0, return_inverse=True, return_counts=True)
    bg_lab = np.median(seeds[inverse.ravel() == int(np.argmax(counts))], axis=0)
    share = float(np.mean(np.linalg.norm(seeds - bg_lab, axis=1) < tolerance))
    info["bg_border_share"] = round(share, 4)
    if share < BG_MIN_BORDER_SHARE:
        return None, info
    lab = (float(bg_lab[0]), float(bg_lab[1]), float(bg_lab[2]))
    return tuple(round(v, 3) for v in lab), info  # type: ignore[return-value]


def remove_background(
    rgb: np.ndarray, alpha: np.ndarray | None, tolerance: float
) -> tuple[np.ndarray | None, dict[str, float]]:
    """Make a uniform, border-connected background transparent.

    The background color comes from :func:`detect_background`. Pixels within ``tolerance``
    (CIE76 Delta-E) of it that are 4-connected to the border become alpha 0; the anti-aliased
    fringe just outside the region gets a proportional alpha ramp. Returns (new alpha, info) or
    (None, info) when nothing was removed (no uniform background, or the border is already
    transparent).
    """
    bg_lab, info = detect_background(rgb, alpha, tolerance)
    info["bg_fraction"] = 0.0
    if bg_lab is None:
        return None, info
    h, w = rgb.shape[:2]
    border = _border_mask((h, w))
    opaque = np.ones((h, w), dtype=bool) if alpha is None else alpha >= 128
    delta = np.linalg.norm(_to_lab(rgb) - np.asarray(bg_lab, np.float32), axis=2)
    close = ((delta < tolerance) & opaque).astype(np.uint8)
    _, labels = cv2.connectedComponents(close, connectivity=4)
    border_labels = np.unique(labels[border & (close > 0)])
    background = np.isin(labels, border_labels[border_labels > 0])
    if not background.any():
        return None, info
    new_alpha = np.full((h, w), 255, dtype=np.uint8) if alpha is None else alpha.copy()
    new_alpha[background] = 0
    fringe = cv2.dilate(background.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~background & opaque
    ramp = np.clip((delta[fringe] - tolerance) / tolerance, 0.0, 1.0)
    new_alpha[fringe] = np.minimum(new_alpha[fringe], np.round(ramp * 255.0)).astype(np.uint8)
    info["bg_fraction"] = round(float(background.mean()), 4)
    return new_alpha, info


def target_scale(width: int, height: int) -> float:
    """Processing scale: 2x if the smaller side < UPSCALE_BELOW, capped so the long side <= MAX_LONG_SIDE."""
    scale = 2.0 if min(width, height) < UPSCALE_BELOW else 1.0
    return min(scale, MAX_LONG_SIDE / max(width, height))


def _resize(arr: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
    return cv2.resize(arr, size, interpolation=interpolation)


# --------------------------------------------------------------------------------------
# Stage entrypoint
# --------------------------------------------------------------------------------------


def preprocess(image: ImageInput, settings: Settings) -> PreprocessResult:
    """Decode, convert to sRGB, split alpha, denoise/deblock, optionally rescale,
    and (if settings.remove_background) make the background transparent."""
    decoded = decode_image(image.path)
    rgb, alpha = decoded.rgb, decoded.alpha
    h, w = rgb.shape[:2]
    if (w, h) != (image.width, image.height):
        raise StageError("preprocess", f"decoded size {w}x{h} != ImageInput {image.width}x{image.height}")
    if alpha is not None and int(alpha.min()) == 255:
        alpha = None
    source = image
    if source.has_alpha != (alpha is not None):
        source = image.model_copy(update={"has_alpha": alpha is not None})

    rgb = fill_transparent_rgb(rgb, alpha)

    scale = target_scale(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if scale < 1.0:
        rgb = _resize(rgb, new_size, cv2.INTER_AREA)
        alpha = None if alpha is None else _resize(alpha, new_size, cv2.INTER_AREA)

    rgb, denoise = _denoise(rgb, decoded.source_format, settings.detail_level)
    extra = dict(denoise.extra)
    extra["scale_factor"] = scale

    tolerance = BG_TOLERANCE[settings.detail_level]
    background_lab, bg_info = detect_background(rgb, alpha, tolerance)
    extra.update(bg_info)
    background_removed = False
    if settings.remove_background:
        new_alpha, bg_info = remove_background(rgb, alpha, tolerance)
        extra.update(bg_info)
        if new_alpha is not None:
            alpha, background_removed = new_alpha, True

    if scale > 1.0:
        # Nearest-neighbour (pixel duplication): the upscale must not invent colors. Bilinear
        # blends 1-3 px strokes/glyphs with the background so exact palette colors vanish and
        # the quantizer learns AA blends (08: #268bd2 2556 px -> 0). Duplication keeps every
        # source color and alpha value exactly; edge smoothing is the vectorizer's job.
        rgb = _resize(rgb, new_size, cv2.INTER_NEAREST_EXACT)
        alpha = None if alpha is None else _resize(alpha, new_size, cv2.INTER_NEAREST_EXACT)

    rgb = fill_transparent_rgb(rgb, alpha)
    return PreprocessResult(
        source=source,
        image=np.ascontiguousarray(rgb, dtype=np.uint8),
        alpha=None if alpha is None else np.ascontiguousarray(alpha, dtype=np.uint8),
        background_removed=background_removed,
        background_lab=background_lab,
        scale_factor=scale,
        denoise=denoise.model_copy(update={"extra": extra}),
    )
