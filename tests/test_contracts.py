"""Contract tests: every invariant enforced by contracts/schemas.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from contracts import fixtures as fx
from contracts.api import ConfigResponse, JobResponse, JobStatus, Limits, PipelineStage
from contracts.schemas import (
    DETAIL_PRESETS,
    DetailLevel,
    ExportBundle,
    ImageClassLabel,
    LineMap,
    MetricCheck,
    MetricName,
    OutputFormat,
    Palette,
    PaletteColor,
    PreprocessResult,
    QualityReport,
    QualityThresholds,
    Settings,
    StageError,
    VectorDocument,
    VectorLayer,
    layer_name,
)

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ settings


def test_settings_defaults_and_svg_always_included() -> None:
    s = Settings()
    assert s.output_formats == list(OutputFormat)
    s2 = Settings(output_formats=[OutputFormat.PNG, OutputFormat.EPS, OutputFormat.PNG])
    assert s2.output_formats == [OutputFormat.SVG, OutputFormat.EPS, OutputFormat.PNG]
    assert s.preset == DETAIL_PRESETS[DetailLevel.MEDIUM]


@pytest.mark.parametrize("field,value", [("max_colors", 1), ("max_colors", 65), ("smoothing", 101), ("mode", "photo")])
def test_settings_rejects_out_of_range(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field: value})


def test_palette_override_deduped_and_validated() -> None:
    assert Settings(palette_override=["#ffffff", "#000000", "#ffffff"]).palette_override == ["#ffffff", "#000000"]
    for bad in ([], ["#FFF"], ["#GGGGGG"]):
        with pytest.raises(ValidationError):
            Settings(palette_override=bad)


def test_settings_json_roundtrip() -> None:
    s = Settings(max_colors=8, remove_background=True)
    assert Settings.model_validate_json(s.model_dump_json()) == s


def test_detail_presets_cover_all_levels_and_are_monotonic() -> None:
    assert set(DETAIL_PRESETS) == set(DetailLevel)
    lo, med, hi = (DETAIL_PRESETS[d] for d in (DetailLevel.LOW, DetailLevel.MEDIUM, DetailLevel.HIGH))
    assert lo.speckle_min_area_px > med.speckle_min_area_px > hi.speckle_min_area_px
    assert lo.simplify_tolerance_px > med.simplify_tolerance_px > hi.simplify_tolerance_px


# ------------------------------------------------------------------ preprocess


def test_preprocess_fixture_valid_and_arrays_readonly() -> None:
    pre = fx.make_preprocess_result()
    assert pre.width == pre.height == 64
    assert pre.opaque_mask.all()
    with pytest.raises(ValueError):
        pre.image[0, 0, 0] = 1


def test_preprocess_does_not_freeze_callers_array() -> None:
    img = np.zeros((8, 8, 3), np.uint8)
    PreprocessResult(
        source=fx.make_image_input(8),
        image=img,
        alpha=None,
        scale_factor=1.0,
        denoise=fx.make_preprocess_result().denoise,
    )
    img[0, 0, 0] = 5  # caller keeps a writeable array


def test_preprocess_alpha_fixture() -> None:
    pre = fx.make_preprocess_result(has_alpha=True)
    assert not pre.opaque_mask[0].any() and pre.opaque_mask[-1].all()


@pytest.mark.parametrize(
    "image,alpha,scale,has_alpha",
    [
        (np.zeros((64, 64, 3), np.float32), None, 1.0, False),  # wrong dtype
        (np.zeros((64, 64), np.uint8), None, 1.0, False),  # wrong ndim
        (np.zeros((64, 64, 4), np.uint8), None, 1.0, False),  # wrong channels
        (np.zeros((64, 64, 3), np.uint8), np.zeros((32, 32), np.uint8), 1.0, True),  # alpha shape
        (np.zeros((64, 64, 3), np.uint8), None, 1.0, True),  # has_alpha but no alpha
        (np.zeros((64, 64, 3), np.uint8), None, 2.0, False),  # scale mismatch
    ],
)
def test_preprocess_rejects_inconsistent(
    image: np.ndarray, alpha: np.ndarray | None, scale: float, has_alpha: bool
) -> None:
    with pytest.raises(ValidationError):
        PreprocessResult(
            source=fx.make_image_input(64, has_alpha),
            image=image,
            alpha=alpha,
            scale_factor=scale,
            denoise=fx.make_preprocess_result().denoise,
        )


def test_preprocess_background_removed_requires_alpha() -> None:
    img = np.zeros((8, 8, 3), np.uint8)
    denoise = fx.make_preprocess_result().denoise
    PreprocessResult(
        source=fx.make_image_input(8),
        image=img,
        alpha=np.zeros((8, 8), np.uint8),
        scale_factor=1.0,
        denoise=denoise,
        background_removed=True,
    )
    with pytest.raises(ValidationError):
        PreprocessResult(
            source=fx.make_image_input(8),
            image=img,
            alpha=None,
            scale_factor=1.0,
            denoise=denoise,
            background_removed=True,
        )


def test_preprocess_accepts_downscale() -> None:
    PreprocessResult(
        source=fx.make_image_input(64),
        image=np.zeros((32, 32, 3), np.uint8),
        alpha=None,
        scale_factor=0.5,
        denoise=fx.make_preprocess_result().denoise,
    )


# ------------------------------------------------------------------ palette


def test_palette_fixture_and_background() -> None:
    pal = fx.make_palette()
    assert pal.background_index == 0
    assert sum(c.pixel_count for c in pal.colors) == 64 * 64


def test_palette_transparent_pixels_excluded() -> None:
    pal = fx.make_palette(fx.make_preprocess_result(has_alpha=True))
    assert (pal.label_map == -1).sum() == 4 * 64
    assert sum(c.pixel_count for c in pal.colors) == 60 * 64


def test_palette_color_hex_must_match_rgb() -> None:
    with pytest.raises(ValidationError):
        PaletteColor(index=0, rgb=(1, 2, 3), lab=(0, 0, 0), hex="#000000", pixel_count=0)
    with pytest.raises(ValidationError):
        PaletteColor(index=0, rgb=(255, 0, 0), lab=(0, 0, 0), hex="#FF0000", pixel_count=0)


def test_palette_rejects_bad_counts_and_labels() -> None:
    pal = fx.make_palette()
    bad_counts = [c.model_copy(update={"pixel_count": c.pixel_count + 1}) for c in pal.colors]
    with pytest.raises(ValidationError, match="pixel_count"):
        Palette(colors=bad_counts, label_map=pal.label_map)
    labels = pal.label_map.copy()
    labels[0, 0] = 7
    with pytest.raises(ValidationError, match="label_map values"):
        Palette(colors=pal.colors, label_map=labels)


# ------------------------------------------------------------------ lines


def test_line_map_invariants() -> None:
    lm = fx.make_line_map()
    assert lm.skeleton.sum() > 0
    mask = np.zeros((8, 8), bool)
    skel = np.zeros((8, 8), bool)
    skel[1, 1] = True
    with pytest.raises(ValidationError, match="subset"):
        LineMap(
            mask=mask, skeleton=skel, width_map=np.zeros((8, 8), np.float32), median_stroke_width=1, color_rgb=(0, 0, 0)
        )
    mask[1, 1] = True
    wm = np.ones((8, 8), np.float32)
    with pytest.raises(ValidationError, match="width_map"):
        LineMap(mask=mask, skeleton=skel, width_map=wm, median_stroke_width=1, color_rgb=(0, 0, 0))


# ------------------------------------------------------------------ vector document


def test_vector_document_fixture() -> None:
    doc = fx.make_vector_document()
    assert doc.view_box == (0.0, 0.0, 64.0, 64.0)
    assert VectorDocument.model_validate_json(doc.model_dump_json(exclude={"view_box"})) == doc


def _layer(**kw: object) -> VectorLayer:
    base: dict[str, object] = dict(id="l0", name="L", role="fill", color_hex="#112233", paths=["M0 0L1 1Z"], z_order=0)
    base.update(kw)
    return VectorLayer(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kw",
    [
        {"id": "1bad"},
        {"color_hex": "red"},
        {"paths": []},
        {"paths": ["L0 0"]},
        {"paths": ["M0 0 <script>"]},
        {"paths": ["m0 0l1 1z"]},  # relative commands
        {"paths": ["M0 0Q1 1 2 2"]},  # quadratic not allowed
        {"paths": ["M0 0A1 1 0 0 1 2 2"]},  # arcs not allowed
        {"is_stroke": True},
        {"stroke_width": 2.0},
    ],
)
def test_vector_layer_rejects(kw: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _layer(**kw)


def test_layer_name_convention() -> None:
    assert layer_name("fill", 1, "#e53935") == ("color_1_E53935", "color_1_#E53935")
    assert layer_name("line", 4, "#000000") == ("line_4_000000", "line_4_#000000")
    with pytest.raises(ValueError):
        layer_name("fill", 1, "red")


def test_vector_document_rejects_bad_order_and_duplicate_ids() -> None:
    doc = fx.make_vector_document()
    with pytest.raises(ValidationError, match="z_order"):
        VectorDocument(width=64, height=64, layers=list(reversed(doc.layers)), metadata=doc.metadata)
    dup = [doc.layers[0], doc.layers[1].model_copy(update={"id": "color_1_FFFFFF"})]
    with pytest.raises(ValidationError, match="unique"):
        VectorDocument(width=64, height=64, layers=dup, metadata=doc.metadata)


# ------------------------------------------------------------------ export / quality


def test_export_bundle_requires_existing_files(tmp_path: Path) -> None:
    svg, png = tmp_path / "output.svg", tmp_path / "preview.png"
    with pytest.raises(ValidationError, match="does not exist"):
        ExportBundle(svg_path=svg, preview_png_path=png)
    svg.write_text("<svg/>")
    png.write_bytes(b"\x89PNG")
    assert ExportBundle(svg_path=svg, preview_png_path=png).ai_path is None


def test_metric_check_consistency_and_report_passed() -> None:
    ok = MetricCheck(name=MetricName.SSIM, value=0.95, threshold=0.9, comparator=">=", passed=True)
    bad = MetricCheck(name=MetricName.MAX_DELTA_E, value=3.5, threshold=3.0, comparator="<", passed=False)
    with pytest.raises(ValidationError, match="contradicts"):
        MetricCheck(name=MetricName.SSIM, value=0.5, threshold=0.9, comparator=">=", passed=True)
    kw = dict(
        ssim=0.95,
        gap_ratio=0.0,
        mean_delta_e=1.0,
        max_delta_e=3.5,
        node_count=10,
        file_size_bytes=100,
        processing_time_s=0.5,
    )
    assert QualityReport(checks=[ok], **kw).passed
    report = QualityReport(checks=[ok, bad], **kw)
    assert not report.passed and report.model_dump()["passed"] is False


def test_thresholds() -> None:
    assert QualityThresholds.SSIM_MIN[ImageClassLabel.FLAT_COLOR] == 0.90
    assert QualityThresholds.SSIM_MIN[ImageClassLabel.LINE_ART] == 0.85
    assert QualityThresholds.MAX_MEAN_DELTA_E == 2.0 and QualityThresholds.MAX_DELTA_E == 3.0
    assert QualityThresholds.MIN_ALPHA_IOU == 0.98
    assert QualityThresholds.time_budget_s(2000, 2000) == 10.0
    assert QualityThresholds.time_budget_s(10, 10) == 2.0


def test_stage_error_message() -> None:
    err = StageError("quantize", "boom")
    assert err.stage == "quantize" and "[quantize] boom" in str(err) and err.code == "stage_failed"


# ------------------------------------------------------------------ api


def test_job_response_roundtrip() -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    job = JobResponse(
        job_id="00000000-0000-0000-0000-000000000000",
        status=JobStatus.QUEUED,
        stage=PipelineStage.UPLOAD,
        progress=0.0,
        created_at=now,
        updated_at=now,
        expires_at=now,
        filename="a.png",
        settings=Settings(),
    )
    assert JobResponse.model_validate_json(job.model_dump_json()) == job
    assert ConfigResponse(defaults=Settings(), limits=Limits()).limits.max_upload_bytes == 20 * 1024 * 1024


def test_openapi_is_up_to_date() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_openapi.py"), "--check"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_stage_entrypoints_cover_every_protocol() -> None:
    from contracts import stages

    assert set(stages.STAGE_ENTRYPOINTS) == {
        "load_image",
        "preprocess",
        "classify",
        "quantize",
        "extract_lines",
        "vectorize",
        "assemble_svg",
        "export",
        "evaluate",
    }
    for target in stages.STAGE_ENTRYPOINTS.values():
        module, func = target.split(":")
        assert module.split(".")[0] in {"pipeline", "eval"} and func.isidentifier()
