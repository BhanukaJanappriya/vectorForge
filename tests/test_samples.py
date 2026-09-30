"""The sample set is present, matches its ground-truth JSON, and ground-truth palettes are real."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
MANIFEST = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SAMPLES.glob("*.json"))]


def test_ten_samples() -> None:
    assert len(MANIFEST) == 10
    assert {s["expected_class"] for s in MANIFEST} == {"flat_color", "line_art", "mixed"}


@pytest.mark.parametrize("entry", MANIFEST, ids=[s["file"] for s in MANIFEST])
def test_sample_matches_ground_truth(entry: dict) -> None:
    with Image.open(SAMPLES / entry["file"]) as img:
        assert (img.width, img.height) == (entry["width"], entry["height"])
        assert ("A" in img.getbands()) == entry["has_alpha"]
        if entry["palette_hex"] and not entry["file"].endswith(".jpg"):
            rgb = np.asarray(img.convert("RGB")).reshape(-1, 3)
            present = {"#{:02x}{:02x}{:02x}".format(*c) for c in np.unique(rgb, axis=0)}
            missing = set(entry["palette_hex"]) - present
            assert not missing, f"ground-truth colors never appear exactly: {missing}"
