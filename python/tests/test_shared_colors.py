"""Tests for generated Python shared colormap and theme definitions."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from caliper import _colors_gen, _convert

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "shared_colors.py"

EXPECTED_NAMES = (
    "viridis", "magma", "rdbu", "greys", "cividis", "inferno", "plasma", "turbo",
    "twilight", "coolwarm", "batlow", "vik", "roma", "oslo", "lajolla", "berlin",
)


def test_colormap_names_ids_and_live_subset():
    assert _colors_gen.COLORMAP_NAMES == EXPECTED_NAMES
    assert _colors_gen.COLORMAP_IDS == {n: i for i, n in enumerate(EXPECTED_NAMES)}
    assert _colors_gen.LIVE_COLORMAPS == {"viridis": 0, "magma": 1, "rdbu": 2}
    assert _convert.CMAPS == _colors_gen.LIVE_COLORMAPS
    for name, idx in _colors_gen.LIVE_COLORMAPS.items():
        assert _convert.cmap_index(name) == idx
        assert _convert.cmap_index(name.upper()) == idx
        assert _convert.cmap_index(idx) == idx
    with pytest.raises(ValueError):
        _convert.cmap_index("cividis")
    with pytest.raises(ValueError):
        _convert.cmap_index(3)


def test_colormap_luts_structure_and_samples():
    assert tuple(_colors_gen.COLORMAP_LUTS.keys()) == EXPECTED_NAMES
    for name, lut in _colors_gen.COLORMAP_LUTS.items():
        assert len(lut) == 256, name
        assert all((c >> 24) == 0xFF for c in lut), name
    greys = _colors_gen.COLORMAP_LUTS["greys"]
    assert greys == tuple(0xFF000000 | (i << 16) | (i << 8) | i for i in range(256))
    # Spot check known matplotlib / cmcrameri entries matching tests/test_gfx_layer.cpp.
    assert _colors_gen.COLORMAP_LUTS["cividis"][0] == 0xFF4E2200
    assert _colors_gen.COLORMAP_LUTS["cividis"][255] == 0xFF38E8FE
    assert _colors_gen.COLORMAP_LUTS["berlin"][0] == 0xFFFFB09E
    assert _colors_gen.COLORMAP_LUTS["berlin"][255] == 0xFFADADFF
    # Spot check RdBu endpoints from ColorBrewer 11-class anchors.
    assert _colors_gen.COLORMAP_LUTS["rdbu"][0] == (0xFF000000 | (31 << 16) | (0 << 8) | 103)
    assert _colors_gen.COLORMAP_LUTS["rdbu"][255] == (0xFF000000 | (97 << 16) | (48 << 8) | 5)


def test_theme_tokens_and_categorical_palette():
    assert len(_colors_gen.THEME_TOKENS) == 16
    assert _colors_gen.THEME_TOKENS["ground"] == (24, 23, 21)
    assert _colors_gen.THEME_TOKENS["bronze"] == (179, 139, 89)
    assert _colors_gen.HOUSE_PIGMENTS == (
        _colors_gen.THEME_TOKENS["vehicle"],
        _colors_gen.THEME_TOKENS["dog"],
        _colors_gen.THEME_TOKENS["bird"],
        _colors_gen.THEME_TOKENS["cat"],
        _colors_gen.THEME_TOKENS["other"],
    )
    assert len(_colors_gen.UNTOKENIZED_THEME_COLORS) == 20
    assert _colors_gen.UNTOKENIZED_THEME_COLORS["flat_point"] == (220, 220, 230, 255)
    assert len(_colors_gen.CATEGORICAL_PALETTE) == 16


def test_scene_uses_generated_colors():
    pytest.importorskip("numpy")
    from caliper import _scene

    assert _scene._CMAPS == _colors_gen.LIVE_COLORMAPS
    assert _scene.PALETTE == _colors_gen.CATEGORICAL_PALETTE


def test_shared_colors_cli_passes_and_detects_drift(tmp_path):
    ok = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(REPO), "--check"],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr

    # Copy managed files into tmp_path and drift one copy.
    sys.path.insert(0, str(REPO / "scripts"))
    import shared_colors  # noqa: E402

    rels = [
        shared_colors.SOURCE_REL,
        shared_colors.STERNBILD_CMAKE_REL,
        *(rel for rel, _ in shared_colors.FULL_TARGETS),
        *(rel for rel, _, _, _ in shared_colors.BLOCK_TARGETS),
    ]
    for rel in rels:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, dst)

    drifted = tmp_path / "web/src/render/colormaps.gen.ts"
    drifted.write_text(drifted.read_text(encoding="utf-8") + "// drift\n", encoding="utf-8")

    bad = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(tmp_path), "--check"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert "web/src/render/colormaps.gen.ts" in bad.stderr
