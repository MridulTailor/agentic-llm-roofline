"""Unit tests for roofline model calculation and plotting."""

from __future__ import annotations

from pathlib import Path

import pytest

from model import HardwareConfig, ModelConfig, RuntimeConfig, estimate
from roofline import (
    plot_roofline,
    points_from_estimate,
    roofline_bound,
    save_points_csv,
)


def _estimate():
    model = ModelConfig(
        n_layers=2,
        hidden_size=32,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        vocab_size=128,
        name="plot-test",
    )
    runtime = RuntimeConfig(batch_size=2, seq_len=32, gen_len=4, weight_dtype="fp16")
    hardware = HardwareConfig(
        name="nominal-test",
        peak_flops=100e12,
        memory_bandwidth=2e12,
        memory_capacity=80e9,
    )
    return estimate(model, runtime, hardware)


def test_roofline_bound_is_min() -> None:
    assert roofline_bound(0.5, peak_flops=100.0, bandwidth=10.0) == pytest.approx(5.0)
    assert roofline_bound(50.0, peak_flops=100.0, bandwidth=10.0) == 100.0
    assert roofline_bound(0.0, peak_flops=100.0, bandwidth=10.0) == 0.0


def test_points_cover_prefill_and_decode() -> None:
    points = points_from_estimate(_estimate())
    labels = {p.label for p in points}
    assert labels == {"prefill", "decode (first)", "decode (mean)"}
    for point in points:
        assert point.bound_flops_per_s > 0
        assert point.intensity > 0


def test_plot_and_csv(tmp_path: Path) -> None:
    est = _estimate()
    png = plot_roofline(est, tmp_path / "roofline.png")
    csv_path = save_points_csv(points_from_estimate(est), tmp_path / "points.csv")
    assert png.exists() and png.stat().st_size > 1000
    assert csv_path.exists()
    header = csv_path.read_text().splitlines()[0]
    assert "intensity" in header and "bound_flops_per_s" in header
