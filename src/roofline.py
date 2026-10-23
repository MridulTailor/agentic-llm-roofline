"""Roofline model calculation and visualization.

The roofline bound defines the maximum achievable throughput:

    P(I) = min(P_peak, B * I)

where I is operational intensity (FLOP/byte), P_peak is peak compute throughput
(FLOP/s), and B is memory bandwidth (byte/s).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import HardwareConfig, InferenceEstimate, format_bytes


@dataclass(frozen=True)
class RooflinePoint:
    label: str
    intensity: float
    bound_flops_per_s: float
    phase: str
    flops: float
    bytes: float


def roofline_bound(
    intensity: float, peak_flops: float, bandwidth: float
) -> float:
    """Calculate maximum throughput in FLOP/s from operational intensity."""
    if intensity <= 0:
        return 0.0
    return min(peak_flops, bandwidth * intensity)


def points_from_estimate(estimate: InferenceEstimate) -> list[RooflinePoint]:
    points = [
        RooflinePoint(
            label="prefill",
            intensity=estimate.prefill.intensity,
            bound_flops_per_s=roofline_bound(
                estimate.prefill.intensity,
                estimate.hardware.effective_peak_flops,
                estimate.hardware.effective_bandwidth,
            ),
            phase="prefill",
            flops=estimate.prefill.flops,
            bytes=estimate.prefill.bytes_total,
        ),
        RooflinePoint(
            label="decode (first)",
            intensity=estimate.decode_first.intensity,
            bound_flops_per_s=roofline_bound(
                estimate.decode_first.intensity,
                estimate.hardware.effective_peak_flops,
                estimate.hardware.effective_bandwidth,
            ),
            phase="decode_first",
            flops=estimate.decode_first.flops,
            bytes=estimate.decode_first.bytes_total,
        ),
        RooflinePoint(
            label="decode (mean)",
            intensity=estimate.decode.intensity,
            bound_flops_per_s=roofline_bound(
                estimate.decode.intensity,
                estimate.hardware.effective_peak_flops,
                estimate.hardware.effective_bandwidth,
            ),
            phase="decode",
            flops=estimate.decode.flops,
            bytes=estimate.decode.bytes_total,
        ),
    ]
    return [p for p in points if p.intensity > 0]


def save_points_csv(points: Sequence[RooflinePoint], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["label", "phase", "intensity", "bound_flops_per_s", "flops", "bytes"]
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "label": point.label,
                    "phase": point.phase,
                    "intensity": point.intensity,
                    "bound_flops_per_s": point.bound_flops_per_s,
                    "flops": point.flops,
                    "bytes": point.bytes,
                }
            )
    return path


def _hardware_title(estimate: InferenceEstimate) -> str:
    return (
        f"{estimate.model.name} | B={estimate.runtime.batch_size}, "
        f"S={estimate.runtime.seq_len}, G={estimate.runtime.gen_len}, "
        f"{estimate.runtime.weight_dtype}"
    )


def plot_roofline(
    estimate: InferenceEstimate,
    out_png: str | Path,
    title: str | None = None,
) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    hw = estimate.hardware
    points = points_from_estimate(estimate)

    peak = hw.effective_peak_flops
    bw = hw.effective_bandwidth
    ridge = hw.ridge_point

    xs = np.logspace(-2, 5, 500)
    ys = np.minimum(peak, bw * xs)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.loglog(xs, ys, color="#111111", linewidth=2.2, label="roofline: min(peak, BW x I)")
    ax.axhline(peak, color="#1f77b4", linestyle="--", linewidth=1.2, label="compute ceiling")
    ax.axvline(ridge, color="#d62728", linestyle=":", linewidth=1.2, label="ridge point")

    colors = {"prefill": "#2ca02c", "decode (first)": "#ff7f0e", "decode (mean)": "#9467bd"}
    for point in points:
        ax.scatter(
            [point.intensity],
            [point.bound_flops_per_s],
            s=85,
            color=colors.get(point.label, "#333333"),
            zorder=5,
            edgecolor="white",
            linewidth=1.0,
        )
        ax.annotate(
            point.label,
            (point.intensity, point.bound_flops_per_s),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )

    ax.set_xlabel("Operational intensity (FLOP / byte)")
    ax.set_ylabel("Predicted achievable throughput (FLOP / s)")
    ax.set_title(title or f"Analytical roofline: {_hardware_title(estimate)}")
    subtitle = (
        f"{hw.name} | peak={peak/1e12:.3g} TFLOP/s, BW={bw/1e12:.3g} TB/s, "
        f"ridge={ridge:.3g} FLOP/B\n"
        f"source: {hw.source}. Points are model estimates, not measured GPU throughput."
    )
    ax.text(
        0.01,
        0.01,
        subtitle,
        transform=ax.transAxes,
        fontsize=7.4,
        va="bottom",
        color="#444444",
    )
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_predicted_vs_measured(
    comparisons: Iterable[dict],
    out_png: str | Path,
    title: str = "Predicted vs measured latency (CPU numpy backend)",
) -> Path:
    """Generate a parity plot comparing predicted and measured execution times."""
    rows = list(comparisons)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    markers = {"prefill": "o", "decode": "s"}
    colors = {"prefill": "#2ca02c", "decode": "#ff7f0e"}
    for phase in ("prefill", "decode"):
        subset = [r for r in rows if r["phase"] == phase and r["measured_ms"] > 0]
        if not subset:
            continue
        x = [r["predicted_ms"] for r in subset]
        y = [r["measured_ms"] for r in subset]
        ax.scatter(
            x,
            y,
            marker=markers[phase],
            color=colors[phase],
            s=55,
            alpha=0.85,
            label=f"{phase} (n={len(subset)})",
        )
    all_x = [r["predicted_ms"] for r in rows if r["predicted_ms"] > 0]
    all_y = [r["measured_ms"] for r in rows if r["measured_ms"] > 0]
    if all_x and all_y:
        lo = min(min(all_x), min(all_y)) * 0.5
        hi = max(max(all_x), max(all_y)) * 2.0
        ax.plot([lo, hi], [lo, hi], color="#111111", linestyle="--", linewidth=1.2, label="parity")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted latency (ms, analytical bound)")
    ax.set_ylabel("Measured latency (ms)")
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_throughput_scaling(
    comparisons: Iterable[dict], out_png: str | Path
) -> Path:
    """Generate a comparison chart of predicted and measured token throughput."""
    rows = list(comparisons)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    labels = [r.get("label", r.get("workload_id", "config")) for r in rows]
    x = np.arange(len(rows))
    for idx, phase in enumerate(("prefill", "decode")):
        subset = [r for r in rows if r["phase"] == phase]
        positions = [x[rows.index(r)] + (-0.18 if phase == "prefill" else 0.18) for r in subset]
        ax.bar(
            [p - 0.08 for p in positions],
            [r["predicted_tokens_per_s"] for r in subset],
            width=0.15,
            label=f"{phase} predicted",
            color="#9ecae1" if phase == "prefill" else "#fdae6b",
        )
        ax.bar(
            [p + 0.08 for p in positions],
            [r["measured_tokens_per_s"] for r in subset],
            width=0.15,
            label=f"{phase} measured",
            color="#3182bd" if phase == "prefill" else "#e6550d",
        )
    ax.set_yscale("log")
    ax.set_ylabel("Tokens / s (log)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title("Predicted vs measured throughput (CPU, not GPU)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_sweep_pareto(rows: Sequence[dict], out_png: str | Path) -> Path:
    """Generate a Pareto plot of decode throughput against peak KV cache pressure."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    hardware_names = sorted({r["hardware"] for r in rows})
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, hw in enumerate(hardware_names):
        subset = [r for r in rows if r["hardware"] == hw]
        xs = [(r["kv_pressure_peak"] or 0.0) * 100.0 for r in subset]
        ys = [r["decode_tokens_per_s"] for r in subset]
        ax.scatter(
            xs,
            ys,
            s=34,
            alpha=0.75,
            marker=markers[i % len(markers)],
            label=hw,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Peak KV-cache pressure (% of device memory)")
    ax.set_ylabel("Predicted decode throughput (tokens / s)")
    ax.set_title("Design-space sweep: decode throughput vs KV pressure")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def plot_sweep_throughput(rows: Sequence[dict], out_png: str | Path) -> Path:
    """Generate a scaling plot of decode throughput against batch size."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    # Select the median sequence length for display.
    seqs = sorted({r["seq_len"] for r in rows})
    if not seqs:
        plt.close(fig)
        return out_png
    seq = seqs[len(seqs) // 2]
    subset = [r for r in rows if r["seq_len"] == seq and r["weight_dtype"] == sorted({x["weight_dtype"] for x in rows})[0]]
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in subset:
        groups.setdefault((row["hardware"], row["attention"]), []).append(row)
    for (hw, attn), values in sorted(groups.items()):
        values = sorted(values, key=lambda r: r["batch_size"])
        ax.plot(
            [v["batch_size"] for v in values],
            [v["decode_tokens_per_s"] for v in values],
            marker="o",
            linewidth=1.4,
            label=f"{hw} / {attn}",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Predicted decode throughput (tokens / s)")
    ax.set_title(f"Batch scaling (seq_len={seq}, first dtype in sweep)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def format_mib(value: float | int) -> str:
    return format_bytes(value)
