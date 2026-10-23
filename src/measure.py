"""CPU performance measurement and calibration.

Benchmarks CPU execution using the cpu-numpy-blas-float32 backend.
Microbenchmarks measure matrix multiplication throughput, memory bandwidth,
and layer dispatch overhead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from cpu_backend import NumpyDecoder, decoder_for_runtime
from model import (
    HardwareConfig,
    ModelConfig,
    RuntimeConfig,
    estimate,
    model_from_dict,
    runtime_from_dict,
    workload_id,
)
from roofline import plot_predicted_vs_measured, plot_throughput_scaling

BACKEND = "cpu-numpy-blas-float32"
DEVICE = "CPU"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cpu_brand() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
            ).strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return 0


def _blas_name() -> str:
    try:
        config = np.__config__  # type: ignore[attr-defined]
        info = config.get_info("blas_opt")
        if isinstance(info, dict):
            return str(info.get("name", "unknown"))
    except Exception:
        pass
    return "unknown"


def nvidia_gpu_status() -> dict[str, Any]:
    """Check NVIDIA GPU availability using nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "checked_at": _now(),
        "available": False,
        "devices": [],
        "reason": "nvidia-smi not found on PATH",
        "nvidia_claims": "none",
    }
    if not smi:
        return result
    try:
        output = subprocess.check_output(
            [
                smi,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:  # pragma: no cover - depends on host
        result["reason"] = f"nvidia-smi failed: {exc}"
        return result
    devices = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            devices.append(
                {"name": parts[0], "memory_total_mib": float(parts[1]), "driver": parts[2]}
            )
    result["devices"] = devices
    result["available"] = bool(devices)
    if devices:
        result["reason"] = "NVIDIA GPU detected; no GPU benchmark was run by this script"
        result["nvidia_claims"] = "none (detection only)"
    return result


def environment_info() -> dict[str, Any]:
    return {
        "timestamp_utc": _now(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _cpu_brand(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "blas": _blas_name(),
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "backend": BACKEND,
        "device": DEVICE,
        "gpu_used": False,
        "num_threads_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(list(values)))


def _timed_call(fn, repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return times


def measure_calibration(
    *,
    repeats: int = 3,
    matmul_size: int = 2048,
    copy_mb: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure dense matrix multiplication throughput and memory copy bandwidth."""
    rng = np.random.default_rng(seed)
    n = int(matmul_size)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)
    _ = a @ b
    matmul_times = _timed_call(lambda: a @ b, repeats=repeats)
    matmul_s = _median(matmul_times)
    peak_flops = 2.0 * n**3 / matmul_s

    copy_bytes = int(copy_mb * 1024 * 1024)
    src = np.ones(copy_bytes // 4, dtype=np.float32)
    dst = np.empty_like(src)
    _ = np.copyto(dst, src)
    copy_times = _timed_call(lambda: np.copyto(dst, src), repeats=repeats)
    copy_s = _median(copy_times)
    bandwidth = 2.0 * src.nbytes / copy_s

    overhead = measure_layer_overhead(repeats=max(7, repeats + 4))
    return {
        "schema_version": "1.0",
        "kind": "cpu_calibration",
        "backend": BACKEND,
        "device": DEVICE,
        "gpu_used": False,
        "environment": environment_info(),
        "matmul": {
            "size": n,
            "repeats": repeats,
            "times_s": matmul_times,
            "median_s": matmul_s,
            "flops": 2.0 * n**3,
            "peak_flops": float(peak_flops),
        },
        "copy": {
            "bytes_copied": int(src.nbytes),
            "bytes_traffic": int(2 * src.nbytes),
            "repeats": repeats,
            "times_s": copy_times,
            "median_s": copy_s,
            "memory_bandwidth": float(bandwidth),
        },
        "layer_overhead_s": float(overhead["layer_overhead_s"]),
        "layer_overhead_method": overhead,
        "notes": [
            "Effective peaks for this NumPy reference implementation, not datasheet peaks.",
            "Dense matmul throughput is measured at one large square shape; small "
            "transformer GEMMs will achieve less.",
            "Layer overhead is the slope of forward latency vs layer count on a "
            "tiny model, used as a per-layer dispatch correction.",
        ],
    }


def measure_layer_overhead(*, repeats: int = 7, seed: int = 0) -> dict[str, Any]:
    """Measure per-layer dispatch overhead using single-layer and multi-layer models."""

    def make(layers: int) -> NumpyDecoder:
        cfg = ModelConfig(
            n_layers=layers,
            hidden_size=64,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=128,
            vocab_size=256,
            tie_embeddings=True,
            gated_mlp=True,
            name=f"overhead-L{layers}",
        )
        return NumpyDecoder(cfg, seed=seed, max_seq_len=4)

    tokens = np.zeros((1, 1), dtype=np.int64)
    times: dict[int, float] = {}
    for layers in (1, 5):
        decoder = make(layers)
        _ = decoder.prefill(tokens)
        samples = _timed_call(lambda d=decoder: d.prefill(tokens), repeats=repeats)
        times[layers] = _median(samples)
    slope = (times[5] - times[1]) / 4.0
    return {
        "repeats": repeats,
        "l1_median_s": times[1],
        "l5_median_s": times[5],
        "layer_overhead_s": max(0.0, float(slope)),
        "method": "slope=(t_L5-t_L1)/(5-1) on a tiny 1-token prefill",
    }


def hardware_from_calibration(
    calibration: dict[str, Any], *, name: str | None = None
) -> HardwareConfig:
    env = calibration.get("environment", {})
    capacity = float(env.get("physical_memory_bytes") or 0)
    return HardwareConfig(
        name=name or f"{env.get('processor', 'CPU')} [{BACKEND}] calibrated",
        peak_flops=float(calibration["matmul"]["peak_flops"]),
        memory_bandwidth=float(calibration["copy"]["memory_bandwidth"]),
        memory_capacity=capacity,
        compute_efficiency=1.0,
        bandwidth_efficiency=1.0,
        layer_overhead_s=float(calibration.get("layer_overhead_s", 0.0)),
        source=(
            "measured microbenchmarks on this host (NumPy/Accelerate BLAS); "
            "not a nominal GPU ceiling"
        ),
        notes="Effective profile for the CPU reference backend only.",
    )


def measure_workload(
    model: ModelConfig,
    runtime: RuntimeConfig,
    *,
    seed: int = 0,
    repeats: int = 3,
    warmup: int = 1,
) -> dict[str, Any]:
    """Benchmark prefill and decode execution times for the workload."""
    for label, dtype in (
        ("weight_dtype", runtime.weight_dtype),
        ("kv_dtype", runtime.kv_dtype or runtime.weight_dtype),
        ("activation_dtype", runtime.activation_dtype or runtime.weight_dtype),
    ):
        if dtype != "fp32":
            raise ValueError(
                f"{label}={dtype}: the NumPy reference backend executes float32 only"
            )

    decoder = decoder_for_runtime(model, runtime, seed=seed, slack=0)
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, model.vocab_size, size=(runtime.batch_size, runtime.seq_len))

    def run_once() -> tuple[float, float, list[float], int]:
        prefill_start = time.perf_counter()
        logits, cache = decoder.prefill(tokens)
        prefill_s = time.perf_counter() - prefill_start
        step_times: list[float] = []
        decode_start = time.perf_counter()
        for _ in range(runtime.gen_len):
            next_tokens = np.argmax(logits, axis=-1)
            step_start = time.perf_counter()
            logits = decoder.decode_step(next_tokens, cache)
            step_times.append(time.perf_counter() - step_start)
        decode_s = time.perf_counter() - decode_start
        return prefill_s, decode_s, step_times, cache.nbytes

    for _ in range(warmup):
        run_once()
    prefill_samples: list[float] = []
    decode_samples: list[float] = []
    step_samples: list[list[float]] = []
    kv_nbytes = 0
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start_all = time.perf_counter()
    for _ in range(repeats):
        p, d, steps, kv = run_once()
        prefill_samples.append(p)
        decode_samples.append(d)
        step_samples.append(steps)
        kv_nbytes = kv
    wall = time.perf_counter() - start_all
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    prefill_s = _median(prefill_samples)
    decode_s = _median(decode_samples)
    per_step_flat = [t for sample in step_samples for t in sample]
    tokens_prefill = runtime.batch_size * runtime.seq_len
    tokens_decode = runtime.batch_size * runtime.gen_len
    # ru_maxrss returns bytes on macOS and kilobytes on Linux.
    rss_scale = 1.0 if sys.platform == "darwin" else 1024.0

    payload = {
        "schema_version": "1.0",
        "kind": "measured_workload",
        "backend": BACKEND,
        "device": DEVICE,
        "gpu_used": False,
        "model": model.__dict__.copy(),
        "runtime": runtime.__dict__.copy(),
        "workload_id": workload_id(model, runtime),
        "seed": seed,
        "repeats": repeats,
        "warmup": warmup,
        "prefill_times_s": prefill_samples,
        "decode_times_s": decode_samples,
        "decode_step_times_s": step_samples,
        "prefill_median_s": prefill_s,
        "decode_median_s": decode_s,
        "decode_step_median_s": _median(per_step_flat),
        "prefill_tokens_per_s": tokens_prefill / prefill_s,
        "decode_tokens_per_s": tokens_decode / decode_s,
        "end_to_end_tokens_per_s": (tokens_prefill + tokens_decode) / (prefill_s + decode_s),
        "kv_bytes_allocated": int(kv_nbytes),
        "param_bytes_measured": int(decoder.weight_bytes()),
        "backend_summary": decoder.summary(),
        "peak_rss_bytes": int(max(rss_before, rss_after) * rss_scale),
        "measurement_wall_time_s": wall,
        "environment": environment_info(),
    }
    digest = hashlib.sha256(
        json.dumps(
            {"model": payload["model"], "runtime": payload["runtime"], "seed": seed},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    payload["measurement_id"] = f"meas-{digest}"
    return payload


def compare_measurement(
    measurement: dict[str, Any], hardware: HardwareConfig
) -> dict[str, Any]:
    model = model_from_dict(measurement["model"])
    runtime = runtime_from_dict(measurement["runtime"])
    est = estimate(model, runtime, hardware)

    def row(phase: str, predicted_s: float, ideal_s: float, measured_s: float, flops: float) -> dict[str, Any]:
        predicted_ms = predicted_s * 1e3
        measured_ms = measured_s * 1e3
        return {
            "phase": phase,
            "predicted_ms": predicted_ms,
            "ideal_ms": ideal_s * 1e3,
            "measured_ms": measured_ms,
            "error_pct": (predicted_ms - measured_ms) / measured_ms * 100.0,
            "speedup_needed": measured_ms / predicted_ms,
            "achieved_fraction_of_ideal": ideal_s / measured_s,
            "predicted_flops_per_s": flops / predicted_s,
            "measured_flops_per_s": flops / measured_s,
            "predicted_tokens_per_s": None,
            "measured_tokens_per_s": None,
        }

    prefill = row(
        "prefill",
        est.prefill.predicted_time_s,
        est.prefill.ideal_time_s,
        measurement["prefill_median_s"],
        est.prefill.flops,
    )
    prefill["predicted_tokens_per_s"] = est.prefill.tokens_per_s
    prefill["measured_tokens_per_s"] = measurement["prefill_tokens_per_s"]
    prefill["bound"] = est.prefill.bound
    prefill["intensity"] = est.prefill.intensity

    decode = row(
        "decode",
        est.decode.predicted_time_s,
        est.decode.ideal_time_s,
        measurement["decode_median_s"],
        est.decode.flops,
    )
    decode["predicted_tokens_per_s"] = est.decode.tokens_per_s
    decode["measured_tokens_per_s"] = measurement["decode_tokens_per_s"]
    decode["bound"] = est.decode.bound
    decode["intensity"] = est.decode.intensity

    return {
        "workload_id": est.workload_id,
        "config_id": est.config_id,
        "measurement_id": measurement["measurement_id"],
        "label": f"{model.name}-b{runtime.batch_size}-s{runtime.seq_len}-g{runtime.gen_len}",
        "model": measurement["model"],
        "runtime": measurement["runtime"],
        "hardware": hardware.name,
        "kv_bytes_peak_predicted": est.kv_bytes_peak,
        "kv_bytes_measured": measurement["kv_bytes_allocated"],
        "param_bytes_predicted": est.param_bytes,
        "param_bytes_measured": measurement["param_bytes_measured"],
        "phases": [prefill, decode],
    }


def _expand_grid(base: dict[str, Any], grid: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(grid)
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    rows: list[dict[str, Any]] = []
    for combo in _product(value_lists):
        row = dict(base)
        row.update(dict(zip(keys, combo)))
        rows.append(row)
    return rows


def _product(lists: list[list[Any]]) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = [()]
    for values in lists:
        result = [prefix + (value,) for prefix in result for value in values]
    return result


def _summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        rows = [p for c in comparisons for p in c["phases"] if p["phase"] == phase]
        if not rows:
            continue
        errors = [abs(r["error_pct"]) for r in rows]
        ratios = [r["measured_ms"] / r["predicted_ms"] for r in rows]
        summary[phase] = {
            "n": len(rows),
            "median_abs_error_pct": _median(errors),
            "min_abs_error_pct": min(errors),
            "max_abs_error_pct": max(errors),
            "median_measured_over_predicted": _median(ratios),
            "median_achieved_fraction_of_ideal": _median(
                [r["achieved_fraction_of_ideal"] for r in rows]
            ),
        }
    return summary


def run_validation(
    config_path: str | Path,
    calibration_path: str | Path,
    out_dir: str | Path,
    *,
    repeats_override: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(config_path.read_text())
    validation = spec.get("validation", {})
    calibration = json.loads(Path(calibration_path).read_text())
    hardware = hardware_from_calibration(calibration)
    seed = int(validation.get("seed", 0))
    repeats = int(repeats_override or validation.get("repeats", 3))
    warmup = int(validation.get("warmup", 1))

    measurements = []
    comparisons = []
    for workload in validation.get("workloads", []):
        model = model_from_dict(workload["model"])
        runtime_base = dict(workload.get("runtime", {}))
        grid = workload.get("runtime_grid", {})
        for runtime_dict in _expand_grid(runtime_base, grid):
            runtime = runtime_from_dict(runtime_dict)
            measurement = measure_workload(
                model, runtime, seed=seed, repeats=repeats, warmup=warmup
            )
            comparison = compare_measurement(measurement, hardware)
            if workload.get("name"):
                comparison["label"] = f"{workload['name']}-b{runtime.batch_size}-s{runtime.seq_len}-g{runtime.gen_len}"
            measurements.append(measurement)
            comparisons.append(comparison)
            print(
                f"[measure] {comparison['label']}: "
                f"prefill {measurement['prefill_median_s']*1e3:.2f} ms, "
                f"decode {measurement['decode_median_s']*1e3:.2f} ms"
            )

    flat_phases = [phase for comp in comparisons for phase in comp["phases"]]
    payload = {
        "schema_version": "1.0",
        "kind": "cpu_validation",
        "backend": BACKEND,
        "device": DEVICE,
        "gpu_used": False,
        "generated_at": _now(),
        "calibration_file": str(calibration_path),
        "hardware": hardware.__dict__,
        "environment": environment_info(),
        "measurements": measurements,
        "comparisons": comparisons,
        "summary": _summary(comparisons),
        "limitations": [
            "This validates methodology on CPU, not NVIDIA GPU performance.",
            "Predictions use measured effective CPU peaks (large matmul/copy), "
            "so they remain an idealized lower bound for this naive backend.",
            "The reference decoder materializes attention scores and pays Python "
            "dispatch overhead; real serving kernels are fused and batched.",
        ],
    }
    (out_dir / "validation.json").write_text(json.dumps(payload, indent=2))
    _write_comparison_csv(comparisons, out_dir / "validation.csv")
    plot_predicted_vs_measured(flat_phases, out_dir / "predicted_vs_measured.png")
    plot_throughput_scaling(flat_phases, out_dir / "throughput_scaling.png")
    (out_dir / "summary.json").write_text(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["summary"], indent=2))
    return payload


def _write_comparison_csv(comparisons: list[dict[str, Any]], path: Path) -> None:
    rows = []
    for comp in comparisons:
        for phase in comp["phases"]:
            rows.append(
                {
                    "label": comp["label"],
                    "phase": phase["phase"],
                    "predicted_ms": phase["predicted_ms"],
                    "ideal_ms": phase["ideal_ms"],
                    "measured_ms": phase["measured_ms"],
                    "error_pct": phase["error_pct"],
                    "predicted_tokens_per_s": phase["predicted_tokens_per_s"],
                    "measured_tokens_per_s": phase["measured_tokens_per_s"],
                    "bound": phase["bound"],
                    "intensity": phase["intensity"],
                    "achieved_fraction_of_ideal": phase["achieved_fraction_of_ideal"],
                    "predicted_flops_per_s": phase["predicted_flops_per_s"],
                    "measured_flops_per_s": phase["measured_flops_per_s"],
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU measurement and validation")
    sub = parser.add_subparsers(dest="command", required=True)

    cal = sub.add_parser("calibrate", help="measure effective CPU peaks")
    cal.add_argument("--out", required=True)
    cal.add_argument("--repeats", type=int, default=3)
    cal.add_argument("--matmul-size", type=int, default=2048)
    cal.add_argument("--copy-mb", type=int, default=64)

    val = sub.add_parser("validation", help="run measured CPU workloads and compare")
    val.add_argument("--config", required=True)
    val.add_argument("--calibration", required=True)
    val.add_argument("--out-dir", required=True)
    val.add_argument("--repeats", type=int, default=None)

    gpu = sub.add_parser("gpu-status", help="detect NVIDIA GPUs without benchmarking")
    gpu.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        payload = measure_calibration(
            repeats=args.repeats, matmul_size=args.matmul_size, copy_mb=args.copy_mb
        )
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(
            json.dumps(
                {
                    "peak_tflops": payload["matmul"]["peak_flops"] / 1e12,
                    "bandwidth_gbps": payload["copy"]["memory_bandwidth"] / 1e9,
                    "layer_overhead_ms": payload["layer_overhead_s"] * 1e3,
                    "backend": BACKEND,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "validation":
        run_validation(
            args.config, args.calibration, args.out_dir, repeats_override=args.repeats
        )
        return 0
    if args.command == "gpu-status":
        status = nvidia_gpu_status()
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
