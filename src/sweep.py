"""Design-space parameter sweep for model, runtime, and hardware configurations.

Executes deterministic evaluations across the configuration grid.
Outputs structured results in JSON, CSV, and Markdown formats.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from measure import hardware_from_calibration
from model import (
    HardwareConfig,
    InferenceEstimate,
    RuntimeConfig,
    estimate,
    hardware_from_dict,
    model_from_dict,
    runtime_from_dict,
)
from roofline import (
    plot_roofline,
    plot_sweep_pareto,
    plot_sweep_throughput,
    points_from_estimate,
    save_points_csv,
)

RUNTIME_AXES = {
    "batch_size",
    "seq_len",
    "gen_len",
    "weight_dtype",
    "kv_dtype",
    "activation_dtype",
    "causal",
    "attention_impl",
    "activation_traffic_factor",
    "logits_last_token_only",
}


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _product(values: list[list[Any]]) -> list[tuple[Any, ...]]:
    result: list[tuple[Any, ...]] = [()]
    for options in values:
        result = [prefix + (option,) for prefix in result for option in options]
    return result


def _runtime_combinations(base: dict[str, Any], grid: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(grid)
    options = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    rows = []
    for combo in _product(options):
        row = dict(base)
        row.update(dict(zip(keys, combo)))
        rows.append(row)
    return rows


def expand_experiment(
    experiment: dict[str, Any], *, calibration_path: str | Path | None = None
) -> list[tuple[Any, Any, Any, str]]:
    """Generate all configuration combinations for the experiment."""
    model_base = model_from_dict(experiment["model"])
    runtime_base = dict(experiment.get("runtime", {}))
    grid = {k: v for k, v in runtime_base.items() if isinstance(v, list)}
    scalar_runtime = {k: v for k, v in runtime_base.items() if not isinstance(v, list)}
    grid.update(experiment.get("runtime_grid", {}) or {})
    variants = experiment.get("attention_variants", [model_base.attention_variant])
    if calibration_path:
        calibration = json.loads(Path(calibration_path).read_text())
        hardware_list = [hardware_from_calibration(calibration)]
    else:
        hardware_list = [hardware_from_dict(h) for h in experiment["hardware"]]

    combos: list[tuple[Any, Any, Any, str]] = []
    for runtime_dict in _runtime_combinations(scalar_runtime, grid):
        runtime = runtime_from_dict(runtime_dict)
        for variant in variants:
            model = model_base.with_attention_variant(variant)
            for hardware in hardware_list:
                combos.append((model, runtime, hardware, variant))
    return combos


def _row(estimate_obj: InferenceEstimate, attention: str, experiment_name: str) -> dict[str, Any]:
    est = estimate_obj
    return {
        "experiment": experiment_name,
        "config_id": est.config_id,
        "workload_id": est.workload_id,
        "model_name": est.model.name,
        "attention": attention,
        "n_layers": est.model.n_layers,
        "hidden_size": est.model.hidden_size,
        "n_heads": est.model.n_heads,
        "n_kv_heads": est.model.n_kv_heads,
        "head_dim": est.model.head_dim,
        "intermediate_size": est.model.intermediate_size,
        "vocab_size": est.model.vocab_size,
        "batch_size": est.runtime.batch_size,
        "seq_len": est.runtime.seq_len,
        "gen_len": est.runtime.gen_len,
        "weight_dtype": est.runtime.weight_dtype,
        "kv_dtype": est.runtime.kv_dtype or est.runtime.weight_dtype,
        "activation_dtype": est.runtime.activation_dtype or est.runtime.weight_dtype,
        "attention_impl": est.runtime.attention_impl,
        "hardware": est.hardware.name,
        "peak_flops": est.hardware.peak_flops,
        "memory_bandwidth": est.hardware.memory_bandwidth,
        "memory_capacity": est.hardware.memory_capacity,
        "hardware_source": est.hardware.source,
        "param_bytes": est.param_bytes,
        "kv_bytes_peak": est.kv_bytes_peak,
        "kv_pressure_peak": est.kv_pressure_peak,
        "fits_memory": (
            est.kv_pressure_peak is not None and est.kv_pressure_peak <= 1.0
        ),
        "prefill_ms": est.prefill.predicted_time_s * 1e3,
        "decode_ms": est.decode.predicted_time_s * 1e3,
        "end_to_end_ms": (est.prefill.predicted_time_s + est.decode.predicted_time_s) * 1e3,
        "prefill_tokens_per_s": est.prefill.tokens_per_s,
        "decode_tokens_per_s": est.decode.tokens_per_s,
        "end_to_end_tokens_per_s": est.end_to_end_tokens_per_s,
        "prefill_bound": est.prefill.bound,
        "decode_bound": est.decode.bound,
        "prefill_intensity": est.prefill.intensity,
        "decode_intensity": est.decode.intensity,
        "prefill_flops": est.prefill.flops,
        "decode_flops": est.decode.flops,
        "prefill_bytes": est.prefill.bytes_total,
        "decode_bytes": est.decode.bytes_total,
        "ridge_point_flops_per_byte": est.ridge_point_flops_per_byte,
        "config": {
            "model": asdict(est.model),
            "runtime": asdict(est.runtime),
            "hardware": asdict(est.hardware),
        },
    }


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate the Pareto optimal set for KV cache pressure and throughput."""
    candidates = [r for r in rows if r["kv_pressure_peak"] is not None and r["fits_memory"]]
    candidates.sort(key=lambda r: (r["kv_pressure_peak"], -r["decode_tokens_per_s"]))
    front: list[dict[str, Any]] = []
    best_tps = -1.0
    for row in candidates:
        if row["decode_tokens_per_s"] > best_tps:
            front.append(row)
            best_tps = row["decode_tokens_per_s"]
    return front


def _ranking_rows(rows: list[dict[str, Any]], limit: int = 15) -> list[dict[str, Any]]:
    return [
        {
            "config_id": r["config_id"],
            "attention": r["attention"],
            "batch_size": r["batch_size"],
            "seq_len": r["seq_len"],
            "weight_dtype": r["weight_dtype"],
            "hardware": r["hardware"],
            "decode_tokens_per_s": r["decode_tokens_per_s"],
            "kv_pressure_peak": r["kv_pressure_peak"],
            "prefill_bound": r["prefill_bound"],
            "decode_bound": r["decode_bound"],
        }
        for r in rows[:limit]
    ]


def run_sweep(
    config_path: str | Path,
    experiment_name: str,
    out_dir: str | Path,
    *,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    experiments = config.get("experiments", {})
    if experiment_name not in experiments:
        raise KeyError(f"unknown experiment {experiment_name!r}; have {sorted(experiments)}")
    experiment = experiments[experiment_name]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model, runtime, hardware, variant in expand_experiment(
        experiment, calibration_path=calibration_path
    ):
        est = estimate(model, runtime, hardware)
        rows.append(_row(est, variant, experiment_name))

    rows_by_tps = sorted(rows, key=lambda r: (-r["decode_tokens_per_s"], r["config_id"]))
    rows_by_kv = sorted(
        [r for r in rows if r["kv_pressure_peak"] is not None],
        key=lambda r: (-(r["kv_pressure_peak"] or 0.0), r["config_id"]),
    )
    front = pareto_front(rows)

    payload = {
        "schema_version": "1.0",
        "kind": "design_space_sweep",
        "experiment": experiment_name,
        "description": experiment.get("description", ""),
        "config_file": str(config_path),
        "calibration_file": str(calibration_path) if calibration_path else None,
        "n_configs": len(rows),
        "disclaimer": (
            "All rows are analytical predictions from the supplied hardware profile "
            "(nominal GPU ceilings or a measured CPU calibration). No row is a "
            "measured workload result."
        ),
        "rows": rows,
        "rankings": {
            "decode_tokens_per_s": _ranking_rows(rows_by_tps),
            "kv_pressure_peak": _ranking_rows(rows_by_kv),
            "pareto": _ranking_rows(front[:30]),
        },
    }
    (out_dir / "sweep.json").write_text(json.dumps(payload, indent=2))
    _write_csv(rows, out_dir / "sweep.csv")
    _write_markdown(payload, out_dir / "summary.md")
    plot_sweep_pareto(rows, out_dir / "pareto.png")
    plot_sweep_throughput(rows, out_dir / "batch_scaling.png")

    # Generate roofline plots for the optimal configuration and the batch-1 baseline.
    best = rows_by_tps[0]
    _plot_row_roofline(best, out_dir / "roofline_best.png")
    batch1 = [r for r in rows if r["batch_size"] == 1]
    if batch1:
        worst_decode = max(batch1, key=lambda r: r["decode_intensity"])
        _plot_row_roofline(worst_decode, out_dir / "roofline_batch1.png")

    print(
        f"[sweep] {len(rows)} configs -> {out_dir}\n"
        f"[sweep] best decode: {best['decode_tokens_per_s']:.2f} tok/s "
        f"({best['hardware']}, {best['attention']}, B={best['batch_size']}, "
        f"S={best['seq_len']}, {best['weight_dtype']})"
    )
    return payload


def _plot_row_roofline(row: dict[str, Any], path: Path) -> None:
    cfg = row["config"]
    est = estimate(
        model_from_dict(cfg["model"]),
        runtime_from_dict(cfg["runtime"]),
        hardware_from_dict(cfg["hardware"]),
    )
    plot_roofline(est, path)
    save_points_csv(points_from_estimate(est), path.with_suffix(".csv"))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    flat_rows = [{k: v for k, v in row.items() if k != "config"} for row in rows]
    fieldnames = list(flat_rows[0]) if flat_rows else []
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    rows = payload["rows"]
    top = payload["rankings"]["decode_tokens_per_s"][:10]
    kv = payload["rankings"]["kv_pressure_peak"][:10]
    lines = [
        f"# Sweep: {payload['experiment']}",
        "",
        payload.get("description", ""),
        "",
        f"Configurations: **{payload['n_configs']}**",
        "",
        "> All numbers are analytical predictions from the supplied hardware profile, "
        "not measured workload performance.",
        "",
        "## Top predicted decode throughput",
        "",
        "| hardware | attn | batch | seq | dtype | decode tok/s | KV pressure | decode bound |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in top:
        pressure = "n/a" if r["kv_pressure_peak"] is None else f"{r['kv_pressure_peak']*100:.1f}%"
        lines.append(
            f"| {r['hardware']} | {r['attention']} | {r['batch_size']} | {r['seq_len']} | "
            f"{r['weight_dtype']} | {r['decode_tokens_per_s']:.1f} | {pressure} | {r['decode_bound']} |"
        )
    lines += [
        "",
        "## Highest peak KV pressure",
        "",
        "| hardware | attn | batch | seq | dtype | decode tok/s | KV pressure |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in kv:
        pressure = "n/a" if r["kv_pressure_peak"] is None else f"{r['kv_pressure_peak']*100:.1f}%"
        lines.append(
            f"| {r['hardware']} | {r['attention']} | {r['batch_size']} | {r['seq_len']} | "
            f"{r['weight_dtype']} | {r['decode_tokens_per_s']:.1f} | {pressure} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- KV pressure counts the KV cache at seq_len + gen_len; it excludes weights.",
        "- Decode throughput is batch * gen_len / summed per-step predicted time.",
        "- FP8 changes weight/KV bytes only; the compute ceiling is held at the FP16 "
        "preset value, so FP8 gains here are memory-side only.",
        "",
        f"Total rows: {len(rows)}",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analytical design-space sweep")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--calibration", default=None)
    args = parser.parse_args(argv)
    run_sweep(
        args.config,
        args.experiment,
        args.out_dir,
        calibration_path=args.calibration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
