"""Unit tests for the design-space sweep."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from sweep import run_sweep


def _experiment() -> dict:
    return {
        "experiments": {
            "unit": {
                "description": "unit sweep",
                "model": {
                    "name": "unit-model",
                    "n_layers": 2,
                    "hidden_size": 64,
                    "n_heads": 4,
                    "n_kv_heads": 2,
                    "intermediate_size": 128,
                    "vocab_size": 256,
                    "tie_embeddings": True,
                    "gated_mlp": True,
                },
                "runtime": {
                    "gen_len": 8,
                    "weight_dtype": ["fp16", "fp8"],
                    "kv_dtype": "fp16",
                    "activation_dtype": "fp16",
                    "causal": True,
                    "attention_impl": "fused",
                    "activation_traffic_factor": 2.0,
                    "logits_last_token_only": True,
                },
                "runtime_grid": {
                    "batch_size": [1, 4],
                    "seq_len": [64, 256],
                },
                "attention_variants": ["mha", "gqa"],
                "hardware": [
                    {
                        "name": "hw-fast",
                        "peak_flops": 100e12,
                        "memory_bandwidth": 2e12,
                        "memory_capacity": 80e9,
                        "source": "unit-test nominal",
                    },
                    {
                        "name": "hw-slow-mem",
                        "peak_flops": 100e12,
                        "memory_bandwidth": 0.5e12,
                        "memory_capacity": 20e9,
                        "source": "unit-test nominal",
                    },
                ],
            }
        }
    }


def test_sweep_outputs_and_rankings(tmp_path: Path) -> None:
    config_path = tmp_path / "configs.yaml"
    config_path.write_text(yaml.safe_dump(_experiment()))
    out = tmp_path / "out"
    payload = run_sweep(config_path, "unit", out)
    assert payload["n_configs"] == 2 * 2 * 2 * 2 * 2  # batch, seq, dtype, attn, hw
    assert (out / "sweep.json").exists()
    assert (out / "sweep.csv").exists()
    assert (out / "summary.md").exists()
    assert (out / "pareto.png").exists()
    assert (out / "roofline_best.png").exists()
    top = payload["rankings"]["decode_tokens_per_s"]
    assert top == sorted(top, key=lambda r: -r["decode_tokens_per_s"])
    kv = payload["rankings"]["kv_pressure_peak"]
    assert kv == sorted(kv, key=lambda r: -(r["kv_pressure_peak"] or 0.0))
    assert all("config" in row for row in payload["rows"])
    assert all("config_id" in row for row in payload["rows"])


def test_sweep_is_deterministic(tmp_path: Path) -> None:
    config_path = tmp_path / "configs.yaml"
    config_path.write_text(yaml.safe_dump(_experiment()))
    first = run_sweep(config_path, "unit", tmp_path / "a")
    second = run_sweep(config_path, "unit", tmp_path / "b")
    assert [r["config_id"] for r in first["rows"]] == [
        r["config_id"] for r in second["rows"]
    ]
    ids = [r["config_id"] for r in first["rows"]]
    assert len(ids) == len(set(ids))


def test_pareto_front_is_monotone(tmp_path: Path) -> None:
    config_path = tmp_path / "configs.yaml"
    config_path.write_text(yaml.safe_dump(_experiment()))
    payload = run_sweep(config_path, "unit", tmp_path / "out")
    front_ids = [r["config_id"] for r in payload["rankings"]["pareto"]]
    assert len(front_ids) == len(set(front_ids))
    rows = {r["config_id"]: r for r in payload["rows"]}
    pressures = [rows[i]["kv_pressure_peak"] for i in front_ids]
    tps = [rows[i]["decode_tokens_per_s"] for i in front_ids]
    assert pressures == sorted(pressures)
    assert tps == sorted(tps)


def test_markdown_disclaims_gpu(tmp_path: Path) -> None:
    config_path = tmp_path / "configs.yaml"
    config_path.write_text(yaml.safe_dump(_experiment()))
    out = tmp_path / "out"
    run_sweep(config_path, "unit", out)
    text = (out / "summary.md").read_text()
    assert "not measured workload performance" in text
