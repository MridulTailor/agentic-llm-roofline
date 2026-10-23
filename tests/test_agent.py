"""Unit tests for the bounded exploration agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from agent import Budget, Candidate, RooflineAgent
from model import HardwareConfig, ModelConfig, RuntimeConfig, estimate


def _hardware() -> HardwareConfig:
    return HardwareConfig(
        name="fake-cpu",
        peak_flops=500e9,
        memory_bandwidth=80e9,
        memory_capacity=16e9,
        layer_overhead_s=1e-5,
        source="unit-test",
    )


def _candidates() -> list[Candidate]:
    base = ModelConfig(
        n_layers=2,
        hidden_size=32,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        vocab_size=128,
        name="agent-test",
    )
    candidates = []
    for variant in ("mha", "gqa"):
        model = base.with_attention_variant(variant)
        for batch in (1, 4):
            for seq in (16, 32):
                runtime = RuntimeConfig(
                    batch_size=batch,
                    seq_len=seq,
                    gen_len=2,
                    weight_dtype="fp32",
                    kv_dtype="fp32",
                    activation_dtype="fp32",
                )
                candidates.append(Candidate(model, runtime, f"{variant}-b{batch}-s{seq}"))
    return candidates


def _simulate(config: dict[str, Any]) -> dict[str, Any]:
    from model import model_from_dict, runtime_from_dict

    return estimate(
        model_from_dict(config["model"]),
        runtime_from_dict(config["runtime"]),
        _hardware(),
    ).to_dict()


def _fake_measurement(config: dict[str, Any], scale: float = 2.0) -> dict[str, Any]:
    from model import model_from_dict, runtime_from_dict

    est = estimate(
        model_from_dict(config["model"]),
        runtime_from_dict(config["runtime"]),
        _hardware(),
    )
    prefill_s = est.prefill.predicted_time_s * scale
    decode_s = est.decode.predicted_time_s * scale
    return {
        "schema_version": "1.0",
        "kind": "measured_workload",
        "measurement_id": "meas-unit",
        "workload_id": est.workload_id,
        "backend": "cpu-numpy-blas-float32",
        "device": "CPU",
        "gpu_used": False,
        "model": config["model"],
        "runtime": config["runtime"],
        "prefill_median_s": prefill_s,
        "decode_median_s": decode_s,
        "prefill_tokens_per_s": est.prefill.tokens / prefill_s,
        "decode_tokens_per_s": est.decode.tokens / decode_s,
        "kv_bytes_allocated": est.kv_bytes_peak,
        "param_bytes_measured": est.param_bytes,
    }


def _profile(config: dict[str, Any]) -> dict[str, Any]:
    return _fake_measurement(config, scale=2.0)


def _build(
    tmp_path: Path,
    *,
    profile_fn: Callable[[dict[str, Any]], dict[str, Any]] = _profile,
    budget: Budget = Budget(max_rounds=3, max_tool_calls=8, max_profiles=2),
    require_nvidia_gpu: bool = False,
    gpu_status_fn=None,
) -> RooflineAgent:
    return RooflineAgent(
        _candidates(),
        _hardware(),
        budget=budget,
        simulate_fn=_simulate,
        profile_fn=profile_fn,
        output_dir=tmp_path,
        require_nvidia_gpu=require_nvidia_gpu,
        gpu_status_fn=gpu_status_fn,
        profile_repeats=1,
    )


def test_agent_runs_bounded_and_cites_evidence(tmp_path: Path) -> None:
    agent = _build(tmp_path)
    report = agent.run()
    assert report["status"] == "complete"
    assert report["budget"]["tool_calls_used"] <= 8
    assert report["budget"]["profile_calls_used"] <= 2
    assert report["validation"]
    assert report["claims"]
    assert report["evidence_ids"]
    assert len(report["top_configs"]) > 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "agent_trace.jsonl").exists()
    assert (tmp_path / "roofline_best.png").exists()
    assert (tmp_path / "predicted_vs_measured.png").exists()

    # Every cited evidence id exists.
    for claim in report["claims"]:
        assert claim["evidence"]
        for eid in claim["evidence"]:
            assert eid in report["evidence_ids"]
    measured_claims = [c for c in report["claims"] if c["kind"] == "measured"]
    assert measured_claims
    for claim in measured_claims:
        assert any(eid.startswith("meas-") for eid in claim["evidence"])

    # Trace has one line per event and every successful call has an evidence id.
    lines = [
        json.loads(line)
        for line in (tmp_path / "agent_trace.jsonl").read_text().splitlines()
    ]
    assert len(lines) == len(agent.trace)
    ok_calls = [e for e in lines if e["status"] == "ok" and e["tool"] in ("simulate", "profile")]
    assert len(ok_calls) == report["budget"]["tool_calls_used"]
    assert all(e["evidence_id"] for e in ok_calls)


def test_agent_never_fabricates_measurements(tmp_path: Path) -> None:
    def broken_profile(config: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("backend offline")

    agent = _build(tmp_path, profile_fn=broken_profile)
    report = agent.run()
    assert report["status"] == "partial"
    assert not [c for c in report["claims"] if c["kind"] == "measured"]
    assert report["validation"] == []
    assert any(e["tool"] == "profile" and e["status"] == "error" for e in agent.trace)
    assert any("No successful profile call" in item for item in report["limitations"])


def test_agent_hard_caps_tool_calls(tmp_path: Path) -> None:
    agent = _build(
        tmp_path,
        budget=Budget(max_rounds=5, max_tool_calls=3, max_profiles=1),
    )
    report = agent.run()
    assert report["budget"]["tool_calls_used"] <= 3
    assert report["budget"]["profile_calls_used"] <= 1
    assert len(agent.trace) <= 3 + report["budget"]["blocked_attempts"]


def test_agent_halts_without_required_gpu(tmp_path: Path) -> None:
    agent = _build(
        tmp_path,
        require_nvidia_gpu=True,
        gpu_status_fn=lambda: {
            "available": False,
            "reason": "no nvidia-smi",
            "nvidia_claims": "none",
        },
    )
    report = agent.run()
    assert report["status"] == "insufficient_evidence"
    assert report["claims"] == []
    assert report["evidence_ids"] == []
    assert report["budget"]["tool_calls_used"] == 0
    assert "NVIDIA GPU required" in (report["halt_reason"] or "")
    assert any(
        e["tool"] == "preflight_nvidia_check" and e["status"] == "unavailable"
        for e in agent.trace
    )
    assert (tmp_path / "report.json").exists()


def test_agent_refuses_to_relabel_cpu_as_gpu(tmp_path: Path) -> None:
    agent = _build(
        tmp_path,
        require_nvidia_gpu=True,
        gpu_status_fn=lambda: {
            "available": True,
            "reason": "GPU present",
            "nvidia_claims": "none",
        },
    )
    report = agent.run()
    assert report["status"] == "insufficient_evidence"
    assert "no configured GPU profiling backend" in (report["halt_reason"] or "")
    assert report["claims"] == []


def test_all_claims_have_evidence_ids(tmp_path: Path) -> None:
    agent = _build(tmp_path)
    report = agent.run()
    assert all(c["evidence"] for c in report["claims"])
    assert all(
        eid.startswith(("sim-", "meas-"))
        for c in report["claims"]
        for eid in c["evidence"]
    )
