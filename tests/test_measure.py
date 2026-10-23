"""Unit tests for CPU measurement and calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cpu_backend import NumpyDecoder
from measure import (
    compare_measurement,
    environment_info,
    hardware_from_calibration,
    measure_calibration,
    measure_workload,
    nvidia_gpu_status,
)
from model import ModelConfig, RuntimeConfig, estimate, kv_cache_bytes


def _tiny() -> tuple[ModelConfig, RuntimeConfig]:
    model = ModelConfig(
        n_layers=1,
        hidden_size=32,
        n_heads=2,
        n_kv_heads=1,
        intermediate_size=64,
        vocab_size=64,
        tie_embeddings=True,
        gated_mlp=True,
        name="measure-tiny",
    )
    runtime = RuntimeConfig(
        batch_size=1,
        seq_len=8,
        gen_len=2,
        weight_dtype="fp32",
        kv_dtype="fp32",
        activation_dtype="fp32",
        attention_impl="fused",
        activation_traffic_factor=2.0,
    )
    return model, runtime


def test_calibration_is_positive() -> None:
    cal = measure_calibration(repeats=1, matmul_size=128, copy_mb=1)
    assert cal["backend"] == "cpu-numpy-blas-float32"
    assert cal["gpu_used"] is False
    assert cal["matmul"]["peak_flops"] > 0
    assert cal["copy"]["memory_bandwidth"] > 0
    assert cal["layer_overhead_s"] >= 0
    hw = hardware_from_calibration(cal)
    assert hw.peak_flops == cal["matmul"]["peak_flops"]
    assert hw.memory_bandwidth == cal["copy"]["memory_bandwidth"]
    assert "measured" in hw.source


def test_decoder_shapes_and_cache() -> None:
    model, runtime = _tiny()
    decoder = NumpyDecoder(model, seed=0, max_seq_len=runtime.max_context)
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, model.vocab_size, size=(runtime.batch_size, runtime.seq_len))
    logits, cache = decoder.prefill(tokens)
    assert logits.shape == (runtime.batch_size, model.vocab_size)
    assert cache.nbytes == kv_cache_bytes(model, runtime, runtime.max_context)
    next_tokens = np.argmax(logits, axis=-1)
    logits2 = decoder.decode_step(next_tokens, cache)
    assert logits2.shape == (runtime.batch_size, model.vocab_size)
    assert cache.length == runtime.seq_len + 1


def test_measure_workload_and_compare() -> None:
    model, runtime = _tiny()
    measurement = measure_workload(model, runtime, seed=0, repeats=1, warmup=0)
    assert measurement["backend"] == "cpu-numpy-blas-float32"
    assert measurement["device"] == "CPU"
    assert measurement["gpu_used"] is False
    assert "cuda" not in measurement["backend"].lower()
    assert measurement["prefill_median_s"] > 0
    assert measurement["decode_median_s"] > 0
    assert measurement["prefill_tokens_per_s"] > 0
    assert measurement["decode_tokens_per_s"] > 0
    assert measurement["kv_bytes_allocated"] == kv_cache_bytes(
        model, runtime, runtime.max_context
    )
    expected_params = model.total_params * 4
    assert measurement["param_bytes_measured"] == expected_params
    assert measurement["workload_id"]

    hardware = hardware_from_calibration(
        measure_calibration(repeats=1, matmul_size=128, copy_mb=1)
    )
    comparison = compare_measurement(measurement, hardware)
    assert [p["phase"] for p in comparison["phases"]] == ["prefill", "decode"]
    for phase in comparison["phases"]:
        assert np.isfinite(phase["error_pct"])
        assert phase["measured_ms"] > 0
        assert phase["predicted_ms"] > 0
        assert phase["measured_tokens_per_s"] > 0


def test_measure_rejects_unexecuted_dtype() -> None:
    model, runtime = _tiny()
    fp16 = RuntimeConfig(**{**runtime.__dict__, "weight_dtype": "fp16"})
    with pytest.raises(ValueError, match="float32"):
        measure_workload(model, fp16, repeats=1, warmup=0)


def test_gpu_status_never_claims_benchmark() -> None:
    status = nvidia_gpu_status()
    assert "available" in status
    assert status["nvidia_claims"].startswith("none")


def test_environment_labels_cpu() -> None:
    env = environment_info()
    assert env["backend"] == "cpu-numpy-blas-float32"
    assert env["device"] == "CPU"
    assert env["gpu_used"] is False


def test_model_kv_prediction_matches_measurement() -> None:
    model, runtime = _tiny()
    measurement = measure_workload(model, runtime, seed=0, repeats=1, warmup=0)
    est = estimate(model, runtime, hardware_from_calibration(
        measure_calibration(repeats=1, matmul_size=128, copy_mb=1)
    ))
    assert est.kv_bytes_peak == measurement["kv_bytes_allocated"]
