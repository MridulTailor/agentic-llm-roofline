"""Unit tests for the analytical cost model.

Tests verify parameter counts, FLOPs, and memory traffic against hand-computed values.
"""

from __future__ import annotations

import math

import pytest

from model import (
    HardwareConfig,
    ModelConfig,
    RuntimeConfig,
    attention_flops_decode,
    attention_flops_prefill,
    estimate,
    kv_cache_bytes,
    workload_id,
)


@pytest.fixture
def tiny_model() -> ModelConfig:
    return ModelConfig(
        n_layers=1,
        hidden_size=4,
        n_heads=2,
        n_kv_heads=1,
        intermediate_size=8,
        vocab_size=10,
        tie_embeddings=True,
        gated_mlp=False,
        name="tiny",
    )


@pytest.fixture
def tiny_runtime() -> RuntimeConfig:
    return RuntimeConfig(
        batch_size=2,
        seq_len=3,
        gen_len=1,
        weight_dtype="fp16",
        kv_dtype="fp16",
        activation_dtype="fp16",
        causal=True,
        attention_impl="fused",
        activation_traffic_factor=0.0,
        logits_last_token_only=True,
    )


@pytest.fixture
def hardware() -> HardwareConfig:
    return HardwareConfig(
        name="unit",
        peak_flops=1e12,
        memory_bandwidth=1e12,
        memory_capacity=1e9,
    )


def test_parameter_count_hand_computed(tiny_model: ModelConfig) -> None:
    assert tiny_model.head_dim == 2
    assert tiny_model.kv_dim == 2
    assert tiny_model.layer_linear_params == 112
    assert tiny_model.layer_norm_params == 8
    assert tiny_model.embedding_params == 40
    assert tiny_model.lm_head_params == 0
    assert tiny_model.total_params == 164


def test_untied_embeddings_add_lm_head(tiny_model: ModelConfig) -> None:
    untied = ModelConfig(**{**tiny_model.__dict__, "tie_embeddings": False})
    assert untied.lm_head_params == 40
    assert untied.total_params == 204


def test_kv_cache_exact(tiny_model: ModelConfig, tiny_runtime: RuntimeConfig) -> None:
    # 2 * L * B * C * Hkv * head_dim * bytes = 2*1*2*C*1*2*2 = 16*C
    assert kv_cache_bytes(tiny_model, tiny_runtime, 3) == 48
    assert kv_cache_bytes(tiny_model, tiny_runtime, 4) == 64


def test_prefill_flops_exact(tiny_model: ModelConfig, tiny_runtime: RuntimeConfig) -> None:
    # layers 2*B*S*112 = 1344; logits (last token) 2*B*d*V = 160;
    # causal attention 2*B*d*S*(S+1) = 192.
    assert attention_flops_prefill(tiny_model, tiny_runtime, 3) == 192
    est = estimate(
        tiny_model,
        tiny_runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    assert est.prefill.flops == 1696
    assert est.prefill.bytes_core == 424  # weights 328 + KV write 48 + read 48
    assert est.prefill.bytes_total == 424
    assert est.prefill.tokens == 6
    assert est.prefill.intensity == pytest.approx(1696 / 424)


def test_prefill_breakdown_exact(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    est = estimate(
        tiny_model,
        tiny_runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    phase = est.prefill
    # 2*B*S*112 = 1344, causal attention 192, last-token logits 160.
    assert phase.linear_flops == 1344
    assert phase.attention_flops == 192
    assert phase.logits_flops == 160
    assert phase.linear_flops + phase.attention_flops + phase.logits_flops == phase.flops
    # Weights 164 params * 2 B = 328; KV write 48 and read 48; no activations.
    assert phase.weight_bytes == 328
    assert phase.kv_write_bytes == 48
    assert phase.kv_read_bytes == 48
    assert phase.activation_bytes == 0
    assert phase.materialized_attention_bytes == 0
    assert (
        phase.weight_bytes + phase.kv_write_bytes + phase.kv_read_bytes
        == phase.bytes_core
        == phase.bytes_total
    )


def test_decode_breakdown_exact(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    est = estimate(
        tiny_model,
        tiny_runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    phase = est.decode
    # 2*B*L*112 = 448, attention 4*B*d*C = 128, logits 2*B*d*V = 160.
    assert phase.linear_flops == 448
    assert phase.attention_flops == 128
    assert phase.logits_flops == 160
    assert phase.linear_flops + phase.attention_flops + phase.logits_flops == phase.flops
    assert phase.weight_bytes == 328
    assert phase.kv_read_bytes == 64
    assert phase.kv_write_bytes == 16
    assert phase.activation_bytes == 0
    assert phase.materialized_attention_bytes == 0
    assert (
        phase.weight_bytes + phase.kv_read_bytes + phase.kv_write_bytes
        == phase.bytes_core
        == phase.bytes_total
    )


def test_breakdown_sums_match_totals(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    runtime = RuntimeConfig(
        **{
            **tiny_runtime.__dict__,
            "gen_len": 3,
            "attention_impl": "materialized",
            "activation_traffic_factor": 2.0,
        }
    )
    est = estimate(
        tiny_model,
        runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    for phase in (est.prefill, est.decode_first, est.decode):
        assert phase.linear_flops + phase.attention_flops + phase.logits_flops == pytest.approx(
            phase.flops
        )
        assert (
            phase.weight_bytes
            + phase.kv_read_bytes
            + phase.kv_write_bytes
            + phase.activation_bytes
            + phase.materialized_attention_bytes
        ) == pytest.approx(phase.bytes_total)
        assert (
            phase.weight_bytes + phase.kv_read_bytes + phase.kv_write_bytes
        ) == pytest.approx(phase.bytes_core)
        assert phase.activation_bytes > 0
        assert phase.materialized_attention_bytes > 0


def test_decode_flops_and_bytes_exact(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    # C = S+1 = 4. Linear 2*B*(112+40)=608; attention 4*B*d*C=128.
    assert attention_flops_decode(tiny_model, tiny_runtime, 4) == 128
    est = estimate(
        tiny_model,
        tiny_runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    assert est.decode.flops == 736
    assert est.decode.bytes_core == 408  # 328 weights + 64 KV read + 16 KV write
    assert est.decode.intensity == pytest.approx(736 / 408)
    assert est.decode_first.flops == est.decode.flops
    assert est.decode.kv_bytes == est.kv_bytes_peak == 64
    assert est.kv_bytes_prefill == 48


def test_decode_aggregate_growing_context(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    runtime = RuntimeConfig(**{**tiny_runtime.__dict__, "gen_len": 3})
    est = estimate(
        tiny_model,
        runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    # Linear 3*608 + attention 32*(4+5+6) = 1824 + 480 = 2304.
    assert est.decode.flops == 2304
    # Bytes (328+16C+16) for C=4,5,6 -> 408+424+440 = 1272.
    assert est.decode.bytes_core == 1272
    assert est.kv_bytes_peak == 96


def test_linear_flops_scale_with_layers(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    from dataclasses import replace

    two_layers = replace(tiny_model, n_layers=2)
    est = estimate(
        two_layers,
        tiny_runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    # 2*B*S*L*112 = 2688, logits 160, causal attention 192.
    assert est.prefill.flops == 3040
    # Decode: 2*B*(L*112 + 40) + 128 = 1056 + 128.
    assert est.decode.flops == 1184


def test_causal_attention_is_half_of_full(tiny_model: ModelConfig) -> None:
    full = RuntimeConfig(batch_size=2, seq_len=3, gen_len=1, causal=False)
    causal = RuntimeConfig(batch_size=2, seq_len=3, gen_len=1, causal=True)
    assert attention_flops_prefill(tiny_model, full, 3) == 288
    assert attention_flops_prefill(tiny_model, causal, 3) == 192


def test_materialized_attention_adds_exact_score_traffic(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    runtime = RuntimeConfig(**{**tiny_runtime.__dict__, "attention_impl": "materialized"})
    est = estimate(
        tiny_model,
        runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    # 2 * B * H * S * S * act_bytes = 2*2*2*3*3*2 = 144 bytes.
    assert est.prefill.bytes_total - est.prefill.bytes_core == 144


def test_quantization_changes_bytes_not_flops(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    fp32_runtime = RuntimeConfig(
        **{
            **tiny_runtime.__dict__,
            "weight_dtype": "fp32",
            "kv_dtype": "fp32",
            "activation_dtype": "fp32",
        }
    )
    fp16 = estimate(
        tiny_model,
        tiny_runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    fp32 = estimate(
        tiny_model,
        fp32_runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    assert fp32.param_bytes == 656
    assert fp32.kv_bytes_prefill == 96
    assert fp32.prefill.flops == fp16.prefill.flops


def test_logits_all_tokens_adds_projection_flops(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    runtime = RuntimeConfig(**{**tiny_runtime.__dict__, "logits_last_token_only": False})
    est = estimate(
        tiny_model,
        runtime,
        HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12),
    )
    # Extra (S-1) token projections: 2*B*2*d*V = 320.
    assert est.prefill.flops == 1696 + 320


def test_batch_amortizes_weight_traffic(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig, hardware: HardwareConfig
) -> None:
    small = estimate(tiny_model, tiny_runtime, hardware)
    large_runtime = RuntimeConfig(**{**tiny_runtime.__dict__, "batch_size": 8})
    large = estimate(tiny_model, large_runtime, hardware)
    assert large.prefill.bytes_core < 4 * small.prefill.bytes_core
    assert large.prefill.intensity > small.prefill.intensity


def test_roofline_regimes(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    compute_hw = HardwareConfig(
        name="compute", peak_flops=1e12, memory_bandwidth=1e12
    )
    memory_hw = HardwareConfig(
        name="memory", peak_flops=1e12, memory_bandwidth=1e9
    )
    compute = estimate(tiny_model, tiny_runtime, compute_hw)
    memory = estimate(tiny_model, tiny_runtime, memory_hw)
    assert compute.ridge_point_flops_per_byte == 1.0
    assert memory.ridge_point_flops_per_byte == 1000.0
    assert compute.prefill.bound == "compute"
    assert compute.decode.bound == "compute"
    assert memory.prefill.bound == "memory"
    assert memory.decode.bound == "memory"
    # max(compute, memory) and token throughput units.
    assert compute.prefill.ideal_time_s == pytest.approx(
        max(
            compute.prefill.flops / 1e12,
            compute.prefill.bytes_total / 1e12,
        )
    )
    assert compute.prefill.tokens_per_s == pytest.approx(
        compute.prefill.tokens / compute.prefill.predicted_time_s
    )


def test_kv_pressure_and_capacity(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig
) -> None:
    est = estimate(
        tiny_model,
        tiny_runtime,
        HardwareConfig(
            name="cap", peak_flops=1e12, memory_bandwidth=1e12, memory_capacity=1000
        ),
    )
    assert est.kv_bytes_peak == 64
    assert est.kv_pressure_peak == pytest.approx(0.064)
    assert est.decode.kv_pressure == pytest.approx(0.064)


def test_gqa_reduces_kv_cache(tiny_model: ModelConfig) -> None:
    mha = tiny_model.with_attention_variant("mha")
    gqa = tiny_model.with_attention_variant("gqa")
    mqa = tiny_model.with_attention_variant("mqa")
    rt = RuntimeConfig(batch_size=1, seq_len=10, gen_len=1)
    assert mha.n_kv_heads == 2 and gqa.n_kv_heads == 1 and mqa.n_kv_heads == 1
    assert kv_cache_bytes(gqa, rt, 10) == kv_cache_bytes(mha, rt, 10) // 2


def test_stable_identifiers(tiny_model: ModelConfig, tiny_runtime: RuntimeConfig) -> None:
    hw = HardwareConfig(name="u", peak_flops=1e12, memory_bandwidth=1e12)
    a = estimate(tiny_model, tiny_runtime, hw)
    b = estimate(tiny_model, tiny_runtime, hw)
    assert a.config_id == b.config_id
    assert a.workload_id == workload_id(tiny_model, tiny_runtime)
    faster = HardwareConfig(name="u2", peak_flops=2e12, memory_bandwidth=1e12)
    assert estimate(tiny_model, tiny_runtime, faster).config_id != a.config_id


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        ModelConfig(
            n_layers=1,
            hidden_size=10,
            n_heads=3,
            n_kv_heads=1,
            intermediate_size=8,
            vocab_size=10,
        )
    with pytest.raises(ValueError):
        RuntimeConfig(batch_size=1, seq_len=1, gen_len=1, weight_dtype="int4")
    with pytest.raises(ValueError):
        HardwareConfig(name="bad", peak_flops=0, memory_bandwidth=1)


def test_predicted_time_is_finite(
    tiny_model: ModelConfig, tiny_runtime: RuntimeConfig, hardware: HardwareConfig
) -> None:
    est = estimate(tiny_model, tiny_runtime, hardware)
    for phase in (est.prefill, est.decode, est.decode_first):
        assert math.isfinite(phase.predicted_time_s)
        assert phase.predicted_time_s >= phase.ideal_time_s
