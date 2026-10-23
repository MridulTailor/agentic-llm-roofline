"""Analytical LLM inference cost model for prefill and decode phases.

Units are standard SI units: bytes, FLOPs, seconds, FLOP/s, and bytes/s.
The model calculates computational work and memory traffic, then evaluates them
against nominal hardware ceilings. The model does not simulate GPU execution.

Simplifying assumptions are listed in ASSUMPTIONS and documented in README.md.
Activation traffic uses a configurable heuristic factor.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

DTYPE_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
}

AttentionVariant = Literal["mha", "gqa", "mqa"]
AttentionImpl = Literal["fused", "materialized"]

FORMULA_VERSION = "1.0"

ASSUMPTIONS: tuple[str, ...] = (
    "Dense decoder-only transformer with pre-norm blocks, no biases, and no "
    "positional-embedding table (RoPE assumed).",
    "Linear-layer FLOPs use 2*weights*tokens (one multiply-add = 2 FLOPs). "
    "Embedding lookups and norm/elementwise FLOPs are ignored.",
    "Attention FLOPs count QK^T and AV only. Causal prefill uses the exact "
    "triangular pair count; decode is non-causal because one query sees the "
    "whole context.",
    "Weights are streamed once per forward (prefill) or once per decode step. "
    "No weight reuse across batches beyond that single stream.",
    "KV-cache traffic is 2 (K and V) * layers * batch * context * kv_heads * "
    "head_dim * kv_bytes. Prefill writes and reads the new KV; decode reads "
    "the existing cache and writes one new token.",
    "Activation traffic is a heuristic: activation_traffic_factor * layers * "
    "batch * tokens * hidden * activation_bytes. Set the factor to 0 for the "
    "core (weights + KV) byte count.",
    "If attention_impl='materialized', the prefill/decode score matrix is "
    "written and read once: 2 * batch * heads * query_len * key_len * "
    "activation_bytes. 'fused' models a FlashAttention-style kernel.",
    "Predicted latency = max(compute time, memory time) + per-layer launch "
    "overhead. It is an idealized lower bound when overhead is zero.",
    "Nominal hardware peaks are datasheet/spec ceilings, not measured "
    "throughput. For measured CPU validation a calibrated hardware profile is "
    "supplied explicitly.",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ModelConfig:
    """Static transformer shape. Sizes are element counts, not bytes."""

    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    intermediate_size: int
    vocab_size: int
    tie_embeddings: bool = True
    gated_mlp: bool = True
    name: str = "model"

    def __post_init__(self) -> None:
        _require(self.n_layers > 0, "n_layers must be > 0")
        _require(self.hidden_size > 0, "hidden_size must be > 0")
        _require(self.n_heads > 0, "n_heads must be > 0")
        _require(self.n_kv_heads > 0, "n_kv_heads must be > 0")
        _require(
            self.hidden_size % self.n_heads == 0,
            "hidden_size must be divisible by n_heads",
        )
        _require(
            self.n_heads % self.n_kv_heads == 0,
            "n_heads must be divisible by n_kv_heads (GQA grouping)",
        )
        _require(self.intermediate_size > 0, "intermediate_size must be > 0")
        _require(self.vocab_size > 0, "vocab_size must be > 0")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def attention_variant(self) -> AttentionVariant:
        if self.n_kv_heads == self.n_heads:
            return "mha"
        if self.n_kv_heads == 1:
            return "mqa"
        return "gqa"

    @property
    def layer_linear_params(self) -> int:
        """Linear layer parameter count (Q, K, V, output projection, and MLP)."""
        q = self.hidden_size * self.hidden_size
        kv = self.hidden_size * self.kv_dim
        o = self.hidden_size * self.hidden_size
        mlp = (3 if self.gated_mlp else 2) * self.hidden_size * self.intermediate_size
        return q + 2 * kv + o + mlp

    @property
    def layer_norm_params(self) -> int:
        return 2 * self.hidden_size

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def lm_head_params(self) -> int:
        return 0 if self.tie_embeddings else self.vocab_size * self.hidden_size

    @property
    def total_params(self) -> int:
        return (
            self.embedding_params
            + self.n_layers * (self.layer_linear_params + self.layer_norm_params)
            + self.lm_head_params
            + self.hidden_size
        )

    def with_attention_variant(self, variant: AttentionVariant) -> "ModelConfig":
        if variant == "mha":
            kv_heads = self.n_heads
        elif variant == "mqa":
            kv_heads = 1
        elif variant == "gqa":
            kv_heads = self.n_kv_heads
        else:  # pragma: no cover - guarded by the type alias
            raise ValueError(f"unknown attention variant: {variant}")
        return replace(self, n_kv_heads=kv_heads)


@dataclass(frozen=True)
class RuntimeConfig:
    """Workload dimensions and precision settings for inference."""

    batch_size: int
    seq_len: int
    gen_len: int
    weight_dtype: str = "fp16"
    kv_dtype: str | None = None
    activation_dtype: str | None = None
    causal: bool = True
    attention_impl: AttentionImpl = "fused"
    activation_traffic_factor: float = 4.0
    logits_last_token_only: bool = True

    def __post_init__(self) -> None:
        _require(self.batch_size > 0, "batch_size must be > 0")
        _require(self.seq_len > 0, "seq_len must be > 0")
        _require(self.gen_len > 0, "gen_len must be > 0")
        _require(self.activation_traffic_factor >= 0, "activation factor must be >= 0")
        for label, dtype in (
            ("weight_dtype", self.weight_dtype),
            ("kv_dtype", self.kv_dtype or self.weight_dtype),
            ("activation_dtype", self.activation_dtype or self.weight_dtype),
        ):
            _require(dtype in DTYPE_BYTES, f"unknown {label}: {dtype}")
        _require(self.attention_impl in ("fused", "materialized"), "bad attention_impl")

    @property
    def weight_bytes(self) -> float:
        return DTYPE_BYTES[self.weight_dtype]

    @property
    def kv_bytes(self) -> float:
        return DTYPE_BYTES[self.kv_dtype or self.weight_dtype]

    @property
    def activation_bytes(self) -> float:
        return DTYPE_BYTES[self.activation_dtype or self.weight_dtype]

    @property
    def max_context(self) -> int:
        return self.seq_len + self.gen_len


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware performance limits and measured dispatch overhead."""

    name: str
    peak_flops: float
    memory_bandwidth: float
    memory_capacity: float = 0.0
    compute_efficiency: float = 1.0
    bandwidth_efficiency: float = 1.0
    layer_overhead_s: float = 0.0
    source: str = "nominal specification (theoretical, not measured)"
    notes: str = ""

    def __post_init__(self) -> None:
        _require(self.peak_flops > 0, "peak_flops must be > 0")
        _require(self.memory_bandwidth > 0, "memory_bandwidth must be > 0")
        _require(0 < self.compute_efficiency <= 1, "compute_efficiency in (0, 1]")
        _require(0 < self.bandwidth_efficiency <= 1, "bandwidth_efficiency in (0, 1]")
        _require(self.layer_overhead_s >= 0, "layer_overhead_s must be >= 0")

    @property
    def effective_peak_flops(self) -> float:
        return self.peak_flops * self.compute_efficiency

    @property
    def effective_bandwidth(self) -> float:
        return self.memory_bandwidth * self.bandwidth_efficiency

    @property
    def ridge_point(self) -> float:
        """Operational intensity where compute and bandwidth ceilings intersect."""
        return self.effective_peak_flops / self.effective_bandwidth


@dataclass(frozen=True)
class PhaseEstimate:
    phase: str
    tokens: int
    flops: float
    bytes_total: float
    bytes_core: float
    intensity: float
    intensity_core: float
    compute_time_s: float
    memory_time_s: float
    ideal_time_s: float
    launch_time_s: float
    predicted_time_s: float
    tokens_per_s: float
    bound: Literal["compute", "memory"]
    kv_bytes: int
    kv_pressure: float | None
    linear_flops: float = 0.0
    attention_flops: float = 0.0
    logits_flops: float = 0.0
    weight_bytes: float = 0.0
    kv_read_bytes: float = 0.0
    kv_write_bytes: float = 0.0
    activation_bytes: float = 0.0
    materialized_attention_bytes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceEstimate:
    config_id: str
    workload_id: str
    model: ModelConfig
    runtime: RuntimeConfig
    hardware: HardwareConfig
    param_bytes: float
    kv_bytes_prefill: int
    kv_bytes_peak: int
    kv_pressure_peak: float | None
    prefill: PhaseEstimate
    decode_first: PhaseEstimate
    decode: PhaseEstimate
    end_to_end_tokens_per_s: float
    ridge_point_flops_per_byte: float
    formulas_version: str = FORMULA_VERSION
    assumptions: tuple[str, ...] = ASSUMPTIONS

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "workload_id": self.workload_id,
            "model": asdict(self.model),
            "runtime": asdict(self.runtime),
            "hardware": asdict(self.hardware),
            "param_bytes": self.param_bytes,
            "kv_bytes_prefill": self.kv_bytes_prefill,
            "kv_bytes_peak": self.kv_bytes_peak,
            "kv_pressure_peak": self.kv_pressure_peak,
            "prefill": self.prefill.to_dict(),
            "decode_first": self.decode_first.to_dict(),
            "decode": self.decode.to_dict(),
            "end_to_end_tokens_per_s": self.end_to_end_tokens_per_s,
            "ridge_point_flops_per_byte": self.ridge_point_flops_per_byte,
            "formulas_version": self.formulas_version,
            "assumptions": list(self.assumptions),
        }


def kv_cache_bytes(model: ModelConfig, runtime: RuntimeConfig, context: int) -> int:
    """Calculate key and value cache size in bytes for a given context length."""
    value = (
        2
        * model.n_layers
        * runtime.batch_size
        * context
        * model.n_kv_heads
        * model.head_dim
        * runtime.kv_bytes
    )
    return int(round(value))


def attention_flops_prefill(
    model: ModelConfig, runtime: RuntimeConfig, seq_len: int
) -> float:
    batch = runtime.batch_size
    if runtime.causal:
        return 2.0 * batch * model.hidden_size * seq_len * (seq_len + 1)
    return 4.0 * batch * model.hidden_size * seq_len * seq_len


def attention_flops_decode(
    model: ModelConfig, runtime: RuntimeConfig, context: int
) -> float:
    return 4.0 * runtime.batch_size * model.hidden_size * context


def _phase(
    *,
    phase: str,
    tokens: int,
    flops: float,
    bytes_total: float,
    bytes_core: float,
    hardware: HardwareConfig,
    launch_time_s: float,
    kv_bytes: int,
    linear_flops: float,
    attention_flops: float,
    logits_flops: float,
    weight_bytes: float,
    kv_read_bytes: float,
    kv_write_bytes: float,
    activation_bytes: float,
    materialized_attention_bytes: float,
) -> PhaseEstimate:
    compute_time = flops / hardware.effective_peak_flops
    memory_time = bytes_total / hardware.effective_bandwidth
    ideal = max(compute_time, memory_time)
    predicted = ideal + launch_time_s
    bound: Literal["compute", "memory"] = (
        "compute" if compute_time >= memory_time else "memory"
    )
    pressure = kv_bytes / hardware.memory_capacity if hardware.memory_capacity > 0 else None
    return PhaseEstimate(
        phase=phase,
        tokens=tokens,
        flops=float(flops),
        bytes_total=float(bytes_total),
        bytes_core=float(bytes_core),
        intensity=float(flops / bytes_total) if bytes_total > 0 else math.inf,
        intensity_core=float(flops / bytes_core) if bytes_core > 0 else math.inf,
        compute_time_s=float(compute_time),
        memory_time_s=float(memory_time),
        ideal_time_s=float(ideal),
        launch_time_s=float(launch_time_s),
        predicted_time_s=float(predicted),
        tokens_per_s=float(tokens / predicted) if predicted > 0 else math.inf,
        bound=bound,
        kv_bytes=int(kv_bytes),
        kv_pressure=float(pressure) if pressure is not None else None,
        linear_flops=float(linear_flops),
        attention_flops=float(attention_flops),
        logits_flops=float(logits_flops),
        weight_bytes=float(weight_bytes),
        kv_read_bytes=float(kv_read_bytes),
        kv_write_bytes=float(kv_write_bytes),
        activation_bytes=float(activation_bytes),
        materialized_attention_bytes=float(materialized_attention_bytes),
    )


def _activation_bytes(
    model: ModelConfig, runtime: RuntimeConfig, tokens: int
) -> float:
    return (
        runtime.activation_traffic_factor
        * model.n_layers
        * runtime.batch_size
        * tokens
        * model.hidden_size
        * runtime.activation_bytes
    )


def _materialized_attention_bytes(
    model: ModelConfig, runtime: RuntimeConfig, query_len: int, key_len: int
) -> float:
    if runtime.attention_impl != "materialized":
        return 0.0
    return (
        2.0
        * runtime.batch_size
        * model.n_heads
        * query_len
        * key_len
        * runtime.activation_bytes
    )


def _prefill_phase(
    model: ModelConfig, runtime: RuntimeConfig, hardware: HardwareConfig
) -> PhaseEstimate:
    s = runtime.seq_len
    b = runtime.batch_size
    layer_linear = 2.0 * b * s * model.n_layers * model.layer_linear_params
    logits_tokens = 1 if runtime.logits_last_token_only else s
    logits = 2.0 * b * logits_tokens * model.vocab_size * model.hidden_size
    attention = attention_flops_prefill(model, runtime, s)
    flops = layer_linear + logits + attention

    weights = model.total_params * runtime.weight_bytes
    kv = kv_cache_bytes(model, runtime, s)
    kv_read = kv_cache_bytes(model, runtime, s)
    activation = _activation_bytes(model, runtime, s)
    materialized = _materialized_attention_bytes(model, runtime, s, s)
    core = weights + kv + kv_read
    total = core + activation + materialized
    return _phase(
        phase="prefill",
        tokens=b * s,
        flops=flops,
        bytes_total=total,
        bytes_core=core,
        hardware=hardware,
        launch_time_s=model.n_layers * hardware.layer_overhead_s,
        kv_bytes=kv,
        linear_flops=layer_linear,
        attention_flops=attention,
        logits_flops=logits,
        weight_bytes=weights,
        kv_read_bytes=kv_read,
        kv_write_bytes=kv,
        activation_bytes=activation,
        materialized_attention_bytes=materialized,
    )


def _decode_step_phase(
    model: ModelConfig,
    runtime: RuntimeConfig,
    hardware: HardwareConfig,
    context: int,
) -> PhaseEstimate:
    b = runtime.batch_size
    linear = 2.0 * b * model.n_layers * model.layer_linear_params
    logits = 2.0 * b * model.vocab_size * model.hidden_size
    attention = attention_flops_decode(model, runtime, context)
    flops = linear + logits + attention

    weights = model.total_params * runtime.weight_bytes
    kv_read = kv_cache_bytes(model, runtime, context)
    kv_write = kv_cache_bytes(model, runtime, 1)
    activation = _activation_bytes(model, runtime, 1)
    materialized = _materialized_attention_bytes(model, runtime, 1, context)
    core = weights + kv_read + kv_write
    total = core + activation + materialized
    return _phase(
        phase="decode_step",
        tokens=b,
        flops=flops,
        bytes_total=total,
        bytes_core=core,
        hardware=hardware,
        launch_time_s=model.n_layers * hardware.layer_overhead_s,
        kv_bytes=kv_read,
        linear_flops=linear,
        attention_flops=attention,
        logits_flops=logits,
        weight_bytes=weights,
        kv_read_bytes=kv_read,
        kv_write_bytes=kv_write,
        activation_bytes=activation,
        materialized_attention_bytes=materialized,
    )


def _decode_aggregate(
    model: ModelConfig, runtime: RuntimeConfig, hardware: HardwareConfig
) -> PhaseEstimate:
    """Aggregate execution metrics across all generation steps."""
    flops = 0.0
    bytes_total = 0.0
    bytes_core = 0.0
    compute_time = 0.0
    memory_time = 0.0
    linear_flops = 0.0
    attention_flops = 0.0
    logits_flops = 0.0
    weight_bytes = 0.0
    kv_read_bytes = 0.0
    kv_write_bytes = 0.0
    activation_bytes = 0.0
    materialized_attention_bytes = 0.0
    for step in range(1, runtime.gen_len + 1):
        phase = _decode_step_phase(model, runtime, hardware, runtime.seq_len + step)
        flops += phase.flops
        bytes_total += phase.bytes_total
        bytes_core += phase.bytes_core
        compute_time += phase.compute_time_s
        memory_time += phase.memory_time_s
        linear_flops += phase.linear_flops
        attention_flops += phase.attention_flops
        logits_flops += phase.logits_flops
        weight_bytes += phase.weight_bytes
        kv_read_bytes += phase.kv_read_bytes
        kv_write_bytes += phase.kv_write_bytes
        activation_bytes += phase.activation_bytes
        materialized_attention_bytes += phase.materialized_attention_bytes
    ideal = max(compute_time, memory_time)
    launch = model.n_layers * hardware.layer_overhead_s * runtime.gen_len
    predicted = ideal + launch
    kv_peak = kv_cache_bytes(model, runtime, runtime.max_context)
    pressure = kv_peak / hardware.memory_capacity if hardware.memory_capacity > 0 else None
    return PhaseEstimate(
        phase="decode",
        tokens=runtime.batch_size * runtime.gen_len,
        flops=float(flops),
        bytes_total=float(bytes_total),
        bytes_core=float(bytes_core),
        intensity=float(flops / bytes_total) if bytes_total > 0 else math.inf,
        intensity_core=float(flops / bytes_core) if bytes_core > 0 else math.inf,
        compute_time_s=float(compute_time),
        memory_time_s=float(memory_time),
        ideal_time_s=float(ideal),
        launch_time_s=float(launch),
        predicted_time_s=float(predicted),
        tokens_per_s=float(runtime.batch_size * runtime.gen_len / predicted),
        bound="compute" if compute_time >= memory_time else "memory",
        kv_bytes=kv_peak,
        kv_pressure=float(pressure) if pressure is not None else None,
        linear_flops=float(linear_flops),
        attention_flops=float(attention_flops),
        logits_flops=float(logits_flops),
        weight_bytes=float(weight_bytes),
        kv_read_bytes=float(kv_read_bytes),
        kv_write_bytes=float(kv_write_bytes),
        activation_bytes=float(activation_bytes),
        materialized_attention_bytes=float(materialized_attention_bytes),
    )


def _stable_id(payload: dict[str, Any], length: int = 12) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]


def workload_id(model: ModelConfig, runtime: RuntimeConfig) -> str:
    """Generate a unique identifier for a model and runtime configuration."""
    return _stable_id({"model": asdict(model), "runtime": asdict(runtime)})


def estimate(
    model: ModelConfig,
    runtime: RuntimeConfig,
    hardware: HardwareConfig,
) -> InferenceEstimate:
    """Compute prefill and decode performance estimates for a workload."""
    param_bytes = float(model.total_params * runtime.weight_bytes)
    kv_prefill = kv_cache_bytes(model, runtime, runtime.seq_len)
    kv_peak = kv_cache_bytes(model, runtime, runtime.max_context)
    pressure = kv_peak / hardware.memory_capacity if hardware.memory_capacity > 0 else None

    prefill = _prefill_phase(model, runtime, hardware)
    decode_first = _decode_step_phase(
        model, runtime, hardware, runtime.seq_len + 1
    )
    decode = _decode_aggregate(model, runtime, hardware)
    e2e = (
        runtime.batch_size * (runtime.seq_len + runtime.gen_len)
        / (prefill.predicted_time_s + decode.predicted_time_s)
    )
    config_id = _stable_id(
        {
            "model": asdict(model),
            "runtime": asdict(runtime),
            "hardware": asdict(hardware),
        }
    )
    return InferenceEstimate(
        config_id=config_id,
        workload_id=workload_id(model, runtime),
        model=model,
        runtime=runtime,
        hardware=hardware,
        param_bytes=param_bytes,
        kv_bytes_prefill=kv_prefill,
        kv_bytes_peak=kv_peak,
        kv_pressure_peak=float(pressure) if pressure is not None else None,
        prefill=prefill,
        decode_first=decode_first,
        decode=decode,
        end_to_end_tokens_per_s=float(e2e),
        ridge_point_flops_per_byte=hardware.ridge_point,
    )


def model_from_dict(data: dict[str, Any]) -> ModelConfig:
    fields = set(ModelConfig.__dataclass_fields__)
    return ModelConfig(**{k: v for k, v in data.items() if k in fields})


def runtime_from_dict(data: dict[str, Any]) -> RuntimeConfig:
    fields = set(RuntimeConfig.__dataclass_fields__)
    return RuntimeConfig(**{k: v for k, v in data.items() if k in fields})


def hardware_from_dict(data: dict[str, Any]) -> HardwareConfig:
    fields = set(HardwareConfig.__dataclass_fields__)
    return HardwareConfig(**{k: v for k, v in data.items() if k in fields})


def estimate_from_dicts(
    model: dict[str, Any],
    runtime: dict[str, Any],
    hardware: dict[str, Any],
) -> InferenceEstimate:
    return estimate(model_from_dict(model), runtime_from_dict(runtime), hardware_from_dict(hardware))


def format_bytes(value: float | int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(value)
    for unit in units:
        if abs(x) < 1024.0 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"
