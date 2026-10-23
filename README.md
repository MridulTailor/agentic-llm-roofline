# Agentic LLM Inference Roofline Lab

An analytical roofline and design-space laboratory for decoder-only LLM inference,
with a bounded autonomous search agent and CPU performance validation.

The project distinguishes three performance metrics:

1. **Analytical predictions** from an explicit cost model and a hardware profile.
2. **Nominal GPU ceilings** from manufacturer specifications (theoretical bounds).
3. **Measured latency** from the local CPU reference backend.

This repository does not benchmark NVIDIA GPUs. The local benchmark backend is
`cpu-numpy-blas-float32` on Apple Silicon.

## System capabilities

| Component | Function | Implementation |
|---|---|---|
| Cost model | Calculates transformer parameters, KV cache size, FLOPs, memory traffic, and operational intensity | Analytical model (`src/model.py`) |
| Roofline analysis | Evaluates operational intensity against bandwidth and compute ceilings | Plots and tabular summaries (`src/roofline.py`) |
| Parameter sweep | Evaluates batch size, sequence length, precision, and attention variants | 162 hardware configurations (`src/sweep.py`) |
| Hardware validation | Measures reference CPU execution times and compares against analytical bounds | NumPy reference backend (`src/measure.py`) |
| Autonomous search agent | Explores configuration spaces with budget-constrained tool calls | Deterministic policy with trace logging (`src/agent.py`) |

## Repository layout

```text
agentic-llm-roofline/
  README.md
  pyproject.toml
  src/model.py          # analytical cost model
  src/roofline.py       # roofline math and plots
  src/sweep.py          # design-space sweep
  src/cpu_backend.py    # NumPy reference decoder (measured workload)
  src/measure.py        # calibration, CPU profiling, validation, GPU detection
  src/agent.py          # bounded agent, trace, evidence-cited report
  tests/                # hand-computed, sweep, measurement, agent tests
  experiments/configs.yaml
  scripts/run_all.sh
  results/              # generated logs, JSON/CSV, plots
```

## Quick start

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'

# 1. Calibrate the CPU reference backend
.venv/bin/python -m measure calibrate --out results/calibration_cpu.json

# 2. Run the analytical parameter sweep
.venv/bin/python -m sweep --config experiments/configs.yaml \
  --experiment llama_analytical --out-dir results/sweep_analytical

# 3. Run the CPU-calibrated sweep
.venv/bin/python -m sweep --config experiments/configs.yaml \
  --experiment tiny_cpu_calibrated --out-dir results/sweep_tiny_cpu \
  --calibration results/calibration_cpu.json

# 4. Run CPU validation and generate comparison charts
.venv/bin/python -m measure validation --config experiments/configs.yaml \
  --calibration results/calibration_cpu.json --out-dir results/validation

# 5. Execute the bounded search agent
.venv/bin/python -m agent --config experiments/configs.yaml \
  --section agent_search --calibration results/calibration_cpu.json \
  --out-dir results/agent

# 6. Check GPU status
.venv/bin/python -m measure gpu-status --out results/gpu_status.json

# 7. Test safe-halt condition (exits with code 2)
.venv/bin/python -m agent --config experiments/configs.yaml \
  --section agent_search --calibration results/calibration_cpu.json \
  --out-dir results/agent_nvidia_required --require-nvidia-gpu

.venv/bin/pytest
```

Run all steps: `bash scripts/run_all.sh`.

## Cost model

### Symbols and units

| Symbol | Meaning | Unit |
|---|---|---|
| `L` | transformer layers | count |
| `d` | hidden size | elements |
| `H`, `H_kv` | query heads, KV heads | count |
| `h_d = d/H` | head dimension | elements |
| `I` | MLP intermediate size | elements |
| `V` | vocabulary size | tokens |
| `B`, `S`, `G` | batch, prompt length, generated tokens | count |
| `C` | context length at a decode step | tokens |
| `P` | parameter count | elements |
| `W` | weight bytes | bytes |
| `K(C)` | KV-cache bytes at context `C` | bytes |
| `F` | floating-point operations | FLOP |
| `I_op` | operational intensity `F / bytes` | FLOP/byte |
| `P_peak`, `BW` | compute ceiling, bandwidth ceiling | FLOP/s, byte/s |
| `t` | latency | s |

### Parameters (bias-free, final norm included)

```text
P_embed  = V*d
P_layer  = d*d              # q
         + d*(H_kv*h_d)     # k
         + d*(H_kv*h_d)     # v
         + d*d              # o
         + (3*d*I if gated_mlp else 2*d*I)
         + 2*d              # two RMSNorm weights
P_total  = P_embed + L*P_layer + (0 if tied else V*d) + d
W        = P_total * weight_bytes
```

`head_dim = d/H`; GQA changes `H_kv`, not `H`. The model supports MHA
(`H_kv=H`), GQA (`1 < H_kv < H`) and MQA (`H_kv=1`).

### KV cache

```text
K(C) = 2 * L * B * C * H_kv * h_d * kv_bytes
```

The factor 2 is K and V. `kv_bytes` is the KV dtype width (1/2/4 bytes).

### Prefill FLOPs

With prompt length `S`:

```text
F_linear  = 2 * B * S * L * P_layer_linear
F_logits  = 2 * B * V * d                  # last-token logits only
F_attn    = 2 * B * d * S * (S + 1)        # causal QK^T + AV
F_prefill = F_linear + F_logits + F_attn
```

The causal attention count uses the exact triangular pair count. If
`logits_last_token_only=false`, add `2*B*(S-1)*V*d`. Full (non-causal)
attention replaces `F_attn` with `4*B*d*S^2`.

### Decode FLOPs

For one step at context `C`:

```text
F_step(C) = 2*B*(L*P_layer_linear + V*d) + 4*B*d*C
```

The attention term is not halved: the single query sees all `C` cached keys.
Generation aggregates over `C = S+1 ... S+G`.

### Traffic model

Weights are streamed once per prefill or once per decode step. KV traffic is
directional.

```text
W = P_total * weight_bytes

K_prefill = K(S)
prefill_core_bytes = W + 2*K_prefill
# 2*K: write new K/V, then read K/V for attention

K_read  = K(C)
K_write = K(1)
decode_core_bytes(C) = W + K_read + K_write

activation_bytes = f * L * B * tokens * d * activation_bytes
materialized_attention_bytes =
    2 * B * H * query_len * key_len * activation_bytes   # if materialized
```

`f = activation_traffic_factor` is an explicit heuristic (default 4). Set it to
0 for the core byte count. `attention_impl="fused"` models a
FlashAttention-style kernel (no score-matrix traffic); `"materialized"` adds
the score-matrix write+read. The model reports both `bytes_core` and
`bytes_total`. Every phase estimate also exposes its components
(`linear_flops`, `attention_flops`, `logits_flops`, `weight_bytes`,
`kv_read_bytes`, `kv_write_bytes`, `activation_bytes`,
`materialized_attention_bytes`); FLOP components sum to `flops` and byte
components sum to `bytes_total`.

### Roofline and latency

```text
I_op  = F / bytes_total
P(I)  = min(P_peak * compute_efficiency, BW * bandwidth_efficiency * I_op)
t_ideal = max(F / (P_peak * e_c), bytes_total / (BW * e_b))
t_pred  = t_ideal + L * layer_overhead_s * forwards
```

`t_pred` is an **idealized lower bound plus dispatch correction**, not a
kernel-level simulation. `layer_overhead_s` is zero for nominal GPU profiles;
for the CPU profile it is measured from an `L=1` vs `L=5` microbenchmark. A
phase is reported as `compute`-bound when compute time >= memory time, else
`memory`-bound.

Ridge point: `P_peak * e_c / (BW * e_b)` FLOP/byte.

## Assumptions

- Dense decoder-only transformer, pre-norm, no biases, RoPE assumed.
- One multiply-add = 2 FLOPs; norm and elementwise FLOPs ignored.
- Embedding lookups ignored; logits projection included.
- All weights are read from HBM once per forward/prefill or per decode step.
- KV-cache traffic is written then read during prefill; read plus one-token
  write during decode.
- Activation traffic is the heuristic above, not a cache simulator.
- Nominal GPU peaks are datasheet ceilings. Real kernels achieve less.
- FP8 changes byte widths but the compute ceiling is held at the FP16 preset;
  FP8 improvements in the sweep are therefore memory-side only.
- No multi-GPU, no tensor/pipeline parallelism, no paged-attention allocator,
  no prefix caching, no batching scheduler.

## Results

Environment: Apple M1 Pro, 8 logical CPU cores, 16 GiB RAM, macOS 15.3.2,
Python 3.13.3, NumPy 2.5.4, backend `cpu-numpy-blas-float32`. No GPU.

### CPU calibration (`results/calibration_cpu.json`)

| Quantity | Measured |
|---|---|
| Dense 2048x2048 FP32 matmul | 1.606 TFLOP/s |
| Large copy bandwidth | 89.2 GB/s |
| Per-layer dispatch slope (tiny model) | 0.0488 ms/layer |

These are **effective CPU implementation** numbers, not hardware datasheet
peaks.

### Measured validation (`results/validation/`)

Eight workloads (two model sizes, batch 1/8, context 64-256, 8 generated
tokens), 3 repeats each, median latency. Full table:
`results/validation/validation.csv`; charts:
`predicted_vs_measured.png`, `throughput_scaling.png`.

| Phase | n | Median absolute error | Median measured / predicted | Median achieved fraction of ideal |
|---|---:|---:|---:|---:|
| Prefill | 8 | 91.7% | 12.1x slower | 5.7% |
| Decode | 8 | 84.3% | 6.4x slower | 6.4% |

Representative points:

| Workload | Phase | Predicted | Measured | Error |
|---|---|---:|---:|---:|
| tiny-4x128, B=1, S=64 | prefill | 0.27 ms | 1.47 ms | -81.3% |
| tiny-4x128, B=1, S=64 | decode | 1.97 ms | 3.57 ms | -44.7% |
| small-8x256, B=8, S=256 | prefill | 20.0 ms | 271.8 ms | -92.7% |
| small-8x256, B=8, S=256 | decode | 6.83 ms | 74.0 ms | -90.8% |

**Analysis:** The analytical model identifies the operational regime correctly
(prefill compute-bound, decode memory-bound). The reference NumPy backend
operates below the theoretical bound due to software overhead:

- **Dispatch overhead:** Python and NumPy execute multiple small kernels per layer. Small matrix multiplications do not reach large-matrix BLAS peak throughput.
- **Materialized attention:** The reference decoder writes and reads full attention score matrices instead of using fused attention kernels.
- **Memory allocations:** Array transposition, concatenation, and reduction allocate temporary memory buffers.
- **Cache hierarchy:** The model assumes uniform memory bandwidth. Real hardware operates across multi-level CPU caches.
- **Sequential execution:** Operations execute sequentially without kernel overlap.
- **Small matrix shapes:** At small hidden dimensions (d=128-256), framework overhead represents a large fraction of total execution time.

### Nominal GPU sweep (`results/sweep_analytical/`)

162 configurations: H100, a bandwidth-constrained H100 variant, and A100;
batch 1/8/32; context 512/2048/8192; fp16/fp8; MHA/GQA/MQA. All analytical.

| Finding | Value |
|---|---|
| Best predicted decode throughput | 13,433 tokens/s (H100, MQA, B=32, S=512, fp8) |
| Highest peak KV pressure | 174.5% of 80 GB (MHA, B=32, S=8192, fp16; does not fit) |
| Same shape with GQA | 43.6% KV pressure |
| H100 GQA B=1, S=8192 | prefill 4,288 FLOP/byte, compute-bound; decode 0.88 FLOP/byte, memory-bound |
| H100 GQA B=32, S=8192 | decode intensity rises to 9.55 FLOP/byte; 2,113 tokens/s predicted |

Charts: `pareto.png`, `batch_scaling.png`, `roofline_best.png`,
`roofline_batch1.png`. Full data: `sweep.csv`, `sweep.json`, `summary.md`.

These are **nominal-spec predictions**. They are not measured H100/A100
results and should not be cited as such.

### CPU-calibrated sweep (`results/sweep_tiny_cpu/`)

27 analytical configurations for the tiny model using the measured CPU
profile. Useful as a sensitivity check; absolute tokens/s remain idealized
upper bounds.

### Bounded agent (`results/agent/`)

Latest run: **12/12 tool calls, 2/2 profiles, 8 evidence-cited claims**.
Top predicted decode configuration: `agent-tiny-4x128/MQA/B=8/S=64/fp32`,
31,901 tokens/s predicted. Profiled top two configurations; measured decode
was 4,308-4,934 tokens/s, giving +547% to +624% error (model over-predicts),
consistent with the validation gap above.

Files: `agent_trace.jsonl` (one JSON object per invocation),
`report.json` (machine-readable claims with evidence IDs),
`report.md`, `roofline_best.png`, `predicted_vs_measured.png`.

Safe halt: `results/agent_nvidia_required/` was produced with
`--require-nvidia-gpu`. It has `status=insufficient_evidence`, zero claims and
zero tool calls.

## Agent design

The agent is a **deterministic bounded policy**, not an LLM planner. It does
not call an external model API. This makes runs reproducible and removes any
path for the planner to hallucinate measurements.

- **Tools:** `simulate` (analytical model) and `profile` (measured CPU
  backend). Every invocation is appended to JSONL with timestamp, arguments,
  status, duration, result and evidence ID.
- **Budgets:** `max_rounds=3`, `max_tool_calls=12`, `max_profiles=2`.
  Attempts beyond budget are logged as blocked and stop the loop.
- **Policy:** coarse batch/attention coverage at the middle context, then
  neighbors of the best simulated configuration, then remaining candidates.
  It ranks only successful simulations by predicted decode throughput.
- **Evidence discipline:** numeric claims are constructed only from stored
  tool results. A failed profile never produces a measured claim. Claims cite
  `[sim-...]` and `[meas-...]` IDs that exist in the trace.
- **Safe halt:** if an NVIDIA GPU is required but unavailable, the agent emits
  `insufficient_evidence` with no claims. If a GPU is detected but no GPU
  profiling backend is configured, it also refuses to relabel CPU data.

## GPU benchmarking protocol (future work)

This repository does not benchmark GPU hardware. For future GPU evaluation,
follow these steps:

1. Record GPU model, driver version, CUDA version, inference backend, and commit hash.
2. Configure fixed clock frequencies and power limits, execute warmup iterations, and run repeated trials.
3. Benchmark the inference backend using identical shapes from the analytical sweep.
4. Collect `nsys profile` timelines and `ncu` kernel metrics (DRAM throughput, SM utilization, tensor core utilization).
5. Compare measured latency with analytical bounds; record variance and throttling metrics.
6. Maintain separate CPU and GPU result directories.

## Tests

`pytest` covers:

- Hand-computed parameter, KV, prefill/decode FLOP and byte counts for a
  one-layer transformer, including the linear/attention/logits and
  weight/KV/activation breakdowns, causal/full attention, materialized
  attention, dtype width, logits scope and layer scaling.
- Roofline `min(peak, BW*I)` behavior and plot/CSV generation.
- Sweep determinism, ranking order, Pareto monotonicity and output files.
- CPU measurement shapes, KV allocation match, dtype guard, backend labels.
- Agent budget caps, trace completeness, evidence existence, no-fabrication on
  profile failure, and safe halt without a required GPU.

## Limitations

- CPU prediction error is significant because the cost model calculates theoretical bounds.
- The repository does not include NVIDIA GPU execution data or Nsight metrics.
- Activation memory traffic uses heuristic calculations rather than hardware cache simulation.
- The model does not include paged attention, prefix caching, speculative decoding, kernel quantization, or multi-GPU parallelism.
- The search agent evaluates a bounded configuration space using fixed rules.
- Sweep throughput rankings depend on nominal datasheet specifications.
