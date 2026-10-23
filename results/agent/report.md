# Bounded agent bottleneck report

**Status:** `complete`  
**Backend:** `cpu-numpy-blas-float32` (CPU; GPU used: False)  
**Tool calls:** 12 / 12  
**Profiles:** 2 / 2

## Method

The search algorithm evaluates candidate configurations with the analytical simulator, ranks candidates by predicted decode throughput, and benchmarks the top configurations on the CPU backend. Tool invocations are recorded in `agent_trace.jsonl` with evidence identifiers.

## Evidence-cited claims

- Highest predicted decode throughput was 30323.9 tokens/s for agent-tiny-4x128/mqa/B=8/S=64/fp32 on Apple M1 Pro [cpu-numpy-blas-float32] calibrated. [sim-0007] (`analytical`)
- Decode was memory-bound with operational intensity 3.54 FLOP/byte versus a ridge point of 22.90 FLOP/byte; prefill was compute-bound at 102.76 FLOP/byte. [sim-0007] (`analytical`)
- Peak KV footprint for the best configuration was 576.00 KiB (0.0% of device memory). [sim-0007] (`analytical`)
- Best predicted prefill throughput was 763453.8 tokens/s (agent-tiny-4x128/mqa/B=8/S=256/fp32, compute-bound), a different operating point from the best decode config. [sim-0008] (`analytical`)
- Measured prefill throughput was 58388.1 tokens/s versus 639002.0 tokens/s predicted (+994.4% error) on cpu-numpy-blas-float32. [sim-0007] [meas-0011] (`measured`)
- Measured decode throughput was 5516.7 tokens/s versus 30323.9 tokens/s predicted (+449.7% error) on cpu-numpy-blas-float32. [sim-0007] [meas-0011] (`measured`)
- Measured prefill throughput was 40805.8 tokens/s versus 722151.8 tokens/s predicted (+1669.7% error) on cpu-numpy-blas-float32. [sim-0006] [meas-0012] (`measured`)
- Measured decode throughput was 4616.6 tokens/s versus 29494.7 tokens/s predicted (+538.9% error) on cpu-numpy-blas-float32. [sim-0006] [meas-0012] (`measured`)

## Top predicted configurations

| evidence | config | hardware | decode tok/s | prefill tok/s | decode bound | KV peak |
|---|---|---|---|---|---|---|
| sim-0007 | agent-tiny-4x128/mqa/B=8/S=64/fp32 | Apple M1 Pro [cpu-numpy-blas-float32] calibrated | 30323.9 | 639002.0 | memory | 576.00 KiB |
| sim-0006 | agent-tiny-4x128/mqa/B=8/S=128/fp32 | Apple M1 Pro [cpu-numpy-blas-float32] calibrated | 29494.7 | 722151.8 | memory | 1.06 MiB |
| sim-0008 | agent-tiny-4x128/mqa/B=8/S=256/fp32 | Apple M1 Pro [cpu-numpy-blas-float32] calibrated | 27965.3 | 763453.8 | memory | 2.06 MiB |
| sim-0005 | agent-tiny-4x128/gqa/B=8/S=128/fp32 | Apple M1 Pro [cpu-numpy-blas-float32] calibrated | 27723.8 | 701638.9 | memory | 2.12 MiB |
| sim-0004 | agent-tiny-4x128/mha/B=8/S=128/fp32 | Apple M1 Pro [cpu-numpy-blas-float32] calibrated | 24751.6 | 663921.3 | memory | 4.25 MiB |
| sim-0003 | agent-tiny-4x128/mqa/B=1/S=128/fp32 | Apple M1 Pro [cpu-numpy-blas-float32] calibrated | 3892.8 | 369266.4 | memory | 136.00 KiB |
| sim-0010 | agent-tiny-4x128/gqa/B=1/S=64/fp32 | Apple M1 Pro [cpu-numpy-blas-float32] calibrated | 3863.7 | 235179.7 | memory | 144.00 KiB |
| sim-0002 | agent-tiny-4x128/gqa/B=1/S=128/fp32 | Apple M1 Pro [cpu-numpy-blas-float32] calibrated | 3836.2 | 363827.4 | memory | 272.00 KiB |

## Predicted vs measured

| phase | predicted tok/s | measured tok/s | error | evidence |
|---|---|---|---|---|
| prefill | 639002.0 | 58388.1 | +994.4% | [sim-0007] [meas-0011] |
| decode | 30323.9 | 5516.7 | +449.7% | [sim-0007] [meas-0011] |
| prefill | 722151.8 | 40805.8 | +1669.7% | [sim-0006] [meas-0012] |
| decode | 29494.7 | 4616.6 | +538.9% | [sim-0006] [meas-0012] |

## Limitations

- All measured data comes from the NumPy CPU backend (cpu-numpy-blas-float32); no NVIDIA GPU was used and no NVIDIA performance is claimed.
- Roofline predictions are idealized bounds from the supplied hardware profile plus a measured per-layer dispatch correction.
- The agent is a deterministic bounded policy; it does not call an external LLM.
- Attention scores are materialized in the reference backend, unlike a production FlashAttention kernel; this contributes to prediction error.

## Evidence appendix

| evidence | tool | summary |
|---|---|---|
| meas-0011 | profile | cpu-numpy-blas-float32: prefill 8.77 ms, decode 11.60 ms |
| meas-0012 | profile | cpu-numpy-blas-float32: prefill 25.09 ms, decode 13.86 ms |
| sim-0001 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 3727.9 tok/s (memory-bound), prefill 353416.3 tok/s (compute-bound) |
| sim-0002 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 3836.2 tok/s (memory-bound), prefill 363827.4 tok/s (compute-bound) |
| sim-0003 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 3892.8 tok/s (memory-bound), prefill 369266.4 tok/s (compute-bound) |
| sim-0004 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 24751.6 tok/s (memory-bound), prefill 663921.3 tok/s (compute-bound) |
| sim-0005 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 27723.8 tok/s (memory-bound), prefill 701638.9 tok/s (compute-bound) |
| sim-0006 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 29494.7 tok/s (memory-bound), prefill 722151.8 tok/s (compute-bound) |
| sim-0007 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 30323.9 tok/s (memory-bound), prefill 639002.0 tok/s (compute-bound) |
| sim-0008 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 27965.3 tok/s (memory-bound), prefill 763453.8 tok/s (compute-bound) |
| sim-0009 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 3780.1 tok/s (memory-bound), prefill 230785.1 tok/s (compute-bound) |
| sim-0010 | simulate | Apple M1 Pro [cpu-numpy-blas-float32] calibrated: decode 3863.7 tok/s (memory-bound), prefill 235179.7 tok/s (compute-bound) |

Full trace: `results/agent/agent_trace.jsonl`
