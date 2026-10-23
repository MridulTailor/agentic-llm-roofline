# Sweep: tiny_cpu_calibrated

Tiny 4x128 decoder swept against the measured CPU effective profile. Analytical predictions only; profiling of selected points is done by the agent, not by the sweep.

Configurations: **27**

> All numbers are analytical predictions from the supplied hardware profile, not measured workload performance.

## Top predicted decode throughput

| hardware | attn | batch | seq | dtype | decode tok/s | KV pressure | decode bound |
|---|---|---|---|---|---|---|---|
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mqa | 32 | 64 | fp32 | 110055.5 | 0.0% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mqa | 32 | 128 | fp32 | 99865.9 | 0.0% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | gqa | 32 | 64 | fp32 | 98512.1 | 0.0% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mqa | 32 | 256 | fp32 | 84262.8 | 0.1% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | gqa | 32 | 128 | fp32 | 83296.9 | 0.1% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mha | 32 | 64 | fp32 | 81430.0 | 0.1% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | gqa | 32 | 256 | fp32 | 63638.9 | 0.1% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mha | 32 | 128 | fp32 | 62543.4 | 0.1% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mha | 32 | 256 | fp32 | 42724.6 | 0.2% | memory |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mqa | 8 | 64 | fp32 | 30323.9 | 0.0% | memory |

## Highest peak KV pressure

| hardware | attn | batch | seq | dtype | decode tok/s | KV pressure |
|---|---|---|---|---|---|---|
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mha | 32 | 256 | fp32 | 42724.6 | 0.2% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mha | 32 | 128 | fp32 | 62543.4 | 0.1% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | gqa | 32 | 256 | fp32 | 63638.9 | 0.1% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mha | 32 | 64 | fp32 | 81430.0 | 0.1% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | gqa | 32 | 128 | fp32 | 83296.9 | 0.1% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mqa | 32 | 256 | fp32 | 84262.8 | 0.1% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mha | 8 | 256 | fp32 | 20912.5 | 0.1% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | gqa | 32 | 64 | fp32 | 98512.1 | 0.0% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mha | 8 | 128 | fp32 | 24751.6 | 0.0% |
| Apple M1 Pro [cpu-numpy-blas-float32] calibrated | mqa | 32 | 128 | fp32 | 99865.9 | 0.0% |

## Notes

- KV pressure counts the KV cache at seq_len + gen_len; it excludes weights.
- Decode throughput is batch * gen_len / summed per-step predicted time.
- FP8 changes weight/KV bytes only; the compute ceiling is held at the FP16 preset value, so FP8 gains here are memory-side only.

Total rows: 27
