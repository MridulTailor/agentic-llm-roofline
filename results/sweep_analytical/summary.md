# Sweep: llama_analytical

LLaMA-style ~7.5B decoder. Nominal H100/A100 ceilings, an explicitly bandwidth-constrained H100 variant, MHA/GQA/MQA, batch/context/dtype axes.

Configurations: **162**

> All numbers are analytical predictions from the supplied hardware profile, not measured workload performance.

## Top predicted decode throughput

| hardware | attn | batch | seq | dtype | decode tok/s | KV pressure | decode bound |
|---|---|---|---|---|---|---|---|
| H100-SXM-80GB-nominal | mqa | 32 | 512 | fp8 | 13433.0 | 0.2% | memory |
| H100-SXM-80GB-nominal | mqa | 32 | 2048 | fp8 | 12787.8 | 0.7% | memory |
| H100-SXM-80GB-nominal | gqa | 32 | 512 | fp8 | 11558.0 | 1.7% | memory |
| H100-SXM-80GB-nominal | mqa | 32 | 8192 | fp8 | 10726.9 | 2.7% | memory |
| H100-SXM-80GB-nominal | gqa | 32 | 2048 | fp8 | 8578.6 | 5.7% | memory |
| A100-SXM-80GB-nominal | mqa | 32 | 512 | fp8 | 8176.1 | 0.2% | memory |
| H100-SXM-80GB-nominal | mha | 32 | 512 | fp8 | 7817.1 | 6.7% | memory |
| A100-SXM-80GB-nominal | mqa | 32 | 2048 | fp8 | 7783.4 | 0.7% | memory |
| A100-SXM-80GB-nominal | gqa | 32 | 512 | fp8 | 7034.9 | 1.7% | memory |
| H100-SXM-80GB-nominal | mqa | 32 | 512 | fp16 | 6730.7 | 0.4% | memory |

## Highest peak KV pressure

| hardware | attn | batch | seq | dtype | decode tok/s | KV pressure |
|---|---|---|---|---|---|---|
| A100-SXM-80GB-nominal | mha | 32 | 8192 | fp16 | 417.6 | 174.5% |
| H100-SXM-80GB-nominal | mha | 32 | 8192 | fp16 | 686.1 | 174.5% |
| H100-BW-constrained-nominal | mha | 32 | 8192 | fp16 | 307.2 | 174.5% |
| H100-BW-constrained-nominal | mha | 32 | 8192 | fp8 | 614.3 | 87.2% |
| A100-SXM-80GB-nominal | mha | 32 | 8192 | fp8 | 835.0 | 87.2% |
| H100-SXM-80GB-nominal | mha | 32 | 8192 | fp8 | 1371.9 | 87.2% |
| H100-SXM-80GB-nominal | mha | 32 | 2048 | fp16 | 2016.4 | 45.6% |
| H100-BW-constrained-nominal | mha | 32 | 2048 | fp16 | 902.9 | 45.6% |
| A100-SXM-80GB-nominal | mha | 32 | 2048 | fp16 | 1227.3 | 45.6% |
| H100-SXM-80GB-nominal | gqa | 32 | 8192 | fp16 | 2113.2 | 43.6% |

## Notes

- KV pressure counts the KV cache at seq_len + gen_len; it excludes weights.
- Decode throughput is batch * gen_len / summed per-step predicted time.
- FP8 changes weight/KV bytes only; the compute ceiling is held at the FP16 preset value, so FP8 gains here are memory-side only.

Total rows: 162
