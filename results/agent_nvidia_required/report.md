# Bounded agent bottleneck report

**Status:** `insufficient_evidence`  
**Backend:** `cpu-numpy-blas-float32` (CPU; GPU used: False)  
**Tool calls:** 0 / 12  
**Profiles:** 0 / 2

> Halted safely: NVIDIA GPU required, but none was detected (nvidia-smi unavailable). No measurement claims were produced.

## Method

The search algorithm evaluates candidate configurations with the analytical simulator, ranks candidates by predicted decode throughput, and benchmarks the top configurations on the CPU backend. Tool invocations are recorded in `agent_trace.jsonl` with evidence identifiers.

## Evidence-cited claims

- No claims: insufficient evidence.

## Top predicted configurations

| evidence | config | hardware | decode tok/s | prefill tok/s | decode bound | KV peak |
|---|---|---|---|---|---|---|

## Predicted vs measured

No successful measured comparison was available.

## Limitations

- All measured data comes from the NumPy CPU backend (cpu-numpy-blas-float32); no NVIDIA GPU was used and no NVIDIA performance is claimed.
- Roofline predictions are idealized bounds from the supplied hardware profile plus a measured per-layer dispatch correction.
- The agent is a deterministic bounded policy; it does not call an external LLM.
- Attention scores are materialized in the reference backend, unlike a production FlashAttention kernel; this contributes to prediction error.
- No successful profile call: the report makes no measured claims.

## Evidence appendix

| evidence | tool | summary |
|---|---|---|

Full trace: `results/agent_nvidia_required/agent_trace.jsonl`
