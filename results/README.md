# Result index

Commands in the root `README.md` generate all files in this directory.

Data categories:

- `*_cpu*`, `validation*`, `calibration*`: Measured CPU data (`cpu-numpy-blas-float32`).
- `sweep_analytical/`: Analytical predictions using theoretical accelerator ceilings.
- `sweep_tiny_cpu/`: Analytical predictions using the calibrated CPU profile.
- `agent/`: Search agent trace, analytical simulations, and CPU benchmark profiles.
- `agent_nvidia_required/`: Safe-halt demonstration record.
- `gpu_status.json`: NVIDIA GPU hardware detection record.

| Path | Contents |
|---|---|
| `calibration_cpu.json` | matmul/copy/layer-overhead microbenchmarks + environment |
| `validation/validation.json` | measurements, comparisons, summary, limitations |
| `validation/validation.csv` | flat predicted/measured table |
| `validation/predicted_vs_measured.png` | log-log latency parity chart |
| `validation/throughput_scaling.png` | predicted vs measured tokens/s |
| `sweep_analytical/sweep.json` | 162 nested configs + rankings |
| `sweep_analytical/sweep.csv` | flat sweep table |
| `sweep_analytical/pareto.png` | decode throughput vs KV pressure |
| `sweep_analytical/batch_scaling.png` | throughput vs batch |
| `sweep_analytical/roofline_best.png` | roofline for best decode config |
| `sweep_analytical/roofline_batch1.png` | roofline for a batch-1 config |
| `sweep_tiny_cpu/` | CPU-calibrated analytical sweep |
| `agent/report.json` | machine-readable evidence-cited report |
| `agent/report.md` | human-readable report |
| `agent/agent_trace.jsonl` | every tool invocation |
| `agent/predicted_vs_measured.png` | agent validation chart |
| `agent_nvidia_required/report.json` | safe halt, zero claims |
| `gpu_status.json` | `available=false`, `nvidia_claims=none` |

Analytical sweep results represent theoretical bounds. Refer to the root README
for methodological details.
