#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  uv venv .venv
  uv pip install --python .venv/bin/python -e '.[dev]'
fi
PY=.venv/bin/python

$PY -m measure calibrate --out results/calibration_cpu.json --repeats 5
$PY -m sweep --config experiments/configs.yaml --experiment llama_analytical \
  --out-dir results/sweep_analytical
$PY -m sweep --config experiments/configs.yaml --experiment tiny_cpu_calibrated \
  --out-dir results/sweep_tiny_cpu --calibration results/calibration_cpu.json
$PY -m measure validation --config experiments/configs.yaml \
  --calibration results/calibration_cpu.json --out-dir results/validation
$PY -m agent --config experiments/configs.yaml --section agent_search \
  --calibration results/calibration_cpu.json --out-dir results/agent
$PY -m measure gpu-status --out results/gpu_status.json

set +e
$PY -m agent --config experiments/configs.yaml --section agent_search \
  --calibration results/calibration_cpu.json \
  --out-dir results/agent_nvidia_required --require-nvidia-gpu
HALT_CODE=$?
set -e
if [ "$HALT_CODE" -ne 2 ]; then
  echo "expected safe-halt exit code 2, got $HALT_CODE" >&2
  exit 1
fi

$PY -m pytest
echo "All experiments and tests passed."
