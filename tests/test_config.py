"""Unit tests for configuration schema and parameters."""

from __future__ import annotations

import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_config_sections_exist() -> None:
    config = yaml.safe_load((ROOT / "experiments" / "configs.yaml").read_text())
    assert {"experiments", "validation", "agent_search"} <= set(config)
    assert {"llama_analytical", "tiny_cpu_calibrated"} <= set(config["experiments"])
    assert len(config["validation"]["workloads"]) >= 1
    assert config["agent_search"]["budget"]["max_tool_calls"] > 0


def test_nominal_hardware_values_are_numeric() -> None:
    config = yaml.safe_load((ROOT / "experiments" / "configs.yaml").read_text())
    for experiment in config["experiments"].values():
        for hardware in experiment["hardware"]:
            assert isinstance(hardware["peak_flops"], (int, float))
            assert isinstance(hardware["memory_bandwidth"], (int, float))
            assert isinstance(hardware["memory_capacity"], (int, float))
            assert hardware["peak_flops"] > 0
            assert hardware["memory_bandwidth"] > 0
            assert math.isfinite(hardware["peak_flops"])
