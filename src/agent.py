"""Bounded roofline exploration agent.

The agent uses a deterministic policy to evaluate configurations with
the simulate and profile tools. Configurable budgets limit execution rounds,
tool calls, and profile runs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from measure import (
    BACKEND,
    hardware_from_calibration,
    measure_workload,
    nvidia_gpu_status,
)
from model import (
    HardwareConfig,
    ModelConfig,
    RuntimeConfig,
    estimate,
    format_bytes,
    model_from_dict,
    runtime_from_dict,
)
from roofline import (
    plot_predicted_vs_measured,
    plot_roofline,
    points_from_estimate,
    save_points_csv,
)


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    max_rounds: int = 3
    max_tool_calls: int = 12
    max_profiles: int = 2

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        if self.max_profiles < 0:
            raise ValueError("max_profiles must be >= 0")


@dataclass(frozen=True)
class Candidate:
    model: ModelConfig
    runtime: RuntimeConfig
    label: str

    def config_dict(self) -> dict[str, Any]:
        return {
            "model": _asdict(self.model),
            "runtime": _asdict(self.runtime),
        }

    @property
    def workload_id(self) -> str:
        from model import workload_id

        return workload_id(self.model, self.runtime)


def _asdict(obj: Any) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(obj)


class RooflineAgent:
    def __init__(
        self,
        candidates: list[Candidate],
        hardware: HardwareConfig,
        *,
        budget: Budget = Budget(),
        simulate_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        profile_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        gpu_status_fn: Callable[[], dict[str, Any]] | None = None,
        output_dir: str | Path = "results/agent",
        seed: int = 0,
        profile_repeats: int = 2,
        require_nvidia_gpu: bool = False,
    ) -> None:
        self.candidates = list(candidates)
        self.hardware = hardware
        self.budget = budget
        self.simulate_fn = simulate_fn or self._default_simulate
        self.profile_fn = profile_fn or self._default_profile
        self.gpu_status_fn = gpu_status_fn or nvidia_gpu_status
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "agent_trace.jsonl"
        self.seed = seed
        self.profile_repeats = profile_repeats
        self.require_nvidia_gpu = require_nvidia_gpu

        self.tool_calls = 0
        self.profile_calls = 0
        self.evidence_counter = 0
        self.trace: list[dict[str, Any]] = []
        self.evidence: dict[str, dict[str, Any]] = {}
        self.simulations: list[dict[str, Any]] = []
        self.measurements: list[dict[str, Any]] = []
        self.simulated_ids: set[str] = set()
        self.profiled_ids: set[str] = set()
        self.blocked: list[dict[str, Any]] = []
        self.halt_reason: str | None = None
        self.required_backend_ok = True

        if require_nvidia_gpu:
            status = self.gpu_status_fn()
            self._log_event(
                round_idx=-1,
                tool="preflight_nvidia_check",
                status="ok" if status.get("available") else "unavailable",
                arguments={},
                result=status,
            )
            if not status.get("available"):
                self.required_backend_ok = False
                self.halt_reason = (
                    "NVIDIA GPU required, but none was detected (nvidia-smi unavailable). "
                    "No measurement claims were produced."
                )
            else:
                # Do not use CPU measurements when a GPU backend is required.
                self.required_backend_ok = False
                self.halt_reason = (
                    "NVIDIA GPU detected, but this repository has no configured GPU "
                    "profiling backend; refusing to treat CPU data as GPU evidence."
                )

    def _default_simulate(self, config: dict[str, Any]) -> dict[str, Any]:
        est = estimate(
            model_from_dict(config["model"]),
            runtime_from_dict(config["runtime"]),
            self.hardware,
        )
        return est.to_dict()

    def _default_profile(self, config: dict[str, Any]) -> dict[str, Any]:
        model = model_from_dict(config["model"])
        runtime = runtime_from_dict(config["runtime"])
        return measure_workload(
            model,
            runtime,
            seed=self.seed,
            repeats=self.profile_repeats,
            warmup=1,
        )

    def _write_trace(self, event: dict[str, Any]) -> None:
        self.trace.append(event)
        with self.trace_path.open("a") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    def _log_event(
        self,
        *,
        round_idx: int,
        tool: str,
        status: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: str | None = None,
        evidence_id: str | None = None,
        duration_ms: float | None = None,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": len(self.trace) + 1,
            "timestamp": time.time(),
            "round": round_idx,
            "tool": tool,
            "status": status,
            "arguments": arguments,
            "result": result,
            "error": error,
            "evidence_id": evidence_id,
            "duration_ms": duration_ms,
            "summary": summary,
        }
        self._write_trace(event)
        return event

    def _call(
        self,
        *,
        round_idx: int,
        tool: str,
        config: dict[str, Any],
        fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> dict[str, Any]:
        if self.tool_calls >= self.budget.max_tool_calls:
            event = self._log_event(
                round_idx=round_idx,
                tool=tool,
                status="blocked_tool_budget",
                arguments=_config_summary(config),
                error=f"max_tool_calls={self.budget.max_tool_calls} exhausted",
            )
            self.blocked.append(event)
            raise BudgetExceeded(event["error"])
        if tool == "profile" and self.profile_calls >= self.budget.max_profiles:
            event = self._log_event(
                round_idx=round_idx,
                tool=tool,
                status="blocked_profile_budget",
                arguments=_config_summary(config),
                error=f"max_profiles={self.budget.max_profiles} exhausted",
            )
            self.blocked.append(event)
            raise BudgetExceeded(event["error"])

        self.tool_calls += 1
        if tool == "profile":
            self.profile_calls += 1
        start = time.perf_counter()
        try:
            result = fn(config)
        except Exception as exc:  # noqa: BLE001 - tool failures are evidence
            self._log_event(
                round_idx=round_idx,
                tool=tool,
                status="error",
                arguments=_config_summary(config),
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - start) * 1e3,
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1e3
        self.evidence_counter += 1
        prefix = "sim" if tool == "simulate" else "meas"
        evidence_id = f"{prefix}-{self.evidence_counter:04d}"
        summary = _evidence_summary(tool, result)
        event = self._log_event(
            round_idx=round_idx,
            tool=tool,
            status="ok",
            arguments=_config_summary(config),
            result=result,
            evidence_id=evidence_id,
            duration_ms=duration_ms,
            summary=summary,
        )
        self.evidence[evidence_id] = {
            "evidence_id": evidence_id,
            "tool": tool,
            "config_id": result.get("config_id"),
            "workload_id": result.get("workload_id"),
            "summary": summary,
            "result": result,
        }
        if tool == "simulate":
            self.simulations.append(self.evidence[evidence_id])
            self.simulated_ids.add(result["workload_id"])
        else:
            self.measurements.append(self.evidence[evidence_id])
            self.profiled_ids.add(result["workload_id"])
        return result

    def _by_tps(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            entries,
            key=lambda e: (
                -float(e["result"]["decode"]["tokens_per_s"]),
                e["evidence_id"],
            ),
        )

    def _coarse_proposals(self) -> list[Candidate]:
        if not self.candidates:
            return []
        seqs = sorted({c.runtime.seq_len for c in self.candidates})
        mid = seqs[len(seqs) // 2]
        batches = sorted({c.runtime.batch_size for c in self.candidates})
        chosen_batches = {batches[0], batches[-1]}
        proposed = [
            c
            for c in self.candidates
            if c.runtime.seq_len == mid and c.runtime.batch_size in chosen_batches
        ]
        return proposed

    def _neighbor_proposals(self) -> list[Candidate]:
        if not self.simulations:
            return []
        best = self._by_tps(self.simulations)[0]
        cfg = best["result"]
        model = model_from_dict(cfg["model"])
        runtime = runtime_from_dict(cfg["runtime"])
        seqs = sorted({c.runtime.seq_len for c in self.candidates})
        batches = sorted({c.runtime.batch_size for c in self.candidates})
        wanted_batches = {runtime.batch_size}
        idx = batches.index(runtime.batch_size) if runtime.batch_size in batches else 0
        if idx + 1 < len(batches):
            wanted_batches.add(batches[idx + 1])
        wanted_seqs = {runtime.seq_len}
        sidx = seqs.index(runtime.seq_len) if runtime.seq_len in seqs else 0
        if sidx > 0:
            wanted_seqs.add(seqs[sidx - 1])
        if sidx + 1 < len(seqs):
            wanted_seqs.add(seqs[sidx + 1])
        proposed = [
            c
            for c in self.candidates
            if c.runtime.batch_size in wanted_batches
            and c.runtime.seq_len in wanted_seqs
            and c.model.n_kv_heads == model.n_kv_heads
        ]
        return proposed

    def _remaining_proposals(self) -> list[Candidate]:
        return [c for c in self.candidates if c.workload_id not in self.simulated_ids]

    def _propose(self, round_idx: int) -> list[Candidate]:
        if round_idx == 0:
            proposals = self._coarse_proposals()
        elif round_idx == 1:
            proposals = self._neighbor_proposals()
        else:
            proposals = self._remaining_proposals()
        fresh = [c for c in proposals if c.workload_id not in self.simulated_ids]
        return _dedupe(fresh)

    def run(self) -> dict[str, Any]:
        if not self.required_backend_ok:
            return self._synthesize()

        simulation_budget = max(0, self.budget.max_tool_calls - self.budget.max_profiles)
        for round_idx in range(self.budget.max_rounds):
            if self.tool_calls >= simulation_budget:
                break
            proposals = self._propose(round_idx)
            if not proposals:
                break
            for candidate in proposals:
                if self.tool_calls >= simulation_budget:
                    break
                try:
                    self._call(
                        round_idx=round_idx,
                        tool="simulate",
                        config=candidate.config_dict(),
                        fn=self.simulate_fn,
                    )
                except BudgetExceeded:
                    break
                except Exception:
                    continue

        # Profile top configurations ranked by decode throughput.
        ranked = self._by_tps(self.simulations)
        for entry in ranked:
            if self.profile_calls >= self.budget.max_profiles:
                break
            if self.tool_calls >= self.budget.max_tool_calls:
                break
            cfg = entry["result"]
            if cfg["workload_id"] in self.profiled_ids:
                continue
            config = {"model": cfg["model"], "runtime": cfg["runtime"]}
            try:
                self._call(
                    round_idx=self.budget.max_rounds,
                    tool="profile",
                    config=config,
                    fn=self.profile_fn,
                )
            except BudgetExceeded:
                break
            except Exception:
                continue
        return self._synthesize()

    def _synthesize(self) -> dict[str, Any]:
        if not self.required_backend_ok:
            status = "insufficient_evidence"
        elif self.measurements:
            status = "complete"
        elif self.simulations:
            status = "partial"
        else:
            status = "insufficient_evidence"

        claims: list[dict[str, Any]] = []
        if self.simulations:
            ranked = self._by_tps(self.simulations)
            best = ranked[0]
            b = best["result"]
            claims.append(
                _claim(
                    f"Highest predicted decode throughput was "
                    f"{b['decode']['tokens_per_s']:.1f} tokens/s for "
                    f"{_config_label(b)} on {b['hardware']['name']}.",
                    [best["evidence_id"]],
                    "analytical",
                )
            )
            claims.append(
                _claim(
                    f"Decode was {b['decode']['bound']}-bound with operational "
                    f"intensity {b['decode']['intensity']:.2f} FLOP/byte versus a "
                    f"ridge point of {b['ridge_point_flops_per_byte']:.2f} FLOP/byte; "
                    f"prefill was {b['prefill']['bound']}-bound at "
                    f"{b['prefill']['intensity']:.2f} FLOP/byte.",
                    [best["evidence_id"]],
                    "analytical",
                )
            )
            pressure = b["kv_pressure_peak"]
            pressure_text = (
                f"{pressure*100:.1f}% of device memory"
                if pressure is not None
                else "unknown capacity"
            )
            claims.append(
                _claim(
                    f"Peak KV footprint for the best configuration was "
                    f"{format_bytes(b['kv_bytes_peak'])} ({pressure_text}).",
                    [best["evidence_id"]],
                    "analytical",
                )
            )
            by_prefill = sorted(
                self.simulations,
                key=lambda e: (-float(e["result"]["prefill"]["tokens_per_s"]), e["evidence_id"]),
            )[0]
            if by_prefill["evidence_id"] != best["evidence_id"]:
                pf = by_prefill["result"]
                claims.append(
                    _claim(
                        f"Best predicted prefill throughput was "
                        f"{pf['prefill']['tokens_per_s']:.1f} tokens/s "
                        f"({_config_label(pf)}, {pf['prefill']['bound']}-bound), "
                        f"a different operating point from the best decode config.",
                        [by_prefill["evidence_id"]],
                        "analytical",
                    )
                )

        validation: list[dict[str, Any]] = []
        sims_by_workload = {
            e["result"]["workload_id"]: e for e in self.simulations
        }
        for meas in self.measurements:
            m = meas["result"]
            sim = sims_by_workload.get(m["workload_id"])
            if not sim:
                continue
            s = sim["result"]
            for phase, pred_key, measured_key in (
                ("prefill", "prefill", "prefill_tokens_per_s"),
                ("decode", "decode", "decode_tokens_per_s"),
            ):
                predicted_tps = float(s[pred_key]["tokens_per_s"])
                measured_tps = float(m[measured_key])
                error = (predicted_tps - measured_tps) / measured_tps * 100.0
                validation.append(
                    {
                        "phase": phase,
                        "workload_id": m["workload_id"],
                        "label": _config_label(s),
                        "predicted_tokens_per_s": predicted_tps,
                        "measured_tokens_per_s": measured_tps,
                        "error_pct": error,
                        "predicted_ms": float(s[pred_key]["predicted_time_s"]) * 1e3,
                        "measured_ms": (
                            float(m["prefill_median_s"]) * 1e3
                            if phase == "prefill"
                            else float(m["decode_median_s"]) * 1e3
                        ),
                        "sim_evidence": sim["evidence_id"],
                        "meas_evidence": meas["evidence_id"],
                        "backend": m["backend"],
                    }
                )
                claims.append(
                    _claim(
                        f"Measured {phase} throughput was {measured_tps:.1f} tokens/s "
                        f"versus {predicted_tps:.1f} tokens/s predicted "
                        f"({error:+.1f}% error) on {m['backend']}.",
                        [sim["evidence_id"], meas["evidence_id"]],
                        "measured",
                    )
                )

        limitations = [
            "All measured data comes from the NumPy CPU backend "
            f"({BACKEND}); no NVIDIA GPU was used and no NVIDIA performance is claimed.",
            "Roofline predictions are idealized bounds from the supplied hardware "
            "profile plus a measured per-layer dispatch correction.",
            "The agent is a deterministic bounded policy; it does not call an external LLM.",
            "Attention scores are materialized in the reference backend, unlike a "
            "production FlashAttention kernel; this contributes to prediction error.",
        ]
        if not self.measurements:
            limitations.append(
                "No successful profile call: the report makes no measured claims."
            )
        failed = [e for e in self.trace if e["tool"] == "profile" and e["status"] == "error"]
        for event in failed:
            limitations.append(f"Profile failure: {event['error']}")

        top_configs = []
        for entry in self._by_tps(self.simulations)[:8]:
            r = entry["result"]
            top_configs.append(
                {
                    "evidence_id": entry["evidence_id"],
                    "config_id": r["config_id"],
                    "label": _config_label(r),
                    "hardware": r["hardware"]["name"],
                    "decode_tokens_per_s": r["decode"]["tokens_per_s"],
                    "prefill_tokens_per_s": r["prefill"]["tokens_per_s"],
                    "decode_bound": r["decode"]["bound"],
                    "prefill_bound": r["prefill"]["bound"],
                    "kv_bytes_peak": r["kv_bytes_peak"],
                    "kv_pressure_peak": r["kv_pressure_peak"],
                }
            )

        report = {
            "schema_version": "1.0",
            "kind": "bottleneck_report",
            "status": status,
            "halt_reason": self.halt_reason,
            "backend": {
                "profile": BACKEND,
                "device": "CPU",
                "gpu_used": False,
                "nvidia_claims": "none",
            },
            "hardware_profile": _asdict(self.hardware),
            "budget": {
                "max_rounds": self.budget.max_rounds,
                "max_tool_calls": self.budget.max_tool_calls,
                "max_profiles": self.budget.max_profiles,
                "tool_calls_used": self.tool_calls,
                "profile_calls_used": self.profile_calls,
                "blocked_attempts": len(self.blocked),
            },
            "claims": claims,
            "top_configs": top_configs,
            "validation": validation,
            "limitations": limitations,
            "trace_path": str(self.trace_path),
            "evidence_ids": sorted(self.evidence),
        }
        (self.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        _write_markdown(report, self.evidence, self.output_dir / "report.md")
        self._write_plots(validation)
        return report

    def _write_plots(self, validation: list[dict[str, Any]]) -> None:
        if self.simulations:
            best = self._by_tps(self.simulations)[0]["result"]
            est = estimate(
                model_from_dict(best["model"]),
                runtime_from_dict(best["runtime"]),
                self.hardware,
            )
            plot_roofline(est, self.output_dir / "roofline_best.png")
            save_points_csv(
                points_from_estimate(est), self.output_dir / "roofline_best.csv"
            )
        if validation:
            plot_predicted_vs_measured(
                validation,
                self.output_dir / "predicted_vs_measured.png",
                title="Agent validation: roofline prediction vs CPU measurement",
            )


def _dedupe(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    result = []
    for candidate in candidates:
        if candidate.workload_id in seen:
            continue
        seen.add(candidate.workload_id)
        result.append(candidate)
    return result


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model", {})
    runtime = config.get("runtime", {})
    return {
        "model": model.get("name"),
        "attention": (
            "mha"
            if model.get("n_kv_heads") == model.get("n_heads")
            else "mqa"
            if model.get("n_kv_heads") == 1
            else "gqa"
        ),
        "batch_size": runtime.get("batch_size"),
        "seq_len": runtime.get("seq_len"),
        "gen_len": runtime.get("gen_len"),
        "weight_dtype": runtime.get("weight_dtype"),
    }


def _config_label(result: dict[str, Any]) -> str:
    model = result["model"]
    runtime = result["runtime"]
    attn = "mha" if model["n_kv_heads"] == model["n_heads"] else "mqa" if model["n_kv_heads"] == 1 else "gqa"
    return (
        f"{model['name']}/{attn}/B={runtime['batch_size']}/S={runtime['seq_len']}/"
        f"{runtime['weight_dtype']}"
    )


def _claim(text: str, evidence: list[str], kind: str) -> dict[str, Any]:
    if not evidence:
        raise ValueError("claims must cite at least one evidence id")
    return {"text": text, "evidence": evidence, "kind": kind}


def _evidence_summary(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    if tool == "simulate":
        return {
            "kind": "simulation",
            "config_id": result.get("config_id"),
            "hardware": result["hardware"]["name"],
            "decode_tokens_per_s": result["decode"]["tokens_per_s"],
            "prefill_tokens_per_s": result["prefill"]["tokens_per_s"],
            "decode_bound": result["decode"]["bound"],
            "prefill_bound": result["prefill"]["bound"],
            "kv_bytes_peak": result["kv_bytes_peak"],
        }
    return {
        "kind": "measurement",
        "measurement_id": result.get("measurement_id"),
        "backend": result.get("backend"),
        "prefill_ms": result.get("prefill_median_s", 0.0) * 1e3,
        "decode_ms": result.get("decode_median_s", 0.0) * 1e3,
        "decode_tokens_per_s": result.get("decode_tokens_per_s"),
        "prefill_tokens_per_s": result.get("prefill_tokens_per_s"),
    }


def _write_markdown(
    report: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# Bounded agent bottleneck report",
        "",
        f"**Status:** `{report['status']}`  ",
        f"**Backend:** `{report['backend']['profile']}` (CPU; GPU used: "
        f"{report['backend']['gpu_used']})  ",
        f"**Tool calls:** {report['budget']['tool_calls_used']} / "
        f"{report['budget']['max_tool_calls']}  ",
        f"**Profiles:** {report['budget']['profile_calls_used']} / "
        f"{report['budget']['max_profiles']}",
        "",
    ]
    if report.get("halt_reason"):
        lines += [f"> Halted safely: {report['halt_reason']}", ""]

    lines += [
        "## Method",
        "",
        "The search algorithm evaluates candidate configurations with the analytical simulator, "
        "ranks candidates by predicted decode throughput, and benchmarks the top configurations "
        "on the CPU backend. Tool invocations are recorded in `agent_trace.jsonl` with evidence identifiers.",
        "",
        "## Evidence-cited claims",
        "",
    ]
    for claim in report["claims"]:
        refs = " ".join(f"[{eid}]" for eid in claim["evidence"])
        lines.append(f"- {claim['text']} {refs} (`{claim['kind']}`)")
    if not report["claims"]:
        lines.append("- No claims: insufficient evidence.")

    lines += [
        "",
        "## Top predicted configurations",
        "",
        "| evidence | config | hardware | decode tok/s | prefill tok/s | decode bound | KV peak |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["top_configs"]:
        lines.append(
            f"| {row['evidence_id']} | {row['label']} | {row['hardware']} | "
            f"{row['decode_tokens_per_s']:.1f} | {row['prefill_tokens_per_s']:.1f} | "
            f"{row['decode_bound']} | {format_bytes(row['kv_bytes_peak'])} |"
        )

    lines += [
        "",
        "## Predicted vs measured",
        "",
    ]
    if report["validation"]:
        lines += [
            "| phase | predicted tok/s | measured tok/s | error | evidence |",
            "|---|---|---|---|---|",
        ]
        for row in report["validation"]:
            lines.append(
                f"| {row['phase']} | {row['predicted_tokens_per_s']:.1f} | "
                f"{row['measured_tokens_per_s']:.1f} | {row['error_pct']:+.1f}% | "
                f"[{row['sim_evidence']}] [{row['meas_evidence']}] |"
            )
    else:
        lines.append("No successful measured comparison was available.")

    lines += [
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in report["limitations"]]

    lines += [
        "",
        "## Evidence appendix",
        "",
        "| evidence | tool | summary |",
        "|---|---|---|",
    ]
    for eid in report["evidence_ids"]:
        entry = evidence[eid]
        summary = entry["summary"]
        if entry["tool"] == "simulate":
            text = (
                f"{summary['hardware']}: decode {summary['decode_tokens_per_s']:.1f} tok/s "
                f"({summary['decode_bound']}-bound), prefill "
                f"{summary['prefill_tokens_per_s']:.1f} tok/s ({summary['prefill_bound']}-bound)"
            )
        else:
            text = (
                f"{summary['backend']}: prefill {summary['prefill_ms']:.2f} ms, "
                f"decode {summary['decode_ms']:.2f} ms"
            )
        lines.append(f"| {eid} | {entry['tool']} | {text} |")
    lines += ["", f"Full trace: `{report['trace_path']}`", ""]
    path.write_text("\n".join(lines))


def build_candidates(section: dict[str, Any]) -> list[Candidate]:
    model_base = model_from_dict(section["model"])
    runtime_base = dict(section.get("runtime", {}))
    grid = {k: v for k, v in runtime_base.items() if isinstance(v, list)}
    scalar_runtime = {k: v for k, v in runtime_base.items() if not isinstance(v, list)}
    grid.update(section.get("runtime_grid", {}) or {})
    keys = list(grid)
    options = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    combos: list[tuple[Any, ...]] = [()]
    for values in options:
        combos = [prefix + (value,) for prefix in combos for value in values]
    variants = section.get("attention_variants", [model_base.attention_variant])
    candidates: list[Candidate] = []
    for combo in combos:
        runtime_dict = dict(scalar_runtime)
        runtime_dict.update(dict(zip(keys, combo)))
        runtime = runtime_from_dict(runtime_dict)
        for variant in variants:
            model = model_base.with_attention_variant(variant)
            candidates.append(
                Candidate(
                    model=model,
                    runtime=runtime,
                    label=f"{model.name}-{variant}-b{runtime.batch_size}-s{runtime.seq_len}",
                )
            )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded roofline exploration agent")
    parser.add_argument("--config", required=True)
    parser.add_argument("--section", default="agent_search")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--out-dir", default="results/agent")
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-tool-calls", type=int, default=None)
    parser.add_argument("--max-profiles", type=int, default=None)
    parser.add_argument("--require-nvidia-gpu", action="store_true")
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    section = config.get(args.section)
    if not section:
        raise KeyError(f"missing config section {args.section!r}")
    calibration = json.loads(Path(args.calibration).read_text())
    hardware = hardware_from_calibration(calibration)
    candidates = build_candidates(section)
    budget_cfg = section.get("budget", {})
    budget = Budget(
        max_rounds=args.max_rounds or int(budget_cfg.get("max_rounds", 3)),
        max_tool_calls=args.max_tool_calls or int(budget_cfg.get("max_tool_calls", 12)),
        max_profiles=(
            args.max_profiles
            if args.max_profiles is not None
            else int(budget_cfg.get("max_profiles", 2))
        ),
    )
    agent = RooflineAgent(
        candidates,
        hardware,
        budget=budget,
        output_dir=args.out_dir,
        seed=int(section.get("seed", 0)),
        profile_repeats=int(section.get("repeats", 2)),
        require_nvidia_gpu=args.require_nvidia_gpu,
    )
    report = agent.run()
    print(
        json.dumps(
            {
                "status": report["status"],
                "tool_calls": report["budget"]["tool_calls_used"],
                "profiles": report["budget"]["profile_calls_used"],
                "claims": len(report["claims"]),
                "report": str(Path(args.out_dir) / "report.json"),
                "trace": str(Path(args.out_dir) / "agent_trace.jsonl"),
            },
            indent=2,
        )
    )
    if report["status"] == "insufficient_evidence":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
