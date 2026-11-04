# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ROCm Roofline Analysis using rocprof hardware counters.

NVIDIA NCU reports Speed of Light as percent of peak. rocprof does not.
Memory SOL is the profiled traffic, ``(FETCH_SIZE + WRITE_SIZE) / duration``,
divided by ``peak_memory_bw_gbps``. VALU and VMEM shares of the counted
instructions stay on ``valu_utilization_pct`` and ``memory_bound_pct``.
They are not percent of peak and are not the efficiency number.

The result uses the same fields as
:class:`kernel_perf_agent.kernel_opt.roofline.ncu_roofline.RooflineAnalyzer`
so the rest of the optimization pipeline can use it transparently.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ROCmRooflineConfig:
    """Configuration for ROCm roofline analysis."""

    threshold_pct: float = 85.0  # Lower than NCU (heuristic, not exact SOL)
    # Instruction-mix ratios are not percent-of-peak. Leaving early stop on
    # would halt optimization when VALU instructions dominate, even if the
    # kernel is far from the HBM or FLOP roof.
    early_stop: bool = False
    convergence_rounds: int = 5
    min_improvement_pct: float = 0.1
    underutilized_threshold: float = 40.0  # Both SOL < this → underutilized
    # Kept so older configs still load. Memory SOL is measured traffic
    # divided by peak bandwidth, so this no longer scales that number.
    miss_rate_amplifier: float = 1.5


@dataclass
class ROCmRooflineResult:
    """Result of ROCm roofline analysis from rocprof counters."""

    # Percent of peak only when that side was measured. An unmeasured
    # side stays 0 so formatters have a number, and prompts omit it.
    compute_sol_pct: float
    compute_sol_measured: bool
    memory_sol_pct: float
    memory_sol_measured: bool
    # True only when memory_sol_pct came from traffic over peak bandwidth.
    memory_sol_from_traffic: bool

    # Derived efficiency
    efficiency_pct: float
    at_roofline: bool
    headroom_pct: float

    # Classification
    bottleneck: str  # "compute" | "memory" | "underutilized"
    uses_tensor_cores: bool  # Always False on ROCm (matrix cores tracked differently)

    # Supporting data
    tcc_cache_hit_rate_pct: float
    valu_utilization_pct: float
    memory_bound_pct: float

    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ROCmRooflineAnalyzer:
    """Analyses ROCm kernel performance using rocprof hardware counters.

    Interface is compatible with
    :class:`kernel_perf_agent.kernel_opt.roofline.ncu_roofline.RooflineAnalyzer`
    so it can be used as a drop-in replacement in the optimization pipeline.
    """

    def __init__(
        self,
        config: ROCmRooflineConfig | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or ROCmRooflineConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._efficiency_history: list[float] = []

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        ncu_metrics: dict[str, Any] | None = None,
        rocm_metrics: dict[str, Any] | None = None,
    ) -> ROCmRooflineResult:
        """Analyse rocprof metrics and return a roofline result.

        The first parameter is named ``ncu_metrics`` so this matches
        ``RooflineAnalyzerBase.analyze`` and NVIDIA call sites that pass
        ``ncu_metrics=``. ``rocm_metrics`` is the same dict under the
        ROCm name.

        Args:
            ncu_metrics: Flat metrics dict from rocprof (shared call-site name).
            rocm_metrics: Same dict, accepted when callers use the ROCm name.

        Returns:
            :class:`ROCmRooflineResult`. Memory SOL is traffic over peak
            when the counters and the peak are present.
        """
        metrics = ncu_metrics if ncu_metrics is not None else rocm_metrics
        if not isinstance(metrics, dict):
            raise TypeError(
                "ROCmRooflineAnalyzer.analyze expected a metrics dict, "
                f"got {type(metrics).__name__}"
            )
        rocm_metrics = metrics
        warnings: list[str] = []

        # compute_sol_pct is percent of peak when a caller already measured
        # it. The instruction mix is a different field and is not copied here.
        compute_measured = "compute_sol_pct" in rocm_metrics
        compute_sol = float(rocm_metrics["compute_sol_pct"]) if compute_measured else 0.0
        traffic_sol = _memory_sol_from_traffic(rocm_metrics)
        memory_from_traffic = traffic_sol is not None
        # A caller-supplied memory_sol_pct is a measured number, but it is
        # not traffic over peak. Missing both leaves memory unmeasured.
        memory_measured = memory_from_traffic or "memory_sol_pct" in rocm_metrics
        if memory_from_traffic:
            memory_sol = float(traffic_sol)
        elif memory_measured:
            memory_sol = float(rocm_metrics["memory_sol_pct"])
        else:
            memory_sol = 0.0

        valu_pct = float(rocm_metrics.get("valu_utilization_pct", 0.0))
        mem_bound_pct = float(rocm_metrics.get("memory_bound_pct", 0.0))
        cache_hit_rate = float(rocm_metrics.get("tcc_cache_hit_rate_pct", 100.0))

        if not compute_measured:
            warnings.append(
                "Compute SOL was not measured. The VALU share is an "
                "instruction mix, not percent of peak."
            )
        if not memory_measured:
            warnings.append(
                "Memory SOL was not measured. It is traffic over peak "
                "bandwidth, not the VMEM instruction share."
            )

        if compute_measured or memory_measured:
            efficiency = max(compute_sol, memory_sol)
            bottleneck = self._classify_bottleneck(compute_sol, memory_sol)
            at_roofline = efficiency >= self.config.threshold_pct
        else:
            # Zeros are placeholders, not a measurement. NCU uses "unknown"
            # when both SOL keys are missing.
            efficiency = 0.0
            bottleneck = "unknown"
            at_roofline = False

        return ROCmRooflineResult(
            compute_sol_pct=round(compute_sol, 2),
            compute_sol_measured=compute_measured,
            memory_sol_pct=round(memory_sol, 2),
            memory_sol_measured=memory_measured,
            memory_sol_from_traffic=memory_from_traffic,
            efficiency_pct=round(efficiency, 2),
            at_roofline=at_roofline,
            headroom_pct=round(max(0.0, 100.0 - efficiency), 2),
            bottleneck=bottleneck,
            uses_tensor_cores=False,  # Matrix core detection not implemented yet
            tcc_cache_hit_rate_pct=round(cache_hit_rate, 2),
            valu_utilization_pct=round(valu_pct, 2),
            memory_bound_pct=round(mem_bound_pct, 2),
            warnings=warnings,
        )

    def _classify_bottleneck(self, compute_sol: float, memory_sol: float) -> str:
        """Classify the bottleneck from the SOL percentages."""
        threshold = self.config.underutilized_threshold
        if compute_sol < threshold and memory_sol < threshold:
            return "underutilized"
        if memory_sol >= compute_sol:
            return "memory"
        return "compute"

    # ------------------------------------------------------------------
    # Convergence tracking (same interface as NvidiaRooflineAnalyzer)
    # ------------------------------------------------------------------

    def should_stop(self, result: ROCmRooflineResult) -> tuple[bool, str]:
        """Check whether optimization should stop.

        Args:
            result: :class:`ROCmRooflineResult` from :meth:`analyze`.

        Returns:
            ``(should_stop, reason)`` tuple.
        """
        self._efficiency_history.append(result.efficiency_pct)

        if self.config.early_stop and result.at_roofline:
            return (
                True,
                f"At roofline ({result.efficiency_pct:.1f}% SOL >= "
                f"{self.config.threshold_pct}%)",
            )

        if len(self._efficiency_history) >= self.config.convergence_rounds:
            recent = self._efficiency_history[-self.config.convergence_rounds :]
            improvement = max(recent) - min(recent)
            if improvement < self.config.min_improvement_pct:
                return (
                    True,
                    f"Converged (improvement {improvement:.2f}% < "
                    f"{self.config.min_improvement_pct}%)",
                )

        return False, ""

    def reset_history(self) -> None:
        """Reset convergence tracking for a new optimization run."""
        self._efficiency_history = []


def _memory_sol_from_traffic(metrics: dict[str, Any]) -> float | None:
    """Percent of peak bandwidth from FETCH_SIZE, WRITE_SIZE, and duration.

    The rocprof sizes are kilobytes. Missing traffic or a missing peak
    returns None so an instruction mix is not reported as SOL.
    """
    peak = metrics.get("peak_memory_bw_gbps")
    duration_ms = metrics.get("duration_ms")
    if peak is None or duration_ms is None:
        return None
    if metrics.get("fetch_size_kb") is None and metrics.get("write_size_kb") is None:
        return None
    try:
        peak_gbps = float(peak)
        duration = float(duration_ms)
        traffic_kb = float(metrics.get("fetch_size_kb") or 0.0) + float(
            metrics.get("write_size_kb") or 0.0
        )
    except (TypeError, ValueError):
        return None
    if peak_gbps <= 0.0 or duration <= 0.0:
        return None
    achieved_gbps = traffic_kb * 1024.0 / (duration / 1000.0) / 1e9
    return min(100.0, 100.0 * achieved_gbps / peak_gbps)


def with_peak_bandwidth(
    metrics: dict[str, Any], gpu_specs: dict[str, Any] | None
) -> dict[str, Any]:
    """Copy ``peak_memory_bw_gbps`` onto profiler metrics when the spec has it."""
    if not gpu_specs or metrics.get("peak_memory_bw_gbps") is not None:
        return metrics
    peak = gpu_specs.get("peak_memory_bw_gbps")
    if peak is None:
        return metrics
    stamped = dict(metrics)
    stamped["peak_memory_bw_gbps"] = peak
    return stamped


def format_rocm_roofline_summary(result: ROCmRooflineResult) -> str:
    """Format a human-readable summary of ROCm roofline analysis."""
    either_measured = result.compute_sol_measured or result.memory_sol_measured
    lines = [
        "=== ROCm Roofline Analysis (rocprof counters) ===",
        *(
            [f"SOL Efficiency: {result.efficiency_pct:.1f}%"]
            if either_measured
            else []
        ),
        *(
            [f"  Compute SOL:  {result.compute_sol_pct:.1f}%"]
            if result.compute_sol_measured
            else []
        ),
        *(
            [
                "  Memory SOL:   "
                f"{result.memory_sol_pct:.1f}%"
                + (
                    "  (traffic / peak bandwidth)"
                    if result.memory_sol_from_traffic
                    else ""
                )
            ]
            if result.memory_sol_measured
            else []
        ),
        f"  VALU share:   {result.valu_utilization_pct:.1f}%  (instruction mix)",
        f"  VMEM share:   {result.memory_bound_pct:.1f}%  (instruction mix)",
        f"  Bottleneck:   {result.bottleneck}",
        f"  L2 Hit Rate:  {result.tcc_cache_hit_rate_pct:.1f}%",
        "",
    ]

    if result.at_roofline:
        lines.append("Status: AT ROOFLINE")
    elif either_measured:
        lines.append(f"Headroom: {result.headroom_pct:.1f}%")

    if result.warnings:
        lines.append(f"Warnings: {'; '.join(result.warnings)}")

    lines.append("")
    if result.memory_sol_from_traffic:
        lines.append("Memory SOL is profiled traffic divided by peak bandwidth.")
    lines.append("The VALU and VMEM shares are an instruction mix, not SOL.")

    return "\n".join(lines)
