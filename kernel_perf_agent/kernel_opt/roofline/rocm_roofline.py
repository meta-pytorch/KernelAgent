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

Unlike the NVIDIA path (which uses NCU's built-in "Speed of Light" percent
metrics), AMD rocprof does not expose normalized SOL values directly.  This
module derives heuristic SOL equivalents from raw hardware counters collected
by :mod:`kernel_perf_agent.kernel_opt.profiler.rocprof_profiler`.

Counter → SOL mapping
---------------------
*Compute SOL*  ← VALU instruction utilization (SQ_INSTS_VALU fraction of
               total shader instructions).  A fully VALU-saturated kernel
               approaches 100 %.

*Memory SOL*   ← Memory instruction fraction (SQ_INSTS_VMEM_RD +
               SQ_INSTS_VMEM_WR as fraction of total instructions), amplified
               by the L2 miss rate (a high miss rate means more HBM traffic,
               pushing the kernel closer to the memory roof).

Both are heuristic and share the same interface as
:class:`kernel_perf_agent.kernel_opt.roofline.ncu_roofline.RooflineAnalyzer`
so the rest of the optimization pipeline can use them transparently.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ROCmRooflineConfig:
    """Configuration for ROCm roofline analysis."""

    threshold_pct: float = 85.0  # Lower than NCU (heuristic, not exact SOL)
    early_stop: bool = True
    convergence_rounds: int = 5
    min_improvement_pct: float = 0.1
    underutilized_threshold: float = 40.0  # Both SOL < this → underutilized
    miss_rate_amplifier: float = 1.5  # Scales memory SOL by L2 miss rate


@dataclass
class ROCmRooflineResult:
    """Result of ROCm roofline analysis from rocprof counters."""

    # Heuristic SOL equivalents (0–100 %)
    compute_sol_pct: float
    memory_sol_pct: float

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

    def analyze(self, rocm_metrics: dict[str, Any]) -> ROCmRooflineResult:
        """Analyse rocprof metrics and return a roofline result.

        Args:
            rocm_metrics: Metrics dict from
                :func:`kernel_perf_agent.kernel_opt.profiler.rocprof_profiler.load_rocm_metrics`.

        Returns:
            :class:`ROCmRooflineResult` with heuristic SOL estimates.
        """
        warnings: list[str] = []

        # Retrieve pre-computed heuristic SOL values (set by rocprof_profiler)
        compute_sol = float(rocm_metrics.get("compute_sol_pct", 0.0))
        memory_sol = float(rocm_metrics.get("memory_sol_pct", 0.0))

        valu_pct = float(rocm_metrics.get("valu_utilization_pct", compute_sol))
        mem_bound_pct = float(rocm_metrics.get("memory_bound_pct", memory_sol))
        cache_hit_rate = float(rocm_metrics.get("tcc_cache_hit_rate_pct", 100.0))

        # Amplify memory SOL by L2 miss rate to better reflect HBM pressure.
        # A 100 % cache hit rate means all traffic is served from L2 → low HBM
        # utilisation despite many VMEM instructions.
        l2_miss_rate = max(0.0, 100.0 - cache_hit_rate)
        amplified_memory_sol = min(
            100.0,
            memory_sol
            * (1.0 + (l2_miss_rate / 100.0) * (self.config.miss_rate_amplifier - 1.0)),
        )

        if compute_sol == 0.0 and amplified_memory_sol == 0.0:
            warnings.append(
                "No compute or memory activity detected in rocprof counters"
            )

        efficiency = max(compute_sol, amplified_memory_sol)
        bottleneck = self._classify_bottleneck(compute_sol, amplified_memory_sol)
        at_roofline = efficiency >= self.config.threshold_pct

        return ROCmRooflineResult(
            compute_sol_pct=round(compute_sol, 2),
            memory_sol_pct=round(amplified_memory_sol, 2),
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
        """Classify bottleneck based on heuristic SOL values."""
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


def format_rocm_roofline_summary(result: ROCmRooflineResult) -> str:
    """Format a human-readable summary of ROCm roofline analysis."""
    lines = [
        "=== ROCm Roofline Analysis (rocprof counters) ===",
        f"SOL Efficiency (heuristic): {result.efficiency_pct:.1f}%",
        f"  Compute SOL:  {result.compute_sol_pct:.1f}%  (VALU utilization)",
        f"  Memory SOL:   {result.memory_sol_pct:.1f}%  (VMEM + L2 miss amplification)",
        f"  Bottleneck:   {result.bottleneck}",
        f"  L2 Hit Rate:  {result.tcc_cache_hit_rate_pct:.1f}%",
        "",
    ]

    if result.at_roofline:
        lines.append("Status: AT ROOFLINE (heuristic)")
    else:
        lines.append(f"Headroom: {result.headroom_pct:.1f}%")

    if result.warnings:
        lines.append(f"Warnings: {'; '.join(result.warnings)}")

    lines.append("")
    lines.append("Note: ROCm SOL values are heuristic estimates from rocprof counters,")
    lines.append("not exact hardware-reported percentages like NCU SOL metrics.")

    return "\n".join(lines)
