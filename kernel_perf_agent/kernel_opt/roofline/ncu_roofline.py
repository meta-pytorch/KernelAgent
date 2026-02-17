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
Roofline Analysis Module using NCU SOL (Speed of Light) Metrics.

This module uses NCU's built-in SOL metrics to determine kernel efficiency
relative to hardware limits

NCU SOL metrics directly measure how close performance is to peak:
- Compute SOL: SM throughput as % of peak
- Memory SOL: DRAM throughput as % of peak

Updated in January 2026
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any


# NCU metrics needed for roofline analysis
# Note: The profiler (ncu_profiler.py) collects these and more metrics.
# This list documents the minimum required for roofline decisions.

NCU_ROOFLINE_METRICS = [
    # Primary SOL metrics (Speed of Light)
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",  # Memory SOL
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # Compute SOL
    # Tensor core detection
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
]


@dataclass
class RooflineConfig:
    """Configuration for roofline analysis."""

    threshold_pct: float = 95.0  # SOL % to consider at roofline
    early_stop: bool = True  # Stop optimization when at roofline
    convergence_rounds: int = 5  # Rounds without improvement to trigger stop
    min_improvement_pct: float = 0.1  # Minimum improvement to continue
    tensor_core_threshold: float = 5.0  # Min TC activity % to consider TC usage
    underutilized_threshold: float = 60.0  # Both SOL < this % = underutilized


@dataclass
class RooflineResult:
    """Result of roofline analysis using NCU SOL metrics."""

    # SOL metrics from NCU (primary)
    compute_sol_pct: float  # SM throughput as % of peak
    memory_sol_pct: float  # DRAM throughput as % of peak

    # Derived efficiency (max of compute/memory SOL)
    efficiency_pct: float  # Primary efficiency metric for decisions
    at_roofline: bool  # True if efficiency >= threshold_pct
    headroom_pct: float  # 100 - efficiency

    # Classification
    bottleneck: str  # "memory" | "compute" | "underutilized"
    uses_tensor_cores: bool  # Whether TC is active

    # Data quality
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class RooflineAnalyzer:
    """Analyzes kernel performance using NCU SOL metrics."""

    def __init__(
        self,
        config: RooflineConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        """
        Initialize the roofline analyzer.

        Args:
            config: Roofline configuration (defaults to RooflineConfig())
            logger: Logger instance
        """
        self.config = config or RooflineConfig()
        self.logger = logger or logging.getLogger(__name__)
        self._efficiency_history: list[float] = []

    def _is_using_tensor_cores(self, ncu_metrics: dict[str, Any]) -> bool:
        """Detect tensor core usage from NCU metrics."""
        tc_cycles = ncu_metrics.get(
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", 0
        )
        return tc_cycles > self.config.tensor_core_threshold

    def _classify_bottleneck(self, compute_sol: float, memory_sol: float) -> str:
        """
        Classify bottleneck based on SOL metrics.

        The LOWER SOL value indicates the bottleneck.
        If both are lower than threshold, the kernel is underutilized (could be occupancy,
        instruction mix, launch config, dependency stalls, etc.).
        """
        threshold = self.config.underutilized_threshold

        # Both low = underutilized (neither resource is saturated)
        if memory_sol < threshold and compute_sol < threshold:
            return "underutilized"

        # Return whichever is lower
        if memory_sol >= compute_sol:
            return "memory"
        else:
            return "compute"

    def analyze(
        self,
        ncu_metrics: dict[str, Any],
    ) -> RooflineResult:
        """
        Analyze kernel performance using NCU SOL metrics.

        Args:
            ncu_metrics: NCU profiling metrics dictionary

        Returns:
            RooflineResult with SOL-based efficiency analysis
        """
        warnings: list[str] = []

        # Extract SOL metrics with missing-key detection
        compute_key = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        memory_key = "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"

        compute_missing = compute_key not in ncu_metrics
        memory_missing = memory_key not in ncu_metrics

        if compute_missing:
            self.logger.warning("Compute SOL metric missing from NCU data")
            warnings.append("Compute SOL metric missing")
        if memory_missing:
            self.logger.warning("Memory SOL metric missing from NCU data")
            warnings.append("Memory SOL metric missing")

        # Fail only if both keys are absent
        if compute_missing and memory_missing:
            return RooflineResult(
                compute_sol_pct=0,
                memory_sol_pct=0,
                efficiency_pct=0,
                at_roofline=False,
                headroom_pct=100,
                bottleneck="unknown",
                uses_tensor_cores=False,
                warnings=["Analysis failed - no SOL metrics in NCU data"],
            )

        compute_sol = ncu_metrics.get(compute_key, 0)
        memory_sol = ncu_metrics.get(memory_key, 0)

        # Primary efficiency: use max of compute/memory
        efficiency = max(compute_sol, memory_sol)

        # Tensor core detection
        uses_tc = self._is_using_tensor_cores(ncu_metrics)

        # Classify bottleneck
        bottleneck = self._classify_bottleneck(compute_sol, memory_sol)

        # Check if at roofline
        at_roofline = efficiency >= self.config.threshold_pct

        return RooflineResult(
            compute_sol_pct=compute_sol,
            memory_sol_pct=memory_sol,
            efficiency_pct=efficiency,
            at_roofline=at_roofline,
            headroom_pct=max(0, 100 - efficiency),
            bottleneck=bottleneck,
            uses_tensor_cores=uses_tc,
            warnings=warnings,
        )

    def should_stop(self, result: RooflineResult) -> tuple[bool, str]:
        """
        Check if optimization should stop based on SOL efficiency and convergence.

        Args:
            result: RooflineResult from analyze()

        Returns:
            Tuple of (should_stop, reason)
        """
        self._efficiency_history.append(result.efficiency_pct)

        # Condition 1: At roofline threshold (if early_stop enabled)
        if self.config.early_stop and result.at_roofline:
            return (
                True,
                f"At roofline ({result.efficiency_pct:.1f}% SOL >= "
                f"{self.config.threshold_pct}%)",
            )

        # Condition 2: Efficiency converged (no improvement for N rounds)
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
        """Reset efficiency history for a new optimization run."""
        self._efficiency_history = []


def format_roofline_summary(result: RooflineResult) -> str:
    """Format a human-readable summary of roofline analysis."""
    lines = [
        "=== Roofline Analysis ===",
        f"SOL Efficiency: {result.efficiency_pct:.1f}%",
        f"  Compute SOL: {result.compute_sol_pct:.1f}%",
        f"  Memory SOL:  {result.memory_sol_pct:.1f}%",
        f"  Bottleneck:  {result.bottleneck}",
        f"  Tensor Cores: {'Yes' if result.uses_tensor_cores else 'No'}",
        "",
    ]

    if result.at_roofline:
        lines.append("Status: AT ROOFLINE")
    else:
        lines.append(f"Headroom: {result.headroom_pct:.1f}%")

    if result.warnings:
        lines.append(f"Warnings: {'; '.join(result.warnings)}")

    return "\n".join(lines)
