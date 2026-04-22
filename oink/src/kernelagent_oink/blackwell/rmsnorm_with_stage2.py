"""Backward-compatible stage-2 RMSNorm facade.

The stage-2 scheduling policy now lives in `._rmsnorm_impl` so the optimized
pointer path and the compatibility fallback share one implementation.
"""

from __future__ import annotations

from ._rmsnorm_impl import RMSNormSM100, rmsnorm_forward

RMSNormSM100WithStage2 = RMSNormSM100


def rmsnorm_forward_with_stage2(*args, **kwargs):
    return rmsnorm_forward(*args, **kwargs)


__all__ = ["RMSNormSM100WithStage2", "rmsnorm_forward_with_stage2"]
