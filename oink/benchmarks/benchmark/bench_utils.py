from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from triton.testing import do_bench as triton_do_bench


@dataclass(frozen=True)
class DeviceMeta:
    device: str
    capability: Tuple[int, int]
    torch: str
    cuda: str
    cute_dsl_arch: str
    git_sha: str
    timestamp: str


def _try_git_sha() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", ".."))
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return ""


def collect_device_meta(device: Optional[torch.device] = None) -> DeviceMeta:
    if device is None:
        device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    timestamp = datetime.now().isoformat(timespec="seconds")
    return DeviceMeta(
        device=str(props.name),
        capability=(int(props.major), int(props.minor)),
        torch=str(torch.__version__),
        cuda=str(getattr(torch.version, "cuda", "unknown")),
        cute_dsl_arch=os.environ.get("CUTE_DSL_ARCH", ""),
        git_sha=_try_git_sha(),
        timestamp=timestamp,
    )


def detect_hbm_peak_gbps(device: Optional[torch.device] = None) -> float:
    """Approximate HBM peak bandwidth in GB/s for roofline fractions."""
    if device is None:
        device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor
    if sm >= 100:
        return 8000.0
    return 2000.0


def do_bench_triton(
    fn: Callable[[], Any], *, warmup_ms: int = 25, rep_ms: int = 100
) -> float:
    """Kernel-only timing consistent with the Oink benchmark harnesses."""
    return float(triton_do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="mean"))


def parse_dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    if s == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def parse_configs(s: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for part in s.split(","):
        m, n = part.lower().split("x")
        out.append((int(m), int(n)))
    return out


def quack_suite_configs() -> List[Tuple[int, int, int]]:
    """Return (batch, seq, hidden) triples following Quack's common grid (hidden=4096)."""
    batch_sizes = [1, 4, 8, 16, 32]
    seq_lengths = [8192, 16384, 32768, 65536, 131072]
    hidden = 4096
    cfgs: List[Tuple[int, int, int]] = []
    for bs in batch_sizes:
        for sl in seq_lengths:
            M = bs * sl
            if M * hidden > (2**31):
                continue
            cfgs.append((bs, sl, hidden))
    return cfgs


def ensure_oink_src_on_path() -> None:
    """Make the in-repo KernelAgent Oink package importable without an editable install."""
    here = os.path.dirname(os.path.abspath(__file__))
    oink_src = os.path.abspath(os.path.join(here, "..", "..", "src"))
    if oink_src not in sys.path:
        sys.path.insert(0, oink_src)


def write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(
    path: str,
    meta: DeviceMeta,
    rows: Sequence[Dict[str, Any]],
    *,
    extra: Dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload: Dict[str, Any] = {
        "meta": {**asdict(meta), **(extra or {})},
        "rows": list(rows),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def iter_row_blocks(M: int, block_rows: int) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) row index ranges for a 2D (M, N) matrix.

    The intent is to make correctness references for large tensors tractable
    without materializing full float32 intermediates.
    """
    if M < 0:
        raise ValueError(f"M must be non-negative, got {M}")
    if block_rows <= 0:
        raise ValueError(f"block_rows must be > 0, got {block_rows}")
    for start in range(0, M, block_rows):
        yield start, min(M, start + block_rows)


@dataclass
class ErrorStats:
    """Numerical error stats between an output and a reference.

    Notes:
    - `max_abs` and `rel_l2` are computed exactly (streamed).
    - `p99_abs` is computed over a deterministic strided sample of abs error
      values (to keep very large tensors tractable).
    """

    max_abs: float
    p99_abs: float
    rel_l2: float
    p99_sample_elems: int
    p99_sample_stride: int


class ErrorStatsAccumulator:
    """Stream error stats over (output_block, ref_block) pairs.

    This is intended for large 2D tensors where we compute reference results
    block-by-block to avoid materializing full float32 intermediates.
    """

    def __init__(self, *, total_elems: int, p99_target_samples: int = 1_000_000):
        if total_elems <= 0:
            raise ValueError(f"total_elems must be > 0, got {total_elems}")
        if p99_target_samples <= 0:
            raise ValueError(
                f"p99_target_samples must be > 0, got {p99_target_samples}"
            )
        self.total_elems = int(total_elems)
        self.p99_target_samples = int(p99_target_samples)
        # Deterministic strided sampling across the flattened tensor order.
        self.sample_stride = max(1, self.total_elems // self.p99_target_samples)
        self._global_offset = 0

        self._max_abs = 0.0
        self._err_sq_sum = 0.0
        self._ref_sq_sum = 0.0
        self._abs_err_samples: List[torch.Tensor] = []

    def update(self, out: torch.Tensor, ref: torch.Tensor) -> None:
        if out.shape != ref.shape:
            raise ValueError(
                f"shape mismatch: out={tuple(out.shape)} ref={tuple(ref.shape)}"
            )

        # Compute error in float32 for stable reductions.
        err_f32 = (out - ref).to(torch.float32)
        abs_err = err_f32.abs()

        # Exact reductions.
        self._max_abs = max(self._max_abs, float(abs_err.max().item()))
        self._err_sq_sum += float((err_f32 * err_f32).sum(dtype=torch.float64).item())
        ref_f32 = ref.to(torch.float32)
        self._ref_sq_sum += float((ref_f32 * ref_f32).sum(dtype=torch.float64).item())

        # Deterministic strided sample for p99_abs.
        flat = abs_err.flatten()
        block_elems = int(flat.numel())
        if block_elems <= 0:
            return

        stride = int(self.sample_stride)
        first = (-int(self._global_offset)) % stride
        if first < block_elems:
            idx = torch.arange(
                first, block_elems, step=stride, device=flat.device, dtype=torch.int64
            )
            # Gather a modest number of values (≈ block_elems/stride).
            vals = (
                flat.index_select(0, idx).detach().to(device="cpu", dtype=torch.float32)
            )
            self._abs_err_samples.append(vals)

        self._global_offset += block_elems

    def finalize(self) -> ErrorStats:
        if self._abs_err_samples:
            samples = torch.cat(self._abs_err_samples, dim=0)
            if samples.numel() > self.p99_target_samples:
                samples = samples[: self.p99_target_samples]
            p99 = (
                float(torch.quantile(samples, 0.99).item())
                if samples.numel() > 0
                else 0.0
            )
            sample_elems = int(samples.numel())
        else:
            p99 = 0.0
            sample_elems = 0

        denom = math.sqrt(self._ref_sq_sum) if self._ref_sq_sum > 0 else 0.0
        rel_l2 = (math.sqrt(self._err_sq_sum) / denom) if denom > 0 else 0.0

        return ErrorStats(
            max_abs=float(self._max_abs),
            p99_abs=float(p99),
            rel_l2=float(rel_l2),
            p99_sample_elems=int(sample_elems),
            p99_sample_stride=int(self.sample_stride),
        )


def error_stats_to_row(prefix: str, stats: ErrorStats) -> Dict[str, Any]:
    """Flatten ErrorStats into JSON-friendly row fields."""
    return {
        f"{prefix}_max_abs": float(stats.max_abs),
        f"{prefix}_p99_abs": float(stats.p99_abs),
        f"{prefix}_rel_l2": float(stats.rel_l2),
        f"{prefix}_p99_sample_elems": int(stats.p99_sample_elems),
        f"{prefix}_p99_sample_stride": int(stats.p99_sample_stride),
    }
