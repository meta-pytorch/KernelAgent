"""
Generate Quack-style SVG performance plots (Oink vs Quack) from the SM100 suite
JSON artifacts under `/tmp/kernelagent_oink_sm100_suite_{bf16,fp16}`.

The intent is to match Quack's README visual style:
  - 3 horizontal panels (suite-dependent):
      - Quack-suite: RMSNorm / Softmax / CrossEntropy
      - DSv3 (hidden-size): Fused Add+RMSNorm / Softmax / LayerNorm
      - DSv3 (all ops, 4-panel): Fused Add+RMSNorm / Softmax / LayerNorm / CrossEntropy
      - DSv3 CrossEntropy: CrossEntropy-only (single panel)
  - y-axis: model memory bandwidth (GB/s) derived from an IO model
  - x-axis: a small set of labeled (M, N) shape points
  - thick lines + markers, dashed y-grid, compact legend
  - optional horizontal roofline line (measured STREAM-like HBM peak)

Example:
  python oink/benchmarks/readme/plot_quack_style_svg.py \\
    --in-dir /tmp/kernelagent_oink_sm100_suite_bf16 \\
    --suite quack_suite \\
    --roofline-json /tmp/hbm_roofline_sm100_bf16.json \\
    --out oink/benchmarks/media/sm100_bf16_oink_vs_quack.svg

For completeness, we can also include LayerNorm as an extra panel (Quack's
own README plot does not include LayerNorm):
  python oink/benchmarks/readme/plot_quack_style_svg.py \\
    --in-dir /tmp/kernelagent_oink_sm100_suite_bf16 \\
    --suite quack_suite \\
    --include-layernorm \\
    --roofline-json /tmp/hbm_roofline_sm100_bf16.json \\
    --out oink/benchmarks/media/sm100_bf16_oink_vs_quack_with_layernorm.svg

Note on DSv3 suite:
- The DSv3 plot intentionally covers only the hidden-size ops (fused Add+RMSNorm,
  Softmax, LayerNorm) which share the same `(M, N)` sweep.
- CrossEntropy in DSv3 uses a vocab-size-like `N` sweep and is plotted separately
  via `--suite dsv3_cross_entropy` to avoid a mixed x-axis with gaps.
- For README embedding convenience, `--suite dsv3_all` renders a 4-panel
  single-row figure where the CrossEntropy panel uses its own x-axis.
- The RMSNorm panel uses the real block primitive (fused residual-add + RMSNorm)
  when available: `fused_add_rmsnorm_dsv3.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _fmt_k(v: int) -> str:
    # Match Quack's x-axis labels: "32K" means 32768 (1024-based).
    if v % 1024 == 0:
        return f"{v // 1024}K"
    return str(v)


def _shape_label(m: int, n: int) -> str:
    return f"({_fmt_k(m)}, {_fmt_k(n)})"


def _gbps_from_row(prefix: str, row: Mapping[str, Any]) -> Optional[float]:
    # Prefer GB/s in the JSON if present; otherwise fall back to TB/s.
    gbps_key = f"{prefix}_gbps"
    tbps_key = f"{prefix}_tbps"
    if gbps_key in row and row[gbps_key] is not None:
        return float(row[gbps_key])
    if tbps_key in row and row[tbps_key] is not None:
        return float(row[tbps_key]) * 1000.0
    return None


def _aggregate_by_shape(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Aggregate duplicate (M, N) rows using median (more robust than mean)."""
    buckets: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        m = int(r["M"])
        n = int(r["N"])
        ours = _gbps_from_row("ours", r)
        quack = _gbps_from_row("quack", r)
        if ours is not None:
            buckets[(m, n)]["ours"].append(ours)
        if quack is not None:
            buckets[(m, n)]["quack"].append(quack)

    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for k, vs in buckets.items():
        if not vs["ours"] or not vs["quack"]:
            continue
        out[k] = dict(ours=float(median(vs["ours"])), quack=float(median(vs["quack"])))
    return out


def _sort_shapes(shapes: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    # Sort by N then M to keep the x-axis stable across panels.
    return sorted(set(shapes), key=lambda x: (x[1], x[0]))


def _read_roofline_gbps(path: str) -> float:
    payload = _load_json(path)
    rows = payload.get("rows", [])
    best_tbps = max(float(r["tbps"]) for r in rows)
    return best_tbps * 1000.0


def _ensure_matplotlib():
    try:
        import matplotlib as mpl  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required to generate SVG plots.\n"
            "Install with: `python -m pip install matplotlib`"
        ) from e


def _plot(
    *,
    panels: Sequence[Tuple[str, Dict[Tuple[int, int], Dict[str, float]]]],
    roofline_gbps: Optional[float],
    out_path: str,
    title: str,
    shape_policy: str,
    per_panel_x: bool,
) -> None:
    _ensure_matplotlib()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            # Quack-style: embed glyphs as paths for consistent rendering.
            "svg.fonttype": "path",
            "font.family": "DejaVu Sans",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 10,
            "ytick.labelsize": 12,
        }
    )

    # Colors roughly matching Quack's SVG palette.
    COLOR_OINK = "#5ba3f5"
    COLOR_QUACK = "#ff4444"
    COLOR_ROOF = "#4d4d4d"

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(panels),
        figsize=(6.0 * len(panels), 5.6),
        constrained_layout=False,
        sharey=True,
    )
    if len(panels) == 1:
        axes = [axes]

    max_y = 0.0
    for ax, (panel_title, data) in zip(axes, panels):
        if per_panel_x:
            shapes = _sort_shapes(data.keys())
        else:
            # Quack-style plots use a single shared x-axis across panels. Prefer
            # the intersection so every panel has a value at every x tick
            # (cleaner than rendering gaps), and fall back to the union if the
            # intersection is empty.
            shape_sets = [set(d.keys()) for _n, d in panels]
            if shape_policy in {"first", "primary"}:
                shapes = _sort_shapes(shape_sets[0]) if shape_sets else []
            elif shape_policy == "intersection" and shape_sets:
                common = set.intersection(*shape_sets)
                shapes = _sort_shapes(common) if common else []
            elif shape_policy == "union":
                shapes = _sort_shapes(s for _n, d in panels for s in d.keys())
            else:
                raise ValueError(f"Unsupported shape_policy: {shape_policy}")
            if not shapes:
                shapes = _sort_shapes(s for _n, d in panels for s in d.keys())

        x = list(range(len(shapes)))
        x_labels = [_shape_label(m, n) for (m, n) in shapes]

        ours_y: List[float] = []
        quack_y: List[float] = []
        for s in shapes:
            rec = data.get(s)
            if rec is None:  # only possible in shared-x mode with union
                ours_y.append(math.nan)
                quack_y.append(math.nan)
                continue
            ours_y.append(float(rec["ours"]))
            quack_y.append(float(rec["quack"]))
        max_y = max(
            max_y,
            *(v for v in ours_y if math.isfinite(v)),
            *(v for v in quack_y if math.isfinite(v)),
        )

        ax.plot(
            x,
            ours_y,
            marker="o",
            linewidth=5,
            markersize=7,
            color=COLOR_OINK,
            label="KernelAgent-Oink (ours)",
        )
        ax.plot(
            x,
            quack_y,
            marker="o",
            linewidth=5,
            markersize=7,
            color=COLOR_QUACK,
            label="Quack",
        )
        if roofline_gbps is not None:
            ax.axhline(
                roofline_gbps,
                color=COLOR_ROOF,
                linewidth=3,
                linestyle=(0, (4, 6)),
                label="HBM peak (measured)" if ax is axes[0] else None,
            )
            max_y = max(max_y, float(roofline_gbps))

        ax.set_title(panel_title)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=-45, ha="left")
        if per_panel_x:
            # DSv3 "all ops" figure: each panel has its own x-axis. Make the
            # semantics explicit so readers don't assume the same `N` meaning
            # across panels (CrossEntropy uses a classes/vocab-shard-like axis).
            if "cross" in panel_title.lower():
                ax.set_xlabel("Shape (M, C classes)")
            else:
                ax.set_xlabel("Shape (M, N hidden)")

        # Quack-like dashed y-grid.
        ax.grid(axis="y", linestyle=(0, (4, 7.2)), linewidth=0.8, color="#b0b0b0")
        ax.set_axisbelow(True)

        # Light spines (Quack SVG uses a light gray frame).
        for spine in ax.spines.values():
            spine.set_color("#d3d3d3")
            spine.set_linewidth(1.5)

    axes[0].set_ylabel("Memory Bandwidth (GB/s)")

    # A little headroom above the tallest curve/roofline.
    ymax = max_y * 1.08 if max_y > 0 else 1.0
    for ax in axes:
        ax.set_ylim(0.0, ymax)

    # Tight layout for the axes area, reserving headroom for the suptitle and a
    # shared legend. In some matplotlib versions, figure-level legends can
    # overlap the middle panel title unless we reserve a slightly taller header
    # band.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.70))

    # Single shared legend across the top (like Quack), but keep it inside the
    # reserved header band so it doesn't overlap the middle panel title.
    handles, labels = axes[0].get_legend_handles_labels()
    # Quack's legend fits nicely in one row because their plots are 3-panel and
    # therefore wide. For single-panel figures, a 3-column legend can overflow
    # the canvas and get clipped in the SVG, so we stack it vertically.
    legend_ncol = min(3, len(labels))
    legend_fontsize = 13
    if len(panels) == 1:
        legend_ncol = 1
        legend_fontsize = 12
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.91),
        fontsize=legend_fontsize,
        handlelength=2.5,
    )
    # Single-panel figures (e.g. DSv3 CrossEntropy) are much narrower than the
    # Quack-style 3-panel plots; use a slightly smaller suptitle font to avoid
    # clipping in the exported SVG.
    suptitle_fs = 22 if len(panels) > 1 else 18
    fig.suptitle(title, y=0.98, fontsize=suptitle_fs)

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Use a tight bounding box so rotated x tick labels and the figure-level
    # legend don't get clipped in SVG exports (matplotlib can be fragile here
    # across versions).
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _panel_files_for_suite(suite: str) -> List[Tuple[str, str]]:
    if suite == "quack_suite":
        return [
            ("RMSNorm (fp32 weight)", "rmsnorm_fwd_quack_suite_wfp32.json"),
            ("Softmax (fwd+bwd)", "softmax_fwd_bwd_quack_suite.json"),
            ("Cross-Entropy (fwd+bwd)", "cross_entropy_fwd_bwd_quack_suite.json"),
        ]
    if suite == "dsv3":
        return [
            ("Fused Add+RMSNorm (fwd)", "fused_add_rmsnorm_dsv3.json"),
            ("Softmax (fwd+bwd)", "softmax_fwd_bwd_dsv3.json"),
            ("LayerNorm (fwd)", "layernorm_fwd_dsv3.json"),
        ]
    if suite == "dsv3_all":
        return [
            ("Fused Add+RMSNorm (fwd)", "fused_add_rmsnorm_dsv3.json"),
            ("Softmax (fwd+bwd)", "softmax_fwd_bwd_dsv3.json"),
            ("LayerNorm (fwd)", "layernorm_fwd_dsv3.json"),
            ("Cross-Entropy (fwd+bwd)", "cross_entropy_fwd_bwd_dsv3.json"),
        ]
    if suite == "dsv3_cross_entropy":
        return [
            ("Cross-Entropy (fwd+bwd)", "cross_entropy_fwd_bwd_dsv3.json"),
        ]
    raise ValueError(f"Unsupported suite: {suite}")


def _layernorm_file_for_suite(suite: str) -> str:
    if suite == "quack_suite":
        return "layernorm_fwd_quack_suite.json"
    raise ValueError(f"Unsupported suite: {suite}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate Quack-style SVG plots from KernelAgent-Oink suite JSONs."
    )
    p.add_argument(
        "--in-dir",
        type=str,
        required=True,
        help="Directory containing suite JSON outputs",
    )
    p.add_argument(
        "--suite",
        type=str,
        default="quack_suite",
        choices=["quack_suite", "dsv3", "dsv3_all", "dsv3_cross_entropy"],
    )
    p.add_argument(
        "--include-layernorm",
        action="store_true",
        help="Add a LayerNorm (fwd) panel (only meaningful for `--suite quack_suite`).",
    )
    p.add_argument(
        "--shape-policy",
        type=str,
        default="intersection",
        choices=["intersection", "union", "first"],
        help=(
            "How to pick x-axis shapes across panels. "
            "`intersection` matches Quack-style (only shapes common to every panel). "
            "`first` uses the first panel's shapes (keeps DSv3 N=7168 visible). "
            "`union` includes every shape across panels (may create gaps)."
        ),
    )
    p.add_argument(
        "--roofline-json",
        type=str,
        default=None,
        help="Optional /tmp/hbm_roofline_sm100_*.json path",
    )
    p.add_argument("--out", type=str, required=True, help="Output SVG path")
    p.add_argument(
        "--title", type=str, default=None, help="Optional figure title override"
    )
    args = p.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    if not os.path.isdir(in_dir):
        raise SystemExit(f"--in-dir is not a directory: {in_dir}")

    roofline_gbps = (
        _read_roofline_gbps(args.roofline_json) if args.roofline_json else None
    )

    panel_files = list(_panel_files_for_suite(str(args.suite)))
    if args.include_layernorm:
        if args.suite != "quack_suite":
            raise SystemExit(
                "--include-layernorm is only supported for `--suite quack_suite`."
            )
        panel_files.append(
            ("LayerNorm (fwd)", _layernorm_file_for_suite(str(args.suite)))
        )

    panels: List[Tuple[str, Dict[Tuple[int, int], Dict[str, float]]]] = []
    for panel_title, filename in panel_files:
        path = os.path.join(in_dir, filename)
        if not os.path.exists(path):
            raise SystemExit(f"Missing required JSON: {path}")
        payload = _load_json(path)
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            rows = []
        panels.append((panel_title, _aggregate_by_shape(rows)))

    if args.title is not None:
        title = str(args.title)
    else:
        # Try to infer dtype from the first panel's JSON.
        first_json = os.path.join(in_dir, panel_files[0][1])
        payload = _load_json(first_json)
        rows = payload.get("rows", [])
        dtype = rows[0].get("dtype", "") if rows else ""
        if args.suite == "quack_suite":
            suite_name = "Quack-suite"
        elif args.suite == "dsv3":
            suite_name = "DSv3 (hidden-size ops)"
        elif args.suite == "dsv3_all":
            suite_name = "DSv3 (4 ops)"
        elif args.suite == "dsv3_cross_entropy":
            # Keep this short: this suite is rendered as a single panel, so the
            # figure is much narrower than the 3-panel plots.
            suite_name = "DSv3 CrossEntropy"
        else:
            suite_name = str(args.suite)
        suffix = (
            " (+LayerNorm)"
            if (args.suite == "quack_suite" and args.include_layernorm)
            else ""
        )
        if args.suite == "dsv3_cross_entropy":
            title = f"SM100 {dtype.upper()} — {suite_name}{suffix}"
        else:
            title = f"SM100 {dtype.upper()} Kernel Benchmarks (Oink vs Quack) — {suite_name}{suffix}"

    _plot(
        panels=panels,
        roofline_gbps=roofline_gbps,
        out_path=str(args.out),
        title=title,
        shape_policy=str(args.shape_policy),
        per_panel_x=(str(args.suite) == "dsv3_all"),
    )
    print(f"Wrote: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
