#!/usr/bin/env python3
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
Generate a combined 2-panel SVG: RMSNorm (Quack-suite) + Fused Add+RMSNorm (DSv3).

Example:
  python oink/benchmarks/readme/plot_combined_rmsnorm.py \\
    --bf16-dir /tmp/kernelagent_oink_sm100_suite_bf16 \\
    --roofline-json /tmp/hbm_roofline_sm100_bf16.json \\
    --out oink/benchmarks/media/sm100_bf16_oink_vs_quack_rmsnorm_combined.svg
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plot_quack_style_svg import (
    _aggregate_by_shape,
    _load_json,
    _plot,
    _read_roofline_gbps,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Combined 2-panel SVG: RMSNorm + Fused Add+RMSNorm."
    )
    p.add_argument(
        "--bf16-dir",
        type=str,
        required=True,
        help="Directory containing bf16 suite JSON outputs",
    )
    p.add_argument(
        "--roofline-json",
        type=str,
        default=None,
        help="Optional HBM roofline JSON path",
    )
    p.add_argument("--out", type=str, required=True, help="Output SVG path")
    p.add_argument("--title", type=str, default=None, help="Optional title override")
    args = p.parse_args()

    in_dir = os.path.abspath(args.bf16_dir)

    panel_specs = [
        ("RMSNorm (Quack-suite)", "rmsnorm_fwd_quack_suite_wfp32.json"),
        ("Fused Add+RMSNorm (DSv3)", "fused_add_rmsnorm_dsv3.json"),
    ]

    panels = []
    for panel_title, filename in panel_specs:
        path = os.path.join(in_dir, filename)
        if not os.path.exists(path):
            raise SystemExit(f"Missing required JSON: {path}")
        payload = _load_json(path)
        rows = payload.get("rows", [])
        panels.append((panel_title, _aggregate_by_shape(rows)))

    roofline_gbps = (
        _read_roofline_gbps(args.roofline_json) if args.roofline_json else None
    )

    title = args.title or "SM100 BF16 — RMSNorm / Fused Add+RMSNorm (Oink vs Quack)"

    _plot(
        panels=panels,
        roofline_gbps=roofline_gbps,
        out_path=str(args.out),
        title=title,
        shape_policy="intersection",
        per_panel_x=True,  # different shape sweeps per panel
    )
    print(f"Wrote: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
