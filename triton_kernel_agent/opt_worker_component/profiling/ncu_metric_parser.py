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

import json
from typing import Any
from typing import Dict
import pandas as pd
from pathlib import Path

from kernel_perf_agent.kernel_opt.profiler.ncu_profiler import (
    load_ncu_metrics,
    metrics_to_prompt,
)


def load_and_filter_metrics(csv_path: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load NCU metrics from CSV and filter out PyTorch auxiliary kernels.
    """

    # PyTorch auxiliary kernel patterns to exclude
    EXCLUDED_KERNEL_PATTERNS = [
        "vectorized_elementwise",
        "fillkernel",
        "copykernel",
        "memcpy",
        "memset",
    ]

    metrics_df = load_ncu_metrics(
        csv_path,
        select="last",
        name_list=None,
    )

    if metrics_df.empty or "Kernel Name" not in metrics_df.columns:
        return metrics_df

    kernel_names = metrics_df["Kernel Name"].tolist()
    triton_kernel_names = [
        name
        for name in kernel_names
        if not any(exclude in str(name).lower() for exclude in EXCLUDED_KERNEL_PATTERNS)
    ]

    if triton_kernel_names:
        metrics_df = load_ncu_metrics(
            csv_path,
            select="last",
            name_list=triton_kernel_names[:1],
        )

    return metrics_df, json.loads(metrics_to_prompt(metrics_df))
