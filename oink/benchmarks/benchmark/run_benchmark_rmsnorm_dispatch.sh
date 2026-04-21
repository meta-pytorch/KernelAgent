#!/bin/bash
# Benchmark aten vs quack vs oink RMSNorm through the same aten API.
#
# Quack is registered via torch._native (requires quack PR in pytorch).
# Oink is registered via kernelagent_oink.register_all_kernels.
#
# Usage:
#   bash oink/benchmarks/benchmark/run_benchmark_rmsnorm_dispatch.sh
#
# Prerequisites:
#   - conda env "nanoGPT" with torch, quack-kernels==0.3.7, kernelagent-oink
#   - torch._native infrastructure installed in the torch package
#   - `import torch._native` at end of torch/__init__.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NORM_DIR="${TORCH_NATIVE_NORM_DIR:-$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "_native/ops/norm")' 2>/dev/null)}"

if [ -z "$NORM_DIR" ] || [ ! -d "$NORM_DIR" ]; then
    echo "ERROR: torch._native/ops/norm/ not found. Set TORCH_NATIVE_NORM_DIR or install torch._native."
    exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-/tmp}"

echo "Using norm dir: $NORM_DIR"
echo "Results dir: $RESULTS_DIR"
echo ""

# --- Aten baseline (no override) ---
echo "Benchmarking aten..."
echo "" > "$NORM_DIR/__init__.py"
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python "$SCRIPT_DIR/benchmark_rmsnorm_dispatch.py" \
    --mode=aten > "$RESULTS_DIR/dispatch_aten.json" 2>/dev/null

# --- Quack via torch._native ---
echo "Benchmarking quack..."
echo "from . import rmsnorm_impl  # noqa: F401" > "$NORM_DIR/__init__.py"
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python "$SCRIPT_DIR/benchmark_rmsnorm_dispatch.py" \
    --mode=quack > "$RESULTS_DIR/dispatch_quack.json" 2>/dev/null

# --- Oink via aten_override (register_all_kernels) ---
echo "Benchmarking oink..."
echo "" > "$NORM_DIR/__init__.py"
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python "$SCRIPT_DIR/benchmark_rmsnorm_dispatch.py" \
    --mode=oink > "$RESULTS_DIR/dispatch_oink.json" 2>/dev/null

# --- Restore ---
echo "from . import rmsnorm_impl  # noqa: F401" > "$NORM_DIR/__init__.py"

# --- Print tables ---
python3 -c "
import json, sys

aten = json.loads(open('$RESULTS_DIR/dispatch_aten.json').read())['results']
quack = json.loads(open('$RESULTS_DIR/dispatch_quack.json').read())['results']
oink = json.loads(open('$RESULTS_DIR/dispatch_oink.json').read())['results']

def table(title, key):
    print(f'**{title}:**')
    print('\`\`\`')
    print('┌──────────────────┬───────────┬───────────┬───────────┬─────────┬─────────┬─────────┐')
    print('│ Shape            │ Aten (ms) │ Quack (ms)│ Oink (ms) │ Q vs A  │ O vs A  │ O vs Q  │')
    print('├──────────────────┼───────────┼───────────┼───────────┼─────────┼─────────┼─────────┤')
    for shape in aten:
        M, N = shape.split('x')
        a = aten[shape][key]
        q = quack[shape][key]
        o = oink[shape][key]
        qa = a / q
        oa = a / o
        oq = q / o
        print(f'│ ({M:>5s}, {N:>5s})     │ {a:>9.3f} │ {q:>9.3f} │ {o:>9.3f} │ {qa:>6.2f}x │ {oa:>6.2f}x │ {oq:>6.2f}x │')
    print('└──────────────────┴───────────┴───────────┴───────────┴─────────┴─────────┴─────────┘')
    print('\`\`\`')
    print()

table('Forward', 'fwd')
table('Forward + Backward', 'fwdbwd')
"

echo "Done."
