#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NORM_DIR="${TORCH_NATIVE_NORM_DIR:-$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "_native/ops/norm")' 2>/dev/null)}"

RESULTS_DIR="${RESULTS_DIR:-/tmp}"

echo "CUDA Graph Benchmark: Aten vs Quack vs Oink RMSNorm"
echo "===================================================="
echo ""

# Aten
echo "Running aten..."
echo "" > "$NORM_DIR/__init__.py"
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python "$SCRIPT_DIR/benchmark_cudagraph_all.py" \
    --mode=aten > "$RESULTS_DIR/cudagraph_all_aten.json" 2>/dev/null

# Quack
echo "Running quack..."
echo "from . import rmsnorm_impl  # noqa: F401" > "$NORM_DIR/__init__.py"
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python "$SCRIPT_DIR/benchmark_cudagraph_all.py" \
    --mode=quack > "$RESULTS_DIR/cudagraph_all_quack.json" 2>/dev/null

# Oink
echo "Running oink..."
echo "" > "$NORM_DIR/__init__.py"
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python "$SCRIPT_DIR/benchmark_cudagraph_all.py" \
    --mode=oink > "$RESULTS_DIR/cudagraph_all_oink.json" 2>/dev/null

# Restore
echo "from . import rmsnorm_impl  # noqa: F401" > "$NORM_DIR/__init__.py"

python3 -c "
import json

aten = json.loads(open('$RESULTS_DIR/cudagraph_all_aten.json').read())['results']
quack = json.loads(open('$RESULTS_DIR/cudagraph_all_quack.json').read())['results']
oink = json.loads(open('$RESULTS_DIR/cudagraph_all_oink.json').read())['results']

print()
print('Forward (CUDA graph, bf16):')
print('┌──────────────────┬───────────┬───────────┬───────────┬─────────┬─────────┬─────────┐')
print('│ Shape            │ Aten (ms) │ Quack (ms)│ Oink (ms) │ Q vs A  │ O vs A  │ O vs Q  │')
print('├──────────────────┼───────────┼───────────┼───────────┼─────────┼─────────┼─────────┤')
for shape in aten:
    M, N = shape.split('x')
    a = aten[shape]['fwd']
    q = quack[shape]['fwd']
    o = oink[shape]['fwd']
    def fmt(v):
        return f'{v:9.4f}' if v > 0 else '     FAIL'
    def ratio(num, den):
        if den <= 0 or num <= 0:
            return '   N/A  '
        return f'{num/den:7.2f}x'
    print(f'│ ({M:>5s}, {N:>5s})     │ {fmt(a)} │ {fmt(q)} │ {fmt(o)} │ {ratio(a,q)} │ {ratio(a,o)} │ {ratio(q,o)} │')
print('└──────────────────┴───────────┴───────────┴───────────┴─────────┴─────────┴─────────┘')
"

echo ""
echo "Done."
