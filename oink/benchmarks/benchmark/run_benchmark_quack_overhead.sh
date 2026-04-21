#!/bin/bash
# Overhead analysis for the quack RMSNorm PR (pytorch#178326).
#
# Compares aten baseline vs quack override through the same
# torch.ops.aten._fused_rms_norm API at various shapes.
#
# Usage:
#   cd KernelAgent
#   conda activate nanoGPT
#   bash oink/benchmarks/benchmark/run_benchmark_quack_overhead.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NORM_DIR="${TORCH_NATIVE_NORM_DIR:-$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "_native/ops/norm")' 2>/dev/null)}"

if [ -z "$NORM_DIR" ] || [ ! -d "$NORM_DIR" ]; then
    echo "ERROR: torch._native/ops/norm/ not found."
    exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-/tmp}"

echo "Quack RMSNorm PR Overhead Analysis"
echo "==================================="
echo "norm dir: $NORM_DIR"
echo ""

# --- Aten baseline (no override) ---
echo "Running aten baseline..."
echo "" > "$NORM_DIR/__init__.py"
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python "$SCRIPT_DIR/benchmark_quack_overhead.py" \
    --mode=aten > "$RESULTS_DIR/quack_overhead_aten.json" 2>/dev/null

# --- Quack via torch._native ---
echo "Running quack override..."
echo "from . import rmsnorm_impl  # noqa: F401" > "$NORM_DIR/__init__.py"
TORCH_NATIVE_SKIP_VERSION_CHECK=1 python "$SCRIPT_DIR/benchmark_quack_overhead.py" \
    --mode=quack > "$RESULTS_DIR/quack_overhead_quack.json" 2>/dev/null

# --- Restore ---
echo "from . import rmsnorm_impl  # noqa: F401" > "$NORM_DIR/__init__.py"

# --- Print report ---
python3 -c "
import json

aten = json.loads(open('$RESULTS_DIR/quack_overhead_aten.json').read())['results']
quack = json.loads(open('$RESULTS_DIR/quack_overhead_quack.json').read())['results']

def print_table(title, key):
    print(f'{title}')
    print('┌──────────────────┬───────────┬───────────┬──────────┬────────────┐')
    print('│ Shape            │ Aten (ms) │ Quack (ms)│ Quack/A  │ Overhead   │')
    print('├──────────────────┼───────────┼───────────┼──────────┼────────────┤')
    for shape in aten:
        M, N = shape.split('x')
        a = aten[shape][key]
        q = quack[shape][key]
        ratio = a / q
        overhead_ms = q - a
        marker = '  ✓ faster' if ratio > 1.0 else '  ✗ slower'
        print(f'│ ({M:>5s}, {N:>5s})     │ {a:>9.4f} │ {q:>9.4f} │ {ratio:>7.2f}x │ {overhead_ms:>+8.4f}ms│')
    print('└──────────────────┴───────────┴───────────┴──────────┴────────────┘')
    print()

print()
print_table('Forward only:', 'fwd')
print_table('Forward + Backward:', 'fwdbwd')

# Summary
print('Summary:')
print('--------')
fwd_crossover = None
bwd_crossover = None
for shape in aten:
    M, N = shape.split('x')
    M = int(M)
    a_fwd = aten[shape]['fwd']
    q_fwd = quack[shape]['fwd']
    a_bwd = aten[shape]['fwdbwd']
    q_bwd = quack[shape]['fwdbwd']
    if fwd_crossover is None and a_fwd / q_fwd >= 1.0:
        fwd_crossover = (M, int(N))
    if bwd_crossover is None and a_bwd / q_bwd >= 1.0:
        bwd_crossover = (M, int(N))

if fwd_crossover:
    print(f'  Forward crossover (quack >= aten): M={fwd_crossover[0]}, N={fwd_crossover[1]}')
else:
    print(f'  Forward: quack is slower than aten at all tested shapes')
if bwd_crossover:
    print(f'  Fwd+Bwd crossover (quack >= aten): M={bwd_crossover[0]}, N={bwd_crossover[1]}')
else:
    print(f'  Fwd+Bwd: quack is slower than aten at all tested shapes')

# Overhead analysis
print()
print('Overhead analysis:')
small_fwd = [quack[s]['fwd'] - aten[s]['fwd'] for s in list(aten)[:6]]
small_bwd = [quack[s]['fwdbwd'] - aten[s]['fwdbwd'] for s in list(aten)[:6]]
avg_fwd_overhead = sum(small_fwd) / len(small_fwd)
avg_bwd_overhead = sum(small_bwd) / len(small_bwd)
print(f'  Avg fwd overhead at small M (1-256):     {avg_fwd_overhead:+.4f} ms/call')
print(f'  Avg fwd+bwd overhead at small M (1-256):  {avg_bwd_overhead:+.4f} ms/call')
print(f'  This overhead is from Python dispatch through torch._native:')
print(f'    _fused_rms_norm_impl → quack_rmsnorm_fwd → _compile_rmsnorm_fwd → kernel')
print(f'  At large M, the faster CuTeDSL kernel overcomes this overhead.')
"

echo ""
echo "Done."
