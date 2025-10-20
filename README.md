# KernelAgent — Code‑to‑Code GPU Kernel Synthesis

KernelAgent is a multi‑agent pipeline that takes PyTorch code, extracts fusable subgraphs, generates and verifies Triton kernels, and composes them into an end‑to‑end program. It emphasizes code‑to‑code transformation, execution‑based verification, and parallel search with early exit.

Blog post: [TBD]

## Examples

- Auto‑route a KernelBench problem (router chooses direct KernelAgent vs full Fuser pipeline):
  - `python -m Fuser.auto_agent --problem /abs/path/to/external/KernelBench/KernelBench/level1/19_ReLU.py --verify`

- Run the full pipeline (extract → dispatch → compose) on a problem:
  - `python -m Fuser.pipeline --problem /abs/path/to/problem.py --extract-model gpt-5 --dispatch-model o4-mini --compose-model o4-mini --dispatch-jobs auto --verify`

- Use KernelAgent directly from Python:
  - `from triton_kernel_agent import TritonKernelAgent`
  - `agent = TritonKernelAgent()`
  - `res = agent.generate_kernel(problem_description="Implement ReLU over a 1D tensor")`

- UIs:
  - Triton KernelAgent UI: `python triton_ui.py`
  - FuserAgent UI: `python -m Fuser.fuser_ui`
  - End‑to‑end pipeline UI: `python -m Fuser.pipeline_ui`

## Requirements

KernelAgent requires or works with:
- Linux or macOS; CUDA‑capable GPU for Triton kernels
- Python 3.8–3.12
- Triton (install separately): `pip install triton` or latest from source
- An LLM provider (one of):
  - OpenAI (`OPENAI_API_KEY`)
  - Anthropic (`ANTHROPIC_API_KEY`)
  - Local relay (OpenAI‑compatible endpoint; see `triton_kernel_agent/providers/relay_provider.py`)

Optional:
- Gradio for UIs (installed via project dependencies)
- Proxy variables when running in restricted environments

## Building KernelAgent

```bash
git clone https://github.com/pytorch-labs/KernelAgent.git
cd KernelAgent
python -m venv .venv && source .venv/bin/activate  # or your preferred env
pip install -e .[dev]
pip install triton  # Triton is not auto‑installed
```

Set provider credentials (for example):
```bash
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY=...
```

Run tests:
```bash
pytest -v
```

## Installing KernelAgent

- Development install: `pip install -e .`
- Triton install (required): `pip install triton`
- Optional nightly: `pip install git+https://github.com/triton-lang/triton.git`

Environment configuration (optional `.env`):
```bash
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-5
NUM_KERNEL_SEEDS=4
MAX_REFINEMENT_ROUNDS=10
LOG_LEVEL=INFO
```

## How KernelAgent works

- Auto‑router (optional): Analyzes the problem and chooses between a direct KernelAgent run or the full Fuser pipeline with fallback.
- Fuser — Extract subgraphs:
  - Orchestrator refactors PyTorch into fusable modules and validates by execution.
  - Subgraph extractor emits JSON with explicit ops, shapes, dtypes, and weights.
- Dispatcher → KernelAgent:
  - For each subgraph, synthesize a precise generation spec and run a fresh TritonKernelAgent instance.
  - Multiple workers explore in parallel; the first passing worker cancels peers.
- Composer:
  - Synthesizes a single end‑to‑end Triton program from the subgraph specs and kernels.
  - Optional verification executes the composed file and requires success (UI defaults on; CLI flag to enable).
- Enforcement & verification:
  - KernelAgent verification blocks common PyTorch compute helpers inside wrappers and gates success on execution (exit code 0 + PASS/sentinel) and bounded tolerances (default rtol/atol 1e‑3; fp16/bf16 never above 1e‑2).

Artifacts are written to a run directory (subgraphs.json, per‑subgraph sessions, composed_kernel.py) for auditability and debugging.

## Full documentation

- Architecture overviews: `docs/kernelfalcon_overview.html`, `docs/kernelfalcon_agents2_overview.html`
- Fuser agent sketches and comparisons: `docs/FuserAgent_sketch.html`, `docs/fuser_agent_compare.html`
- Blog post: [TBD]

## Join the KernelAgent community

- Issues: https://github.com/pytorch-labs/KernelAgent/issues
- Discussions: [TBD]
- Blog: [TBD]

See the `CONTRIBUTING.md` file for how to help out.

## License

KernelAgent is licensed under the Apache License 2.0, as found in the LICENSE file.
