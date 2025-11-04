# Contributing to KernelAgent

Thanks for your interest in improving KernelAgent! We aim to make
contributions straightforward, transparent, and welcoming.

## Code of Conduct
By participating, you agree to abide by our Code of Conduct. See `CODE_OF_CONDUCT.md`.

## Ways to Contribute
- Report bugs and performance issues
- Propose features and improvements
- Improve documentation and examples
- Add tests and fix bugs
- Optimize kernels or refine prompts/templates

## Development Setup

Supported Python: 3.8–3.12. A CUDA‑enabled GPU and an API key are only required
for end‑to‑end runs; basic tests do not require them.

1) Clone and install (choose one):

```bash
git clone https://github.com/pytorch-labs/KernelAgent.git
cd KernelAgent

# Using pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"

# Or using uv (matches CI)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

2) Optional: set environment for e2e runs (LLM + GPU):

```bash
cp .env.example .env   # then edit with your API key(s)
```

## Running Tests

Unit tests:

```bash
uv run pytest tests/ -v
```

Coverage (optional):

```bash
uv run pytest tests/ -v --cov=triton_kernel_agent
```

End‑to‑end example (requires network + GPU):

```bash
uv run python e2e_test.py
```

## Linting and Style

We follow PEP 8 with type hints and docstrings.

- Indentation: 4 spaces
- Line length: aim for ~100 chars where practical
- Imports: standard library, third‑party, then local
- Logging: prefer `logging` over `print`
- Types: use annotations for public APIs

Ruff configuration is included in `pyproject.toml`. If you have `ruff` installed, run:

```bash
ruff check .
```

## Pull Requests
We actively welcome your pull requests. Small, focused PRs are easier to review.

1. Fork the repo and branch from `main`.
2. Make your change with tests where appropriate.
3. Update docs/README/examples if behavior or APIs change.
4. Run tests locally and ensure CI passes.
5. Optionally run `ruff check .` and address issues.
6. Submit a clear PR description with context and rationale.

### PR Checklist
- Tests added or updated when behavior changes
- Docs/examples updated when user‑facing behavior changes
- No unrelated formatting or drive‑by refactors
- CI green (see `.github/workflows/ci.yml`)

## Issues
Use GitHub Issues for bugs and feature requests.

When reporting a bug, please include:
- What you did (minimal reproducible example)
- What you expected to happen
- What actually happened (full error, logs, screenshots as relevant)
- Environment details (OS, Python, GPU/CUDA, package versions)

## Security
If you believe you’ve found a security vulnerability, do not open a public issue.
Please follow the process described in Meta’s [bug bounty program](https://bugbounty.meta.com/).

## Contributor License Agreement (CLA)
Some contributions may require signing a CLA before they can be merged. If
prompted by the bot during your PR, please complete the form here:
https://code.facebook.com/cla

## License
By contributing to KernelAgent, you agree that your contributions will be
licensed under the `LICENSE` file in the root of this repository.

## Notes for Kernel/Template Contributions
- Changes to Jinja2 templates in [`triton_kernel_agent/templates/`](https://github.com/meta-pytorch/KernelAgent/tree/main/triton_kernel_agent/templates) should be validated with an
  end‑to‑end run (see e2e section) since they affect generation behavior.
- For GPU kernels or performance‑sensitive code, include a short note about
  observed correctness and performance if available.
