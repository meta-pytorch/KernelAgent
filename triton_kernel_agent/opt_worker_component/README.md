# Opt Worker Components

High-level components used by `OptimizationWorker`.

These components are **thin wrappers** around low-level utilities
from `kernel_perf_agent` that provide:
- Logging integration
- Error handling
- Worker-specific configuration

## Dependency Flow
worker_components → kernel_perf_agent (implementation)
