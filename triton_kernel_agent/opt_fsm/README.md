# FSM Implementation of OptimizationManager

## Overview

Reimplements the `OptimizationManager` → `OptimizationWorker` → `OptimizationOrchestrator` optimization pipeline as two explicit Finite State Machines (outer manager FSM + inner per-worker FSM). Produces identical output to the existing implementation. All existing components (profiler, benchmarker, bottleneck analyzer, verification worker, strategies, etc.) are reused as-is.

## Files

| File | State / Role | Description |
|---|---|---|
| **Infrastructure** | | |
| `__init__.py` | Public API | Exports `FSMOptimizationManager` |
| `engine.py` | `State` ABC + `FSMEngine` | Base state class (`execute(ctx) -> next_state`) and engine loop |
| `context.py` | `OptimizationContext` | Dataclass holding all mutable state for the outer FSM |
| `fsm_manager.py` | `FSMOptimizationManager` | Drop-in replacement for `OptimizationManager` — same constructor args, same `run_optimization()` return dict |
| **Outer FSM States** (`states/`) | | |
| `states/__init__.py` | — | Package init |
| `states/verify_initial.py` | `VERIFY_INITIAL_KERNEL` | Run correctness test on starting kernel; abort on failure |
| `states/benchmark_baselines.py` | `BENCHMARK_BASELINES` | Time PyTorch eager, torch.compile, and initial kernel |
| `states/select_candidates.py` | `SELECT_CANDIDATES` | Increment round, ask strategy for candidates |
| `states/run_workers.py` | `RUN_WORKERS` | Spawn parallel `mp.Process` workers (30-min timeout, kill on hang) |
| `states/update_strategy.py` | `UPDATE_STRATEGY` | Feed results to strategy, collect shared history/reflexions |
| `states/check_termination.py` | `CHECK_TERMINATION` | Check `strategy.should_terminate()`; loop or finalize |
| `states/finalize.py` | `FINALIZE` | Build final result dict (identical keys to original) |
| **Worker FSM Infrastructure** (`worker/`) | | |
| `worker/__init__.py` | — | Package init |
| `worker/context.py` | `WorkerContext` | Dataclass holding all mutable state for the inner FSM |
| `worker/worker_fsm.py` | `build_worker_fsm()` + `worker_process_entry()` | Builds FSM with all 15 states; process entry point (same signature as original `_worker_process`) |
| **Inner FSM States** (`worker/states/`) | | |
| `worker/states/__init__.py` | — | Package init |
| `worker/states/benchmark_baseline.py` | `BENCHMARK_BASELINE` | Use known time or benchmark; profile baseline SOL |
| `worker/states/profile_kernel.py` | `PROFILE_KERNEL` | Run NCU profiling on current kernel |
| `worker/states/analyze_bottleneck.py` | `ANALYZE_BOTTLENECK` | LLM bottleneck analysis (or override); select primary bottleneck |
| `worker/states/generate_kernel.py` | `GENERATE_KERNEL` | Build optimization prompt (with RAG, history, reflexions), call LLM |
| `worker/states/check_patterns.py` | `CHECK_DISALLOWED_PATTERNS` | Scan for forbidden PyTorch patterns (`torch.nn`, `matmul`, etc.) |
| `worker/states/run_tests.py` | `RUN_TESTS` | Write kernel+test files, run test subprocess |
| `worker/states/check_attempts.py` | `CHECK_REFINE_ATTEMPTS` | Check refinement budget (max 3 attempts) |
| `worker/states/refine_kernel.py` | `REFINE_WITH_LLM` | Send error output to LLM for kernel fix |
| `worker/states/benchmark_new.py` | `BENCHMARK_NEW_KERNEL` | Benchmark the verified optimized kernel |
| `worker/states/profile_sol.py` | `PROFILE_FOR_SOL` | NCU profiling for roofline SOL efficiency |
| `worker/states/generate_reflexion.py` | `GENERATE_REFLEXION` | LLM self-reflection; populates attempt history |
| `worker/states/update_kernels.py` | `UPDATE_BEST_KERNELS` | Two-kernel tracking (best-runtime + best-SOL), divergence reversion |
| `worker/states/check_early_stop.py` | `CHECK_EARLY_STOP` | Roofline-based early termination (≥95% SOL or converged) |
| `worker/states/record_failure.py` | `RECORD_FAILURE` | Record failed generation/verification attempt |
| `worker/states/worker_finalize.py` | `WORKER_FINALIZE` | Profile final best kernel, build result metrics, save `best_kernel.py` |

## State Transition Maps

### Outer FSM

```
VerifyInitialKernel ──pass──→ BenchmarkBaselines → SelectCandidates
       │                          → RunWorkers → UpdateStrategy → CheckTermination
       fail                                                          │       │
       ↓                                                       continue   terminate
    Finalize ←──────────────────────────────────────── (loop) ←──┘       ↓
                                                                      Finalize
```

### Inner (Worker) FSM

```
BenchmarkBaseline → ProfileKernel → AnalyzeBottleneck → GenerateKernel
                                         │                   │
                                      no data             fail
                                         ↓                   ↓
                                   WorkerFinalize      RecordFailure → GenerateReflexion → WorkerFinalize
                                                             ↑
GenerateKernel ──ok──→ CheckPatterns ──clean──→ RunTests ──fail──→ CheckAttempts
                           │                      │                    │        │
                        violation                pass              remaining  exhausted
                           ↓                      ↓                    ↓        ↓
                      RecordFailure         BenchmarkNew        RefineKernel  RecordFailure
                                                 ↓                    ↓
                                            ProfileSol        CheckPatterns (loop)
                                                 ↓
                                         GenerateReflexion
                                                 ↓
                                          UpdateKernels
                                            │         │
                                        at roofline   not
                                            ↓         ↓
                                      CheckEarlyStop  WorkerFinalize
                                            ↓
                                      WorkerFinalize
```
