# RMSNorm concision plan

1. Validate the current split facade plus shared-launcher refactor with `py_compile` and focused forward, fused-add, and backward benchmarks.
2. Recount Oink and Quack RMSNorm LOC after the latest backward fast-launch restoration.
3. Trim only low-risk host-side duplication that does not change kernel math, schedule routing, or small-`M` backward behavior.
4. Re-run targeted performance and correctness checks after each concision step and revert any regression immediately.
5. Summarize final LOC, benchmark deltas, and remaining gap to Quack-sized readability.
