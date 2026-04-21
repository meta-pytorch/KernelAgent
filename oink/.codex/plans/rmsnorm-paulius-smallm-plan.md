# RMSNorm Paulius small-M plan

1. Profile the no-weight `N=4096` forward gap versus Paulius on B200 and identify whether launch overhead, schedule shape, or kernel work dominates.
2. Add the smallest possible specialized fast path for the Paulius-style regime (`weight=None`, no bias/residual/rstd) without perturbing Quack-suite and DSv3 paths.
3. Benchmark the tuned path against both Paulius and the prior Oink baseline across representative small-M and large-M shapes.
4. Re-run focused correctness checks for the new specialization and confirm it does not regress the validated Oink RMSNorm suites.
5. Summarize the measured gain, remaining gap to Paulius if any, and the code-size/perf tradeoff.
