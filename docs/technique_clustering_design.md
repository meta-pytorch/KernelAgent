# Technique-vector clustering for beam-search dedup

This document is the design for a *semantic* deduplication / diversity layer
that complements the existing PTX-hash dedup in `BeamSearchStrategy`.

PTX-hash dedup answers "are two kernels byte-identical at the compiler
level?" — which is correct but conservative (different code that uses the
same techniques does not collapse). This layer answers "are two kernels
applying the same set of optimization techniques?", as judged by an LLM,
and uses that signal to keep beam diversity even after PTX-dedup.

## Architecture

```
update_with_results pipeline (round N):

  pool = top_kernels  +  new_entries
   │
   ▼
  _dedup_by_ptx          ← byte-identical kernels collapse to fastest
   │
   ▼
  classify survivors     ← LLM emits binary technique vector per kernel
   │                       (only for entries without a cached vector)
   ▼
  diversity-aware select ← walk sorted-by-time pool, keep one per cluster,
                            backfill remaining slots with fastest unaccepted
   │
   ▼
  next round's top_kernels
```

PTX-dedup runs first because it's free (already cached on `ProgramEntry`)
and provably correct. The technique-vector layer runs only on the
survivors — typically 12–15 per round in our production runs — so the
extra LLM cost is small.

## The technique taxonomy (expandable)

Techniques are defined in a YAML file. Each entry has:

```yaml
- name: split_k_reduce          # short identifier (must be unique, stable)
  description: |                # one-line human description
    Split-K with separate reduce kernel.
  llm_hint: |                   # how the LLM should detect this technique
    Two kernels in the same module: one writes partial K-slice dot
    products to a scratch buffer; a separate reduce kernel sums them.
```

The default list (`examples/configs/techniques_default.yaml`) covers
Triton-visible techniques observed in our production runs and the
common-techniques checklist:

1. split_k_reduce
2. tensor_cores              (`tl.dot`, WMMA, WGMMA)
3. software_pipelining       (`num_stages >= 2`)
4. swizzled_load              (XOR / blocked-swizzle to dodge bank conflicts)
5. persistent_kernel          (one-block-per-SM + work loop)
6. vectorized_load            (vector dtype loads, `tl.load(...)` width >1)
7. shared_memory_tiling
8. register_tiling            (small `BLOCK_M`/`BLOCK_N`, large `BLOCK_K`)
9. autotuned_config           (`@triton.autotune`)
10. masked_access             (`tl.load(..., mask=...)`)
11. atomic_reduction          (`tl.atomic_add` for cross-block reductions)
12. precision_split           (fp32 accumulator, lower-precision operands)
13. warp_specialization       (producer/consumer split inside one kernel)
14. thread_block_clusters     (Hopper cluster launch / DSMEM)
15. grid_swizzling            (custom rasterization order for L2 reuse)
16. epilogue_fusion           (post-matmul ops fused inside the kernel)
17. loop_unrolling            (`#pragma unroll`, `tl.range(num_stages=…)`)
18. async_copy_tma            (`cp.async`, TMA descriptor loads)

Adding a technique = adding a YAML entry. The vector dimension is
`len(techniques)` at runtime; persisted vectors with mismatched length
are discarded and re-classified on the next round.

## When the LLM classification fires

Manager-side, after PTX dedup. Only entries whose `technique_vector` is
`None` (newly arrived this round, or whose vector was invalidated by a
schema change) get classified. Existing beam members keep their cached
vector — surviving from prior rounds is free.

Classifications run in a `ThreadPoolExecutor` with a small concurrency
cap so the manager doesn't sit on the LLM provider sequentially. With
~12 new survivors per round and 4-way concurrency, expect 10–30 s of
manager-side wall time added per round.

## Clustering rule: diversity-aware selection

Not exact-vector dedup (too aggressive — would discard same-technique
kernels even when they have different runtimes). Not Hamming-threshold
clustering (extra parameter to tune, brittle). Instead: **keep the
fastest representative of each cluster, then backfill remaining beam
slots from the fastest unaccepted entries**.

Algorithm (`select_diverse_top_k`):

```
sort pool by time_ms ascending
beam, seen_clusters = [], set()
for entry in pool:
    cluster_id = tuple(entry.technique_vector or ())
    if cluster_id not in seen_clusters:
        beam.append(entry); seen_clusters.add(cluster_id)
    if len(beam) == K: break
# backfill if fewer than K distinct clusters exist
for entry in pool:
    if len(beam) == K: break
    if entry not in beam: beam.append(entry)
return beam
```

Equivalent semantics: every distinct technique vector represented in the
pool gets at least one beam slot (capped at K). This guarantees diverse
expansion in the next round. Entries whose `technique_vector` is `None`
(classification failed or disabled) are treated as their own singleton
cluster — never merged.

## Configuration

YAML preset (`examples/configs/beam_search_diverse_clustered.yaml`):

```yaml
strategy: beam_search
strategy_config:
  num_top_kernels: 10
  num_expanding_parents: 2
  num_bottlenecks: 3
  samples_per_prompt: 5
  models: [claude-opus-4.6, gpt-5-4, gemini-2-5-pro]
  technique_clustering:
    enabled: true
    techniques_yaml: examples/configs/techniques_default.yaml
    classifier_model: claude-opus-4.6   # one model for stable vectors
    max_concurrency: 4
```

Default behavior (no `technique_clustering` block): clustering is
disabled, behavior matches the prior PTX-only dedup.

## Cost / payoff

- **Cost.** ~12 new classifications × 8 rounds = ~96 extra LLM calls per
  run, parallelized 4-way. ~5–10% of the run's existing LLM budget.
  Persistence of vectors across rounds means surviving beam members
  don't re-classify.
- **Payoff (hypothesis).** In the concentrated production run, rounds
  5–7 just re-discovered the round-4 winner. PTX-dedup correctly
  collapsed those re-discoveries, but the *beam* still filled with
  near-clones because every kernel in the pool was the leader. Diversity
  -aware selection forces the other 9 beam slots to represent different
  technique vectors, which changes what gets expanded in subsequent
  rounds — potentially producing a second improvement axis.

## Open questions

- Schema migration when techniques YAML changes: V1 just discards old
  vectors with mismatched length. A more durable approach hashes the
  YAML and stores `(schema_id, vector)` so old vectors are correctly
  identified as stale. Add only if needed.
- LLM-vector inconsistency: the same kernel classified by two different
  models can produce different vectors. Mitigated by using a single
  `classifier_model` per run; can revisit if vectors look noisy.
- Future: replace LLM with regex/AST-based classifier for deterministic
  detection of structural techniques (split-K, autotune, atomic). Cheap
  and consistent. LLM remains the fallback for fuzzy cases like "warp
  specialization" or "swizzled load pattern".
