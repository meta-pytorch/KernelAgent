# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Technique-vector clustering for beam-search dedup.

Companion to ``ptx_fingerprint.py``. Where PTX-hash dedup answers
"are two kernels byte-identical at the compiler level?", this module
answers "do two kernels apply the same set of optimization techniques?",
as judged by an LLM, and uses that signal to keep beam diversity even
after PTX-dedup has fired.

The technique taxonomy is loaded from a YAML file at run start and is
expandable: appending a new entry to the YAML widens the vector
dimension by one. Persisted vectors with mismatched length are
discarded and re-classified on the next round.

Public API:
- ``load_techniques(path)`` — read taxonomy from YAML.
- ``classify_kernel(kernel_code, techniques, provider, model, ...)`` —
  one LLM call, returns a tuple of 0/1 ints of length ``len(techniques)``,
  or ``None`` if the call fails.
- ``classify_many(...)`` — thread-pooled fan-out for a batch of kernels.
- ``select_diverse_top_k(entries, k)`` — diversity-aware truncation that
  keeps the fastest representative of each technique cluster.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .history.models import ProgramEntry


# --- Taxonomy ----------------------------------------------------------------


@dataclass(frozen=True)
class TechniqueDefinition:
    """One bit of the binary technique vector.

    The ``name`` is a stable identifier persisted with vectors.  The
    ``llm_hint`` is the prompt fragment that tells the classifier model
    what to look for; it should be a short, concrete description that
    distinguishes this technique from others on the list.
    """

    name: str
    description: str
    llm_hint: str


def load_techniques(path: Path | str) -> list[TechniqueDefinition]:
    """Read the technique taxonomy from a YAML file.

    The YAML must have a top-level ``techniques`` key whose value is a
    list of ``{name, description, llm_hint}`` entries.  Order of the
    list is the bit ordering of the vector — appending is safe;
    reordering invalidates persisted vectors.
    """
    import yaml

    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or "techniques" not in raw:
        raise ValueError(
            f"Techniques YAML at {path} must have a top-level 'techniques' key"
        )
    out: list[TechniqueDefinition] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw["techniques"]):
        name = entry.get("name")
        if not name:
            raise ValueError(f"Technique entry #{i} in {path} is missing 'name'")
        if name in seen:
            raise ValueError(f"Duplicate technique name {name!r} in {path}")
        seen.add(name)
        out.append(
            TechniqueDefinition(
                name=name,
                description=str(entry.get("description", "")).strip(),
                llm_hint=str(entry.get("llm_hint", "")).strip(),
            )
        )
    return out


# --- Prompt + parser --------------------------------------------------------


_PROMPT_HEADER = """\
You are classifying a Triton GPU kernel against a fixed list of optimization
techniques.  For each technique, answer 0 (not used) or 1 (used) based on
the kernel source below.  Be conservative — only mark a bit as 1 if the
technique is unambiguously present in the source.

Return your answer as a JSON object with one key per technique, in the
exact same order the techniques are listed.  Do not include any
commentary, prose, or markdown fencing.  Example shape:

    {"split_k_reduce": 0, "tensor_cores": 1, ...}
"""


def _build_prompt(kernel_code: str, techniques: Sequence[TechniqueDefinition]) -> str:
    lines = [_PROMPT_HEADER, "", "Techniques:"]
    for i, t in enumerate(techniques):
        lines.append(f"{i + 1}. {t.name}: {t.description}")
        if t.llm_hint:
            for hint_line in t.llm_hint.splitlines():
                lines.append(f"     {hint_line.strip()}")
    lines.append("")
    lines.append("Kernel source:")
    lines.append("```python")
    lines.append(kernel_code)
    lines.append("```")
    return "\n".join(lines)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_response(
    text: str, techniques: Sequence[TechniqueDefinition]
) -> tuple[int, ...] | None:
    """Extract a length-N binary vector from the LLM response.

    Tolerates surrounding prose and common markdown fences.  Returns
    ``None`` if the response can't be parsed into the expected shape.
    """
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    bits: list[int] = []
    for t in techniques:
        v = obj.get(t.name)
        if v is None:
            return None
        try:
            bit = int(v)
        except (TypeError, ValueError):
            return None
        if bit not in (0, 1):
            return None
        bits.append(bit)
    return tuple(bits)


# --- Classification ---------------------------------------------------------


def classify_kernel(
    kernel_code: str,
    techniques: Sequence[TechniqueDefinition],
    provider: Any,
    model: str,
    logger: logging.Logger | None = None,
    max_tokens: int = 1024,
) -> tuple[int, ...] | None:
    """One LLM call → binary technique vector for a single kernel.

    Returns ``None`` on any failure (LLM error, parse error, dimension
    mismatch).  Callers should treat ``None`` as "unclassified" — these
    kernels are kept as singleton clusters by ``select_diverse_top_k``.
    """
    log = logger or logging.getLogger(__name__)
    try:
        from triton_kernel_agent.worker_util import _call_llm
    except Exception as e:  # pragma: no cover — only when the import path moves
        log.warning(f"technique_vector: cannot import _call_llm ({e})")
        return None

    prompt = _build_prompt(kernel_code, techniques)
    try:
        response = _call_llm(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            logger=log,
            max_tokens=max_tokens,
        )
    except Exception as e:
        log.warning(f"technique_vector: LLM call failed ({e})")
        return None

    vec = _parse_response(response, techniques)
    if vec is None:
        log.warning(
            f"technique_vector: failed to parse response (first 200 chars: "
            f"{response[:200]!r})"
        )
    return vec


def classify_many(
    items: Iterable[tuple[str, str]],
    techniques: Sequence[TechniqueDefinition],
    provider: Any,
    model: str,
    logger: logging.Logger | None = None,
    max_concurrency: int = 4,
) -> dict[str, tuple[int, ...] | None]:
    """Classify a batch of (program_id, kernel_code) pairs in parallel.

    Returns ``{program_id: vector_or_None}``.  Uses a small thread pool
    so the manager's wall-clock isn't dominated by sequential LLM
    latency.
    """
    log = logger or logging.getLogger(__name__)
    items_list = list(items)
    out: dict[str, tuple[int, ...] | None] = {}
    if not items_list:
        return out

    workers = max(1, min(max_concurrency, len(items_list)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                classify_kernel,
                kernel_code,
                techniques,
                provider,
                model,
                log,
            ): pid
            for pid, kernel_code in items_list
        }
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                out[pid] = fut.result()
            except Exception as e:
                log.warning(f"technique_vector: classify failed for {pid}: {e}")
                out[pid] = None
    return out


# --- Diversity-aware selection ---------------------------------------------


def select_diverse_top_k(
    entries: Sequence[ProgramEntry], k: int
) -> list[ProgramEntry]:
    """Pick top-k from ``entries`` while preserving cluster diversity.

    Walks the input sorted by ``time_ms`` ascending; first occurrence
    of each technique-vector cluster is accepted into the result.
    Once every distinct cluster has a representative or the result has
    ``k`` members, remaining slots are backfilled from the fastest
    unaccepted entries.

    Entries with ``technique_vector is None`` are treated as their own
    singleton cluster (never merged with anything else).
    """
    if k <= 0 or not entries:
        return []
    ordered = sorted(entries, key=lambda e: e.metrics.time_ms)
    accepted: list[ProgramEntry] = []
    accepted_ids: set[str] = set()
    seen_clusters: set[Any] = set()
    none_count = 0

    def _cluster_key(e: ProgramEntry) -> Any:
        v = e.technique_vector
        if v is None:
            nonlocal none_count
            none_count += 1
            return ("__none__", none_count)  # singleton: each None is unique
        return tuple(v)

    # First pass: one per cluster.
    for entry in ordered:
        if len(accepted) >= k:
            break
        cid = _cluster_key(entry)
        if cid in seen_clusters:
            continue
        accepted.append(entry)
        accepted_ids.add(entry.program_id)
        seen_clusters.add(cid)

    # Backfill remaining slots from the fastest unaccepted.
    if len(accepted) < k:
        for entry in ordered:
            if len(accepted) >= k:
                break
            if entry.program_id in accepted_ids:
                continue
            accepted.append(entry)
            accepted_ids.add(entry.program_id)

    return accepted
