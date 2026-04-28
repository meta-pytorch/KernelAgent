# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PTX-based kernel fingerprinting for dedup.

Two Triton kernels that lower to identical (normalized) PTX are doing the
same work on the GPU — a strong equivalence relation for "is this the same
kernel?" that is invariant to variable renaming, comments, whitespace, and
most source-level cosmetic changes.

Usage:
    fingerprint = ptx_hash_from_cache(Path(triton_cache_dir))
    if fingerprint is None:
        # PTX capture failed — treat as a singleton for dedup purposes.
        ...

Caveats:
- Raw PTX is not fully canonical; this module applies a normalization pass
  (strip comments, debug, headers; canonicalize register/label names) before
  hashing.  Normalization is best-effort: the failure mode is a false
  negative (two true duplicates hashed differently), which only reduces the
  dedup rate — it does not cause incorrect merges.
- A kernel module may define multiple ``@triton.jit`` functions.  This
  module hashes the sorted concatenation of all normalized PTX strings
  under the cache dir, so the fingerprint reflects the whole module.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# --- Normalization regex patterns -------------------------------------------

# Strip ``//`` line comments and ``/* */`` block comments.
_RE_LINE_COMMENT = re.compile(r"//[^\n]*")
_RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

# PTX directives we want to drop entirely (debug + version/target headers).
# These vary with compile-time context but not with kernel behavior.
_DROP_DIRECTIVE_PREFIXES = (
    ".version",
    ".target",
    ".address_size",
    ".loc",
    ".file",
    ".section .debug",
)

# PTX register classes we canonicalize.  Each class gets its own dense
# numbering starting from 0, assigned in first-occurrence order.
#   %r    — 32-bit unsigned
#   %rd   — 64-bit unsigned
#   %rs   — 16-bit unsigned
#   %f    — 32-bit float
#   %fd   — 64-bit float
#   %p    — predicate
_REGISTER_CLASSES = ("rd", "rs", "fd", "r", "f", "p")
# Regex fragment matching any register of any class, with the class captured.
# Order matters: longer prefixes ("rd", "rs", "fd") before shorter ones so we
# do not partial-match %rd42 as %r+"d42".
_RE_REGISTER = re.compile(r"%(rd|rs|fd|r|f|p)(\d+)\b")

# PTX labels look like ``$L__BB0_3`` or ``$Ltmp1``; normalize by interning.
_RE_LABEL = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")

# Collapse runs of whitespace (including at start of line) to a single space;
# drop pure-blank lines.
_RE_WS = re.compile(r"[ \t]+")


def normalize_ptx(ptx: str) -> str:
    """Return a canonical form of *ptx* suitable for hashing.

    Stripped: comments, debug directives, version/target headers.
    Canonicalized: register names and labels (renamed to dense sequences).
    Normalized: whitespace.

    The output is deterministic given the same input and (importantly)
    identical for PTX strings that differ only in those cosmetic axes.
    """
    text = _RE_BLOCK_COMMENT.sub("", ptx)
    text = _RE_LINE_COMMENT.sub("", text)

    kept_lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if any(stripped.startswith(p) for p in _DROP_DIRECTIVE_PREFIXES):
            continue
        kept_lines.append(stripped)
    body = "\n".join(kept_lines)

    # Canonicalize registers.  We do a single pass building a {old: new}
    # mapping per class; substitution is applied once using re.sub.
    reg_maps: dict[str, dict[str, int]] = {cls: {} for cls in _REGISTER_CLASSES}

    def _register_sub(m: re.Match[str]) -> str:
        cls, num = m.group(1), m.group(2)
        mapping = reg_maps[cls]
        if num not in mapping:
            mapping[num] = len(mapping)
        return f"%{cls}{mapping[num]}"

    body = _RE_REGISTER.sub(_register_sub, body)

    # Canonicalize labels.  Same first-occurrence renaming scheme.
    label_map: dict[str, int] = {}

    def _label_sub(m: re.Match[str]) -> str:
        name = m.group(0)
        if name not in label_map:
            label_map[name] = len(label_map)
        return f"$L{label_map[name]}"

    body = _RE_LABEL.sub(_label_sub, body)

    # Final whitespace collapse.
    body = _RE_WS.sub(" ", body)

    return body


def _find_ptx_files(cache_dir: Path) -> list[Path]:
    """Return every ``*.ptx`` file under *cache_dir* (recursive)."""
    if not cache_dir.exists():
        return []
    return sorted(cache_dir.rglob("*.ptx"))


def ptx_hash_from_cache(cache_dir: Path) -> str | None:
    """Compute a stable hash of all PTX files under *cache_dir*.

    Returns ``None`` if no PTX files are present (e.g. compilation failed
    or the cache dir was not populated).  A ``None`` result tells callers
    to treat the kernel as a dedup singleton.

    The hash covers the normalized PTX of every compiled Triton function
    in the module, sorted by relative path for determinism across runs.
    """
    ptx_files = _find_ptx_files(cache_dir)
    if not ptx_files:
        return None

    hasher = hashlib.sha256()
    # Sort by relative path so the fingerprint is stable regardless of
    # filesystem iteration order.
    rel_sorted = sorted(
        (p.relative_to(cache_dir).as_posix(), p) for p in ptx_files
    )
    for rel, path in rel_sorted:
        try:
            normalized = normalize_ptx(path.read_text(errors="replace"))
        except OSError:
            continue
        # Include the relative filename in the hash so two kernels with the
        # same PTX content under different function names still differ.
        # Strip the leading hash-bucket directory (Triton caches files in
        # content-addressed subdirs, which we do not want in the fingerprint).
        basename = Path(rel).name
        hasher.update(basename.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(normalized.encode("utf-8"))
        hasher.update(b"\0")

    return hasher.hexdigest()
