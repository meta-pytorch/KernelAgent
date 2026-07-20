#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PTX-based kernel fingerprinting and deduplication."""

import os

import pytest

from triton_kernel_agent.opt_worker_component.searching import ptx_fingerprint as _ptx
normalize_ptx = _ptx.normalize_ptx
ptx_hash_from_cache = _ptx.ptx_hash_from_cache

BASE_PTX = """\
.version 8.2
.target sm_90
// a comment
.visible .entry kern()
{
    .reg .b32 %r<3>;
    mov.u32 %r1, 5;
    add.s32 %r2, %r1, 7;
$L__BB0_1:
    bra $L__BB0_1;
}
"""


class TestNormalizePtx:
    """Tests for PTX normalization invariants."""

    def test_register_renumbering_invariant(self):
        """Different register numbering should not change the fingerprint."""
        renumbered = BASE_PTX.replace("%r1", "%r7").replace("%r2", "%r4")
        assert normalize_ptx(BASE_PTX) == normalize_ptx(renumbered)

    def test_comment_and_whitespace_invariant(self):
        """Comments and extra whitespace should not affect the fingerprint."""
        noisy = BASE_PTX.replace("// a comment", "// completely different comment\n\n   ")
        assert normalize_ptx(BASE_PTX) == normalize_ptx(noisy)

    def test_version_and_target_directives_ignored(self):
        """PTX version and target directives should not affect the fingerprint."""
        other_version = BASE_PTX.replace(".version 8.2", ".version 8.4").replace(
            "sm_90", "sm_80"
        )
        assert normalize_ptx(BASE_PTX) == normalize_ptx(other_version)

    def test_label_renaming_invariant(self):
        """Different label names should not change the fingerprint."""
        relabeled = BASE_PTX.replace("$L__BB0_1", "$L__BB0_9")
        assert normalize_ptx(BASE_PTX) == normalize_ptx(relabeled)

    def test_constant_difference_is_preserved(self):
        """Kernels differing only in a constant value must produce different fingerprints.

        This is the load-bearing invariant: canonicalizing numeric constants away
        would cause incorrect kernel-equivalence merges in beam search deduplication.
        """
        changed_constant = BASE_PTX.replace("mov.u32 %r1, 5;", "mov.u32 %r1, 6;")
        assert normalize_ptx(BASE_PTX) != normalize_ptx(changed_constant)

    def test_register_classes_not_conflated(self):
        """32-bit and 64-bit register classes must remain distinct after normalization."""
        b32 = "mov.u32 %r10, 1;"
        b64 = "mov.u64 %rd10, 1;"
        assert normalize_ptx(b32) != normalize_ptx(b64)


class TestFingerprintKernelDir:
    """Tests for directory-level PTX fingerprinting."""

    def test_missing_directory_returns_none(self, tmp_path):
        """A nonexistent directory should return None."""
        
        assert ptx_hash_from_cache(tmp_path / "does_not_exist") is None

    def test_empty_directory_returns_none(self, tmp_path):
        """An empty directory with no PTX files should return None."""
        
        assert ptx_hash_from_cache(tmp_path) is None

    def test_single_file_returns_fingerprint(self, tmp_path):
        """A directory with one PTX file should return a non-None fingerprint."""
        
        (tmp_path / "kernel.ptx").write_text(BASE_PTX)
        result = ptx_hash_from_cache(tmp_path)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_identical_content_produces_same_fingerprint(self, tmp_path):
        """Two directories with identical PTX content should produce the same fingerprint."""
        
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        (dir_a / "kernel.ptx").write_text(BASE_PTX)
        (dir_b / "kernel.ptx").write_text(BASE_PTX)
        assert ptx_hash_from_cache(dir_a) == ptx_hash_from_cache(dir_b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])