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


def test_imports():
    """Test that main modules can be imported."""
    import triton_kernel_agent
    from triton_kernel_agent.agent import TritonKernelAgent
    from triton_kernel_agent.manager import WorkerManager
    from triton_kernel_agent.worker import VerificationWorker
    from triton_kernel_agent.prompt_manager import PromptManager

    assert triton_kernel_agent is not None
    assert TritonKernelAgent is not None
    assert WorkerManager is not None
    assert VerificationWorker is not None
    assert PromptManager is not None
