# Copyright (c) Meta Platforms, Inc. and affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def reference_dropout_residual(x, residual, p=0.1, training=True):
    """
    Reference implementation of dropout + residual.

    This is a correct but potentially slow implementation using PyTorch.
    The Triton kernel should produce the same results but run faster.
    """
    import torch.nn.functional as F

    if training:
        # Apply dropout
        dropout_out = F.dropout(x, p=p, training=True)
        # Add residual
        output = dropout_out + residual
    else:
        # No dropout during inference
        output = x + residual

    return output
