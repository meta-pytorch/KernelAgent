Create a Triton kernel for fused dropout + residual addition.

The kernel should:
1. Apply dropout to input tensor with probability p
2. Add the result to a residual tensor
3. Return the final output

Input tensors shape: (batch_size, seq_len, hidden_dim)
All tensors are float32 on CUDA device.
