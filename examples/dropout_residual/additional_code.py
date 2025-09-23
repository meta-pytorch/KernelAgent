def reference_dropout_residual(x, residual, p=0.1, training=True):
    """
    Reference implementation of dropout + residual.

    This is a correct but potentially slow implementation using PyTorch.
    The Triton kernel should produce the same results but run faster.
    """
    import torch
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
