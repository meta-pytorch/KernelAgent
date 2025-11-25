import torch
import torch.nn as nn

def test_kernel():
    """Test the Triton kernel implementation for square matrix multiplication (C = A * B)."""
    try:
        from kernel import kernel_function
        
        # Sanity check: kernel should be callable and self-contained
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        # Create test data using EXACT specifications from problem description
        # Original problem specified N = 2048 * 2 = 4096
        N = 4096
        
        # Create random matrices on CUDA device with bfloat16 dtype
        # Using non-zero random data to ensure computation is actually performed
        A = torch.randn(N, N, dtype=torch.bfloat16, device='cuda')
        B = torch.randn(N, N, dtype=torch.bfloat16, device='cuda')
        
        print(f"Created input matrices: A {A.shape} {A.dtype}, B {B.shape} {B.dtype}")
        print(f"A device: {A.device}, B device: {B.device}")
        
        # Call kernel_function as a normal Python function
        result = kernel_function(A, B)
        
        # Verify output is a tensor
        if not isinstance(result, torch.Tensor):
            print(f"Expected torch.Tensor output, got {type(result)}")
            return False
        
        # Verify output device matches input device
        if result.device != A.device:
            print(f"Output device {result.device} doesn't match input device {A.device}")
            return False
        
        # Verify output shape and dtype
        expected_shape = (N, N)
        if result.shape != expected_shape:
            print(f"Output shape {result.shape} doesn't match expected shape {expected_shape}")
            return False
        
        if result.dtype != torch.bfloat16:
            print(f"Output dtype {result.dtype} doesn't match expected dtype torch.bfloat16")
            return False
        
        # Compute reference result using PyTorch matmul
        # Note: We compute reference in bfloat16 to match the kernel's precision
        with torch.no_grad():
            expected = torch.matmul(A, B)
        
        print(f"Computed reference result: {expected.shape} {expected.dtype}")
        
        # For bfloat16 with large accumulation dimension (4096), use relaxed tolerances
        # Large accumulation can lead to significant rounding errors in lower precision
        rtol = 1e-2  # Relaxed due to bfloat16 precision and large accumulation
        atol = 2e-2  # Relaxed due to bfloat16 precision and large accumulation
        
        # Check if results are numerically close
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            print(f"NUMERICAL MISMATCH:")
            print(f"Input A shape: {A.shape}, dtype: {A.dtype}")
            print(f"Input B shape: {B.shape}, dtype: {B.dtype}")
            print(f"Expected output shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            
            # Sample some values for debugging
            sample_indices = [(0, 0), (0, 1), (1, 0), (N//2, N//2), (-1, -1)]
            
            print("\nSample value comparison:")
            for i, j in sample_indices:
                exp_val = expected[i, j].item()
                res_val = result[i, j].item()
                abs_diff = abs(exp_val - res_val)
                rel_diff = abs_diff / (abs(exp_val) + 1e-8)
                print(f"  [{i},{j}]: Expected {exp_val:.6f}, Got {res_val:.6f}, "
                      f"Abs diff: {abs_diff:.6f}, Rel diff: {rel_diff:.6f}")
            
            # Overall error statistics
            abs_diff = torch.abs(result - expected)
            max_abs_diff = torch.max(abs_diff).item()
            mean_abs_diff = torch.mean(abs_diff).item()
            
            rel_diff = torch.abs((result - expected) / (expected + 1e-8))
            max_rel_diff = torch.max(rel_diff).item()
            mean_rel_diff = torch.mean(rel_diff).item()
            
            print(f"\nError statistics:")
            print(f"Max absolute difference: {max_abs_diff:.6f}")
            print(f"Mean absolute difference: {mean_abs_diff:.6f}")
            print(f"Max relative difference: {max_rel_diff:.6f}")
            print(f"Mean relative difference: {mean_rel_diff:.6f}")
            print(f"Used tolerances: rtol={rtol}, atol={atol}")
            
            return False
        
        print("Test passed! Kernel output matches reference implementation.")
        return True
        
    except NameError as e:
        # Surface undefined helper issues from kernel.py clearly
        print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
        return False
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)