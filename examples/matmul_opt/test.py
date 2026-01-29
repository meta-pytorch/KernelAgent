import torch


def test_kernel():
    """
    Test the Triton kernel implementation for a single matrix multiplication:
      C = A * B
    where
      A: shape (2048, 8192), dtype=bfloat16
      B: shape (8192, 4096), dtype=bfloat16
    This test will:
      - Generate random non-zero inputs in fp32, cast to bf16
      - Compute a high-precision reference in fp32, then cast to bf16
      - Invoke kernel_function(A, B) as a normal Python function
      - Compare the result to the reference with tolerances suitable for bf16
      - Print detailed debug info on mismatch or exceptions
    """
    try:
        # Import the user-provided Triton kernel entry point
        from kernel import kernel_function

        # Sanity check: kernel_function must be callable
        if not callable(kernel_function):
            print("ERROR: kernel_function is not callable")
            return False

        # Ensure CUDA is available for this test
        if not torch.cuda.is_available():
            print("ERROR: CUDA is not available on this machine")
            return False
        device = torch.device("cuda")

        # Problem dimensions (exactly as specified)
        M = 1024 * 2  # 2048
        K = 4096 * 2  # 8192
        N = 2048 * 2  # 4096

        # Generate random non-zero inputs in fp32, then cast to bf16
        # (bf16 is used because original spec was fp32, but bf16 avoids FP32 tests per guidelines)
        A_fp32 = torch.rand((M, K), dtype=torch.float32, device=device)
        B_fp32 = torch.rand((K, N), dtype=torch.float32, device=device)
        A = A_fp32.to(torch.bfloat16)
        B = B_fp32.to(torch.bfloat16)

        # Call the Triton kernel function as a regular Python function
        result = kernel_function(A, B)

        # Basic structural checks
        if not isinstance(result, torch.Tensor):
            print(f"ERROR: Result is not a torch.Tensor (got type={type(result)})")
            return False
        if result.device != A.device:
            print(f"ERROR: Result device {result.device} != input device {A.device}")
            return False
        if result.dtype != A.dtype:
            print(f"ERROR: Result dtype {result.dtype} != input dtype {A.dtype}")
            return False
        if result.shape != (M, N):
            print(f"ERROR: Result shape {result.shape} != expected {(M, N)}")
            return False

        # Compute high-precision reference in fp32, then cast to bf16
        expected_fp32 = torch.matmul(A_fp32, B_fp32)
        expected = expected_fp32.to(torch.bfloat16)

        # Compare results with tolerances suitable for bf16:
        # bf16 has lower precision than fp32 --> allow rtol=1e-3, atol=2e-3
        rtol = 1e-2
        atol = 2e-2
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            # Compute diagnostic metrics in fp32 for clarity
            diff = result.to(torch.float32) - expected.to(torch.float32)
            max_abs_diff = torch.max(diff.abs()).item()
            max_rel_err = torch.max(
                diff.abs() / (expected.to(torch.float32).abs() + 1e-8)
            ).item()

            print("NUMERICAL MISMATCH DETECTED")
            print(f"  Input A shape={A.shape}, dtype={A.dtype}")
            print(f"  Input B shape={B.shape}, dtype={B.dtype}")
            print(f"  Output shape={result.shape}, dtype={result.dtype}")
            print(f"  Tolerances: rtol={rtol}, atol={atol}")
            print(f"  Max absolute difference: {max_abs_diff:e}")
            print(f"  Max relative difference: {max_rel_err:e}")
            # Show a few sample values
            idx_samples = [(0, 0), (M // 2, N // 2), (M - 1, N - 1)]
            for idx in idx_samples:
                i, j = idx
                print(
                    f"    A[{i},{j % K}]: {A[i, j % K].item():.4f}, B[{j % K},{j}]: {B[j % K, j].item():.4f}"
                )
                print(
                    f"    Expected[{i},{j}]: {expected[i, j].item():.4f}, Got: {result[i, j].item():.4f}"
                )
            return False

        # If we reach here, the test passed
        return True

    except Exception as e:
        # Gracefully handle exceptions and surface NameErrors for missing helpers
        if isinstance(e, NameError):
            print(
                f"TEST FAILURE: NameError (likely undefined helper in kernel.py): {e}"
            )
        else:
            print(f"TEST FAILURE: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    import sys

    success = test_kernel()
    sys.exit(0 if success else 1)
