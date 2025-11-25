import torch

def test_kernel():
    """Test the Triton kernel implementation for ReLU activation."""
    try:
        from kernel import kernel_function
        
        # Sanity check: kernel should be callable and self-contained
        if not callable(kernel_function):
            print("kernel_function is not callable")
            return False

        # Create test data using EXACT specifications from problem description
        # Original problem specified batch_size=4096, dim=393216, and used torch.rand (uniform [0,1))
        # Convert to bfloat16 as per requirements (avoid FP32)
        batch_size = 4096
        dim = 393216
        
        # Generate input tensor with non-zero random values to avoid hiding missing computation
        # Using torch.rand to match the original problem's get_inputs() function
        input_tensor = torch.rand(batch_size, dim, dtype=torch.bfloat16, device='cuda')
        
        # Store original device for comparison
        original_device = input_tensor.device
        
        # Call kernel_function as a normal Python function
        result = kernel_function(input_tensor)
        
        # Verify output is a tensor
        if not isinstance(result, torch.Tensor):
            print(f"Expected torch.Tensor output, got {type(result)}")
            return False
            
        # Verify device consistency (avoid comparing to literal 'cuda')
        if result.device != original_device:
            print(f"Device mismatch: input on {original_device}, output on {result.device}")
            return False
            
        # Verify shape consistency
        if result.shape != input_tensor.shape:
            print(f"Shape mismatch: input {input_tensor.shape}, output {result.shape}")
            return False
            
        # Verify dtype consistency
        if result.dtype != input_tensor.dtype:
            print(f"Dtype mismatch: input {input_tensor.dtype}, output {result.dtype}")
            return False
            
        # Compute expected result using torch.relu for comparison
        # Convert to bfloat16 to match our input dtype
        with torch.no_grad():
            expected = torch.relu(input_tensor)
        
        # Use adjusted tolerances for bfloat16: lower precision requires looser tolerances
        # bfloat16 has ~7-8 bits of mantissa precision vs 23 for float32
        rtol = 1e-2  # 1% relative tolerance
        atol = 2e-2  # 0.02 absolute tolerance
        
        # Check if results match expected values
        if not torch.allclose(result, expected, rtol=rtol, atol=atol):
            print("NUMERICAL MISMATCH:")
            print(f"Input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            print(f"Expected shape: {expected.shape}, dtype: {expected.dtype}")
            print(f"Result shape: {result.shape}, dtype: {result.dtype}")
            
            # Sample some values for debugging
            flat_input = input_tensor.flatten()
            flat_expected = expected.flatten()
            flat_result = result.flatten()
            
            print(f"Input sample (first 10): {flat_input[:10].cpu()}")
            print(f"Expected sample (first 10): {flat_expected[:10].cpu()}")
            print(f"Got sample (first 10): {flat_result[:10].cpu()}")
            
            # Find indices where there are significant differences
            diff_mask = ~torch.isclose(result, expected, rtol=rtol, atol=atol)
            if diff_mask.any():
                diff_indices = torch.nonzero(diff_mask, as_tuple=False)
                print(f"Found {len(diff_indices)} elements with significant differences")
                
                # Show first few problematic elements
                for i in range(min(5, len(diff_indices))):
                    idx = diff_indices[i]
                    print(f"  Element {idx}: input={input_tensor[tuple(idx)]:.6f}, "
                          f"expected={expected[tuple(idx)]:.6f}, got={result[tuple(idx)]:.6f}")
            
            # Compute error metrics
            abs_diff = torch.abs(result - expected)
            max_abs_diff = torch.max(abs_diff)
            mean_abs_diff = torch.mean(abs_diff)
            
            # Handle division by zero for relative error
            rel_error = torch.where(expected != 0, abs_diff / torch.abs(expected), abs_diff)
            max_rel_error = torch.max(rel_error)
            mean_rel_error = torch.mean(rel_error)
            
            print(f"Max absolute difference: {max_abs_diff:.6f}")
            print(f"Mean absolute difference: {mean_abs_diff:.6f}")
            print(f"Max relative error: {max_rel_error:.6f}")
            print(f"Mean relative error: {mean_rel_error:.6f}")
            
            return False
        
        # Additional sanity checks for ReLU behavior
        # Check that all negative inputs produce zero outputs
        negative_mask = input_tensor < 0
        if negative_mask.any():
            negative_outputs = result[negative_mask]
            if not torch.all(negative_outputs == 0):
                print("ReLU violation: negative inputs should produce zero outputs")
                non_zero_count = torch.sum(negative_outputs != 0)
                print(f"Found {non_zero_count} negative inputs with non-zero outputs")
                return False
        
        # Check that non-negative inputs are preserved
        non_negative_mask = input_tensor >= 0
        if non_negative_mask.any():
            input_non_negative = input_tensor[non_negative_mask]
            output_non_negative = result[non_negative_mask]
            if not torch.allclose(input_non_negative, output_non_negative, rtol=rtol, atol=atol):
                print("ReLU violation: non-negative inputs should be preserved")
                return False
        
        print("Test passed: kernel produces correct ReLU results")
        return True
        
    except Exception as e:
        # Surface undefined helper issues from kernel.py clearly
        if isinstance(e, NameError):
            print(f"Test failed: NameError (likely undefined helper in kernel.py): {e}")
            print("This suggests kernel_function references undefined variables/functions")
        elif isinstance(e, ImportError):
            print(f"Test failed: ImportError - cannot import kernel_function: {e}")
        else:
            print(f"Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    success = test_kernel()
    sys.exit(0 if success else 1)