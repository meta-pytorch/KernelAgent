#!/usr/bin/env python3
"""
Test script for loading DeepSeek R1 models with llama-cpp-python.

This script demonstrates how to:
1. Install necessary dependencies
2. Download DeepSeek R1 models
3. Load models using llama-cpp-python
4. Perform basic inference tests
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check if necessary dependencies are installed."""
    print("Checking dependencies...")
    
    try:
        import llama_cpp
        print("✓ llama-cpp-python is installed")
        version = getattr(llama_cpp, '__version__', 'unknown')
        print(f"  Version: {version}")
        return True
    except ImportError:
        print("✗ llama-cpp-python is not installed")
        print("\nInstallation methods:")
        print("# CPU version")
        print("pip install llama-cpp-python")
        print("\n# GPU version (CUDA)")
        print("CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\" pip install llama-cpp-python")
        print("\n# GPU version (Metal - macOS)")
        print("CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python")
        return False


def show_download_guide():
    """Display model download guide."""
    print("\n" + "="*60)
    print("DeepSeek R1 Model Download Guide")
    print("="*60)
    
    print("\n1. Install huggingface-hub:")
    print("pip install huggingface-hub")
    
    print("\n2. Download models (choose one version):")
    print("# 4-bit quantized version (recommended, ~1GB)")
    print("huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-GGUF \\")
    print("  deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf --local-dir ./models/")
    
    print("\n# 7B model 4-bit quantized version (~4.5GB)")
    print("huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B-GGUF \\")
    print("  deepseek-r1-distill-qwen-7b-q4_k_m.gguf --local-dir ./models/")
    
    print("\n# 14B model 4-bit quantized version (~8GB)")
    print("huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF \\")
    print("  deepseek-r1-distill-qwen-14b-q4_k_m.gguf --local-dir ./models/")
    
    print("\n3. Set environment variable:")
    print("export DEEPSEEK_R1_MODEL_PATH=/path/to/your/model.gguf")


def test_direct_llama_cpp():
    """Test direct llama-cpp-python usage."""
    print("\n" + "="*60)
    print("Testing Direct llama-cpp-python Usage")
    print("="*60)
    
    model_path = os.getenv("DEEPSEEK_R1_MODEL_PATH")
    if not model_path:
        print("Please set DEEPSEEK_R1_MODEL_PATH environment variable")
        print("Example: export DEEPSEEK_R1_MODEL_PATH=/path/to/deepseek-r1-model.gguf")
        return False
    
    if not Path(model_path).exists():
        print(f"Model file does not exist: {model_path}")
        return False
    
    try:
        from llama_cpp import Llama
        
        print(f"Loading model: {model_path}")
        print("This may take a few minutes...")
        
        # Initialize model
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,        # Context window size
            n_gpu_layers=-1,   # Use all GPU layers (if GPU available)
            verbose=False      # Disable verbose logging
        )
        
        print("✓ Model loaded successfully!")
        
        # Test inference
        prompt = "Implement a simple matrix multiplication function in Python:"
        
        print(f"\nTest prompt: {prompt}")
        print("Generating response...")
        
        # Format prompt (DeepSeek R1 format)
        formatted_prompt = f"<｜begin▁of▁sentence｜>User: {prompt}\n\nAssistant: "
        
        response = llm(
            formatted_prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["<｜end▁of▁sentence｜>", "User:", "\n\nUser:"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        print(f"\nModel response:\n{answer}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_backend_integration():
    """Test KernelAgent backend integration."""
    print("\n" + "="*60)
    print("Testing KernelAgent Backend Integration")
    print("="*60)
    
    try:
        from triton_kernel_agent.model_backends import DeepSeekR1Backend, get_available_backends
        
        # Check backend availability
        backends = get_available_backends()
        if not backends.get('deepseek-r1', False):
            print("✗ DeepSeek R1 backend not available (llama-cpp-python not installed)")
            return False
        
        print("✓ DeepSeek R1 backend is available")
        
        model_path = os.getenv("DEEPSEEK_R1_MODEL_PATH")
        if not model_path or not Path(model_path).exists():
            print("Please set correct DEEPSEEK_R1_MODEL_PATH environment variable")
            return False
        
        # Create backend instance
        print(f"Initializing DeepSeek R1 backend: {model_path}")
        backend = DeepSeekR1Backend(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,
            verbose=False
        )
        
        if not backend.is_available():
            print("✗ Backend not available")
            return False
        
        print("✓ Backend initialized successfully")
        
        # Test single completion
        prompt = "Implement a Triton kernel for vector addition"
        print(f"\nTest prompt: {prompt}")
        
        response = backend.generate_completion(
            prompt,
            max_tokens=512,
            temperature=0.7
        )
        
        print(f"\nGenerated response:\n{response}")
        
        # Test multiple completions
        print("\nTesting multiple completions generation...")
        responses = backend.generate_multiple_completions(
            prompt,
            n=2,
            max_tokens=256,
            temperature=0.8
        )
        
        for i, resp in enumerate(responses):
            print(f"\nCompletion {i+1}:\n{resp[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_kernel_generation():
    """Test complete kernel generation workflow."""
    print("\n" + "="*60)
    print("Testing Complete Kernel Generation Workflow")
    print("="*60)
    
    try:
        from triton_kernel_agent import TritonKernelAgent
        
        model_path = os.getenv("DEEPSEEK_R1_MODEL_PATH")
        if not model_path or not Path(model_path).exists():
            print("Please set correct DEEPSEEK_R1_MODEL_PATH environment variable")
            return False
        
        # Configure DeepSeek R1 backend
        config = {
            "backend_type": "deepseek-r1",
            "deepseek-r1": {
                "model_path": model_path,
                "n_ctx": 8192,
                "n_gpu_layers": -1,
                "verbose": False
            }
        }
        
        print("Creating TritonKernelAgent...")
        agent = TritonKernelAgent(
            model_backend_config=config,
            num_workers=2,  # Reduce worker count to save resources
            max_rounds=3    # Reduce optimization rounds to save time
        )
        
        # Test kernel generation
        problem = "Implement a simple vector addition kernel that adds two vectors"
        
        print(f"Problem description: {problem}")
        print("Starting kernel generation... (this may take several minutes)")
        
        result = agent.generate_kernel(problem_description=problem)
        
        if result["success"]:
            print("✓ Kernel generation successful!")
            print(f"Worker process used: {result['worker_id']}")
            print(f"Optimization rounds: {result['rounds']}")
            print(f"Session directory: {result['session_dir']}")
            print("\nGenerated kernel code:")
            print("-" * 40)
            code_preview = result["kernel_code"]
            if len(code_preview) > 1000:
                print(code_preview[:1000] + "...")
            else:
                print(code_preview)
        else:
            print("✗ Kernel generation failed")
            print(f"Error message: {result.get('message', 'Unknown error')}")
        
        return result["success"]
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    print("DeepSeek R1 + llama-cpp-python Test Script")
    print("=" * 60)
    
    # 1. Check dependencies
    if not check_dependencies():
        return
    
    # 2. Show download guide
    show_download_guide()
    
    # 3. Test direct llama-cpp-python usage
    print("\nPress Enter to continue testing direct llama-cpp-python usage...")
    input()
    
    if not test_direct_llama_cpp():
        print("Direct test failed, please check model path and dependencies")
        return
    
    # 4. Test backend integration
    print("\nPress Enter to continue testing backend integration...")
    input()
    
    if not test_backend_integration():
        print("Backend integration test failed")
        return
    
    # 5. Test complete workflow (optional)
    print("\nTest complete kernel generation workflow? (this may take a long time) [y/N]: ", end="")
    if input().lower().startswith('y'):
        test_kernel_generation()
    
    print("\nAll tests completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUser interrupted test")
    except Exception as e:
        print(f"\nError occurred during testing: {e}")
        import traceback
        traceback.print_exc()
