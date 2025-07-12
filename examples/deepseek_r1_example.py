#!/usr/bin/env python3
"""
DeepSeek R1 Usage Examples.

This script demonstrates how to use DeepSeek R1 models to generate Triton kernels.
"""

import os
import sys
import time
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from triton_kernel_agent import TritonKernelAgent
from triton_kernel_agent.model_backends import get_available_backends


def check_backends():
    """Check available model backends."""
    print("Checking available model backends...")
    backends = get_available_backends()
    
    for backend_name, available in backends.items():
        status = "✓ Available" if available else "✗ Not available"
        print(f"  {backend_name}: {status}")
    
    return backends


def example_with_config_file():
    """Example using configuration file."""
    print("\n" + "="*60)
    print("Example 1: Using Configuration File")
    print("="*60)
    
    # Configuration file path
    config_path = project_root / "config_examples" / "deepseek_r1_config.json"
    
    if not config_path.exists():
        print(f"Configuration file does not exist: {config_path}")
        return
    
    try:
        # Create agent
        agent = TritonKernelAgent(backend_config_path=str(config_path))
        
        # Generate kernel
        problem = "Implement an efficient vector addition kernel with broadcasting support"
        
        print(f"Problem description: {problem}")
        print("Starting kernel generation...")
        
        result = agent.generate_kernel(problem_description=problem)
        
        if result["success"]:
            print("✓ Kernel generation successful!")
            print(f"Worker process used: {result['worker_id']}")
            print(f"Optimization rounds: {result['rounds']}")
            print(f"Session directory: {result['session_dir']}")
            print("\nGenerated kernel code:")
            print("-" * 40)
            print(result["kernel_code"])
        else:
            print("✗ Kernel generation failed")
            print(f"Error message: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please check if the model path in the configuration file is correct")


def example_with_direct_config():
    """Example with direct configuration."""
    print("\n" + "="*60)
    print("Example 2: Direct DeepSeek R1 Configuration")
    print("="*60)
    
    # Get model path from environment variable
    model_path = os.getenv("DEEPSEEK_R1_MODEL_PATH")
    
    if not model_path:
        print("Please set DEEPSEEK_R1_MODEL_PATH environment variable")
        print("Example: export DEEPSEEK_R1_MODEL_PATH=/path/to/deepseek-r1-0528.gguf")
        return
    
    if not Path(model_path).exists():
        print(f"Model file does not exist: {model_path}")
        return
    
    # Direct configuration
    config = {
        "backend_type": "deepseek-r1",
        "deepseek-r1": {
            "model_path": model_path,
            "n_ctx": 32768,
            "n_gpu_layers": -1,  # Use all GPU layers
            "verbose": True      # Enable verbose logging
        }
    }
    
    try:
        # Create agent
        agent = TritonKernelAgent(model_backend_config=config)
        
        # Generate kernel
        problem = "Implement a matrix transpose kernel with optimized memory access patterns"
        
        print(f"Problem description: {problem}")
        print("Starting kernel generation...")
        
        result = agent.generate_kernel(problem_description=problem)
        
        if result["success"]:
            print("✓ Kernel generation successful!")
            print(f"Worker process used: {result['worker_id']}")
            print(f"Optimization rounds: {result['rounds']}")
            print(f"Session directory: {result['session_dir']}")
            print("\nGenerated kernel code:")
            print("-" * 40)
            print(result["kernel_code"])
        else:
            print("✗ Kernel generation failed")
            print(f"Error message: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_with_fallback():
    """Example with fallback mechanism."""
    print("\n" + "="*60)
    print("Example 3: Configuration with Fallback Mechanism")
    print("="*60)
    
    # Configure multiple backends, prioritize DeepSeek R1, fallback to OpenAI
    config = {
        "backend_type": "deepseek-r1",  # Prioritize DeepSeek R1
        "deepseek-r1": {
            "model_path": os.getenv("DEEPSEEK_R1_MODEL_PATH", "/nonexistent/path.gguf"),
            "n_ctx": 32768,
            "n_gpu_layers": -1
        },
        "openai": {  # Fallback option
            "model_name": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            "high_reasoning_effort": False
        }
    }
    
    try:
        # Create agent
        agent = TritonKernelAgent(model_backend_config=config)
        
        # Generate kernel
        problem = "Implement a softmax activation function kernel"
        
        print(f"Problem description: {problem}")
        print("Starting kernel generation...")
        
        result = agent.generate_kernel(problem_description=problem)
        
        if result["success"]:
            print("✓ Kernel generation successful!")
            print(f"Worker process used: {result['worker_id']}")
            print(f"Optimization rounds: {result['rounds']}")
            print(f"Session directory: {result['session_dir']}")
            print("\nGenerated kernel code:")
            print("-" * 40)
            print(result["kernel_code"])
        else:
            print("✗ Kernel generation failed")
            print(f"Error message: {result.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_performance_comparison():
    """Performance comparison example."""
    print("\n" + "="*60)
    print("Example 4: Performance Comparison (DeepSeek R1 vs OpenAI)")
    print("="*60)
    
    problem = "Implement an efficient ReLU activation function kernel"
    
    # DeepSeek R1 configuration
    deepseek_config = {
        "backend_type": "deepseek-r1",
        "deepseek-r1": {
            "model_path": os.getenv("DEEPSEEK_R1_MODEL_PATH", "/nonexistent/path.gguf"),
            "n_ctx": 16384,  # Smaller context for faster speed
            "n_gpu_layers": -1
        }
    }
    
    # OpenAI configuration
    openai_config = {
        "backend_type": "openai",
        "openai": {
            "model_name": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            "high_reasoning_effort": False
        }
    }
    
    configs = [
        ("DeepSeek R1", deepseek_config),
        ("OpenAI GPT-4", openai_config)
    ]
    
    for name, config in configs:
        print(f"\nTesting {name}...")
        
        try:
            start_time = time.time()
            agent = TritonKernelAgent(model_backend_config=config, num_workers=2)
            
            result = agent.generate_kernel(problem_description=problem)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result["success"]:
                print(f"✓ {name} successful - Duration: {duration:.2f}s")
                print(f"  Optimization rounds: {result['rounds']}")
            else:
                print(f"✗ {name} failed - Duration: {duration:.2f}s")
                
        except Exception as e:
            print(f"✗ {name} error: {e}")


def main():
    """Main function."""
    print("DeepSeek R1 KernelAgent Usage Examples")
    print("=" * 60)
    
    # Check backend availability
    backends = check_backends()
    
    if not any(backends.values()):
        print("\nWarning: No available model backends!")
        print("Please install necessary dependencies:")
        print("  pip install openai  # OpenAI API support")
        print("  pip install llama-cpp-python  # DeepSeek R1 support")
        return
    
    # Run examples
    try:
        # Example 1: Configuration file
        example_with_config_file()
        
        # Example 2: Direct configuration
        if backends.get('deepseek-r1', False):
            example_with_direct_config()
        else:
            print("\nSkipping DeepSeek R1 direct configuration example (llama-cpp-python not installed)")
        
        # Example 3: Fallback mechanism
        example_with_fallback()
        
        # Example 4: Performance comparison
        if backends.get('deepseek-r1', False) and backends.get('openai', False):
            example_performance_comparison()
        else:
            print("\nSkipping performance comparison example (requires both DeepSeek R1 and OpenAI to be available)")
            
    except KeyboardInterrupt:
        print("\n\nUser interrupted execution")
    except Exception as e:
        print(f"\nError occurred during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
