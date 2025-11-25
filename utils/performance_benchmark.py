"""
Triton内核性能基准测试模块
专门用于验证成功的算子进行性能测试
"""

import torch
import time
import logging
import tempfile
import os
import sys
import importlib.util
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class TritonPerformanceBenchmark:
    """Triton内核性能基准测试器"""

    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 50):
        """
        初始化性能基准测试器
        
        Args:
            warmup_runs: 预热运行次数
            benchmark_runs: 基准测试运行次数
        """
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            # 记录原始TF32设置
            self.original_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            self.original_tf32_cudnn = torch.backends.cudnn.allow_tf32

    def benchmark_kernel_from_file(self, kernel_file_path: str, test_file_path: str) -> Dict[str, Any]:
        """
        从文件路径加载并测试内核性能
        
        Args:
            kernel_file_path: 内核文件路径 (kernel.py)
            test_file_path: 测试文件路径 (test.py)
            
        Returns:
            性能测试结果
        """
        results = {
            "success": False,
            "error": None,
            "pytorch_time_ms": None,
            "triton_time_ms": None,
            "speedup": None,
            "memory_usage_mb": None
        }
        
        try:
            # 1. 加载内核函数
            kernel_func = self._load_kernel_function(kernel_file_path)
            if kernel_func is None:
                results["error"] = "无法加载内核函数"
                return results
            
            # 2. 从测试文件中提取测试输入
            test_inputs, pytorch_reference = self._extract_test_info(test_file_path)
            if test_inputs is None:
                results["error"] = "无法提取测试输入"
                return results
            
            # 3. 运行性能基准测试
            perf_results = self._run_benchmark(kernel_func, pytorch_reference, test_inputs)
            results.update(perf_results)
            
            if results["success"]:
                logger.info(f"🚀 性能测试完成")
                logger.info(f"   PyTorch: {results['pytorch_time_ms']:.3f}ms")
                logger.info(f"   Triton:  {results['triton_time_ms']:.3f}ms")
                logger.info(f"   加速比:  {results['speedup']:.2f}x")
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"性能测试失败: {e}")
        
        return results

    def _load_kernel_function(self, kernel_file_path: str) -> Optional[Any]:
        """
        从kernel.py文件加载kernel_function
        
        Args:
            kernel_file_path: kernel.py文件路径
            
        Returns:
            kernel_function函数对象
        """
        try:
            # 读取内核代码
            with open(kernel_file_path, 'r', encoding='utf-8') as f:
                kernel_code = f.read()
            
            # 创建临时模块
            spec = importlib.util.spec_from_file_location("kernel_module", kernel_file_path)
            module = importlib.util.module_from_spec(spec)
            
            # 执行模块
            spec.loader.exec_module(module)
            
            # 查找kernel_function
            if hasattr(module, 'kernel_function'):
                return module.kernel_function
            else:
                logger.error("kernel.py中未找到kernel_function函数")
                return None
                
        except Exception as e:
            logger.error(f"加载内核函数失败: {e}")
            return None

    def _extract_test_info(self, test_file_path: str) -> Tuple[Optional[List[torch.Tensor]], Optional[Any]]:
        """
        从test.py文件中提取测试输入和PyTorch参考实现
        
        Args:
            test_file_path: test.py文件路径
            
        Returns:
            (测试输入列表, PyTorch参考函数)
        """
        try:
            # 读取测试代码
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_code = f.read()
            
            # 执行测试代码获取输入
            namespace = {}
            exec(test_code, namespace)
            
            # 查找测试输入创建逻辑
            test_inputs = self._create_test_inputs_from_code(test_code)
            pytorch_ref = self._create_pytorch_reference_from_code(test_code, test_inputs)
            
            return test_inputs, pytorch_ref
            
        except Exception as e:
            logger.error(f"提取测试信息失败: {e}")
            return None, None

    def _create_test_inputs_from_code(self, test_code: str) -> Optional[List[torch.Tensor]]:
        """
        从测试代码中创建测试输入
        
        Args:
            test_code: 测试代码字符串
            
        Returns:
            测试输入张量列表
        """
        try:
            # 简单的模式匹配来提取张量创建
            import re
            
            # 查找torch.randn, torch.rand等调用，包括更复杂的参数模式
            tensor_patterns = [
                r'torch\.randn\(([^)]+)\)',
                r'torch\.rand\(([^)]+)\)',
                r'torch\.zeros\(([^)]+)\)',
                r'torch\.ones\(([^)]+)\)',
                r'torch\.tensor\(([^)]+)\)'
            ]
            
            test_inputs = []
            detected_dtype = torch.float32  # 默认类型
            detected_device = self.device
            
            # 首先检测测试代码中使用的数据类型
            if 'torch.bfloat16' in test_code or 'dtype=torch.bfloat16' in test_code:
                detected_dtype = torch.bfloat16
            elif 'torch.float16' in test_code or 'dtype=torch.float16' in test_code:
                detected_dtype = torch.float16
            elif 'torch.float32' in test_code or 'dtype=torch.float32' in test_code:
                detected_dtype = torch.float32
            
            logger.info(f"检测到测试代码中的数据类型: {detected_dtype}")
            
            for pattern in tensor_patterns:
                matches = re.findall(pattern, test_code)
                for match in matches:
                    try:
                        # 解析参数 - 更智能的解析
                        args_str = match.strip()
                        
                        # 提取形状参数
                        shape_args = []
                        dtype = detected_dtype  # 使用检测到的类型
                        device = detected_device
                        
                        # 分割参数，但要处理嵌套的情况
                        args = []
                        paren_count = 0
                        current_arg = ""
                        
                        for char in args_str + ",":
                            if char == "," and paren_count == 0:
                                if current_arg.strip():
                                    args.append(current_arg.strip())
                                current_arg = ""
                            else:
                                if char in "([":
                                    paren_count += 1
                                elif char in ")]":
                                    paren_count -= 1
                                current_arg += char
                        
                        for arg in args:
                            arg = arg.strip()
                            
                            # 检查数据类型
                            if 'dtype=' in arg:
                                if 'bfloat16' in arg:
                                    dtype = torch.bfloat16
                                elif 'float16' in arg:
                                    dtype = torch.float16
                                elif 'float32' in arg:
                                    dtype = torch.float32
                            
                            # 检查设备
                            elif 'device=' in arg:
                                if 'cuda' in arg:
                                    device = 'cuda'
                                elif 'cpu' in arg:
                                    device = 'cpu'
                            
                            # 提取数字形状参数
                            elif arg.isdigit():
                                shape_args.append(int(arg))
                            
                            # 处理变量形状（如 N = 1024）
                            elif arg in ['N'] and 'N = 1024' in test_code:
                                shape_args.append(1024)
                        
                        # 如果没有找到形状，尝试从常见模式中提取
                        if not shape_args:
                            # 查找 N = 数字 的模式
                            n_match = re.search(r'N\s*=\s*(\d+)', test_code)
                            if n_match:
                                shape_args = [int(n_match.group(1))]
                            else:
                                # 默认形状
                                shape_args = [1024]
                        
                        if shape_args:
                            if device == 'cuda' and torch.cuda.is_available():
                                tensor = torch.randn(shape_args, dtype=dtype, device='cuda')
                            else:
                                tensor = torch.randn(shape_args, dtype=dtype)
                            test_inputs.append(tensor)
                            logger.info(f"创建测试张量: shape={shape_args}, dtype={dtype}, device={device}")
                            
                    except Exception as e:
                        logger.debug(f"解析张量参数失败: {e}")
                        continue
            
            # 如果没有找到，创建默认输入（使用检测到的类型）
            if not test_inputs:
                logger.info(f"未找到张量创建模式，使用默认输入: dtype={detected_dtype}")
                if self.device == 'cuda':
                    test_inputs = [torch.randn(1024, dtype=detected_dtype, device='cuda')]
                else:
                    test_inputs = [torch.randn(1024, dtype=detected_dtype)]
            
            return test_inputs
            
        except Exception as e:
            logger.error(f"创建测试输入失败: {e}")
            return None

    def _create_pytorch_reference_from_code(self, test_code: str, test_inputs: List[torch.Tensor]) -> Optional[Any]:
        """
        根据测试代码推断PyTorch参考实现
        
        Args:
            test_code: 测试代码
            test_inputs: 测试输入
            
        Returns:
            PyTorch参考函数
        """
        try:
            # 根据测试代码中的操作推断
            code_lower = test_code.lower()
            
            if 'relu' in code_lower:
                def pytorch_relu(*inputs):
                    # 确保输入和输出类型一致
                    input_tensor = inputs[0]
                    result = torch.relu(input_tensor)
                    # 确保结果与输入有相同的dtype和device
                    return result.to(dtype=input_tensor.dtype, device=input_tensor.device)
                return pytorch_relu
                
            elif 'softmax' in code_lower:
                def pytorch_softmax(*inputs):
                    input_tensor = inputs[0]
                    result = torch.softmax(input_tensor, dim=-1)
                    return result.to(dtype=input_tensor.dtype, device=input_tensor.device)
                return pytorch_softmax
                
            elif 'sigmoid' in code_lower:
                def pytorch_sigmoid(*inputs):
                    input_tensor = inputs[0]
                    result = torch.sigmoid(input_tensor)
                    return result.to(dtype=input_tensor.dtype, device=input_tensor.device)
                return pytorch_sigmoid
                
            elif 'add' in code_lower and len(test_inputs) >= 2:
                def pytorch_add(*inputs):
                    result = torch.add(inputs[0], inputs[1])
                    return result.to(dtype=inputs[0].dtype, device=inputs[0].device)
                return pytorch_add
                
            elif 'matmul' in code_lower and len(test_inputs) >= 2:
                def pytorch_matmul(*inputs):
                    result = torch.matmul(inputs[0], inputs[1])
                    return result.to(dtype=inputs[0].dtype, device=inputs[0].device)
                return pytorch_matmul
            
            else:
                # 默认返回输入（用于测试）
                def pytorch_identity(*inputs):
                    return inputs[0].clone()
                return pytorch_identity
                
        except Exception as e:
            logger.error(f"创建PyTorch参考失败: {e}")
            return None

    def _run_benchmark(self, triton_func: Any, pytorch_func: Any, test_inputs: List[torch.Tensor]) -> Dict[str, Any]:
        """
        运行性能基准测试
        
        Args:
            triton_func: Triton内核函数
            pytorch_func: PyTorch参考函数
            test_inputs: 测试输入
            
        Returns:
            性能测试结果
        """
        results = {
            "success": False,
            "pytorch_time_ms": None,
            "triton_time_ms": None,
            "speedup": None,
            "memory_usage_mb": None
        }
        
        try:
            # 测试PyTorch性能
            pytorch_time = self._benchmark_pytorch(pytorch_func, test_inputs)
            
            # 测试Triton性能
            triton_time = self._benchmark_triton(triton_func, test_inputs)
            
            if pytorch_time is not None and triton_time is not None:
                speedup = pytorch_time / triton_time if triton_time > 0 else 0
                
                results.update({
                    "success": True,
                    "pytorch_time_ms": pytorch_time,
                    "triton_time_ms": triton_time,
                    "speedup": speedup
                })
                
                # # 计算内存使用
                # if self.device == 'cuda':
                #     memory_mb = sum(t.numel() * t.element_size() for t in test_inputs) / (1024 * 1024)
                #     results["memory_usage_mb"] = memory_mb
            
        except Exception as e:
            results["error"] = str(e)
            logger.error(f"基准测试执行失败: {e}")
        
        return results

    def _benchmark_pytorch(self, pytorch_func: Any, test_inputs: List[torch.Tensor]) -> Optional[float]:
        """PyTorch性能测试"""
        try:
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 预热
            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    pytorch_func(*test_inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 基准测试
            start_time = time.perf_counter()
            for _ in range(self.benchmark_runs):
                with torch.no_grad():
                    pytorch_func(*test_inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / self.benchmark_runs * 1000
            return avg_time
            
        except Exception as e:
            logger.error(f"PyTorch基准测试失败: {e}")
            return None

    def _benchmark_triton(self, triton_func: Any, test_inputs: List[torch.Tensor]) -> Optional[float]:
        """Triton性能测试"""
        try:
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 预热
            for _ in range(self.warmup_runs):
                triton_func(*test_inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 基准测试
            start_time = time.perf_counter()
            for _ in range(self.benchmark_runs):
                triton_func(*test_inputs)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / self.benchmark_runs * 1000
            return avg_time
            
        except Exception as e:
            logger.error(f"Triton基准测试失败: {e}")
            return None

    def format_performance_summary(self, results: Dict[str, Any]) -> str:
        """
        格式化性能测试结果摘要
        
        Args:
            results: 性能测试结果
            
        Returns:
            格式化的摘要字符串
        """
        if not results["success"]:
            return f"❌ 性能测试失败: {results.get('error', 'Unknown error')}"
        
        pytorch_time = results["pytorch_time_ms"]
        triton_time = results["triton_time_ms"]
        speedup = results["speedup"]
        
        
        summary = f"""🚀 性能基准测试结果:
   PyTorch: {pytorch_time:.3f}ms
   Triton:  {triton_time:.3f}ms
   加速比:  {speedup:.2f}x
   评级:    {grade}"""
        
        if results.get("memory_usage_mb"):
            summary += f"\n   内存:    {results['memory_usage_mb']:.1f}MB"
        
        return summary


def benchmark_successful_kernel(session_dir: str) -> Optional[Dict[str, Any]]:
    """
    对成功验证的内核进行性能基准测试
    
    Args:
        session_dir: 会话目录路径，包含kernel.py和test.py
        
    Returns:
        性能测试结果，失败返回None
    """
    try:
        session_path = Path(session_dir)
        kernel_file = session_path / "final_kernel.py"
        test_file = session_path / "test.py"
        
        # 检查文件是否存在
        if not kernel_file.exists():
            logger.warning(f"内核文件不存在: {kernel_file}")
            return None
            
        if not test_file.exists():
            logger.warning(f"测试文件不存在: {test_file}")
            return None
        
        # 创建基准测试器
        benchmark = TritonPerformanceBenchmark(warmup_runs=3, benchmark_runs=20)
        
        # 运行性能测试
        results = benchmark.benchmark_kernel_from_file(str(kernel_file), str(test_file))
        
        # 保存结果
        if results["success"]:
            perf_file = session_path / "performance_results.json"
            with open(perf_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        logger.error(f"性能基准测试失败: {e}")
        return None