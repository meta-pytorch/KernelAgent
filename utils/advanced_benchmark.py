"""
高级性能基准测试模块
提供更详细的性能分析和优化建议
"""

import torch
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from .performance_benchmark import TritonPerformanceBenchmark

logger = logging.getLogger(__name__)


class AdvancedTritonBenchmark(TritonPerformanceBenchmark):
    """高级Triton性能基准测试器，提供详细分析"""

    def __init__(self, warmup_runs: int = 10, benchmark_runs: int = 100):
        super().__init__(warmup_runs, benchmark_runs)
        self.detailed_results = {}

    def run_comprehensive_analysis(self, kernel_file_path: str, test_file_path: str) -> Dict[str, Any]:
        """
        运行全面的性能分析
        
        Args:
            kernel_file_path: 内核文件路径
            test_file_path: 测试文件路径
            
        Returns:
            详细的性能分析结果
        """
        results = {
            "basic_performance": None,
            "memory_analysis": None,
            "scaling_analysis": None,
            "optimization_suggestions": [],
            "performance_grade": None
        }
        
        try:
            # 1. 基础性能测试
            logger.info("🔍 运行基础性能测试...")
            basic_perf = self.benchmark_kernel_from_file(kernel_file_path, test_file_path)
            results["basic_performance"] = basic_perf
            
            if not basic_perf["success"]:
                return results
            
            # 2. 内存分析
            logger.info("📊 分析内存使用模式...")
            memory_analysis = self._analyze_memory_patterns(kernel_file_path, test_file_path)
            results["memory_analysis"] = memory_analysis
            
            # 3. 扩展性分析
            logger.info("📈 分析性能扩展性...")
            scaling_analysis = self._analyze_scaling_performance(kernel_file_path, test_file_path)
            results["scaling_analysis"] = scaling_analysis
            
            # 4. 生成优化建议
            logger.info("💡 生成优化建议...")
            suggestions = self._generate_optimization_suggestions(basic_perf, memory_analysis, scaling_analysis)
            results["optimization_suggestions"] = suggestions
            
            # 5. 综合评级
            grade = self._calculate_comprehensive_grade(basic_perf, memory_analysis, scaling_analysis)
            results["performance_grade"] = grade
            
            logger.info("✅ 高级性能分析完成")
            
        except Exception as e:
            logger.error(f"高级性能分析失败: {e}")
            results["error"] = str(e)
        
        return results

    def _analyze_memory_patterns(self, kernel_file_path: str, test_file_path: str) -> Dict[str, Any]:
        """分析内存访问模式"""
        try:
            # 加载内核和测试
            kernel_func = self._load_kernel_function(kernel_file_path)
            test_inputs, _ = self._extract_test_info(test_file_path)
            
            if not kernel_func or not test_inputs:
                return {"success": False, "error": "无法加载内核或测试"}
            
            # 测试不同内存布局的性能
            memory_results = {
                "success": True,
                "contiguous_performance": None,
                "strided_performance": None,
                "memory_efficiency": None,
                "bandwidth_utilization": None
            }
            
            # 连续内存测试
            contiguous_input = test_inputs[0].contiguous()
            contiguous_time = self._benchmark_triton(kernel_func, [contiguous_input])
            memory_results["contiguous_performance"] = contiguous_time
            
            # 步长内存测试（如果可能）
            try:
                strided_input = test_inputs[0][::2].contiguous()  # 每隔一个元素
                strided_time = self._benchmark_triton(kernel_func, [strided_input])
                memory_results["strided_performance"] = strided_time
            except:
                memory_results["strided_performance"] = None
            
            # 计算内存效率
            tensor_size_bytes = test_inputs[0].numel() * test_inputs[0].element_size()
            if contiguous_time and contiguous_time > 0:
                # 假设读写各一次
                bandwidth_gb_s = (2 * tensor_size_bytes / (1024**3)) / (contiguous_time / 1000)
                memory_results["bandwidth_utilization"] = bandwidth_gb_s
                
                # 内存效率评分（相对于理论峰值）
                theoretical_bandwidth = 900  # GB/s for modern GPUs
                efficiency = min(bandwidth_gb_s / theoretical_bandwidth * 100, 100)
                memory_results["memory_efficiency"] = efficiency
            
            return memory_results
            
        except Exception as e:
            logger.error(f"内存分析失败: {e}")
            return {"success": False, "error": str(e)}

    def _analyze_scaling_performance(self, kernel_file_path: str, test_file_path: str) -> Dict[str, Any]:
        """分析性能扩展性"""
        try:
            kernel_func = self._load_kernel_function(kernel_file_path)
            if not kernel_func:
                return {"success": False, "error": "无法加载内核"}
            
            scaling_results = {
                "success": True,
                "size_performance": [],
                "scaling_efficiency": None,
                "optimal_size": None
            }
            
            # 测试不同大小的性能
            test_sizes = [1024, 4096, 16384, 65536, 262144]
            
            for size in test_sizes:
                try:
                    # 创建测试输入
                    test_input = torch.randn(size, dtype=torch.float32, device='cuda')
                    
                    # 测试性能
                    triton_time = self._benchmark_triton(kernel_func, [test_input])
                    pytorch_time = self._benchmark_pytorch(lambda x: torch.relu(x), [test_input])
                    
                    if triton_time and pytorch_time:
                        speedup = pytorch_time / triton_time
                        throughput = size / (triton_time / 1000)  # elements per second
                        
                        scaling_results["size_performance"].append({
                            "size": size,
                            "triton_time_ms": triton_time,
                            "pytorch_time_ms": pytorch_time,
                            "speedup": speedup,
                            "throughput_elements_per_sec": throughput
                        })
                except Exception as e:
                    logger.debug(f"跳过大小 {size}: {e}")
                    continue
            
            # 分析扩展效率
            if len(scaling_results["size_performance"]) >= 2:
                perfs = scaling_results["size_performance"]
                
                # 找到最佳性能点
                best_speedup = max(p["speedup"] for p in perfs)
                optimal_size = next(p["size"] for p in perfs if p["speedup"] == best_speedup)
                scaling_results["optimal_size"] = optimal_size
                
                # 计算扩展效率（理想情况下吞吐量应该随大小线性增长）
                throughputs = [p["throughput_elements_per_sec"] for p in perfs]
                sizes = [p["size"] for p in perfs]
                
                # 简单的线性拟合来评估扩展性
                if len(throughputs) >= 2:
                    correlation = np.corrcoef(sizes, throughputs)[0, 1]
                    scaling_results["scaling_efficiency"] = max(0, correlation * 100)
            
            return scaling_results
            
        except Exception as e:
            logger.error(f"扩展性分析失败: {e}")
            return {"success": False, "error": str(e)}

    def _generate_optimization_suggestions(self, basic_perf: Dict, memory_analysis: Dict, scaling_analysis: Dict) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        try:
            # 基于基础性能的建议
            if basic_perf and basic_perf["success"]:
                speedup = basic_perf["speedup"]
                
                if speedup < 0.8:
                    suggestions.append("🔧 内核性能低于PyTorch，考虑优化算法或增加并行度")
                elif speedup < 1.2:
                    suggestions.append("⚡ 性能接近PyTorch，可尝试调整block size或内存访问模式")
                elif speedup > 2.0:
                    suggestions.append("🏆 性能优秀！可考虑进一步优化以达到更高加速比")
            
            # 基于内存分析的建议
            if memory_analysis and memory_analysis["success"]:
                if memory_analysis.get("memory_efficiency"):
                    efficiency = memory_analysis["memory_efficiency"]
                    if efficiency < 30:
                        suggestions.append("💾 内存带宽利用率较低，检查内存访问模式和合并")
                    elif efficiency < 60:
                        suggestions.append("📈 内存效率中等，可优化数据布局或访问模式")
                
                # 比较连续和步长性能
                cont_perf = memory_analysis.get("contiguous_performance")
                stride_perf = memory_analysis.get("strided_performance")
                if cont_perf and stride_perf and stride_perf > cont_perf * 1.5:
                    suggestions.append("🔄 步长访问性能较差，确保数据连续性")
            
            # 基于扩展性分析的建议
            if scaling_analysis and scaling_analysis["success"]:
                scaling_eff = scaling_analysis.get("scaling_efficiency")
                if scaling_eff and scaling_eff < 70:
                    suggestions.append("📊 扩展性较差，检查是否存在性能瓶颈或同步问题")
                
                optimal_size = scaling_analysis.get("optimal_size")
                if optimal_size:
                    suggestions.append(f"🎯 最佳性能出现在大小 {optimal_size}，考虑针对此大小优化")
            
            # 通用建议
            if not suggestions:
                suggestions.append("✨ 性能表现良好，可考虑测试更大的数据集或复杂场景")
            
        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            suggestions.append("❓ 无法生成具体建议，请检查性能数据")
        
        return suggestions

    def _calculate_comprehensive_grade(self, basic_perf: Dict, memory_analysis: Dict, scaling_analysis: Dict) -> Dict[str, Any]:
        """计算综合性能评级"""
        try:
            grade_info = {
                "overall_score": 0,
                "performance_score": 0,
                "memory_score": 0,
                "scaling_score": 0,
                "grade": "未知",
                "details": {}
            }
            
            scores = []
            
            # 基础性能评分 (40%)
            if basic_perf and basic_perf["success"]:
                speedup = basic_perf["speedup"]
                if speedup >= 2.0:
                    perf_score = 100
                elif speedup >= 1.5:
                    perf_score = 85
                elif speedup >= 1.0:
                    perf_score = 70
                elif speedup >= 0.8:
                    perf_score = 55
                else:
                    perf_score = 30
                
                grade_info["performance_score"] = perf_score
                scores.append(("performance", perf_score, 0.4))
            
            # 内存效率评分 (30%)
            if memory_analysis and memory_analysis["success"]:
                mem_eff = memory_analysis.get("memory_efficiency", 50)
                mem_score = min(mem_eff * 2, 100)  # 转换为0-100分
                
                grade_info["memory_score"] = mem_score
                scores.append(("memory", mem_score, 0.3))
            
            # 扩展性评分 (30%)
            if scaling_analysis and scaling_analysis["success"]:
                scale_eff = scaling_analysis.get("scaling_efficiency", 50)
                scale_score = min(scale_eff, 100)
                
                grade_info["scaling_score"] = scale_score
                scores.append(("scaling", scale_score, 0.3))
            
            # 计算加权总分
            if scores:
                total_weight = sum(weight for _, _, weight in scores)
                weighted_sum = sum(score * weight for _, score, weight in scores)
                overall_score = weighted_sum / total_weight
                
                grade_info["overall_score"] = overall_score
                
                # 确定等级
                if overall_score >= 90:
                    grade_info["grade"] = "优秀 🏆"
                elif overall_score >= 80:
                    grade_info["grade"] = "良好 ✅"
                elif overall_score >= 70:
                    grade_info["grade"] = "中等 ⚡"
                elif overall_score >= 60:
                    grade_info["grade"] = "一般 ⚠️"
                else:
                    grade_info["grade"] = "需优化 ❌"
                
                grade_info["details"] = {
                    "performance_weight": "40%",
                    "memory_weight": "30%", 
                    "scaling_weight": "30%",
                    "total_components": len(scores)
                }
            
            return grade_info
            
        except Exception as e:
            logger.error(f"计算评级失败: {e}")
            return {"overall_score": 0, "grade": "错误 ❌", "error": str(e)}

    def generate_performance_report(self, results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """生成详细的性能报告"""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("🚀 TRITON 内核性能分析报告")
            report_lines.append("=" * 80)
            
            # 基础性能
            if results.get("basic_performance"):
                basic = results["basic_performance"]
                if basic["success"]:
                    report_lines.append("\n📊 基础性能指标:")
                    report_lines.append(f"   PyTorch时间: {basic['pytorch_time_ms']:.3f}ms")
                    report_lines.append(f"   Triton时间:  {basic['triton_time_ms']:.3f}ms")
                    report_lines.append(f"   加速比:      {basic['speedup']:.2f}x")
                    # if basic.get("memory_usage_mb"):
                    #     report_lines.append(f"   内存使用:    {basic['memory_usage_mb']:.1f}MB")
            
            # 内存分析
            if results.get("memory_analysis"):
                mem = results["memory_analysis"]
                if mem["success"]:
                    report_lines.append("\n💾 内存分析:")
                    if mem.get("memory_efficiency"):
                        report_lines.append(f"   内存效率:    {mem['memory_efficiency']:.1f}%")
                    if mem.get("bandwidth_utilization"):
                        report_lines.append(f"   带宽利用:    {mem['bandwidth_utilization']:.1f} GB/s")
                    if mem.get("contiguous_performance") and mem.get("strided_performance"):
                        ratio = mem["strided_performance"] / mem["contiguous_performance"]
                        report_lines.append(f"   步长性能比:  {ratio:.2f}x")
            
            # 扩展性分析
            if results.get("scaling_analysis"):
                scale = results["scaling_analysis"]
                if scale["success"]:
                    report_lines.append("\n📈 扩展性分析:")
                    if scale.get("scaling_efficiency"):
                        report_lines.append(f"   扩展效率:    {scale['scaling_efficiency']:.1f}%")
                    if scale.get("optimal_size"):
                        report_lines.append(f"   最佳大小:    {scale['optimal_size']} 元素")
                    
                    # 性能数据表格
                    if scale.get("size_performance"):
                        report_lines.append("\n   性能数据:")
                        report_lines.append("   大小      Triton(ms)  PyTorch(ms)  加速比")
                        report_lines.append("   " + "-" * 45)
                        for perf in scale["size_performance"]:
                            report_lines.append(
                                f"   {perf['size']:8d}  {perf['triton_time_ms']:8.3f}  "
                                f"{perf['pytorch_time_ms']:9.3f}  {perf['speedup']:6.2f}x"
                            )
            
            # 综合评级
            if results.get("performance_grade"):
                grade = results["performance_grade"]
                report_lines.append("\n🏆 综合评级:")
                report_lines.append(f"   总体评分:    {grade['overall_score']:.1f}/100")
                report_lines.append(f"   性能评分:    {grade['performance_score']:.1f}/100")
                report_lines.append(f"   内存评分:    {grade['memory_score']:.1f}/100")
                report_lines.append(f"   扩展评分:    {grade['scaling_score']:.1f}/100")
                report_lines.append(f"   最终等级:    {grade['grade']}")
            
            # 优化建议
            if results.get("optimization_suggestions"):
                report_lines.append("\n💡 优化建议:")
                for i, suggestion in enumerate(results["optimization_suggestions"], 1):
                    report_lines.append(f"   {i}. {suggestion}")
            
            report_lines.append("\n" + "=" * 80)
            
            report_text = "\n".join(report_lines)
            
            # 保存报告
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                logger.info(f"性能报告已保存到: {output_path}")
            
            return report_text
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return f"报告生成失败: {e}"


def run_advanced_benchmark(session_dir: str) -> Optional[Dict[str, Any]]:
    """
    运行高级性能基准测试
    
    Args:
        session_dir: 会话目录路径
        
    Returns:
        高级性能分析结果
    """
    try:
        session_path = Path(session_dir)
        kernel_file = session_path / "final_kernel.py"
        test_file = session_path / "test.py"
        
        if not kernel_file.exists() or not test_file.exists():
            logger.warning("缺少必要的文件进行高级分析")
            return None
        
        # 创建高级基准测试器
        benchmark = AdvancedTritonBenchmark(warmup_runs=5, benchmark_runs=30)
        
        # 运行全面分析
        results = benchmark.run_comprehensive_analysis(str(kernel_file), str(test_file))
        
        # 生成报告
        report = benchmark.generate_performance_report(results)
        
        # 保存结果和报告
        if results:
            with open(session_path / "advanced_performance.json", 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            with open(session_path / "performance_report.txt", 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info("高级性能分析完成")
            logger.info("\n" + report)
        
        return results
        
    except Exception as e:
        logger.error(f"高级性能基准测试失败: {e}")
        return None