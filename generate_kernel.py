#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KernelAgent主入口脚本
支持通过--level和--problem_id从KernelBench数据集生成Triton内核
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

# 导入KernelAgent组件
from triton_kernel_agent import TritonKernelAgent
from Fuser.auto_agent import AutoKernelRouter
from utils.kernelbench_loader import KernelBenchLoader


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def create_problem_file(problem_info: Dict[str, Any], output_dir: Path) -> Path:
    """
    从KernelBench问题信息创建临时问题文件
    
    Args:
        problem_info: 问题信息字典
        output_dir: 输出目录
        
    Returns:
        创建的问题文件路径
    """
    problem_file = output_dir / f"level{problem_info['level']}_problem{problem_info['problem_id']}.py"
    problem_file.write_text(problem_info['pytorch_code'], encoding='utf-8')
    return problem_file


def save_results(result: Dict[str, Any], output_dir: Path, problem_info: Dict[str, Any]) -> None:
    """
    保存生成结果
    
    Args:
        result: 生成结果
        output_dir: 输出目录
        problem_info: 问题信息
    """
    # 创建结果目录
    result_dir = output_dir / f"level{problem_info['level']}_problem{problem_info['problem_id']}_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存问题信息
    problem_file = result_dir / "problem_info.json"
    with open(problem_file, 'w', encoding='utf-8') as f:
        json.dump(problem_info, f, indent=2, ensure_ascii=False)
    
    # 保存原始PyTorch代码
    pytorch_file = result_dir / "original_pytorch.py"
    pytorch_file.write_text(problem_info['pytorch_code'], encoding='utf-8')
    
    # 保存生成结果
    result_file = result_dir / "generation_result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 如果生成成功，保存内核代码
    if result.get('success') and result.get('kernel_code'):
        kernel_file = result_dir / "generated_kernel.py"
        kernel_file.write_text(result['kernel_code'], encoding='utf-8')
        print(f"✓ 生成的内核已保存到: {kernel_file}")
    
    print(f"✓ 所有结果已保存到: {result_dir}")


def generate_kernel_direct(
    problem_info: Dict[str, Any], 
    args: argparse.Namespace,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    使用KernelAgent直接生成内核
    
    Args:
        problem_info: 问题信息
        args: 命令行参数
        logger: 日志器
        
    Returns:
        生成结果
    """
    logger.info("使用KernelAgent直接路径生成内核...")
    
    # 创建KernelAgent实例
    agent = TritonKernelAgent(
        num_workers=args.workers,
        max_rounds=args.max_rounds,
        model_name=args.model
    )
    
    try:
        # 生成内核
        start_time = time.time()
        result = agent.generate_kernel(
            problem_description=problem_info['pytorch_code'],
            test_code=None
        )
        generation_time = time.time() - start_time
        
        result['generation_time'] = generation_time
        result['method'] = 'kernelagent_direct'
        
        if result['success']:
            logger.info(f"✓ KernelAgent成功生成内核 (耗时: {generation_time:.2f}s)")
            logger.info(f"  工作器 {result['worker_id']} 在 {result['rounds']} 轮中找到解决方案")
        else:
            logger.warning(f"✗ KernelAgent生成失败: {result.get('message', 'Unknown error')}")
        
        return result
        
    finally:
        agent.cleanup()


def generate_kernel_auto_router(
    problem_info: Dict[str, Any], 
    args: argparse.Namespace,
    logger: logging.Logger,
    temp_dir: Path
) -> Dict[str, Any]:
    """
    使用AutoRouter自动选择路径生成内核
    
    Args:
        problem_info: 问题信息
        args: 命令行参数
        logger: 日志器
        temp_dir: 临时目录
        
    Returns:
        生成结果
    """
    logger.info("使用AutoRouter自动路径选择...")
    
    # 创建临时问题文件
    problem_file = create_problem_file(problem_info, temp_dir)
    
    # 创建AutoRouter实例
    router = AutoKernelRouter(
        ka_model=args.model,
        ka_num_workers=args.workers,
        ka_max_rounds=args.max_rounds,
        verify=args.verify,
        allow_fallback=True
    )
    
    # 生成内核
    start_time = time.time()
    result = router.solve(problem_file)
    generation_time = time.time() - start_time
    
    # 构建返回结果
    return_result = {
        'success': result.success,
        'method': 'auto_router',
        'route': result.route,
        'generation_time': generation_time,
        'details': result.details
    }
    
    if result.kernel_code:
        return_result['kernel_code'] = result.kernel_code
    
    if result.success:
        logger.info(f"✓ AutoRouter成功生成内核 (路径: {result.route}, 耗时: {generation_time:.2f}s)")
    else:
        logger.warning(f"✗ AutoRouter生成失败 (路径: {result.route})")
    
    return return_result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="KernelAgent - 从KernelBench数据集生成Triton内核"
    )
    
    # 基本参数
    parser.add_argument("--level", type=int, required=True, help="KernelBench问题级别 (1-4)")
    parser.add_argument("--problem_id", type=int, required=True, help="问题ID")
    parser.add_argument("--output_dir", type=str, default="./kernel_outputs", 
                       help="输出目录 (默认: ./kernel_outputs)")
    
    # 生成方法选择
    parser.add_argument("--method", choices=["direct", "auto"], default="auto",
                       help="生成方法: direct=KernelAgent直接, auto=AutoRouter自动选择 (默认: auto)")
    
    # KernelAgent参数
    parser.add_argument("--model", type=str, default=None,
                       help="LLM模型名称 (默认从环境变量获取)")
    parser.add_argument("--workers", type=int, default=4,
                       help="KernelAgent并行工作器数量 (默认: 4)")
    parser.add_argument("--max_rounds", type=int, default=10,
                       help="KernelAgent最大优化轮数 (默认: 10)")
    parser.add_argument("--verify", action="store_true",
                       help="验证生成的内核")
    
    # 其他参数
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="日志级别 (默认: INFO)")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    # 加载环境变量
    load_dotenv()
    
    # 初始化KernelBench加载器
    try:
        logger.info("初始化KernelBench数据库加载器...")
        loader = KernelBenchLoader()
    except Exception as e:
        logger.error(f"初始化KernelBench加载器失败: {e}")
        return 1
    
    try:
        # 获取问题信息
        try:
            problem_info = loader.get_problem(args.level, args.problem_id)
            logger.info(f"加载问题: Level {args.level} Problem {args.problem_id} - {problem_info['operation_name']}")
            logger.info(f"描述: {problem_info['description']}")
        except ValueError as e:
            logger.error(str(e))
            return 1
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成内核
        result = None
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if args.method == "direct":
                result = generate_kernel_direct(problem_info, args, logger)
            elif args.method == "auto":
                result = generate_kernel_auto_router(problem_info, args, logger, temp_path)
        
        # 保存结果
        if result:
            save_results(result, output_dir, problem_info)
            
            # 打印结果摘要
            print("\n" + "=" * 60)
            print("生成结果摘要")
            print("=" * 60)
            print(f"问题: Level {args.level} Problem {args.problem_id} - {problem_info['operation_name']}")
            print(f"方法: {result.get('method', 'unknown')}")
            if 'route' in result:
                print(f"路由: {result['route']}")
            print(f"成功: {'✓' if result['success'] else '✗'}")
            print(f"耗时: {result.get('generation_time', 0):.2f}s")
            
            if result['success']:
                print("✓ 内核生成成功！")
                return 0
            else:
                print("✗ 内核生成失败")
                return 1
        else:
            logger.error("生成过程中发生未知错误")
            return 1
            
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"发生错误: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        return 1
    finally:
        loader.close()


if __name__ == "__main__":
    sys.exit(main())