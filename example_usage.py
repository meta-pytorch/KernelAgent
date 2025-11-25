#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KernelAgent使用示例
演示如何使用generate_kernel.py生成内核
"""

import subprocess
import sys
from pathlib import Path


def run_example():
    """运行示例"""
    print("KernelAgent + KernelBench 使用示例")
    print("=" * 50)
    
    # 示例1: 生成简单的ReLU内核
    print("\n示例1: 生成ReLU内核 (Level 1, Problem 19)")
    print("-" * 40)
    cmd = [
        sys.executable, "generate_kernel.py",
        "--level", "1",
        "--problem_id", "19", 
        "--method", "direct",
        "--workers", "2"
    ]
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, timeout=300)
        if result.returncode == 0:
            print("✓ ReLU内核生成成功！")
        else:
            print("✗ ReLU内核生成失败")
    except subprocess.TimeoutExpired:
        print("⚠ 生成超时")
    except Exception as e:
        print(f"✗ 执行失败: {e}")
    
    # 示例2: 使用AutoRouter生成更复杂的内核
    print("\n示例2: 使用AutoRouter生成内核 (Level 1, Problem 1)")
    print("-" * 40)
    cmd = [
        sys.executable, "generate_kernel.py",
        "--level", "1",
        "--problem_id", "1",
        "--method", "auto"
    ]
    print(f"命令: {' '.join(cmd)}")
    print("注意: 这个示例需要配置LLM API密钥")
    
    print("\n使用说明:")
    print("1. 确保已配置.env文件中的API密钥")
    print("2. 运行: python generate_kernel.py --level 1 --problem_id 19 --method direct")
    print("3. 查看结果: ls kernel_outputs/")


if __name__ == "__main__":
    run_example()