# -*- coding: utf-8 -*-
"""
KernelBench数据库加载器
基于aikg项目的数据库设计，提供KernelBench数据集的接入
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


class KernelBenchLoader:
    """KernelBench数据库加载器"""

    def __init__(self, database_path: str = None):
        """
        初始化数据库加载器
        
        Args:
            database_path: 数据库文件路径，如果为None则使用默认路径
        """
        if database_path is None:
            # 默认数据库路径
            self.database_path = Path(__file__).parent.parent / "kernelbench.db"
        else:
            self.database_path = Path(database_path)
        
        self.connection = None
        self.dataset = None
        self._init_database()
        logger.info(f"KernelBench数据库加载器初始化完成: {self.database_path}")

    def _init_database(self):
        """初始化数据库连接"""
        try:
            # 首先尝试加载Hugging Face数据集
            self._load_huggingface_dataset()
            
            # 如果本地数据库不存在，从HF数据集创建
            if not self.database_path.exists():
                logger.info("本地数据库不存在，从Hugging Face数据集创建...")
                self._create_database_from_hf()
            
            self.connection = sqlite3.connect(str(self.database_path))
            self.connection.row_factory = sqlite3.Row  # 使结果可以按列名访问
            
            # 验证数据库结构
            self._verify_database_structure()
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            logger.info("回退到模拟数据库...")
            self._create_mock_database()

    def _load_huggingface_dataset(self):
        """加载Hugging Face KernelBench数据集"""
        try:
            logger.info("正在加载Hugging Face KernelBench数据集...")
            self.dataset = load_dataset("ScalingIntelligence/KernelBench", trust_remote_code=True)
            logger.info(f"成功加载数据集，包含 {len(self.dataset)} 个分割")
            
            # 打印数据集信息
            for split_name, split_data in self.dataset.items():
                logger.info(f"  {split_name}: {len(split_data)} 个样本")
                
        except Exception as e:
            logger.warning(f"加载Hugging Face数据集失败: {e}")
            self.dataset = None

    def _create_database_from_hf(self):
        """从Hugging Face数据集创建本地数据库"""
        if not self.dataset:
            raise Exception("Hugging Face数据集未加载")
        
        logger.info("从Hugging Face数据集创建本地数据库...")
        
        # 创建数据库文件
        self.connection = sqlite3.connect(str(self.database_path))
        self.connection.row_factory = sqlite3.Row
        
        # 创建表结构
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS problems (
                id INTEGER PRIMARY KEY,
                level INTEGER NOT NULL,
                problem_id INTEGER NOT NULL,
                operation_name TEXT NOT NULL,
                pytorch_code TEXT NOT NULL,
                input_shapes TEXT NOT NULL,
                output_shapes TEXT NOT NULL,
                data_types TEXT NOT NULL,
                description TEXT,
                difficulty TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(level, problem_id)
            )
        """)
        
        # 从数据集中提取数据并插入数据库
        problems_inserted = 0
        for split_name, split_data in self.dataset.items():
            logger.info(f"处理分割: {split_name} ({len(split_data)} 个样本)")
            
            for idx, example in enumerate(split_data):
                try:
                    # 解析数据集中的字段 - 基于真实的KernelBench结构
                    level = example.get('level', 1)
                    problem_id = example.get('problem_id', idx + 1)
                    operation_name = example.get('name', f'problem_{idx+1}')
                    pytorch_code = example.get('code', '')
                    
                    # 如果无法提取必要信息，跳过
                    if not pytorch_code or not operation_name:
                        logger.warning(f"跳过样本 {idx}: 缺少必要字段")
                        continue
                    
                    # 从代码中推断其他信息
                    input_shapes, output_shapes, data_types = self._infer_shapes_from_code(pytorch_code)
                    description = f"{operation_name} operation"
                    difficulty = self._infer_difficulty_from_level(level)
                    category = self._infer_category_from_name(operation_name)
                    
                    # 插入数据库
                    cursor.execute("""
                        INSERT OR REPLACE INTO problems 
                        (level, problem_id, operation_name, pytorch_code, input_shapes, 
                         output_shapes, data_types, description, difficulty, category)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (level, problem_id, operation_name, pytorch_code,
                          input_shapes, output_shapes, data_types,
                          description, difficulty, category))
                    
                    problems_inserted += 1
                    if problems_inserted <= 5:  # 打印前5个样本的信息
                        logger.info(f"  样本 {problems_inserted}: Level {level} Problem {problem_id} - {operation_name}")
                        
                except Exception as e:
                    logger.warning(f"处理样本 {idx} 失败: {e}")
                    continue
        
        self.connection.commit()
        logger.info(f"成功从Hugging Face数据集创建数据库，插入了 {problems_inserted} 个问题")

    def _infer_shapes_from_code(self, code: str) -> Tuple[str, str, str]:
        """从代码中推断输入输出形状和数据类型"""
        # 简单的形状推断逻辑
        input_shapes = "[]"
        output_shapes = "[]"
        data_types = "[\"torch.float32\"]"
        
        try:
            import re
            
            # 首先提取变量定义
            variables = {}
            var_lines = re.findall(r'(\w+)\s*=\s*(\d+)', code)
            for var_name, var_value in var_lines:
                variables[var_name] = int(var_value)
            
            # 查找torch.rand或torch.randn的调用
            tensor_patterns = [
                r'torch\.rand\(([^)]+)\)',
                r'torch\.randn\(([^)]+)\)',
                r'torch\.zeros\(([^)]+)\)',
                r'torch\.ones\(([^)]+)\)'
            ]
            
            shapes = []
            for pattern in tensor_patterns:
                calls = re.findall(pattern, code)
                for call in calls:
                    # 解析参数
                    args = [arg.strip() for arg in call.split(',')]
                    shape = []
                    for arg in args:
                        # 跳过非形状参数
                        if any(keyword in arg for keyword in ['device', 'dtype', 'requires_grad']):
                            continue
                        # 尝试解析为数字
                        if arg.isdigit():
                            shape.append(int(arg))
                        # 尝试解析为变量
                        elif arg in variables:
                            shape.append(variables[arg])
                        # 尝试解析简单表达式
                        elif '*' in arg or '+' in arg or '-' in arg:
                            try:
                                # 替换已知变量
                                expr = arg
                                for var_name, var_value in variables.items():
                                    expr = expr.replace(var_name, str(var_value))
                                # 安全计算表达式
                                shape.append(eval(expr))
                            except:
                                pass
                    
                    if shape:
                        shapes.append(shape)
            
            if shapes:
                input_shapes = json.dumps(shapes)
                # 假设输出形状与第一个输入相同（对于elementwise操作）
                output_shapes = json.dumps([shapes[0]])
            
            # 检查数据类型
            if "dtype=torch.float16" in code:
                data_types = "[\"torch.float16\"]"
            elif "dtype=torch.int32" in code:
                data_types = "[\"torch.int32\"]"
            elif "dtype=torch.int64" in code:
                data_types = "[\"torch.int64\"]"
                
        except Exception as e:
            logger.debug(f"形状推断失败: {e}")
        
        return input_shapes, output_shapes, data_types

    def _infer_difficulty_from_level(self, level: int) -> str:
        """从level推断难度"""
        difficulty_map = {1: "easy", 2: "medium", 3: "hard", 4: "expert"}
        return difficulty_map.get(level, "medium")

    def _infer_category_from_name(self, name: str) -> str:
        """从操作名称推断类别"""
        name_lower = name.lower()
        if any(pattern in name_lower for pattern in ["relu", "sigmoid", "tanh", "gelu", "add", "mul", "div"]):
            return "elementwise"
        elif any(pattern in name_lower for pattern in ["sum", "mean", "max", "min", "softmax", "norm"]):
            return "reduction"
        elif any(pattern in name_lower for pattern in ["matmul", "mm", "bmm", "linear"]):
            return "matmul"
        elif any(pattern in name_lower for pattern in ["conv", "convolution"]):
            return "conv"
        else:
            return "unknown"

    def _create_mock_database(self):
        """创建模拟数据库用于测试"""
        logger.info("创建模拟KernelBench数据库")
        
        # 创建数据库文件
        self.connection = sqlite3.connect(str(self.database_path))
        self.connection.row_factory = sqlite3.Row
        
        # 创建表结构
        cursor = self.connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS problems (
                id INTEGER PRIMARY KEY,
                level INTEGER NOT NULL,
                problem_id INTEGER NOT NULL,
                operation_name TEXT NOT NULL,
                pytorch_code TEXT NOT NULL,
                input_shapes TEXT NOT NULL,
                output_shapes TEXT NOT NULL,
                data_types TEXT NOT NULL,
                description TEXT,
                difficulty TEXT,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(level, problem_id)
            )
        """)
        
        # 插入示例数据
        sample_problems = [
            {
                "level": 1,
                "problem_id": 1,
                "operation_name": "relu",
                "pytorch_code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    x = torch.randn(1024, 512, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []
""",
                "input_shapes": "[[1024, 512]]",
                "output_shapes": "[[1024, 512]]",
                "data_types": "[\"torch.float32\"]",
                "description": "ReLU激活函数",
                "difficulty": "easy",
                "category": "elementwise"
            },
            {
                "level": 1,
                "problem_id": 19,
                "operation_name": "relu",
                "pytorch_code": """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    x = torch.randn(1024, 1024, device='cuda', dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []
""",
                "input_shapes": "[[1024, 1024]]",
                "output_shapes": "[[1024, 1024]]",
                "data_types": "[\"torch.float32\"]",
                "description": "ReLU激活函数",
                "difficulty": "easy",
                "category": "elementwise"
            }
        ]
        
        for problem in sample_problems:
            cursor.execute("""
                INSERT OR REPLACE INTO problems 
                (level, problem_id, operation_name, pytorch_code, input_shapes, 
                 output_shapes, data_types, description, difficulty, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (problem["level"], problem["problem_id"], problem["operation_name"],
                  problem["pytorch_code"], problem["input_shapes"], problem["output_shapes"],
                  problem["data_types"], problem["description"], problem["difficulty"],
                  problem["category"]))
        
        self.connection.commit()
        logger.info(f"创建了 {len(sample_problems)} 个示例问题")

    def _verify_database_structure(self):
        """验证数据库结构"""
        cursor = self.connection.cursor()
        
        # 检查problems表是否存在
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='problems'
        """)
        
        if not cursor.fetchone():
            raise Exception("数据库中缺少problems表")
        
        logger.info("数据库结构验证通过")

    def get_problem(self, level: int, problem_id: int) -> Dict[str, Any]:
        """
        获取指定的问题
        
        Args:
            level: 难度级别
            problem_id: 问题ID
            
        Returns:
            问题信息字典
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM problems 
            WHERE level = ? AND problem_id = ?
        """, (level, problem_id))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"未找到问题: Level {level} Problem {problem_id}")
        
        # 转换为字典
        problem_info = dict(row)
        
        # 解析JSON字段
        try:
            problem_info["input_shapes"] = json.loads(problem_info["input_shapes"])
            problem_info["output_shapes"] = json.loads(problem_info["output_shapes"])
            problem_info["data_types"] = json.loads(problem_info["data_types"])
        except json.JSONDecodeError as e:
            logger.warning(f"JSON解析失败: {e}")
        
        return problem_info

    def get_problems_by_level(self, level: int) -> List[Dict[str, Any]]:
        """
        获取指定级别的所有问题
        
        Args:
            level: 难度级别
            
        Returns:
            问题列表
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM problems 
            WHERE level = ?
            ORDER BY problem_id
        """, (level,))
        
        problems = []
        for row in cursor.fetchall():
            problem_info = dict(row)
            # 解析JSON字段
            try:
                problem_info["input_shapes"] = json.loads(problem_info["input_shapes"])
                problem_info["output_shapes"] = json.loads(problem_info["output_shapes"])
                problem_info["data_types"] = json.loads(problem_info["data_types"])
            except json.JSONDecodeError:
                pass
            problems.append(problem_info)
        
        return problems

    def close(self):
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("数据库连接已关闭")

    def __del__(self):
        """析构函数"""
        self.close()