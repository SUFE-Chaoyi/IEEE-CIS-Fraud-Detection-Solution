from __future__ import annotations

"""
通用工具函数（Utilities）。

包含：
- seed_everything：固定随机种子，保证结果可复现
- reduce_mem_usage：通过 downcast 降低 pandas DataFrame 内存占用
- log / ensure_dir：简单日志打印与目录创建

在大规模表格数据任务里，工程层面的“可运行、可复现、可扩展”非常关键。
"""


import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """固定随机种子，保证可复现（尽量）。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(p: Path) -> None:
    """
    确保目录存在，不存在则创建。
    
    Args:
        p: 目录路径（str 或 Path）。
    
    Returns:
        Path: 规范化后的 Path 对象。
    """
    p.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    """
    简单的控制台日志打印，统一输出格式。
    
    Args:
        msg: 要打印的内容。
    """
    print(msg, flush=True)


def human_size(num_bytes: int) -> str:
    """仅用于打印友好大小。"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f}PB"


def df_mem_usage_mb(df) -> float:
    """DataFrame 内存占用（MB）。"""
    return float(df.memory_usage(deep=True).sum() / (1024**2))


def cpu_count() -> int:
    """
    获取可用 CPU 核心数（用于并行/线程数配置）。
    
    Returns:
        int: 逻辑 CPU 数。
    """
    res = os.cpu_count()
    if res > 4:
        return res
    return 4