from __future__ import annotations

"""
用于生成 Kaggle 提交文件
当前 notebook 方案的核心是：每次训练都会重新跑 GroupKFold 并输出 submission。
如果未来把模型保存到 outputs/models/，就可以在这里实现“加载模型推理”。

目前先保留一个最小工具：把已生成的 preds 数组写成 submission。
"""

from pathlib import Path
import numpy as np
import pandas as pd

from .config import PATHS
from .utils import ensure_dir, log


def save_submission(
    preds: np.ndarray, 
    out_csv: Path, 
    sample_path: Path | None = None
    ) -> None:
    """
    保存 Kaggle 提交文件格式：TransactionID + isFraud（预测概率）。
    
    Args:
        path: 输出 CSV 路径。
        transaction_id: 测试集 TransactionID 数组/Series。
        pred: 模型预测概率。
    
    Returns:
        str: 保存的文件路径。
    """
    ensure_dir(out_csv.parent)
    if sample_path is None:
        sample_path = PATHS.DATA_RAW / "sample_submission.csv"

    sub = pd.read_csv(sample_path)
    sub["isFraud"] = preds
    sub.to_csv(out_csv, index=False)
    log(f"[infer] 提交文件保存: {out_csv}")