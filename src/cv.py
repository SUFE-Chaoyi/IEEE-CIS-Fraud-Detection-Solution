from __future__ import annotations

"""
交叉验证（Cross Validation, CV）策略定义。

本项目数据具有“时间漂移/概念漂移”特征（不同月份欺诈分布可能不同）。
因此不能随便随机打乱划分训练/验证集，否则会产生“时间泄露”，导致线下指标虚高。

本文件实现的策略：
- 基于 DT_M（月粒度时间特征）做“按月留出”的时间切分：
  每个 fold 留出一个月份作为验证集，其余月份作为训练集。
- 这种策略更贴近线上：用历史数据训练，预测未来月份/未来样本。
"""


from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import N_SPLITS
from .utils import log


def groupkfold_month_splits(X_train: pd.DataFrame, y_train: pd.Series) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    按月分组的 GroupKFold 时间交叉验证
    X_train 特征
    y_train 标签
    """
    if "DT_M" not in X_train.columns:
        # 检查 X_train 是否包含 DT_M 列
        raise ValueError("DT_M not found. Please call compute_DT_M(train, test) first.")

    skf = GroupKFold(n_splits=N_SPLITS)     # 初始化。 6 折， GroupKFold 是按照月份拆分
    for i, (idxT, idxV) in enumerate( skf.split(X_train, y_train, groups=X_train["DT_M"])):
        # 获取当前 fold 验证集对应的月份
        month = int(X_train.iloc[idxV]["DT_M"].iloc[0])
        log(f"[cv] 当前折数 = {i} , 验证集对应的月份 = {month} 训练集行数 = {len(idxT)} 验证集行数 = {len(idxV)}")
        yield idxT, idxV