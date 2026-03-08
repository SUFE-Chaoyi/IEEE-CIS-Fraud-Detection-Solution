from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score       # 导入 ROC-AUC函数 

from .config import EARLY_STOPPING_ROUNDS, VERBOSE_EVAL, XGB_PARAMS
from .utils import cpu_count


@dataclass
class FoldResult:
    booster: xgb.Booster        # 训练完成的 XGB 模型
    valid_pred: np.ndarray      # 验证集预测概率 用于评估单折效果
    test_pred: np.ndarray       # 测试集预测概率 用于多折集成
    best_iteration: int         # 最优迭代次数
    valid_auc: float            # 验证集 ROC-AUC 分数


def _to_native_params(xgb_params: Dict) -> tuple[Dict, int]:
    """
    输入: XGB 参数
    输出: xgb.train 参数 best_iteration 最大迭代轮数
    """
    # 提取最大迭代
    best_iter = int(xgb_params.get("n_estimators", 2000))

    # 构建xgb.train参数字典 用于适配 XGB API
    params = {
        # 目标函数 二分类逻辑回归
        "objective": "binary:logistic",
        # 评估指标
        "eval_metric": xgb_params.get("eval_metric", "auc"),
        # 数最大深度 默认8
        "max_depth": int(xgb_params.get("max_depth", 8)),
        # 学习率
        "eta": float(xgb_params.get("learning_rate", 0.05)),  
        # 样本采样率 
        "subsample": float(xgb_params.get("subsample", 0.8)),
        # 特征采样率
        "colsample_bytree": float(xgb_params.get("colsample_bytree", 0.8)),
        # 树构建方法 hist 直方图算法
        "tree_method": xgb_params.get("tree_method", "hist"),
        # 并行线程数
        "nthread": int(xgb_params.get("nthread", cpu_count())),
        # 缺失值处理：原 notebook missing=-1（我们 DataFrame 已 fill -1）
        # 这里不需要单独设置 missing，DMatrix 会认为 NaN 是 missing
        
        "scale_pos_weight" : float(xgb_params.get("scale_pos_weight", 1.0))
    }
    return params, best_iter

# 定义单折训练函数
def train_one_fold(
    X_tr: pd.DataFrame,     # 训练集特征
    y_tr: pd.Series,        # 训练集标签/结果
    X_va: pd.DataFrame,     # 验证集特征
    y_va: pd.Series,        # 验证集标签
    X_test: pd.DataFrame,   # 测试集特征
    params: Optional[Dict] = None,
) -> FoldResult:
    """
    用原生 xgb.train 训练
    """
    # 合并参数
    merged = dict(XGB_PARAMS)
    if params:
        merged.update(params)

    # 正负样本权重正则
    # 计算正负样本数量
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos   # 0-1 标签  
    
    # 覆盖 config 默认值
    if n_pos > 0:
        merged["scale_pos_weight"] = n_neg / n_pos
    else:
        merged["scale_pos_weight"] = 1.0
    
    # 转化为XGB API参数
    native_params, num_boost_round = _to_native_params(merged)

    # 构建数据结构
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    # 验证集标签
    dva = xgb.DMatrix(X_va, label=y_va)
    # 测试集标签
    dte = xgb.DMatrix(X_test)

    evals = [(dtr, "train"), (dva, "valid")]

    # 执行训练
    booster = xgb.train(
        params=native_params,
        dtrain=dtr,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=VERBOSE_EVAL,
    )

    # 最佳迭代次数获取
    best_it = getattr(booster, "best_iteration", None)

    # 如果版本不兼容 尝试best_ntree_limit
    if best_it is None:
        best_it = getattr(booster, "best_ntree_limit", None)
    
    # 兜底 无早停时用最大迭代次数
    if best_it is None:
        best_it = num_boost_round

    # 预测：使用 best_iteration 限制树数量
    try:
        valid_pred = booster.predict(dva, iteration_range=(0, int(best_it) + 1))
        test_pred = booster.predict(dte, iteration_range=(0, int(best_it) + 1))
    except TypeError:
        # 不兼容 尝试用 ntree_limit
        ntree_limit = int(best_it) + 1
        valid_pred = booster.predict(dva, ntree_limit=ntree_limit)
        test_pred = booster.predict(dte, ntree_limit=ntree_limit)

    # 计算验证集 AUC 单折模型效果
    valid_auc = float(roc_auc_score(y_va, valid_pred))
    
    return FoldResult(
        booster=booster,
        valid_pred=valid_pred,
        test_pred=test_pred,
        best_iteration=int(best_it),
        valid_auc=valid_auc,
    )