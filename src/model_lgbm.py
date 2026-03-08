from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from .config import EARLY_STOPPING_ROUNDS, VERBOSE_EVAL, N_SPLITS
from .utils import cpu_count

# LightGBM 参数 (对齐 XGBoost 的逻辑，针对不平衡优化)
LGBM_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=64,           # 也就是 2^6，控制复杂度
    max_depth=12,            # 限制树深
    tree_learner="serial",
    colsample_bytree=0.4,    # 特征采样
    subsample=0.8,           # 行采样
    subsample_freq=1,        # 每一轮都重采样
    reg_alpha=0.1,           # L1 正则
    reg_lambda=0.1,          # L2 正则
    random_state=42,
    n_jobs=-1,
)

@dataclass
class FoldResultLGB:
    booster: lgb.Booster
    valid_pred: np.ndarray
    test_pred: np.ndarray
    best_iteration: int
    valid_auc: float

def train_one_fold_lgbm(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    X_test: pd.DataFrame,
    params: Optional[Dict] = None,
) -> FoldResultLGB:
    """
    LightGBM 单折训练函数
    """
    merged_params = dict(LGBM_PARAMS)
    if params:
        merged_params.update(params)

    # 转换为 LightGBM 数据集
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    # 回调函数用于早停和日志
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=VERBOSE_EVAL)
    ]

    # 训练
    gbm = lgb.train(
        merged_params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # 预测
    best_iter = gbm.best_iteration
    valid_pred = gbm.predict(X_va, num_iteration=best_iter)
    test_pred = gbm.predict(X_test, num_iteration=best_iter)

    valid_auc = float(roc_auc_score(y_va, valid_pred))

    return FoldResultLGB(
        booster=gbm,
        valid_pred=valid_pred,
        test_pred=test_pred,
        best_iteration=best_iter,
        valid_auc=valid_auc,
    )