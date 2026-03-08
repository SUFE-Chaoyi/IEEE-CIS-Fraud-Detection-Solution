from __future__ import annotations

"""
评估指标
- AUC(ROC-AUC) 计算：用于衡量二分类模型的排序能力；
- PR 曲线/PR-AUC：在“极度类别不平衡”的反欺诈中更贴近业务关注；
- 特征重要性保存：帮助解释模型与做后续特征迭代。

"""

from typing import Optional, Dict, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix as sk_confusion_matrix
from .utils import log

def auc(y_true, y_pred, name: str = "AUC") -> float:
    """
    计算 ROC-AUC。
    """
    s = float(roc_auc_score(y_true, y_pred))
    log(f"[eval] 指标 {name} = {s:.6f}")
    return s


def describe_pred(pred: np.ndarray, title: str = "") -> None:
    """
    打印基本统计指标
    """
    if title:
        # 打印标题
        log(f"[pred] {title}")
    log(f"  min={pred.min():.6f} max={pred.max():.6f} mean={pred.mean():.6f} std={pred.std():.6f}")


def confusion_matrix(y_true: np.ndarray, y_pred_hard: np.ndarray) -> Dict[str, int]:
    """
    计算混淆矩阵TP/FP/TN/FN。
    
    Args:
        y_true: 真实标签（0/1），数组。
        y_pred_hard: 硬预测标签（0/1），由概率阈值转换而来。
    
    Returns:
        Dict[str, int]: 包含TP/FP/TN/FN的字典。
    """
    # 调用sklearn混淆矩阵，格式：[[TN, FP], [FN, TP]]
    tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred_hard).ravel()
    cm_dict = {"TP": tp, "FP": fp, "TN": tn, "FN": fn}
    log(f"[confusion] 混淆矩阵： TP={tp} FP={fp} TN={tn} FN={fn}")
    return cm_dict


def precision_recall_fpr(y_true: np.ndarray, y_pred_hard: np.ndarray) -> Tuple[float, float, float]:
    """
    计算业务核心指标：Precision（精确率）、Recall（召回率）、FPR（假阳性率）。
    
    Args:
        y_true: 真实标签（0/1），数组。
        y_pred_hard: 硬预测标签（0/1），由概率阈值转换而来。
    
    Returns:
        Tuple[float, float, float]: (Precision, Recall, FPR)
    """
    cm = confusion_matrix(y_true, y_pred_hard)
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    
    # 精确率：避免分母为0（无预测正例时，精确率为0）
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # 召回率：避免分母为0（无真实正例时，召回率为0）
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    # 假阳性率：避免分母为0（无真实负例时，FPR为0）
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    log(f"[Eval] Precision = {precision:.6f} (欺诈判定准确率)")
    log(f"[Eval] Recall = {recall:.6f} (欺诈识别覆盖率)")
    log(f"[Eval] FPR = {fpr:.6f} (正常交易误判率)")
    return precision, recall, fpr


def calculate_cost(
    y_true: np.ndarray,
    y_pred_hard: np.ndarray,
    cost_fp: float = 10.0,      # 假阳性成本
    cost_fn: float = 1000.0     # 假阴性成本
) -> float:                     # 返回总成本

    cm = confusion_matrix(y_true, y_pred_hard)
    total_cost = cost_fp * cm["FP"] + cost_fn * cm["FN"]
    log(f"[cost] 总成本 = {total_cost:.2f} (误拦成本={cost_fp*cm['FP']:.2f}, 漏欺诈成本={cost_fn*cm['FN']:.2f})")
    return total_cost


def evaluate_business_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,     # 阈值
    cost_fp: float = 10.0,
    cost_fn: float = 1000.0
) -> Dict[str, float]:
    """
    Args:
        y_true: 真实标签（0/1），数组。
        y_pred_prob: 预测为欺诈的概率（0~1），数组。
        threshold: 概率阈值（>=阈值判定为欺诈）。
        cost_fp: 假阳性单笔成本。
        cost_fn: 假阴性单笔成本。
    
    Returns:
        Dict[str, float]: 包含AUC/Precision/Recall/FPR/Cost的指标字典。
    """
    # 1. 计算ROC-AUC
    auc_score = auc(y_true, y_pred_prob, name="Business AUC")
    
    # 2. 概率转硬标签
    y_pred_hard = (y_pred_prob >= threshold).astype(int)
    
    # 3. 计算Precision/Recall/FPR
    precision, recall, fpr = precision_recall_fpr(y_true, y_pred_hard)
    
    # 4. 计算业务成本
    total_cost = calculate_cost(y_true, y_pred_hard, cost_fp, cost_fn)
    
    # 5. 返回所有指标
    metrics = {
        "auc": auc_score,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "total_cost": total_cost
    }
    return metrics


