
from __future__ import annotations

"""
训练树模型

职责：
- 串联整个离线训练 pipeline
  1) 读取数据
  2) 基础预处理（缺失、编码、数值位移）
  3) 特征工程 XGB95 / XGB96 / IsolationForest
  4) 时间切分 CV 循环训练
  5) 计算 OOF 指标，输出测试集预测与 submission
  6) 保存元信息 特征列表、OOF、submission

运行方式：
- 在项目根目录执行：`python -m src.train`
"""


import gc
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap         # SHAP 解释包

from .config import BUILD95_DEFAULT, BUILD96_DEFAULT, PATHS, RANDOM_SEED
from .data import get_notebook_schema, load_ieee_raw
from .features import (
    add_base_features_xgb95,
    add_magic_features_xgb96,
    add_isolation_forest_features,  # 新增导入
    build_cols_xgb95,
    # build_cols_xgb96,             # features.py 中未定义此函数，改为本地逻辑
    compute_DT_M,
    label_encode_and_shift_numeric,
    normalize_D_columns,
)
from .cv import groupkfold_month_splits
from .model_xgb import train_one_fold
from .model_lgbm import train_one_fold_lgbm
from .evaluation import auc, describe_pred, confusion_matrix, precision_recall_fpr, calculate_cost, evaluate_business_metrics
from .utils import ensure_dir, log, seed_everything

from sklearn.metrics import confusion_matrix as sk_confusion_matrix

def _save_feature_list(cols: List[str], path: Path) -> None:
    """
    将本次训练使用的特征列名保存到文本文件，便于复现与排查。
    """
    ensure_dir(path.parent)
    
    path.write_text("\n".join(cols), encoding="utf-8")
    log(f"[meta] 训练使用的特征已保存 路径: {path} (n={len(cols)})")


def _save_oof(oof: np.ndarray, index: pd.Index, out_csv: Path) -> None:
    """
    保存 out-of-fold 预测结果
    """
    ensure_dir(out_csv.parent)
    df = pd.DataFrame({"TransactionID": index.values, "oof": oof})
    df.to_csv(out_csv, index=False)
    log(f"[oof] out of fold 预测结果已保存 路径:{out_csv}")


def _save_submission(preds: np.ndarray, out_csv: Path) -> None:
    """
    保存提交文件（测试集预测）。
    """
    ensure_dir(out_csv.parent)
    sample_path = PATHS.DATA_RAW / "sample_submission.csv"
    sub = pd.read_csv(sample_path)
    sub["isFraud"] = preds
    sub.to_csv(out_csv, index=False)
    log(f"[sub] submission 文件已保存 路径: {out_csv}")


# SHAP 解释分析函数
def run_shap_analysis(booster, X_sample: pd.DataFrame, fold: int) -> None:
    """
    使用 TreeExplainer 对 XGBoost/LightGBM 进行解释性分析。
    """
    log(f"[SHAP] 开始对 Fold {fold} 模型进行解释性分析")
    # TreeExplainer 速度较快，适合树模型
    explainer = shap.TreeExplainer(booster)
    # 计算 shap_values
    shap_values = explainer.shap_values(X_sample)
    
    # 由于 shap 绘图依赖 matplotlib/js，这里我们只打印最重要的 top 5 特征
    # 计算每个特征的平均绝对 SHAP 值作为重要性
    if isinstance(shap_values, list): # LightGBM binary case sometimes returns list
        vals = np.abs(shap_values[1]).mean(0)
    else:
        vals = np.abs(shap_values).mean(0)
        
    feature_importance = pd.DataFrame(list(zip(X_sample.columns, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    print(f"\n")
    log(f"[SHAP] Top 5 最重要特征 (Fold {fold}):")
    for idx, row in feature_importance.head(5).iterrows():
        log(f"   - {row['col_name']}: {row['feature_importance_vals']:.4f}")
    
    # 在实际项目中，这里可以 plt.savefig 保存 summary_plot


def run_xgb_pipeline(build_baseline: bool = True, build_new: bool = True) -> None:
    """
    运行完整的 XGBoost + LightGBM 训练流水线
    
    流程概览：
        1) load_data -> 得到 train/test 宽表
        2) preprocess -> 缺失/编码/数值处理
        3) build_features_xgb_95/96 -> 生成特征
        4) make_time_folds -> 按月留出 CV
        5) train_one_fold -> 训练并收集 OOF/test 预测
        6) 评估与落盘（OOF、submission、特征列表）
    
    Args:
        build95: 是否运行 XGB95 特征 + 训练。
        build96: 是否运行 XGB96 特征 + 训练。
    """
    seed_everything(RANDOM_SEED)        # 随机种子选取
    ensure_dir(PATHS.SUBMISSIONS)
    ensure_dir(PATHS.OOF)
    ensure_dir(PATHS.META)

    # 获取数据 schema
    schema = get_notebook_schema()
    # 读取原始数据
    X_train, X_test, y_train = load_ieee_raw(schema)

    # ======= 数据预处理 ========
    # 1 标准化 D 列
    normalize_D_columns(X_train, X_test)
    # 2 类别列编码 + 数值列平移
    label_encode_and_shift_numeric(X_train, X_test)

    # ======= 对齐 notebook =======
    # 生成 XGB95 基础特征
    add_base_features_xgb95(X_train, X_test)

    # ======= 生成 DT_M 用于 按月cv =======
    compute_DT_M(X_train, X_test)

    # =========================
    # XGB95：GroupKFold 训练 + 预测
    # =========================
    if build_baseline:
        log("[xgb baseline] xgb baseline 开始训练")
        cols95 = build_cols_xgb95(X_train)
        _save_feature_list(cols95, PATHS.META / "features_xgb_baseline.txt")
        log(f"[xgb baseline] features={len(cols95)}")

        # 初始化 oof 预测结果 测试集预测结果 数组
        oof95 = np.zeros(len(X_train), dtype=np.float32)
        preds95 = np.zeros(len(X_test), dtype=np.float32)
 
        best_fold, best_valid_auc = 0, 0
        for fold, (idxT, idxV) in enumerate(groupkfold_month_splits(X_train, y_train)):
            res = train_one_fold(
                X_tr=X_train[cols95].iloc[idxT],
                y_tr=y_train.iloc[idxT],
                X_va=X_train[cols95].iloc[idxV],
                y_va=y_train.iloc[idxV],
                X_test=X_test[cols95],
            )
            oof95[idxV] += res.valid_pred.astype(np.float32)
            preds95 += (res.test_pred.astype(np.float32) / 6.0)
            log(f"[xgb baseline] fold={fold} valid_auc={res.valid_auc:.6f} best_iter={res.best_iteration}")

            # ======== 策略优化：阈值搜索 (XGB95) ========
            print("\n")
            print(f"======== [Fold {fold}] 最佳阈值多策略搜索 (XGB95) ==========")
            print("\n")
            
            y_true = y_train.iloc[idxV].values
            
            
            # --- 策略一变量：加码误拦成本 (Strategy 1) ---
            # 将 FP Cost 从 10/50 提高到 100，迫使模型关注精确率
            best_thre_cost = 0.5
            min_cost = float("inf")
            cost_fp_high = 100.0  
            cost_fn_std = 1000.0
            
            # --- 策略二变量：硬约束 FPR (Strategy 2) ---
            # 寻找在 FPR <= 5% 的前提下，Recall 最大的阈值
            best_thre_fpr = 0.5
            max_recall = -1.0
            fpr_limit = 0.05
            
            # 静默搜索
            threshold_list = [x / 200 for x in range(1,200)]
            for thre in threshold_list:
                y_pred_bin = (res.valid_pred >= thre).astype(int)
                tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred_bin).ravel()
                
                # 策略一计算：最小总成本
                curr_cost = fp * cost_fp_high + fn * cost_fn_std
                if curr_cost < min_cost:
                    min_cost = curr_cost
                    best_thre_cost = thre
                
                # 策略二计算：FPR 硬约束
                curr_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                curr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if curr_fpr <= fpr_limit:
                    if curr_recall > max_recall:
                        max_recall = curr_recall
                        best_thre_fpr = thre

            # --- 打印策略一结果 ---
            print(f"\n[策略一] 最小成本优化 (FP={cost_fp_high}) -> 最佳阈值: {best_thre_cost:.3f}, 最低成本: {min_cost:,.0f}")
            evaluate_business_metrics(y_true, res.valid_pred, threshold=best_thre_cost, cost_fp=cost_fp_high, cost_fn=cost_fn_std)

            # --- 打印策略二结果 (最终采纳) ---
            print(f"\n[策略二] FPR硬约束优化 (FPR<={fpr_limit}) -> 最佳阈值: {best_thre_fpr:.3f}, 最大Recall: {max_recall:.4f}")
            eval_matrix = evaluate_business_metrics(y_true, res.valid_pred, threshold=best_thre_fpr, cost_fp=cost_fp_high, cost_fn=cost_fn_std)
            
            if eval_matrix["auc"] > best_valid_auc:
                best_fold, best_valid_auc = fold, eval_matrix["auc"]
            print(f"==================================")

            del res
            gc.collect()

        auc(y_train, oof95, name="XGB_baseline OOF AUC")
        describe_pred(preds95, "XGB_baseline test preds")

        # notebook 里会把 oof 写进 X_train['oof']，然后保存 csv
        X_train["oof"] = oof95
        X_train = X_train.copy()
        
        print(f"======== 训练最佳折数 ==========")
        print(f"折数 = {best_fold} ,验证集 AUC = {best_valid_auc} ")
        print(f"======== 训练最佳折数 ==========")

        # 保存文件
        _save_oof(oof95, X_train.index, PATHS.OOF / "oof_xgb_baseline.csv")
        log(f"oof 结果已保存 路径:{PATHS.OOF}/oof_xgb_95.csv")
        # _save_submission(preds95, PATHS.SUBMISSIONS / "sub_xgb_baseline.csv")
        # log(f"submission 结果已保存 路径:{PATHS.SUBMISSIONS}/sub_xgb_baseline.csv")
        
        log("XGB_baseline 训练结束")
    else:
        # notebook 也会给 X_train['oof']=0，保证后续代码不炸
        X_train["oof"] = 0.0
        X_train = X_train.copy()
        
    # =========================
    # XGB_new + LightGBM 集成：Magic 特征 + GroupKFold
    # =========================
    if build_new:
        log("[xgb+lgb] XGB_new + LightGBM Ensemble 开始训练")
        
        # 1. 添加 XGB_new 增强特征
        add_magic_features_xgb96(X_train, X_test)
        
        # 2. 添加 Isolation Forest 无监督特征
        add_isolation_forest_features(X_train, X_test)
        
        # 3. 构建特征列表
        # 由于 features.py 中移除了 build_cols_xgb96，这里使用逻辑生成：
        # 使用当前所有列，排除掉非特征列
        cols_new = list(X_train.columns)
        cols_to_drop = ["TransactionDT", "DT_M", "oof", "isFraud"] # isFraud 已经被 drop 了，这里防守性编程
        for c in cols_to_drop:
            if c in cols_new:
                cols_new.remove(c)
        
        _save_feature_list(cols_new, PATHS.META / "features_xgb_lgb.txt")
        log(f"[xgb+lgb] features={len(cols_new)}")

        # 初始化 oof 预测结果 (集成：XGB + LGB)
        oof_ensemble = np.zeros(len(X_train), dtype=np.float32)
        preds_ensemble = np.zeros(len(X_test), dtype=np.float32)
        
        best_fold, best_valid_auc = 0, 0
        
        for fold, (idxT, idxV) in enumerate(groupkfold_month_splits(X_train, y_train)):
            # 1. 训练 XGBoost
            log(f"[Fold {fold}] 开始训练 XGBoost")
            res_xgb = train_one_fold(
                X_tr=X_train[cols_new].iloc[idxT],
                y_tr=y_train.iloc[idxT],
                X_va=X_train[cols_new].iloc[idxV],
                y_va=y_train.iloc[idxV],
                X_test=X_test[cols_new],
            )
            
            # 2. 训练 LightGBM
            log(f"[Fold {fold}] 开始训练 LightGBM")
            res_lgb = train_one_fold_lgbm(
                X_tr=X_train[cols_new].iloc[idxT],
                y_tr=y_train.iloc[idxT],
                X_va=X_train[cols_new].iloc[idxV],
                y_va=y_train.iloc[idxV],
                X_test=X_test[cols_new],
            )
            
            # 3. 模型融合 (简单的 0.5 加权，生产环境可由验证集确定权重)
            # Ensemble: 0.5 * XGB + 0.5 * LGB
            # 通过不同模型的残差互补提升 AUC
            fold_valid_pred = 0.5 * res_xgb.valid_pred + 0.5 * res_lgb.valid_pred
            fold_test_pred = 0.5 * res_xgb.test_pred + 0.5 * res_lgb.test_pred
            
            # 记录 OOF
            oof_ensemble[idxV] += fold_valid_pred.astype(np.float32)
            preds_ensemble += (fold_test_pred.astype(np.float32) / 6.0)
            
            # 计算当前折集成后的 AUC
            fold_auc = auc(y_train.iloc[idxV], fold_valid_pred, name=f"Fold {fold} Ensemble AUC")

            # SHAP 解释性分析 (只在最后一折做，节省时间)
            # 解决特征工程“解释性不足”的痛点
            if fold == 5:
                # 采样背景数据，因为计算 SHAP 很慢
                # 这里只解释 XGBoost 模型作为示例，LGBM 同理
                X_sample = X_train[cols_new].iloc[idxV].sample(1000, random_state=42)
                run_shap_analysis(res_xgb.booster, X_sample, fold)

            # ======== 策略优化：阈值搜索 (使用集成后的预测值) ========
            print(f"======== [Fold {fold}] 最佳阈值多策略搜索 ==========")
            
            y_true = y_train.iloc[idxV].values
            
            # 策略一变量
            best_thre_cost = 0.5
            min_cost = float("inf")
            cost_fp_high = 100.0  
            cost_fn_std = 1000.0
            
            # 策略二变量：硬约束 FPR (Strategy 2)
            best_thre_fpr = 0.5
            max_recall = -1.0
            fpr_limit = 0.05
            
            # 静默搜索
            threshold_list = [x / 200 for x in range(1,200)]
            for thre in threshold_list:
                y_pred_bin = (fold_valid_pred >= thre).astype(int)
                tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred_bin).ravel()
                
                # 策略一
                curr_cost = fp * cost_fp_high + fn * cost_fn_std
                if curr_cost < min_cost:
                    min_cost = curr_cost
                    best_thre_cost = thre
                
                # 策略二
                curr_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                curr_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if curr_fpr <= fpr_limit:
                    if curr_recall > max_recall:
                        max_recall = curr_recall
                        best_thre_fpr = thre

            # --- 打印策略一结果 ---
            print(f"\n[策略一] 最小成本优化 (FP={cost_fp_high}) -> 最佳阈值: {best_thre_cost:.3f}, 最低成本: {min_cost:,.0f}")
            evaluate_business_metrics(y_true, fold_valid_pred, threshold=best_thre_cost, cost_fp=cost_fp_high, cost_fn=cost_fn_std)

            # --- 打印策略二结果 (最终采纳) ---
            print(f"\n[策略二] FPR硬约束优化 (FPR<={fpr_limit}) -> 最佳阈值: {best_thre_fpr:.3f}, 最大Recall: {max_recall:.4f}")
            eval_matrix = evaluate_business_metrics(y_true, fold_valid_pred, threshold=best_thre_fpr, cost_fp=cost_fp_high, cost_fn=cost_fn_std)
            
            if fold_auc > best_valid_auc:
                best_fold, best_valid_auc = fold, fold_auc
                
            print(f"==================================")
            
            del res_xgb, res_lgb
            gc.collect()

        auc(y_train, oof_ensemble, name="Ensemble OOF AUC")
        describe_pred(preds_ensemble, "Ensemble test preds")
        
        print(f"======== 训练最佳折数 ==========")
        print(f"折数 = {best_fold} ,验证集 AUC = {best_valid_auc} ")
        print(f"======== 训练最佳折数 ==========")
        
        X_train["oof"] = oof_ensemble
        _save_oof(oof_ensemble, X_train.index, PATHS.OOF / "oof_ensemble.csv")
        log(f"oof 结果已保存 路径:{PATHS.OOF}/oof_ensemble.csv")

        # _save_submission(preds_ensemble, PATHS.SUBMISSIONS / "sub_ensemble.csv")
        # log(f"submission 结果已保存 路径:{PATHS.SUBMISSIONS}/sub_ensemble.csv")

        log("XGB + LightGBM 训练结束")


def main():
    """
    脚本入口：解析配置并启动训练。
    """
    run_xgb_pipeline(build_baseline = BUILD95_DEFAULT, build_new = BUILD96_DEFAULT)


if __name__ == "__main__":
    main()