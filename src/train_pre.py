from __future__ import annotations

"""
训练树模型

职责：
- 串联整个离线训练 pipeline
  1) 读取数据
  2) 基础预处理（缺失、编码、数值位移）
  3) 特征工程 XGB95 / XGB96
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

from .config import BUILD95_DEFAULT, BUILD96_DEFAULT, PATHS, RANDOM_SEED
from .data import get_notebook_schema, load_ieee_raw
from .features import (
    add_base_features_xgb95,
    add_magic_features_xgb96,
    build_cols_xgb95,
    build_cols_xgb96,
    compute_DT_M,
    label_encode_and_shift_numeric,
    normalize_D_columns,
)
from .cv import groupkfold_month_splits
from .model_xgb import train_one_fold
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


def run_xgb_pipeline(build95: bool = True, build96: bool = True) -> None:
    """
    运行完整的 XGBoost 训练流水线
    
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
    if build95:
        log("[xgb95] XGB95 开始训练")
        cols95 = build_cols_xgb95(X_train)
        _save_feature_list(cols95, PATHS.META / "features_xgb_95.txt")
        log(f"[xgb96] 训练所需特征已保存 路径:{PATHS.META}/features_xgb_96.txt")

        log(f"[xgb95] features={len(cols95)}")

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
            log(f"[xgb95] fold={fold} valid_auc={res.valid_auc:.6f} best_iter={res.best_iteration}")

            
            print(f"======== [xgb95] 评估指标 ==========")

            y_true = y_train.iloc[idxV].values 
            eval_matrix = evaluate_business_metrics(y_true, res.valid_pred, threshold=0.5)
            
            if eval_matrix["auc"] > best_valid_auc:
                best_fold, best_valid_auc = fold, eval_matrix["auc"]
            print(f"==================================")

            del res
            gc.collect()

        auc(y_train, oof95, name="XGB95 OOF AUC")
        describe_pred(preds95, "XGB95 test preds")

        # notebook 里会把 oof 写进 X_train['oof']，然后保存 csv
        X_train["oof"] = oof95
        
        
        print(f"======== 训练最佳折数 ==========")
        print(f"折数 = {best_fold} ,验证集 AUC = {best_valid_auc} ")
        print(f"======== 训练最佳折数 ==========")

        # 保存文件
        _save_oof(oof95, X_train.index, PATHS.OOF / "oof_xgb_95.csv")
        log(f"oof 结果已保存 路径:{PATHS.OOF}/oof_xgb_95.csv")
        _save_submission(preds95, PATHS.SUBMISSIONS / "sub_xgb_95.csv")
        log(f"submission 结果已保存 路径:{PATHS.SUBMISSIONS}/sub_xgb_95.csv")
        
        log("XGB95 训练结束")
    else:
        # notebook 也会给 X_train['oof']=0，保证后续代码不炸
        X_train["oof"] = 0.0
        
    # =========================
    # XGB96：加入 Magic 特征 + GroupKFold
    # =========================
    if build96:
        log("[xgb96] XGB96 开始训练")
        
        # 添加 XGB96 增强特征
        add_magic_features_xgb96(X_train, X_test)
        cols96 = build_cols_xgb96(X_train)
        _save_feature_list(cols96, PATHS.META / "features_xgb_96.txt")
        log(f"[xgb96] 训练所需特征已保存 路径:{PATHS.META}/features_xgb_96.txt")
        
        log(f"[xgb96] features={len(cols96)}")

        # 初始化 oof 预测结果 测试集预测结果 树组
        oof96 = np.zeros(len(X_train), dtype=np.float32)
        preds96 = np.zeros(len(X_test), dtype=np.float32)
        
        best_fold, best_valid_auc = 0, 0
        for fold, (idxT, idxV) in enumerate(groupkfold_month_splits(X_train, y_train)):
            res = train_one_fold(
                X_tr=X_train[cols96].iloc[idxT],
                y_tr=y_train.iloc[idxT],
                X_va=X_train[cols96].iloc[idxV],
                y_va=y_train.iloc[idxV],
                X_test=X_test[cols96],
            )
            oof96[idxV] += res.valid_pred.astype(np.float32)
            preds96 += (res.test_pred.astype(np.float32) / 6.0)
            log(f"[xgb96] fold={fold} valid_auc={res.valid_auc:.6f} best_iter={res.best_iteration}")

            print(f"======== [xgb95] 评估指标 ==========")

            y_true = y_train.iloc[idxV].values

            eval_matrix = evaluate_business_metrics(y_true, res.valid_pred, threshold=0.5)
            
            if eval_matrix["auc"] > best_valid_auc:
                best_fold, best_valid_auc = fold, eval_matrix["auc"]
                
            print(f"==================================")
            
            del res
            gc.collect()

        auc(y_train, oof96, name="XGB96 OOF AUC")
        describe_pred(preds96, "XGB96 test preds")
        
        print(f"======== 训练最佳折数 ==========")
        print(f"折数 = {best_fold} ,验证集 AUC = {best_valid_auc} ")
        print(f"======== 训练最佳折数 ==========")
        
        X_train["oof"] = oof96
        _save_oof(oof96, X_train.index, PATHS.OOF / "oof_xgb_96.csv")
        log(f"oof 结果已保存 路径:{PATHS.OOF}/oof_xgb_96.csv")

        _save_submission(preds96, PATHS.SUBMISSIONS / "sub_xgb_96.csv")
        log(f"submission 结果已保存 路径:{PATHS.SUBMISSIONS}/sub_xgb_96.csv")

        log("XGB96 训练结束")


def search_best_threshold(build95: bool = True, build96: bool = True) -> None:
    """
    运行完整的 XGBoost 训练流水线
    
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
    if build95:
        log("[xgb95] XGB95 开始训练")
        cols95 = build_cols_xgb95(X_train)
        _save_feature_list(cols95, PATHS.META / "features_xgb_95.txt")
        log(f"[xgb96] 训练所需特征已保存 路径:{PATHS.META}/features_xgb_96.txt")

        log(f"[xgb95] features={len(cols95)}")

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
            log(f"[xgb95] fold={fold} valid_auc={res.valid_auc:.6f} best_iter={res.best_iteration}")

            print(f"======== [Fold {fold}] 最佳阈值网格搜索 ==========")
            
            y_true = y_train.iloc[idxV].values
            
            # 1. 定义搜索范围：0.01 到 0.99，步长 0.01
            # 这里的 range(1, 100, 1) 表示从 1% 到 99%
            threshold_list = [x/1000 for x in range(1, 201)]
            
            best_threshold = 0.5
            min_cost = float("inf")
            
            # 2. 静默搜索：只计算成本，不打印日志
            for thre in threshold_list:
                # 将概率转换为 0/1
                y_pred_temp = (res.valid_pred >= thre).astype(int)
                
                # 计算混淆矩阵 (ravel顺序: tn, fp, fn, tp)
                tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred_temp).ravel()
                
                # 计算业务成本
                cost_fp = 50
                cost_fn = 10000
                current_cost = fp * cost_fp + fn * cost_fn
                
                # 记录最优
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_threshold = thre
            print(f"\n")
            print(f"搜索完成：最佳阈值 = {best_threshold:.2f}, 最低成本 = {min_cost:,.0f}")
            print(f"\n")

            final_metrics = evaluate_business_metrics(
                y_true, 
                res.valid_pred, 
                threshold=best_threshold,
                cost_fp=cost_fp,
                cost_fn=cost_fn
                )
            
            # 4. 记录最佳 AUC (AUC不随阈值变化，但在最佳折数逻辑中需要用到)
            if final_metrics["auc"] > best_valid_auc:
                best_fold, best_valid_auc = fold, final_metrics["auc"]

            print(f"==================================================")
            
            # ... (后续代码)

            del res
            gc.collect()

        auc(y_train, oof95, name="XGB95 OOF AUC")
        describe_pred(preds95, "XGB95 test preds")

        # notebook 里会把 oof 写进 X_train['oof']，然后保存 csv
        X_train["oof"] = oof95
        
        
        print(f"======== 训练最佳折数 ==========")
        print(f"折数 = {best_fold} ,验证集 AUC = {best_valid_auc} ")
        print(f"======== 训练最佳折数 ==========")

        # 保存文件
        _save_oof(oof95, X_train.index, PATHS.OOF / "oof_xgb_95.csv")
        log(f"oof 结果已保存 路径:{PATHS.OOF}/oof_xgb_95.csv")
        _save_submission(preds95, PATHS.SUBMISSIONS / "sub_xgb_95.csv")
        log(f"submission 结果已保存 路径:{PATHS.SUBMISSIONS}/sub_xgb_95.csv")
        
        log("XGB95 训练结束")
    else:
        # notebook 也会给 X_train['oof']=0，保证后续代码不炸
        X_train["oof"] = 0.0
        
    # =========================
    # XGB96：加入 Magic 特征 + GroupKFold
    # =========================
    if build96:
        log("[xgb96] XGB96 开始训练")
        
        # 添加 XGB96 增强特征
        add_magic_features_xgb96(X_train, X_test)
        cols96 = build_cols_xgb96(X_train)
        _save_feature_list(cols96, PATHS.META / "features_xgb_96.txt")
        log(f"[xgb96] 训练所需特征已保存 路径:{PATHS.META}/features_xgb_96.txt")
        
        log(f"[xgb96] features={len(cols96)}")

        # 初始化 oof 预测结果 测试集预测结果 树组
        oof96 = np.zeros(len(X_train), dtype=np.float32)
        preds96 = np.zeros(len(X_test), dtype=np.float32)
        
        best_fold, best_valid_auc = 0, 0
        for fold, (idxT, idxV) in enumerate(groupkfold_month_splits(X_train, y_train)):
            res = train_one_fold(
                X_tr=X_train[cols96].iloc[idxT],
                y_tr=y_train.iloc[idxT],
                X_va=X_train[cols96].iloc[idxV],
                y_va=y_train.iloc[idxV],
                X_test=X_test[cols96],
            )
            oof96[idxV] += res.valid_pred.astype(np.float32)
            preds96 += (res.test_pred.astype(np.float32) / 6.0)
            log(f"[xgb96] fold={fold} valid_auc={res.valid_auc:.6f} best_iter={res.best_iteration}")

            print(f"======== [Fold {fold}] 最佳阈值网格搜索 ==========")
            
            y_true = y_train.iloc[idxV].values
            
            # 1. 定义搜索范围：0.01 到 0.99，步长 0.01
            # 这里的 range(1, 100, 1) 表示从 1% 到 99%
            threshold_list = [x/1000 for x in range(1, 201)]
            
            best_threshold = 0.5
            min_cost = float("inf")
            
            # 2. 静默搜索：只计算成本，不打印日志
            for thre in threshold_list:
                # 将概率转换为 0/1
                y_pred_temp = (res.valid_pred >= thre).astype(int)
                
                # 计算混淆矩阵 (ravel顺序: tn, fp, fn, tp)
                tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred_temp).ravel()
                
                # 计算业务成本
                cost_fp = 50
                cost_fn = 10000
                current_cost = fp * cost_fp + fn * cost_fn
                
                # 记录最优
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_threshold = thre
            print(f"\n")
            print(f"搜索完成：最佳阈值 = {best_threshold:.2f}, 最低成本 = {min_cost:,.0f}")
            print(f"\n")

            final_metrics = evaluate_business_metrics(
                y_true, 
                res.valid_pred, 
                threshold=best_threshold,
                cost_fp=cost_fp,
                cost_fn=cost_fn
                )
            
            # 4. 记录最佳 AUC (AUC不随阈值变化，但在最佳折数逻辑中需要用到)
            if final_metrics["auc"] > best_valid_auc:
                best_fold, best_valid_auc = fold, final_metrics["auc"]

            print(f"==================================================")
            
            del res
            gc.collect()

        auc(y_train, oof96, name="XGB96 OOF AUC")
        describe_pred(preds96, "XGB96 test preds")
        
        print(f"======== 训练最佳折数 ==========")
        print(f"折数 = {best_fold} ,验证集 AUC = {best_valid_auc} ")
        print(f"======== 训练最佳折数 ==========")
        
        X_train["oof"] = oof96
        _save_oof(oof96, X_train.index, PATHS.OOF / "oof_xgb_96.csv")
        log(f"oof 结果已保存 路径:{PATHS.OOF}/oof_xgb_96.csv")

        _save_submission(preds96, PATHS.SUBMISSIONS / "sub_xgb_96.csv")
        log(f"submission 结果已保存 路径:{PATHS.SUBMISSIONS}/sub_xgb_96.csv")

        log("XGB96 训练结束")


def main():
    """
    脚本入口：解析配置并启动训练。
    """
    run_xgb_pipeline(build95=BUILD95_DEFAULT, build96=BUILD96_DEFAULT)


if __name__ == "__main__":
    search_best_threshold(build95 = False, build96 = True)