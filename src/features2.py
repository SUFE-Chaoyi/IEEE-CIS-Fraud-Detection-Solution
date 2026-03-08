from __future__ import annotations

"""
核心目标：把“原始交易字段”转成能被树模型有效利用的数值特征。
优化重点：使用 pd.concat 替代逐列赋值，彻底解决 DataFrame 碎片化问题。
"""

import gc
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from sklearn.ensemble import IsolationForest  

import numpy as np
import pandas as pd

from .config import START_DATE
from .utils import log


# ======= 特征工具函数 (纯函数化改造) =======

def get_FE(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    cols: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算频率编码，返回新特征的 DataFrame，不修改原数据。
    """
    new_train_cols = {}
    new_test_cols = {}
    
    for col in cols:
        # 拼接训练集和测试集
        df = pd.concat([train[col], test[col]], axis=0)
        # 计算频率
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = f"{col}_FE"
        
        # 映射
        new_train_cols[nm] = train[col].map(vc).astype("float32")
        new_test_cols[nm] = test[col].map(vc).astype("float32")
        log(f"[频率编码 FE ] {nm} 计算完成")
        
    return pd.DataFrame(new_train_cols, index=train.index), pd.DataFrame(new_test_cols, index=test.index)


def get_LE_codes(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    col: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    获取标签编码后的数组，不修改原数据。
    """
    df_comb = pd.concat([train[col], test[col]], axis=0)
    codes, _ = df_comb.factorize(sort=True)
    
    if codes.max() > 32000 or codes.min() < -32000:
        dtype = "int32"
    else:
        dtype = "int16"
        
    tr_codes = codes[: len(train)].astype(dtype)
    te_codes = codes[len(train) :].astype(dtype)
    return tr_codes, te_codes


def encode_LE(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    col: str, 
    verbose: bool = True
) -> None:
    """
    [保留原地修改] 标签编码通常替换原列或用于预处理，不涉及大量新增列，
    且通常在特征工程前完成，保持原地修改以节省内存拷贝开销。
    """
    tr_codes, te_codes = get_LE_codes(train, test, col)
    train[col] = tr_codes
    test[col] = te_codes
    if verbose:
        log(f"[标签编码 LE ] {col} 列编码完成")


def get_AG(
    train: pd.DataFrame,
    test: pd.DataFrame,
    main_columns: Sequence[str],
    uids: Sequence[str],
    aggregations: Sequence[str] = ("mean",),
    fillna: bool = True,
    usena: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算分组聚合特征，返回新 DataFrame。
    """
    new_train_cols = {}
    new_test_cols = {}

    for main_column in main_columns:
        for uid_col in uids:
            for agg_type in aggregations:
                new_col = f"{main_column}_{uid_col}_{agg_type}"
                temp = pd.concat(
                    [train[[uid_col, main_column]], test[[uid_col, main_column]]]
                )
                if usena:
                    temp.loc[temp[main_column] == -1, main_column] = np.nan
                
                # 聚合
                agg = (
                    temp.groupby(uid_col)[main_column]
                    .agg([agg_type])
                    .reset_index()
                    .rename(columns={agg_type: new_col})
                )
                
                mp = pd.Series(agg[new_col].values, index=agg[uid_col]).to_dict()
                
                tr_series = train[uid_col].map(mp).astype("float32")
                te_series = test[uid_col].map(mp).astype("float32")
                
                if fillna:
                    tr_series = tr_series.fillna(-1)
                    te_series = te_series.fillna(-1)
                
                new_train_cols[new_col] = tr_series
                new_test_cols[new_col] = te_series
                log(f"[聚合编码 AG1] {new_col} 计算完成")
    
    return pd.DataFrame(new_train_cols, index=train.index), pd.DataFrame(new_test_cols, index=test.index)


def get_CB(
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    col1: str, 
    col2: str,
    s1_override: Optional[pd.Series] = None
) -> Tuple[str, pd.Series, pd.Series]:
    """
    计算列组合，返回 (新列名, train_series, test_series)。
    """
    nm = f"{col1}_{col2}"
    
    # 获取 col1 数据，优先使用 override (用于处理尚未合并进 DataFrame 的中间特征)
    if s1_override is not None:
        # s1_override 应该是包含了 train+test 的长 Series，这里为了简化，
        # 我们假设调用方处理好了，或者我们依然依赖 DataFrame 中的列
        # 为了代码稳健性，这里还是从 DataFrame 取值，因为现在的逻辑是 concat 后才进行下一步
        pass

    s1 = train[col1].astype(str) + "_" + train[col2].astype(str)
    s2 = test[col1].astype(str) + "_" + test[col2].astype(str)
    
    full_s = pd.concat([s1, s2], axis=0)
    codes, _ = full_s.factorize(sort=True)
    
    if codes.max() > 32000 or codes.min() < -32000:
        dtype = "int32"
    else:
        dtype = "int16"
        
    return nm, pd.Series(codes[:len(train)], index=train.index, dtype=dtype), pd.Series(codes[len(train):], index=test.index, dtype=dtype)


def get_AG2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    main_columns: Sequence[str],
    uids: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """计算唯一值计数特征"""
    new_train_cols = {}
    new_test_cols = {}

    for main_column in main_columns:
        for uid_col in uids:
            comb = pd.concat(
                [train[[uid_col, main_column]], test[[uid_col, main_column]]],
                axis=0,
            )
            mp = comb.groupby(uid_col)[main_column].nunique().to_dict()
            new_col = f"{uid_col}_{main_column}_ct"
            
            new_train_cols[new_col] = train[uid_col].map(mp).astype("float32")
            new_test_cols[new_col] = test[uid_col].map(mp).astype("float32")
            log(f"[唯一编码 AG2] {new_col} 计算完成")

    return pd.DataFrame(new_train_cols, index=train.index), pd.DataFrame(new_test_cols, index=test.index)


# ======= notebook 的预处理与特征处理 =======

def normalize_D_columns(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """标准化 D 列时间 (原地修改，不涉及新增列，保持现状)"""
    for i in range(1, 16):
        if i in [1, 2, 3, 5, 9]:
            continue
        col = f"D{i}"
        train[col] = train[col] - train["TransactionDT"] / np.float32(24 * 60 * 60)
        test[col] = test[col] - test["TransactionDT"] / np.float32(24 * 60 * 60)
        log(f"[prep] {col} 列时间标准化完成")


def label_encode_and_shift_numeric(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """预处理 (原地修改，保持现状)"""
    for f in train.columns:
        is_cat = (str(train[f].dtype) == "category") or (train[f].dtype == "object")
        if is_cat:
            encode_LE(train, test, f, verbose=False)
        elif f not in ["TransactionAmt", "TransactionDT"]:
            mn = np.min((train[f].min(), test[f].min()))
            train[f] = (train[f] - np.float32(mn)).fillna(-1)
            test[f] = (test[f] - np.float32(mn)).fillna(-1)
        
        log(f"[prep] {f} 列数值预处理完成")
    gc.collect()


def add_base_features_xgb_base(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建XGB95模型的基础特征集
    [优化] 使用 pd.concat 代替逐列赋值
    Returns: 更新后的 (train, test)
    """
    
    # --- 批次 0: 核心 ID 组合 (card1_addr1) ---
    log("[XGB Base] 构建核心组合特征 card1_addr1...")
    nm1, s1_tr, s1_te = get_CB(train, test, "card1", "addr1")
    
    # [优化] 使用 concat 合并这第一列，因为后续计算依赖它
    # 注意：这里会产生新的 DataFrame 对象
    train = pd.concat([train, pd.DataFrame({nm1: s1_tr}, index=train.index)], axis=1)
    test = pd.concat([test, pd.DataFrame({nm1: s1_te}, index=test.index)], axis=1)


    # --- 批次 1: 独立特征 & 第二级组合 ---
    log("[XGB Base] 开始构建第一批特征...")
    b1_tr = pd.DataFrame(index=train.index)
    b1_te = pd.DataFrame(index=test.index)

    # 1. Cents
    b1_tr["cents"] = (train["TransactionAmt"] - np.floor(train["TransactionAmt"])).astype("float32")
    b1_te["cents"] = (test["TransactionAmt"] - np.floor(test["TransactionAmt"])).astype("float32")

    # 2. 基础列的 FE
    fe_tr, fe_te = get_FE(train, test, ["addr1", "card1", "card2", "card3", "P_emaildomain"])
    b1_tr = pd.concat([b1_tr, fe_tr], axis=1)
    b1_te = pd.concat([b1_te, fe_te], axis=1)

    # 3. 第二级组合列 (依赖 nm1，此时 nm1 已经在 train/test 中了)
    nm2, s2_tr, s2_te = get_CB(train, test, nm1, "P_emaildomain")
    b1_tr = pd.concat([b1_tr, pd.DataFrame({nm2: s2_tr}, index=train.index)], axis=1)
    b1_te = pd.concat([b1_te, pd.DataFrame({nm2: s2_te}, index=test.index)], axis=1)

    # [优化] 将批次 1 合并回主表
    train = pd.concat([train, b1_tr], axis=1)
    test = pd.concat([test, b1_te], axis=1)
    
    log("[XGB Base] 第一批特征赋值完成")


    # --- 批次 2: 依赖特征 (依赖于 nm1, nm2) ---
    log("[XGB Base] 开始构建第二批特征...")
    b2_tr = pd.DataFrame(index=train.index)
    b2_te = pd.DataFrame(index=test.index)

    # 1. 组合列的 FE
    fe2_tr, fe2_te = get_FE(train, test, [nm1, nm2])
    b2_tr = pd.concat([b2_tr, fe2_tr], axis=1)
    b2_te = pd.concat([b2_te, fe2_te], axis=1)

    # 2. 分组聚合 AG (依赖 nm1, nm2)
    ag_tr, ag_te = get_AG(
        train,
        test,
        main_columns=["TransactionAmt", "D9", "D11"],
        uids=["card1", nm1, nm2],
        aggregations=["mean", "std"],
        usena=True,
    )
    b2_tr = pd.concat([b2_tr, ag_tr], axis=1)
    b2_te = pd.concat([b2_te, ag_te], axis=1)

    # [优化] 将批次 2 合并回主表
    train = pd.concat([train, b2_tr], axis=1)
    test = pd.concat([test, b2_te], axis=1)

    log("[XGB Base] 基础特征块全部构建完成")
    
    return train, test


def compute_DT_M(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算 DT_M
    Returns: 更新后的 (train, test)
    """
    tr_dt = pd.to_datetime(START_DATE) + pd.to_timedelta(train["TransactionDT"], unit="s")
    te_dt = pd.to_datetime(START_DATE) + pd.to_timedelta(test["TransactionDT"], unit="s")

    dt_m_tr = (tr_dt.dt.year - 2017) * 12 + tr_dt.dt.month    
    dt_m_te = (te_dt.dt.year - 2017) * 12 + te_dt.dt.month     
    
    # [优化] 使用 concat 添加单列
    train = pd.concat([train, pd.DataFrame({"DT_M": dt_m_tr}, index=train.index)], axis=1)
    test = pd.concat([test, pd.DataFrame({"DT_M": dt_m_te}, index=test.index)], axis=1)

    log("[DT_M] DT_M 交易月份数值列已添加 ")
    return train, test


def build_cols_xgb95(train: pd.DataFrame) -> List[str]:
    cols = list(train.columns)
    cols.remove("TransactionDT")
    for c in ["D6", "D7", "D8", "D9", "D12", "D13", "D14"]:
        if c in cols: cols.remove(c)
    for c in ["C3", "M5", "id_08", "id_33"]:
        if c in cols: cols.remove(c)
    for c in ["card4", "id_07", "id_14", "id_21", "id_30", "id_32", "id_34"]:
        if c in cols: cols.remove(c)
    for c in [f"id_{x}" for x in range(22, 28)]:
        if c in cols: cols.remove(c)
    if "DT_M" in cols: cols.remove("DT_M")
    return cols


def add_magic_features_xgb_lgb(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建 XGB 增强特征
    Returns: 更新后的 (train, test)
    """
    
    # 检查依赖
    if "card1_addr1" not in train.columns:
        log("[XGB + LGB] 缺少 card1_addr1，补充生成...")
        nm1, s1_tr, s1_te = get_CB(train, test, "card1", "addr1")
        # 补充合并
        train = pd.concat([train, pd.DataFrame({nm1: s1_tr}, index=train.index)], axis=1)
        test = pd.concat([test, pd.DataFrame({nm1: s1_te}, index=test.index)], axis=1)

    # --- 批次 1: 核心 ID 构建 ---
    b1_tr = pd.DataFrame(index=train.index)
    b1_te = pd.DataFrame(index=test.index)

    # Day
    tr_day = train["TransactionDT"] / (24 * 60 * 60)
    te_day = test["TransactionDT"] / (24 * 60 * 60)
    b1_tr["day"] = tr_day
    b1_te["day"] = te_day

    # UID
    # 注意：这里如果列还没合并进去，直接取会报错。但根据上面依赖检查，card1_addr1 肯定在了。
    b1_tr["uid"] = train["card1_addr1"].astype(str) + "_" + np.floor(tr_day - train["D1"]).astype(str)
    b1_te["uid"] = test["card1_addr1"].astype(str) + "_" + np.floor(te_day - test["D1"]).astype(str)

    # Outsider15
    b1_tr["outsider15"] = (np.abs(train["D1"] - train["D15"]) > 3).astype("int8")
    b1_te["outsider15"] = (np.abs(test["D1"] - test["D15"]) > 3).astype("int8")

    # [优化] 将批次 1 合并
    train = pd.concat([train, b1_tr], axis=1)
    test = pd.concat([test, b1_te], axis=1)
    
    log("[XGB + LGB] 核心 ID 特征 (day, uid) 添加完成")

    # --- 批次 2: 基于 UID 的聚合特征 ---
    log("[XGB + LGB] 开始构建聚合特征...")
    b2_tr = pd.DataFrame(index=train.index)
    b2_te = pd.DataFrame(index=test.index)

    # 1. UID FE
    fe_tr, fe_te = get_FE(train, test, ["uid"])
    b2_tr = pd.concat([b2_tr, fe_tr], axis=1)
    b2_te = pd.concat([b2_te, fe_te], axis=1)

    # 2. 各种 AG
    ag1_tr, ag1_te = get_AG(
        train, test,
        main_columns=["TransactionAmt", "D4", "D9", "D10", "D15"],
        uids=["uid"], aggregations=["mean", "std"], fillna=True, usena=True
    )
    
    c_cols = [f"C{x}" for x in range(1, 15) if x != 3]
    ag2_tr, ag2_te = get_AG(train, test, main_columns=c_cols, uids=["uid"], fillna=True, usena=True)

    m_cols = [f"M{x}" for x in range(1, 10)]
    ag3_tr, ag3_te = get_AG(train, test, main_columns=m_cols, uids=["uid"], fillna=True, usena=True)
    
    ag2_uniq_tr, ag2_uniq_te = get_AG2(
        train, test,
        main_columns=["P_emaildomain", "dist1", "DT_M", "id_02", "cents"],
        uids=["uid"]
    )
    
    ag_ex_tr, ag_ex_te = get_AG(train, test, ["C14"], ["uid"], ["std"], True, True)
    ag_ex2_tr, ag_ex2_te = get_AG2(train, test, ["C13", "V314"], ["uid"])
    ag_ex3_tr, ag_ex3_te = get_AG2(train, test, ["V127", "V136", "V309", "V307", "V320"], ["uid"])

    # 合并
    for df_pair in [(ag1_tr, ag1_te), (ag2_tr, ag2_te), (ag3_tr, ag3_te), 
                    (ag2_uniq_tr, ag2_uniq_te), (ag_ex_tr, ag_ex_te), 
                    (ag_ex2_tr, ag_ex2_te), (ag_ex3_tr, ag_ex3_te)]:
        b2_tr = pd.concat([b2_tr, df_pair[0]], axis=1)
        b2_te = pd.concat([b2_te, df_pair[1]], axis=1)

    # [优化] 将批次 2 合并
    train = pd.concat([train, b2_tr], axis=1)
    test = pd.concat([test, b2_te], axis=1)
    
    log("[XGB + LGB] 所有聚合特征构建完成")
    return train, test


def add_isolation_forest_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    添加 Isolation Forest 特征
    Returns: 更新后的 (train, test)
    """
    cols_to_use = ["TransactionAmt", "dist1", "card1", "addr1"] + \
                  [f"C{x}" for x in range(1, 15) if f"C{x}" in train.columns]
    X_comb = pd.concat([train[cols_to_use], test[cols_to_use]], axis=0).fillna(-999)
    iso = IsolationForest(n_estimators=100, max_samples=0.5, random_state=42, n_jobs=-1)
    iso.fit(X_comb)
    
    score_tr = iso.decision_function(train[cols_to_use].fillna(-999)).astype("float32")
    score_te = iso.decision_function(test[cols_to_use].fillna(-999)).astype("float32")
    
    # [优化] 使用 concat
    train = pd.concat([train, pd.DataFrame({"iso_score": score_tr}, index=train.index)], axis=1)
    test = pd.concat([test, pd.DataFrame({"iso_score": score_te}, index=test.index)], axis=1)
    
    log("[IsoForest] iso_score 特征已添加")
    return train, test