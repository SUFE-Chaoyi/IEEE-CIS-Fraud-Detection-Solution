from __future__ import annotations
#  启用向后兼容性，允许类型注解中使用尚未定义的类
"""
数据 IO 层

业务背景：交易反欺诈通常同时有
- 交易流水（transaction）：金额、商品、渠道、时间等
- 身份信息（identity）：设备、浏览器、地理等（缺失较多）

本文件作用：标准化读取和预处理原始数据集
- 从 data/raw 读取 train/test 的 transaction + identity 四个表；
- 以 TransactionID 为键合并交易与身份信息；
- 返回训练集(train_df)与测试集(test_df)的宽表（wide table），供后续特征工程使用。
"""


import gc       # 垃圾回收库，手动释放内存
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .config import PATHS
from .utils import log, df_mem_usage_mb


@dataclass(frozen=True)
class NotebookSchema:
    """完全对齐 Kaggle notebook 的列选择与 dtype 设定。"""
    str_type: List[str]
    cols: List[str]
    v: List[int]
    dtypes: Dict[str, str]


def get_notebook_schema() -> NotebookSchema:
    """     
    返回 NotebookSchema 
    dict[str, str]: 列名到类型字符串的映射。
    """
    # 需要按照 字符串/类别 处理的列名
    str_type = [
        "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
        "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
        "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_30",
        "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
        "DeviceType", "DeviceInfo",
    ]
    str_type += [
        "id-12", "id-15", "id-16", "id-23", "id-27", "id-28", "id-29", "id-30",
        "id-31", "id-33", "id-34", "id-35", "id-36", "id-37", "id-38",
    ]

    # 交易特征列表 包含交易 ID、时间戳、金额、产品类型、卡信息、地址、C/D/M 系列统计列等共 53 列，对齐原始 notebook 的列选择
    cols = [
        "TransactionID", "TransactionDT", "TransactionAmt",
        "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
        "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
        "C12", "C13", "C14", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
        "D9", "D10", "D11", "D12", "D13", "D14", "D15", "M1", "M2", "M3", "M4",
        "M5", "M6", "M7", "M8", "M9",
    ]

    # 需要保留的 V 特征序号列表
    v = [1, 3, 4, 6, 8, 11]
    v += [13, 14, 17, 20, 23, 26, 27, 30]
    v += [36, 37, 40, 41, 44, 47, 48]
    v += [54, 56, 59, 62, 65, 67, 68, 70]
    v += [76, 78, 80, 82, 86, 88, 89, 91]
    v += [107, 108, 111, 115, 117, 120, 121, 123]
    v += [124, 127, 129, 130, 136]
    v += [138, 139, 142, 147, 156, 162]
    v += [165, 160, 166]
    v += [178, 176, 173, 182]
    v += [187, 203, 205, 207, 215]
    v += [169, 171, 175, 180, 185, 188, 198, 210, 209]
    v += [218, 223, 224, 226, 228, 229, 235]
    v += [240, 258, 257, 253, 252, 260, 261]
    v += [264, 266, 267, 274, 277]
    v += [220, 221, 234, 238, 250, 271]
    v += [294, 284, 285, 286, 291, 297]
    v += [303, 305, 307, 309, 310, 320]
    v += [281, 283, 289, 296, 301, 314]

    cols += [f"V{x}" for x in v]    # V 特征添加到交易特征列表

    # 身份表 包含 id_01~id_33 和 id-01~id-33 兼容两种列名格式
    dtypes: Dict[str, str] = {}
    id_cols = [f"id_0{x}" for x in range(1, 10)] + [f"id_{x}" for x in range(10, 34)]
    id_cols += [f"id-0{x}" for x in range(1, 10)] + [f"id-{x}" for x in range(10, 34)]

    # 类型设置，节省内存
    for c in cols + id_cols:
        dtypes[c] = "float32"
    for c in str_type:
        dtypes[c] = "category"   # 将 object 类型 更改为 category 类型，节省内存

    return NotebookSchema(str_type=str_type, cols=cols, v=v, dtypes=dtypes)


def load_ieee_raw(schema: NotebookSchema) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    输入 NotebookSchema 实例，标准化列和类型
    - 读取 transaction + identity
    - test_identity 列名强制对齐 train_identity（zip 重命名）
    - merge
    - 分离 y_train
    """
    
    # 读取原始文件路径 tr训练集 te测试集 tr交易表 id身份表
    tr_tr_path = PATHS.DATA_RAW / "train_transaction.csv"
    tr_id_path = PATHS.DATA_RAW / "train_identity.csv"
    te_tr_path = PATHS.DATA_RAW / "test_transaction.csv"
    te_id_path = PATHS.DATA_RAW / "test_identity.csv"

    # ======训练集操作======
    log(f"[data] 正在读取训练集交易表，路径: {tr_tr_path}")
    X_train = pd.read_csv(
        tr_tr_path,
        index_col="TransactionID",              # 交易 id 设为索引
        dtype=schema.dtypes,                    # 按照定义的类型读取
        usecols=schema.cols + ["isFraud"],      # 只读取核心列 + 标签列
    )

    log(f"[data] 正在读取训练集id表，路径: {tr_id_path}")
    train_id = pd.read_csv(
        tr_id_path, 
        index_col="TransactionID", 
        dtype=schema.dtypes
    )
    # 合并表，train_id 表的所有行都保留，基于索引 TransactionID 合并
    X_train = X_train.merge(train_id, how="left", left_index=True, right_index=True)

    # ======测试集操作======
    
    log(f"[data] 正在读取测试集交易表，路径: {te_tr_path}")
    X_test = pd.read_csv(
        te_tr_path,
        index_col="TransactionID",
        dtype=schema.dtypes,
        usecols=schema.cols,
    )

    log(f"[data] 正在读取测试集id表，路径: {te_id_path}")
    test_id = pd.read_csv(
        te_id_path, 
        index_col="TransactionID", 
        dtype=schema.dtypes
    )

    # test_identity 列名对齐 train_identity
    """
    原始数据集的测试集身份表列名可能与训练集不一致（如顺序 / 命名差异）；
    生成列名映射字典fix：将测试集身份表 test_id 的列名 o 映射为训练集身份表的列名 n
    重命名测试集身份表的列：inplace=True直接修改,保证和训练集列名完全一致,避免合并时列名错位。
    """
    fix = {o: n for o, n in zip(test_id.columns, train_id.columns)}         # 列名映射字典
    test_id.rename(columns=fix, inplace=True)                               # 重命名 test_id 列名

    # 合并测试表
    X_test = X_test.merge(test_id, how="left", left_index=True, right_index=True)

    y_train = X_train["isFraud"].copy()
    X_train.drop(columns=["isFraud"], inplace=True)

    del train_id, test_id   # 删除临时变量
    gc.collect()            # 释放未引用的内存

    log(f"[data] train shape={X_train.shape}, test shape={X_test.shape}")
    log(f"[data] train mem={df_mem_usage_mb(X_train):.2f}MB, test mem={df_mem_usage_mb(X_test):.2f}MB")
    return X_train, X_test, y_train