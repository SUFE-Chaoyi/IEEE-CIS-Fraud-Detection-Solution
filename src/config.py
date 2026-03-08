from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import datetime

# ============== 路径配置 ==============
@dataclass(frozen=True)
class Paths:
    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_RAW: Path = ROOT / "data" / "raw"                  # 原始数据路径
    DATA_PROCESSED: Path = ROOT / "data" / "processed"      # 处理后数据路径
    OUTPUTS: Path = ROOT / "outputs"                        # 输出路径
    SUBMISSIONS: Path = OUTPUTS / "submissions"             # 
    OOF: Path = OUTPUTS / "oof"                             # out of fold 预测数据
    META: Path = OUTPUTS / "meta"                           # 元数据路径 描述数据的数据

PATHS = Paths()

# 起始日
START_DATE = datetime.datetime.strptime("2017-11-30", "%Y-%m-%d")

# ============== 训练开关 ==============
# True 表示训练该树
BUILD95_DEFAULT = True
BUILD96_DEFAULT = True

# ============== XGBoost 参数（对齐 notebook） ==============

XGB_PARAMS = dict(
    n_estimators=5000,      # 迭代总次数，生成数的总数
    max_depth=12,           # 单颗树的最大深度
    learning_rate=0.02,     # 学习率
    subsample=0.8,          # 行采样比例，每轮训练生成新树时，随机抽取 80% 的训练样本参与训练，剩余 20% 不参与。
    colsample_bytree=0.4,   # 列采样比例：每轮训练生成新树时，随机抽取 40% 的特征参与该树的分裂
    missing=-1,
    eval_metric="auc",      # 评估指标，ROC-AUC是二分类任务的指标，适用于类别不平衡场景
    tree_method="hist",     # 直方图算法，构建决策树
    scale_pos_weight = 1    # loss function 权重
)

EARLY_STOPPING_ROUNDS = 200     # 早停次数
VERBOSE_EVAL = 100              # 模型训练 VERBOSE_EVAL 次，打印一次评估指标 AUG
N_SPLITS = 6                    # cv 折数
RANDOM_SEED = 42                # 随机数种子，用于复线结果

# 早停：
# 训练时会同时监控验证集的评估指标（本项目是 AUC），
# 如果连续 200 轮迭代后，验证集指标没有提升，则直接停止训练，不再继续生成新树；
# 该场景的数据存在时间漂移，早停能够减少过拟合
# best_iteration：最终模型会选择使用 验证集指标最优 时的迭代次数

# 时间漂移
# 随着时间推移，数据的分布（特征分布、标签分布、特征与标签的关系）发生变化。过去能有效识别的特征，在未来不适用。
# 对抗方案：1. CV 2. 特征层面归一化 3. 模型早停


