"""
utils/metrics.py
实验通用度量函数。
"""

import numpy as np
from itertools import combinations


def jaccard_stability(feature_lists: list[list]) -> float:
    """
    计算多次特征选择结果的平均 Pairwise Jaccard 稳定性指数。
    
    Parameters
    ----------
    feature_lists : list of list[str]
        每次选择结果的特征列名列表。
    
    Returns
    -------
    float — 平均 Jaccard 相似度（0~1，越大越稳定）
    """
    if len(feature_lists) < 2:
        return 1.0
    scores = []
    for a, b in combinations(feature_lists, 2):
        sa, sb = set(a), set(b)
        union = sa | sb
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(sa & sb) / len(union))
    return float(np.mean(scores))


def spearman_stability(feature_lists: list[list], all_features: list) -> float:
    """
    计算多次特征排名结果的平均 Pairwise Spearman 秩相关。
    将特征列表转为全特征空间上的排名向量（未选中特征排名设为末位）。

    Parameters
    ----------
    feature_lists : list of list[str]  — 每次选择结果（有序，第0个最重要）
    all_features  : list[str]          — 完整特征列表

    Returns
    -------
    float — 平均 Spearman ρ（-1~1，越大越稳定）
    """
    from scipy.stats import spearmanr

    n_all = len(all_features)
    feat_idx = {f: i for i, f in enumerate(all_features)}

    def to_rank_vector(selected: list) -> np.ndarray:
        rank_vec = np.full(n_all, n_all, dtype=float)  # 未选中：末位
        for rank, feat in enumerate(selected):
            if feat in feat_idx:
                rank_vec[feat_idx[feat]] = rank
        return rank_vec

    rank_vecs = [to_rank_vector(fl) for fl in feature_lists]
    scores = []
    for a, b in combinations(rank_vecs, 2):
        rho, _ = spearmanr(a, b)
        scores.append(rho)
    return float(np.mean(scores)) if scores else 1.0


def fdr_tpr(selected: list, true_features: set) -> tuple[float, float]:
    """
    计算特征选择的 FDR 和 TPR（用于噪声注入实验）。

    Parameters
    ----------
    selected      : list[str]  — 被选中的特征
    true_features : set[str]   — 真实有效特征集合

    Returns
    -------
    (fdr, tpr) : float, float
        FDR = FP / (FP + TP)  — 误报率（选中噪声特征的比例）
        TPR = TP / |true|     — 召回率（真实特征被选中的比例）
    """
    selected_set = set(selected)
    tp = len(selected_set & true_features)
    fp = len(selected_set - true_features)
    fn = len(true_features - selected_set)

    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    tpr = tp / len(true_features) if true_features else 0.0
    return fdr, tpr
