"""
feature_selectors.py
9 种特征选择方法的统一接口。

所有方法签名：
    select_*(X, y, k, random_state) -> list[str]

返回值：长度为 k 的特征列名列表（按重要性降序）。

统一入口：
    run_selector(method_name, X, y, k) -> list[str]
    ALL_METHODS  -> 所有可用方法名的列表

注意（官方库替换说明，见论文 S1 Appendix "Reimplementations and official libraries" 一节）：
- powershap / shap_select / borutashap 三个方法现在通过 official_bridge/
  调用各自的官方库（PowerShap、shap-select、BorutaShap），而不是本文件早
  期版本的内联重实现。三个官方库的依赖（尤其是 shap 与 numpy 的版本）互
  不兼容，也与本项目主环境的 shap 0.52 / numpy 2.x 不兼容，因此各自被安
  装在 official_bridge/venvs/ 下的独立虚拟环境中，通过子进程 + 临时 CSV
  文件调用（见 official_bridge/bridge.py、official_bridge/worker.py）。
- 早期的内联重实现已证实与官方库存在实质性差异（详见论文 S1 Appendix
  同一节）：SHAP-Select 的重实现在两个多分类数据集上会因为一处未处理的
  三维 SHAP 数组广播错误而返回空集，且与官方库的平均 Jaccard 重叠只有
  0.195；PowerSHAP 的重实现选择的特征数系统性地多出近一倍（平均 Jaccard
  0.592）；BorutaSHAP 的重实现相对忠实（平均 Jaccard 0.735）。旧实现保留
  在 feature_selectors_v1_reimplementation.py.bak 中以备查考，不再用于
  正式实验。
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "official_bridge"))
from bridge import call_official  # noqa: E402

from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2, RFE
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

import config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────

def _topk_cols(feature_names, scores, k: int) -> list:
    """根据分数数组，返回 Top-K 列名（降序）。"""
    feature_names = list(feature_names)
    k = min(k, len(feature_names))
    idx = np.argsort(scores)[::-1][:k]
    return [feature_names[i] for i in idx]


def _get_xgb(random_state: int, n_classes: int):
    """返回配置好的 XGBClassifier。"""
    from xgboost import XGBClassifier
    params = dict(config.XGB_PARAMS)
    params["random_state"] = random_state
    params["missing"] = float("nan")
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    return XGBClassifier(**params)


def _mean_abs_shap(clf, X: pd.DataFrame) -> np.ndarray:
    """
    计算 mean|SHAP| 向量，兼容二分类和多分类，兼容新旧版 shap。
    返回 shape (n_features,) 的数组。
    """
    import shap
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        result = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
    else:
        sv = np.array(sv)
        if sv.ndim == 3:
            result = np.abs(sv).mean(axis=(0, 2))
        else:
            result = np.abs(sv).mean(axis=0)
    # 清洗 inf/nan（NIDS 极端值可能导致 SHAP 溢出）
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 方法 1 — SHAP Top-K
# ─────────────────────────────────────────────────────────────────────────────

def select_shap_topk(X: pd.DataFrame, y: pd.Series, k: int,
                     random_state: int = config.RANDOM_SEED) -> list:
    """用 XGBoost + TreeExplainer 计算 mean|SHAP|，取 Top-K。"""
    n_classes = y.nunique()
    clf = _get_xgb(random_state, n_classes)
    clf.fit(X, y)
    scores = _mean_abs_shap(clf, X)
    return _topk_cols(X.columns, scores, k)


# ─────────────────────────────────────────────────────────────────────────────
# 方法 2 — BorutaSHAP (Keany 2020) — official `BorutaShap` library via
# official_bridge/ (isolated venv: shap 0.44.1, numpy<2, scipy<1.12).
# See the module docstring and S1 Appendix, section 'Reimplementations and official libraries'.
# ─────────────────────────────────────────────────────────────────────────────

def select_borutashap(X: pd.DataFrame, y: pd.Series, k: int,
                      random_state: int = config.RANDOM_SEED,
                      native: bool = False) -> list:
    """Official BorutaShap library: shadow-feature competition + SHAP importance."""
    return call_official("boruta", X, y, k, random_state, native=native,
                         pad_fn=lambda Xr, yr, n: select_shap_topk(Xr, yr, n, random_state))


# ─────────────────────────────────────────────────────────────────────────────
# 方法 3 — PowerSHAP (Verhaeghe et al. 2023) — official `powershap` library
# via official_bridge/ (isolated venv: shap 0.45.1, numpy<2).
# ─────────────────────────────────────────────────────────────────────────────

def select_powershap(X: pd.DataFrame, y: pd.Series, k: int,
                     random_state: int = config.RANDOM_SEED,
                     native: bool = False) -> list:
    """Official PowerShap library: added-random-feature power analysis, out-of-sample SHAP."""
    return call_official("powershap", X, y, k, random_state, native=native,
                         pad_fn=lambda Xr, yr, n: select_shap_topk(Xr, yr, n, random_state))


# ─────────────────────────────────────────────────────────────────────────────
# 方法 4 — SHAP-Select — official `shap-select` library via official_bridge/
# (this library runs unmodified on the project's own shap 0.52 / numpy 2.x;
# there was never a version conflict for this one).
# ─────────────────────────────────────────────────────────────────────────────

def select_shap_select(X: pd.DataFrame, y: pd.Series, k: int,
                       random_state: int = config.RANDOM_SEED,
                       native: bool = False) -> list:
    """Official shap-select library: signed SHAP ~ true target regression on a
    held-out validation split, ranked by t-value. Supports multiclass natively."""
    return call_official("shapselect", X, y, k, random_state, native=native,
                         pad_fn=lambda Xr, yr, n: select_shap_topk(Xr, yr, n, random_state))


# ─────────────────────────────────────────────────────────────────────────────
# 方法 5 — Mutual Information
# ─────────────────────────────────────────────────────────────────────────────

def select_mi(X: pd.DataFrame, y: pd.Series, k: int,
              random_state: int = config.RANDOM_SEED) -> list:
    scores = mutual_info_classif(X, y, random_state=random_state)
    return _topk_cols(X.columns, scores, k)


# ─────────────────────────────────────────────────────────────────────────────
# 方法 6 — Chi-Square
# ─────────────────────────────────────────────────────────────────────────────

def select_chi2(X: pd.DataFrame, y: pd.Series, k: int,
                random_state: int = config.RANDOM_SEED) -> list:
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    scores, _ = chi2(X_scaled, y)
    scores = np.nan_to_num(scores, nan=0.0)
    return _topk_cols(X.columns, scores, k)


# ─────────────────────────────────────────────────────────────────────────────
# 方法 7 — RFE with XGBoost
# ─────────────────────────────────────────────────────────────────────────────

def select_rfe(X: pd.DataFrame, y: pd.Series, k: int,
               random_state: int = config.RANDOM_SEED) -> list:
    n_classes = y.nunique()
    clf = _get_xgb(random_state, n_classes)
    rfe = RFE(estimator=clf, n_features_to_select=k, step=1)
    rfe.fit(X, y)
    selected = [col for col, r in zip(X.columns, rfe.ranking_) if r == 1]
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# 方法 8 — Lasso (L1 Logistic Regression)
# ─────────────────────────────────────────────────────────────────────────────

def select_lasso(X: pd.DataFrame, y: pd.Series, k: int,
                 random_state: int = config.RANDOM_SEED) -> list:
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = list(X.columns)

    # saga 同时支持二分类和多分类的 L1，sklearn 1.6+ 已移除 multi_class 参数
    for C in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]:
        lr = LogisticRegression(
            penalty="l1", C=C, solver="saga",
            max_iter=2000, random_state=random_state
        )
        lr.fit(X_scaled, y)
        coef = np.max(np.abs(lr.coef_), axis=0)
        if np.sum(coef > 0) >= k:
            return _topk_cols(feature_names, coef, k)

    # 极端稀疏回退
    lr = LogisticRegression(
        penalty="l1", C=1.0, solver="saga",
        max_iter=2000, random_state=random_state
    )
    lr.fit(X_scaled, y)
    coef = np.max(np.abs(lr.coef_), axis=0)
    return _topk_cols(feature_names, coef, k)


# ─────────────────────────────────────────────────────────────────────────────
# 方法 9 — Tree-based Importance (XGBoost 内置)
# ─────────────────────────────────────────────────────────────────────────────

def select_tree_importance(X: pd.DataFrame, y: pd.Series, k: int,
                           random_state: int = config.RANDOM_SEED) -> list:
    n_classes = y.nunique()
    clf = _get_xgb(random_state, n_classes)
    clf.fit(X, y)
    return _topk_cols(X.columns, clf.feature_importances_, k)


# ─────────────────────────────────────────────────────────────────────────────
# 路由表 & 统一入口
# ─────────────────────────────────────────────────────────────────────────────

ALL_METHODS = [
    "shap_topk",
    "borutashap",
    "powershap",
    "shap_select",
    "mi",
    "chi2",
    "rfe",
    "lasso",
    "tree_importance",
]

_SELECTOR_MAP = {
    "shap_topk":       select_shap_topk,
    "borutashap":      select_borutashap,
    "powershap":       select_powershap,
    "shap_select":     select_shap_select,
    "mi":              select_mi,
    "chi2":            select_chi2,
    "rfe":             select_rfe,
    "lasso":           select_lasso,
    "tree_importance": select_tree_importance,
}


def run_selector(method: str, X: pd.DataFrame, y: pd.Series, k: int,
                 random_state: int = config.RANDOM_SEED) -> list:
    """
    统一特征选择入口。

    Parameters
    ----------
    method       : str       — 方法名，需在 ALL_METHODS 中
    X            : DataFrame — 特征矩阵
    y            : Series    — 标签
    k            : int       — 选取特征数
    random_state : int

    Returns
    -------
    list[str] — 选中的特征列名，长度 == k
    """
    if method not in _SELECTOR_MAP:
        raise KeyError(f"未知方法: '{method}'。可用: {ALL_METHODS}")
    selected = _SELECTOR_MAP[method](X, y, k, random_state)
    # 去重 + 截断保险
    seen, deduped = set(), []
    for c in selected:
        if c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped[:k]


# ─────────────────────────────────────────────────────────────────────────────
# 冒烟测试
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loader import load_dataset

    print("=" * 60)
    print("冒烟测试：使用 diabetes 数据集，k=4")
    print("=" * 60)

    X, y, meta = load_dataset("diabetes")
    k = 4

    for method in ALL_METHODS:
        try:
            selected = run_selector(method, X, y, k)
            status = "✓" if len(selected) == k else f"警告:仅选出{len(selected)}个"
            print(f"  [{method:20s}] {status}  {selected}")
        except Exception as e:
            print(f"  [{method:20s}] ERROR: {e}")

    print("\n所有方法测试完成。")
