"""
experiments/exp1_performance.py
实验1：分类性能对比

流程：
  对每个数据集 × 每种方法：
    1. 用该方法选出 Top-K 特征
    2. 用选出的特征训练 XGBoost，5折交叉验证
    3. 记录 Accuracy、F1-macro、AUC-ROC

结果保存至：
  results/exp1_performance.csv   — 完整明细（每行一个数据集×方法）
  results/exp1_summary.csv       — 按方法汇总（均值±标准差）

运行方式：
  python3 experiments/exp1_performance.py
  python3 experiments/exp1_performance.py --datasets diabetes kc1  # 只跑指定数据集
  python3 experiments/exp1_performance.py --methods shap_topk mi   # 只跑指定方法
  python3 experiments/exp1_performance.py --resume                 # 跳过已完成的行
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from xgboost import XGBClassifier

# 将项目根目录加入路径（支持从任意目录运行）
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data_loader import load_dataset, list_datasets
from feature_selectors import run_selector, ALL_METHODS

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 评估函数
# ─────────────────────────────────────────────────────────────────────────────

def _make_clf(n_classes: int) -> XGBClassifier:
    params = dict(config.XGB_PARAMS)
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    return XGBClassifier(**params)


def _auc_scorer(n_classes: int):
    """构造与二/多分类兼容的 AUC scorer。"""
    if n_classes == 2:
        return make_scorer(roc_auc_score, response_method="predict_proba")
    else:
        return make_scorer(
            roc_auc_score,
            response_method="predict_proba",
            multi_class="ovr",
            average="macro",
        )


def evaluate_feature_subset(X_sub: pd.DataFrame, y: pd.Series,
                             n_classes: int) -> dict:
    """
    对给定特征子集做 5折CV，返回 accuracy / f1_macro / auc 的均值和标准差。
    """
    clf = _make_clf(n_classes)
    cv  = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                          random_state=config.RANDOM_SEED)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "auc":      _auc_scorer(n_classes),
    }

    scores = cross_validate(clf, X_sub, y, cv=cv, scoring=scoring,
                            n_jobs=-1, error_score="raise")

    return {
        "accuracy_mean": scores["test_accuracy"].mean(),
        "accuracy_std":  scores["test_accuracy"].std(),
        "f1_mean":       scores["test_f1_macro"].mean(),
        "f1_std":        scores["test_f1_macro"].std(),
        "auc_mean":      scores["test_auc"].mean(),
        "auc_std":       scores["test_auc"].std(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_exp1(dataset_names: list, methods: list, resume: bool = False):
    out_path = config.RESULTS_DIR / "exp1_performance.csv"

    # resume 模式：读取已有结果，跳过已完成的 (dataset, method) 对
    done_pairs = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        existing_ok = existing[existing["status"] == "ok"]
        done_pairs = set(zip(existing_ok["dataset"], existing_ok["method"]))
        print(f"[resume] 已有 {len(done_pairs)} 条记录，跳过已完成项。")

    rows = []
    total = len(dataset_names) * len(methods)
    counter = 0

    for ds_name in dataset_names:
        print(f"\n{'='*60}")
        print(f"数据集: {ds_name}")
        print(f"{'='*60}")

        try:
            X, y, meta = load_dataset(ds_name)
        except Exception as e:
            print(f"  [跳过] 加载失败: {e}")
            continue

        k         = meta["top_k"]
        n_classes = meta["n_classes"]

        for method in methods:
            counter += 1
            tag = f"[{counter}/{total}] {ds_name} × {method}"

            if (ds_name, method) in done_pairs:
                print(f"  {tag}  →  已完成，跳过")
                continue

            print(f"  {tag}", end="", flush=True)
            t0 = time.time()

            row = {
                "dataset":    ds_name,
                "method":     method,
                "n_samples":  meta["n_samples"],
                "n_features": meta["n_features"],
                "n_classes":  n_classes,
                "k":          k,
            }

            # ── 特征选择 ──
            try:
                selected = run_selector(method, X, y, k,
                                        random_state=config.RANDOM_SEED)
                row["selected_features"] = "|".join(selected)
                row["n_selected"] = len(selected)
            except Exception as e:
                print(f"  →  特征选择失败: {e}")
                row.update({
                    "selected_features": "",
                    "n_selected": 0,
                    "accuracy_mean": np.nan, "accuracy_std": np.nan,
                    "f1_mean": np.nan, "f1_std": np.nan,
                    "auc_mean": np.nan, "auc_std": np.nan,
                    "time_select_s": np.nan, "time_eval_s": np.nan,
                    "status": f"select_error: {e}",
                })
                rows.append(row)
                continue

            row["time_select_s"] = round(time.time() - t0, 2)

            # ── 分类性能评估 ──
            t1 = time.time()
            try:
                X_sub = X[selected]
                metrics = evaluate_feature_subset(X_sub, y, n_classes)
                row.update(metrics)
                row["time_eval_s"] = round(time.time() - t1, 2)
                row["status"] = "ok"

                print(f"  →  acc={metrics['accuracy_mean']:.4f}  "
                      f"f1={metrics['f1_mean']:.4f}  "
                      f"auc={metrics['auc_mean']:.4f}  "
                      f"({row['time_select_s']+row['time_eval_s']:.1f}s)")

            except Exception as e:
                print(f"  →  评估失败: {e}")
                row.update({
                    "accuracy_mean": np.nan, "accuracy_std": np.nan,
                    "f1_mean": np.nan, "f1_std": np.nan,
                    "auc_mean": np.nan, "auc_std": np.nan,
                    "time_eval_s": np.nan,
                    "status": f"eval_error: {e}",
                })

            rows.append(row)

        # 每个数据集跑完就追加保存，防止中断丢失
        if rows:
            df_new = pd.DataFrame(rows)
            if resume and out_path.exists():
                df_old = pd.read_csv(out_path)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(out_path, index=False)
            rows = []  # 清空缓冲，避免重复追加

    # ── 生成汇总表 ──
    if out_path.exists():
        df = pd.read_csv(out_path)
        df_ok = df[df["status"] == "ok"]
        if not df_ok.empty:
            summary = df_ok.groupby("method").agg(
                accuracy_mean=("accuracy_mean", "mean"),
                accuracy_std =("accuracy_mean", "std"),
                f1_mean      =("f1_mean", "mean"),
                f1_std       =("f1_mean", "std"),
                auc_mean     =("auc_mean", "mean"),
                auc_std      =("auc_mean", "std"),
                n_datasets   =("dataset", "count"),
            ).round(4)
            summary_path = config.RESULTS_DIR / "exp1_summary.csv"
            summary.to_csv(summary_path)
            print(f"\n{'='*60}")
            print("实验1汇总（跨数据集均值）：")
            print(summary.to_string())
            print(f"\n结果已保存：\n  {out_path}\n  {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 实验1 使用全部 21 个通用数据集（OpenML + UCI + Kaggle）
    # NIDS 数据集留给实验4（跨数据集迁移）
    ALL_GENERAL = (
        [e["name"] for e in config.OPENML_DATASETS] +
        [e["name"] for e in config.UCI_DATASETS] +
        [e["name"] for e in config.KAGGLE_DATASETS]
    )

    parser = argparse.ArgumentParser(description="实验1：分类性能对比")
    parser.add_argument("--datasets", nargs="+", default=ALL_GENERAL,
                        help="指定数据集名称（默认：全部21个通用数据集）")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        help="指定方法名称（默认：全部9种方法）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已完成的 (dataset, method) 对，续跑")
    args = parser.parse_args()

    print(f"实验1 配置：")
    print(f"  数据集: {len(args.datasets)} 个")
    print(f"  方法:   {args.methods}")
    print(f"  CV折数: {config.CV_FOLDS}")
    print(f"  Resume: {args.resume}")

    run_exp1(args.datasets, args.methods, resume=args.resume)
