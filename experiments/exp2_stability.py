"""
experiments/exp2_stability.py
实验2：Bootstrap 稳定性分析

流程：
  对每个数据集（10个代表性数据集）× 每种方法：
    1. 做 50 次 bootstrap 重采样（80% 无放回抽样）
    2. 每次独立运行特征选择，记录选出的特征列表
    3. 计算 Pairwise Jaccard 稳定性指数（50次结果两两比较的均值）
    4. 计算 Pairwise Spearman 秩相关均值

结果保存至：
  results/exp2_stability.csv   — 完整明细（每行一个数据集×方法）
  results/exp2_summary.csv     — 按方法汇总（均值±标准差）

运行方式：
  python3 experiments/exp2_stability.py
  python3 experiments/exp2_stability.py --datasets diabetes kc1
  python3 experiments/exp2_stability.py --methods shap_topk mi
  python3 experiments/exp2_stability.py --resume
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import resample

import sys
# 将项目根目录加入路径（无论从哪里调用本脚本都能找到模块）
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from data_loader import load_dataset
from feature_selectors import run_selector, ALL_METHODS

# 内联导入 metrics（避免 utils 包未初始化时的 ModuleNotFoundError）
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("metrics", _ROOT / "utils" / "metrics.py")
_metrics = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
jaccard_stability  = _metrics.jaccard_stability
spearman_stability = _metrics.spearman_stability

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 单个 (数据集 × 方法) 的稳定性评估
# ─────────────────────────────────────────────────────────────────────────────

def compute_stability(X: pd.DataFrame, y: pd.Series, k: int,
                      method: str, n_boot: int, boot_ratio: float,
                      random_state: int) -> dict:
    """
    对给定数据集和方法做 n_boot 次 bootstrap，
    返回 Jaccard 稳定性和 Spearman 秩相关均值。
    """
    all_features = list(X.columns)
    feature_lists = []
    failed = 0

    for b in range(n_boot):
        seed = random_state + b
        rng  = np.random.RandomState(seed)

        # 无放回分层采样（保持类别比例）
        n_sample = max(int(len(X) * boot_ratio), k + 1)

        # 分层采样：按类别比例
        idx_all = np.arange(len(X))
        classes = y.unique()
        sampled_idx = []
        for cls in classes:
            cls_idx = idx_all[y.values == cls]
            n_cls = max(1, int(round(len(cls_idx) * boot_ratio)))
            chosen = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)),
                                replace=False)
            sampled_idx.append(chosen)
        sampled_idx = np.concatenate(sampled_idx)
        rng.shuffle(sampled_idx)

        X_boot = X.iloc[sampled_idx].reset_index(drop=True)
        y_boot = y.iloc[sampled_idx].reset_index(drop=True)

        # 类别数过少时跳过（极端不平衡采样可能只剩1类）
        if y_boot.nunique() < 2:
            failed += 1
            continue

        try:
            selected = run_selector(method, X_boot, y_boot, k,
                                    random_state=seed)
            feature_lists.append(selected)
        except Exception:
            failed += 1

    n_valid = len(feature_lists)
    if n_valid < 2:
        return {
            "n_valid":         n_valid,
            "n_failed":        failed,
            "jaccard_mean":    np.nan,
            "spearman_mean":   np.nan,
        }

    jaccard  = jaccard_stability(feature_lists)
    spearman = spearman_stability(feature_lists, all_features)

    return {
        "n_valid":       n_valid,
        "n_failed":      failed,
        "jaccard_mean":  round(jaccard,  4),
        "spearman_mean": round(spearman, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_exp2(dataset_names: list, methods: list, resume: bool = False):
    out_path = config.RESULTS_DIR / "exp2_stability.csv"

    done_pairs = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        done_pairs = set(zip(existing["dataset"], existing["method"]))
        print(f"[resume] 已有 {len(done_pairs)} 条记录，跳过已完成项。")

    rows = []
    total   = len(dataset_names) * len(methods)
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

        k = meta["top_k"]

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
                "k":          k,
                "n_boot":     config.BOOTSTRAP_N,
            }

            try:
                result = compute_stability(
                    X, y, k, method,
                    n_boot      = config.BOOTSTRAP_N,
                    boot_ratio  = config.BOOTSTRAP_RATIO,
                    random_state= config.RANDOM_SEED,
                )
                row.update(result)
                row["time_s"] = round(time.time() - t0, 1)
                row["status"] = "ok"

                print(f"  →  Jaccard={row['jaccard_mean']:.4f}  "
                      f"Spearman={row['spearman_mean']:.4f}  "
                      f"({row['time_s']}s, "
                      f"{result['n_valid']}/{config.BOOTSTRAP_N} 有效)")
            except Exception as e:
                row.update({
                    "n_valid": 0, "n_failed": config.BOOTSTRAP_N,
                    "jaccard_mean": np.nan, "spearman_mean": np.nan,
                    "time_s": round(time.time() - t0, 1),
                    "status": f"error: {e}",
                })
                print(f"  →  ERROR: {e}")

            rows.append(row)

        # 每个数据集完成后追加保存
        if rows:
            df_new = pd.DataFrame(rows)
            if resume and out_path.exists():
                df_old = pd.read_csv(out_path)
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new if not out_path.exists() else \
                         pd.concat([pd.read_csv(out_path), df_new],
                                   ignore_index=True)
            df_all.to_csv(out_path, index=False)
            rows = []

    # ── 汇总表 ──
    if out_path.exists():
        df = pd.read_csv(out_path)
        df_ok = df[df["status"] == "ok"]
        if not df_ok.empty:
            summary = df_ok.groupby("method").agg(
                jaccard_mean  =("jaccard_mean",  "mean"),
                jaccard_std   =("jaccard_mean",  "std"),
                spearman_mean =("spearman_mean", "mean"),
                spearman_std  =("spearman_mean", "std"),
                n_datasets    =("dataset",       "count"),
            ).round(4).sort_values("jaccard_mean", ascending=False)

            summary_path = config.RESULTS_DIR / "exp2_summary.csv"
            summary.to_csv(summary_path)

            print(f"\n{'='*60}")
            print("实验2汇总（跨数据集均值，按 Jaccard 降序）：")
            print(summary.to_string())
            print(f"\n结果已保存：\n  {out_path}\n  {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验2：Bootstrap 稳定性分析")
    parser.add_argument("--datasets", nargs="+",
                        default=config.STABILITY_DATASETS,
                        help="指定数据集（默认：config.STABILITY_DATASETS 的10个）")
    parser.add_argument("--methods",  nargs="+", default=ALL_METHODS,
                        help="指定方法（默认：全部9种）")
    parser.add_argument("--resume", action="store_true",
                        help="跳过已完成项，续跑")
    args = parser.parse_args()

    print("实验2 配置：")
    print(f"  数据集:    {args.datasets}")
    print(f"  方法:      {args.methods}")
    print(f"  Bootstrap: {config.BOOTSTRAP_N} 次 × {config.BOOTSTRAP_RATIO} 采样比")
    print(f"  Resume:    {args.resume}")

    run_exp2(args.datasets, args.methods, resume=args.resume)
