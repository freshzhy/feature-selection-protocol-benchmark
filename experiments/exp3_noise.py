"""
experiments/exp3_noise.py
实验3：噪声注入抗噪实验

流程：
  对每个数据集 × 每种噪声比例 × 每种方法：
    1. 向原始数据集注入 r% 随机噪声特征（N(0,1)，列名前缀 __noise__）
    2. 运行特征选择，选出 Top-K 特征
    3. 计算 FDR（选中噪声特征的比例）和 TPR（真实特征被选中的比例）

数据集策略：
  - Madelon：已知真实特征（前20列），天然噪声数据集，作为核心验证集
  - 其他数据集：原始特征视为"真实特征"，注入额外噪声列

结果保存至：
  results/exp3_noise.csv    — 完整明细
  results/exp3_summary.csv  — 按 (方法 × 噪声比) 汇总

运行方式：
  python3 experiments/exp3_noise.py
  python3 experiments/exp3_noise.py --datasets madelon diabetes
  python3 experiments/exp3_noise.py --ratios 0.1 0.5
  python3 experiments/exp3_noise.py --resume
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from data_loader import load_dataset
from feature_selectors import run_selector, ALL_METHODS

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("metrics", _ROOT / "utils" / "metrics.py")
_metrics = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
fdr_tpr = _metrics.fdr_tpr

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 噪声注入
# ─────────────────────────────────────────────────────────────────────────────

def inject_noise(X: pd.DataFrame, noise_ratio: float,
                 random_state: int) -> tuple[pd.DataFrame, set]:
    """
    向 X 注入 ceil(n_features * noise_ratio) 列 N(0,1) 噪声特征。
    返回 (X_noisy, noise_col_names)。
    """
    rng = np.random.RandomState(random_state)
    n_noise = max(1, int(np.ceil(X.shape[1] * noise_ratio)))
    noise_cols = {f"__noise__{i}" for i in range(n_noise)}
    noise_data = rng.randn(len(X), n_noise)
    noise_df = pd.DataFrame(noise_data, columns=list(noise_cols),
                            index=X.index)
    X_noisy = pd.concat([X, noise_df], axis=1)
    return X_noisy, noise_cols


# Madelon 的真实特征（推定相关特征集，见论文 S1 Appendix 与 madelon_putative_set.py）：
# UCI 官方说明列序已被随机打乱，"前20列 = f0..f19 为信息特征"这一假设是错的
# ——用它算出的 FDR 曾经全部落在 0.95-1.00，与随机选择的期望 FDR = 1-20/500 =
# 0.96 完全一致，说明九种方法当时都是在随机噪声水平上被评分。
#
# 真实特征集改为数据驱动地识别：Madelon 的设计是 5 个基础信息特征 + 15 个它们
# 的线性组合（共 20 个"真实"特征）+ 480 个 iid 噪声探针。基础特征与其线性组合
# 彼此高度相关，而 480 个探针互相独立、也与标签独立。对全部 500×500 特征做
# 皮尔逊相关，取 |corr| > 0.3 的边（噪声探针间的最大 |corr| 上界约 0.11，真实
# 簇内的最小 |corr| 下界约 0.69，中间有清晰空隙）做连通分量聚类，得到唯一一个
# 大小恰好为 20 的簇——与 UCI 文档给出的"20 个相关特征"精确吻合，且该簇与标签
# 的平均 |corr| 是其余特征的 5 倍（0.0945 vs 0.0186）。复现脚本：
# 见本包根目录 madelon_putative_set.py。
MADELON_TRUE_FEATURES = {
    "f28", "f48", "f64", "f105", "f128", "f153", "f241", "f281", "f318",
    "f336", "f338", "f378", "f433", "f442", "f451", "f453", "f455", "f472",
    "f475", "f493",
}
assert len(MADELON_TRUE_FEATURES) == 20


# ─────────────────────────────────────────────────────────────────────────────
# 单组评估
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_noise(X: pd.DataFrame, y: pd.Series, k: int,
                   method: str, noise_ratio: float,
                   is_madelon: bool, random_state: int) -> dict:
    """
    注入噪声后运行特征选择，计算 FDR 和 TPR。
    """
    if is_madelon:
        # Madelon：数据集自带噪声，直接在原始特征上运行，
        # 真实特征 = f0..f19，其余 f20..f499 视为噪声
        true_features = MADELON_TRUE_FEATURES
        X_eval = X.copy()
        # 额外注入额外噪声（以模拟不同噪声比对madelon的影响）
        n_extra = max(0, int(np.ceil(X.shape[1] * noise_ratio)) - (X.shape[1] - 20))
        if n_extra > 0:
            rng = np.random.RandomState(random_state)
            extra = pd.DataFrame(
                rng.randn(len(X), n_extra),
                columns=[f"__extra__{i}" for i in range(n_extra)],
                index=X.index
            )
            X_eval = pd.concat([X_eval, extra], axis=1)
        # 真实特征仍为 f0..f19，其他全部为噪声
        noise_features = set(X_eval.columns) - true_features
    else:
        # 普通数据集：原始特征全部视为真实特征，额外注入噪声列
        X_eval, noise_features = inject_noise(X, noise_ratio, random_state)
        true_features = set(X.columns)

    # 调整 k：不超过总特征数
    k_eval = min(k, len(X_eval.columns))

    selected = run_selector(method, X_eval, y, k_eval, random_state)
    fdr, tpr = fdr_tpr(selected, true_features)

    return {
        "n_original":   X.shape[1],
        "n_noise_injected": len(noise_features),
        "n_total":      X_eval.shape[1],
        "k_used":       k_eval,
        "n_selected":   len(selected),
        "fdr":          round(fdr, 4),
        "tpr":          round(tpr, 4),
        "selected_features": "|".join(selected),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

# 实验3使用的数据集：Madelon（核心） + 8个覆盖不同规模和领域的通用数据集
NOISE_DATASETS = [
    "madelon",
    "diabetes", "credit-g", "kc1", "qsar-biodeg",
    "spambase", "ionosphere", "sonar", "adult",
]


def run_exp3(dataset_names: list, methods: list,
             noise_ratios: list, resume: bool = False):
    out_path = config.RESULTS_DIR / "exp3_noise.csv"

    done_triples = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        done_triples = set(
            zip(existing["dataset"], existing["method"],
                existing["noise_ratio"].astype(str))
        )
        print(f"[resume] 已有 {len(done_triples)} 条记录，跳过已完成项。")

    rows = []
    total   = len(dataset_names) * len(methods) * len(noise_ratios)
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

        k          = meta["top_k"]
        is_madelon = (ds_name == "madelon")

        for noise_ratio in noise_ratios:
            for method in methods:
                counter += 1
                tag = f"[{counter}/{total}] {ds_name} × r={noise_ratio} × {method}"
                key = (ds_name, method, str(noise_ratio))

                if key in done_triples:
                    print(f"  {tag}  →  已完成，跳过")
                    continue

                print(f"  {tag}", end="", flush=True)
                t0 = time.time()

                row = {
                    "dataset":     ds_name,
                    "method":      method,
                    "noise_ratio": noise_ratio,
                    "n_samples":   meta["n_samples"],
                    "n_features":  meta["n_features"],
                    "k":           k,
                }

                try:
                    result = evaluate_noise(
                        X, y, k, method, noise_ratio,
                        is_madelon, config.RANDOM_SEED
                    )
                    row.update(result)
                    row["time_s"] = round(time.time() - t0, 1)
                    row["status"] = "ok"
                    print(f"  →  FDR={row['fdr']:.4f}  TPR={row['tpr']:.4f}"
                          f"  ({row['time_s']}s)")
                except Exception as e:
                    row.update({
                        "n_original": meta["n_features"],
                        "n_noise_injected": np.nan, "n_total": np.nan,
                        "k_used": k, "n_selected": np.nan,
                        "fdr": np.nan, "tpr": np.nan,
                        "selected_features": "",
                        "time_s": round(time.time() - t0, 1),
                        "status": f"error: {e}",
                    })
                    print(f"  →  ERROR: {e}")

                rows.append(row)

        # 每个数据集完成后追加保存
        if rows:
            df_new = pd.DataFrame(rows)
            if out_path.exists():
                df_all = pd.concat([pd.read_csv(out_path), df_new],
                                   ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(out_path, index=False)
            rows = []

    # ── 汇总表：按 (方法 × 噪声比) 取均值 ──
    if out_path.exists():
        df = pd.read_csv(out_path)
        df_ok = df[df["status"] == "ok"]
        if not df_ok.empty:
            summary = df_ok.groupby(["method", "noise_ratio"]).agg(
                fdr_mean=("fdr", "mean"),
                fdr_std =("fdr", "std"),
                tpr_mean=("tpr", "mean"),
                tpr_std =("tpr", "std"),
                n_datasets=("dataset", "count"),
            ).round(4)

            summary_path = config.RESULTS_DIR / "exp3_summary.csv"
            summary.to_csv(summary_path)

            print(f"\n{'='*60}")
            print("实验3汇总（按方法 × 噪声比，均值）：")
            # 打印 FDR 透视表（越低越好）
            fdr_pivot = df_ok.pivot_table(
                index="method", columns="noise_ratio",
                values="fdr", aggfunc="mean"
            ).round(4)
            print("\nFDR（误报率，越低越好）：")
            print(fdr_pivot.to_string())

            tpr_pivot = df_ok.pivot_table(
                index="method", columns="noise_ratio",
                values="tpr", aggfunc="mean"
            ).round(4)
            print("\nTPR（召回率，越高越好）：")
            print(tpr_pivot.to_string())
            print(f"\n结果已保存：\n  {out_path}\n  {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验3：噪声注入抗噪实验")
    parser.add_argument("--datasets",     nargs="+", default=NOISE_DATASETS)
    parser.add_argument("--methods",      nargs="+", default=ALL_METHODS)
    parser.add_argument("--ratios",       nargs="+", type=float,
                        default=config.NOISE_RATIOS,
                        help="噪声比例列表，默认 [0.1, 0.2, 0.5]")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("实验3 配置：")
    print(f"  数据集:   {args.datasets}")
    print(f"  方法:     {args.methods}")
    print(f"  噪声比例: {args.ratios}")
    print(f"  Resume:   {args.resume}")

    run_exp3(args.datasets, args.methods, args.ratios, resume=args.resume)
