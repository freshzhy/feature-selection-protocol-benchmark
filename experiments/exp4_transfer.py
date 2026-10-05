"""
experiments/exp4_transfer.py
实验4：跨数据集特征迁移实验（完整重写版，内置 inf 防护）
"""

import argparse
import time
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from xgboost import XGBClassifier

import sys
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from data_loader import load_dataset
from feature_selectors import run_selector, ALL_METHODS

warnings.filterwarnings("ignore")

NIDS_NAMES = [e["name"] for e in config.NIDS_DATASETS]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def sanitize(X: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 中所有 inf/-inf/NaN 替换为 0，
    并把超大值 clip 到 float32 安全范围。
    这是唯一的清洗入口，所有进入 XGBoost 的数据都必须经过这里。
    """
    MAX_VAL = np.finfo(np.float32).max  # ~3.4e38
    arr = X.values.astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -MAX_VAL, MAX_VAL)
    return pd.DataFrame(arr, columns=X.columns, index=X.index)


def _make_clf(n_classes: int) -> XGBClassifier:
    params = dict(config.XGB_PARAMS)
    params["missing"] = float("nan")
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    return XGBClassifier(**params)


def _cv_score(X: pd.DataFrame, y: pd.Series, n_classes: int) -> dict:
    """5折CV，返回 f1_macro 和 auc 的均值。"""
    X = sanitize(X)
    clf = _make_clf(n_classes)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_SEED)
    scoring = {
        "f1": make_scorer(f1_score, average="macro", zero_division=0),
        "auc": make_scorer(
            roc_auc_score,
            response_method="predict_proba",
            **({"multi_class": "ovr", "average": "macro"}
               if n_classes > 2 else {})
        ),
    }
    scores = cross_validate(clf, X, y, cv=cv, scoring=scoring,
                            n_jobs=-1, error_score="raise")
    return {
        "f1_mean":  round(scores["test_f1"].mean(),  4),
        "f1_std":   round(scores["test_f1"].std(),   4),
        "auc_mean": round(scores["test_auc"].mean(), 4),
        "auc_std":  round(scores["test_auc"].std(),  4),
    }


def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run_exp4(methods: list, resume: bool = False):
    out_path = config.RESULTS_DIR / "exp4_transfer.csv"

    done_triples = set()
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        done_triples = set(zip(
            existing["source"], existing["target"], existing["method"]
        ))
        print(f"[resume] 已有 {len(done_triples)} 条记录，跳过已完成项。")

    # ── 预加载并清洗全部4个 NIDS 数据集 ──
    print("预加载 NIDS 数据集...")
    nids_data = {}
    for name in NIDS_NAMES:
        try:
            X, y, meta = load_dataset(name)
            X = sanitize(X)  # 预加载时就清洗一次
            nids_data[name] = (X, y, meta)
            # 验证无 inf
            inf_count = np.isinf(X.values.astype(float)).sum()
            print(f"  {name}: {meta['n_samples']} × {meta['n_features']}, "
                  f"top_k={meta['top_k']}, inf残留={inf_count}")
        except Exception as e:
            print(f"  [跳过] {name}: {e}")

    if len(nids_data) < 2:
        print("有效 NIDS 数据集不足2个，无法进行迁移实验。")
        return

    # ── 公共特征集 ──
    common_features = None
    for name, (X, y, meta) in nids_data.items():
        if common_features is None:
            common_features = set(X.columns)
        else:
            common_features &= set(X.columns)
    common_features = sorted(common_features)
    print(f"公共特征数: {len(common_features)}")

    # ── 统一 top_k ──
    all_k = [nids_data[n][2]["top_k"] for n in nids_data]
    k = min(min(all_k), len(common_features) // 2, config.TOP_K_MAX)
    print(f"统一 top_k = {k}")

    # ── 所有 source→target 有序对 ──
    pairs = list(permutations(nids_data.keys(), 2))
    total = len(pairs) * len(methods)
    counter = 0
    rows = []

    for src_name, tgt_name in pairs:
        X_src, y_src, meta_src = nids_data[src_name]
        X_tgt, y_tgt, meta_tgt = nids_data[tgt_name]

        X_src_c = sanitize(X_src[common_features])
        X_tgt_c = sanitize(X_tgt[common_features])

        n_classes = max(y_src.nunique(), y_tgt.nunique())

        print(f"\n{'='*60}")
        print(f"迁移对: {src_name} → {tgt_name}")
        print(f"{'='*60}")

        for method in methods:
            counter += 1
            tag = f"[{counter}/{total}] {src_name}→{tgt_name} × {method}"
            key = (src_name, tgt_name, method)

            if key in done_triples:
                print(f"  {tag}  →  已完成，跳过")
                continue

            print(f"  {tag}", end="", flush=True)
            t0 = time.time()

            row = {
                "source": src_name, "target": tgt_name, "method": method,
                "n_src": meta_src["n_samples"], "n_tgt": meta_tgt["n_samples"],
                "n_common": len(common_features), "k": k,
            }

            try:
                # Step 1: source 上做特征选择
                t_sel = time.time()
                src_selected = run_selector(method, X_src_c, y_src, k,
                                            config.RANDOM_SEED)
                row["time_select_s"] = round(time.time() - t_sel, 1)
                row["src_selected"] = "|".join(src_selected)

                # Step 2: target 上独立做特征选择（计算 Jaccard）
                tgt_selected = run_selector(method, X_tgt_c, y_tgt, k,
                                            config.RANDOM_SEED)
                row["tgt_selected"] = "|".join(tgt_selected)
                row["feature_jaccard"] = round(_jaccard(src_selected,
                                                        tgt_selected), 4)

                # Step 3: in-domain（source 特征在 source 上 CV）
                in_domain = _cv_score(X_src_c[src_selected], y_src, n_classes)
                row["indomain_f1"] = in_domain["f1_mean"]
                row["indomain_auc"] = in_domain["auc_mean"]

                # Step 4: cross-domain（source 特征在 target 上 CV）
                avail = [f for f in src_selected if f in X_tgt_c.columns]
                if len(avail) < 2:
                    raise ValueError(f"迁移特征不足: {len(avail)} 列")

                cross_domain = _cv_score(X_tgt_c[avail], y_tgt, n_classes)
                row["cross_f1"] = cross_domain["f1_mean"]
                row["cross_auc"] = cross_domain["auc_mean"]
                row["n_transferred"] = len(avail)

                # Step 5: 性能降幅
                row["delta_f1"] = round(in_domain["f1_mean"] -
                                        cross_domain["f1_mean"], 4)
                row["delta_auc"] = round(in_domain["auc_mean"] -
                                         cross_domain["auc_mean"], 4)

                row["time_s"] = round(time.time() - t0, 1)
                row["status"] = "ok"

                print(f"  →  Jaccard={row['feature_jaccard']:.3f}  "
                      f"ΔF1={row['delta_f1']:+.4f}  "
                      f"cross_F1={row['cross_f1']:.4f}  "
                      f"({row['time_s']}s)")

            except Exception as e:
                row.update({
                    "time_select_s": np.nan,
                    "src_selected": "", "tgt_selected": "",
                    "feature_jaccard": np.nan,
                    "indomain_f1": np.nan, "indomain_auc": np.nan,
                    "cross_f1": np.nan, "cross_auc": np.nan,
                    "n_transferred": 0,
                    "delta_f1": np.nan, "delta_auc": np.nan,
                    "time_s": round(time.time() - t0, 1),
                    "status": f"error: {e}",
                })
                print(f"  →  ERROR: {e}")

            rows.append(row)

        # 每个迁移对完成后保存
        if rows:
            df_new = pd.DataFrame(rows)
            if out_path.exists():
                df_all = pd.concat([pd.read_csv(out_path), df_new],
                                   ignore_index=True)
            else:
                df_all = df_new
            df_all.to_csv(out_path, index=False)
            rows = []

    # ── 汇总 ──
    if out_path.exists():
        df = pd.read_csv(out_path)
        df_ok = df[df["status"] == "ok"]
        if not df_ok.empty:
            summary = df_ok.groupby("method").agg(
                feature_jaccard_mean=("feature_jaccard", "mean"),
                feature_jaccard_std=("feature_jaccard", "std"),
                indomain_f1_mean=("indomain_f1", "mean"),
                cross_f1_mean=("cross_f1", "mean"),
                delta_f1_mean=("delta_f1", "mean"),
                delta_f1_std=("delta_f1", "std"),
                cross_auc_mean=("cross_auc", "mean"),
                delta_auc_mean=("delta_auc", "mean"),
                n_pairs=("source", "count"),
            ).round(4).sort_values("delta_f1_mean")

            summary_path = config.RESULTS_DIR / "exp4_summary.csv"
            summary.to_csv(summary_path)

            print(f"\n{'='*60}")
            print("实验4汇总（按 ΔF1 升序）：")
            print(summary[["feature_jaccard_mean", "indomain_f1_mean",
                           "cross_f1_mean", "delta_f1_mean",
                           "delta_auc_mean", "n_pairs"]].to_string())
            print(f"\n结果已保存：\n  {out_path}\n  {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="实验4：跨数据集特征迁移")
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print("实验4 配置：")
    print(f"  NIDS 数据集: {NIDS_NAMES}")
    print(f"  方法: {args.methods}")
    print(f"  CV折数: {config.CV_FOLDS}")
    print(f"  Resume: {args.resume}")

    run_exp4(args.methods, resume=args.resume)
