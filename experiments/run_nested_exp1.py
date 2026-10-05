#!/usr/bin/env python3
"""
experiments/run_nested_exp1.py -- nested arm of Experiment 1 (and 5).

Selection is repeated inside every outer fold, so Experiment 1's nested arm
and Experiment 5's nested arm share exactly the same selection work: the three
classifiers (XGBoost, L2 logistic regression, 64-unit MLP) score the *same*
per-fold subsets.  Running them together halves the compute and, more
importantly, removes any chance that the P1 comparison is confounded by
different subsets.

Outputs (never overwriting the leaky arm):
    results/exp1_performance_nested.csv   189 rows  (21 datasets x 9 methods)
    results/exp5_classifier_nested.csv    378 rows  (21 x 9 x {logreg, mlp})
    results/exp1_folds_nested.csv         945 rows  (189 x 5 folds)

Usage:
    python3 experiments/run_nested_exp1.py
    python3 experiments/run_nested_exp1.py --datasets sonar diabetes
    python3 experiments/run_nested_exp1.py --resume
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import feature_selectors as fs  # noqa: E402
from data_loader import load_dataset  # noqa: E402
from nested_eval import (CLASSIFIERS, append_rows, load_done,  # noqa: E402
                         nested_evaluate)

warnings.filterwarnings("ignore")

RES = Path(config.RESULTS_DIR)
OUT_E1 = RES / "exp1_performance_nested.csv"
OUT_E5 = RES / "exp5_classifier_nested.csv"
OUT_FOLDS = RES / "exp1_folds_nested.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    leaky = pd.read_csv(RES / "exp1_performance.csv")
    leaky = leaky[leaky.status == "ok"].drop_duplicates(["dataset", "method"])

    datasets = args.datasets or sorted(leaky.dataset.unique())
    methods = args.methods or list(fs.ALL_METHODS)
    print(f"{len(datasets)} datasets x {len(methods)} methods = "
          f"{len(datasets) * len(methods)} nested runs")

    done = load_done(OUT_E1, ["dataset", "method"]) if args.resume else set()
    if done:
        print(f"[resume] skipping {len(done)} completed pairs")

    counter, total = 0, len(datasets) * len(methods)
    t_start = time.time()

    for ds in datasets:
        try:
            X, y, meta = load_dataset(ds)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {ds}: {e}")
            continue

        k = meta["top_k"]
        rows_e1, rows_e5, rows_fold = [], [], []

        for method in methods:
            counter += 1
            tag = f"[{counter}/{total}] {ds} x {method}"
            if (ds, method) in done:
                print(f"  {tag}  ->  done, skipped")
                continue
            print(f"  {tag}", end="", flush=True)

            base = {
                "dataset": ds, "method": method,
                "n_samples": meta["n_samples"],
                "n_features": meta["n_features"],
                "n_classes": meta["n_classes"], "k": k,
            }

            try:
                res = nested_evaluate(X, y, partial(fs.run_selector, method),
                                      k, CLASSIFIERS)
            except Exception as e:  # noqa: BLE001
                print(f"  ->  failed: {e}")
                rows_e1.append({**base, "status": f"error: {e}"})
                continue

            agg = res["aggregated"]
            # union of the five fold subsets, for reference
            union = sorted({c for s in res["feature_sets"] for c in s})

            rows_e1.append({
                **base,
                "n_selected_mean": float(np.mean(res["n_selected"])),
                "n_selected_union": len(union),
                "union_features": "|".join(union),
                "time_select_s": res["time_select_s"],
                "accuracy_mean": agg["xgb"]["accuracy_mean"],
                "accuracy_std": agg["xgb"]["accuracy_std"],
                "f1_mean": agg["xgb"]["f1_macro_mean"],
                "f1_std": agg["xgb"]["f1_macro_std"],
                "auc_mean": agg["xgb"]["auc_mean"],
                "auc_std": agg["xgb"]["auc_std"],
                "status": "ok",
            })

            for clf in ("logreg", "mlp"):
                rows_e5.append({
                    "dataset": ds, "method": method, "classifier": clf,
                    "n_samples": meta["n_samples"], "k": k,
                    "n_selected_mean": float(np.mean(res["n_selected"])),
                    "accuracy": agg[clf]["accuracy_mean"],
                    "f1_macro": agg[clf]["f1_macro_mean"],
                    "auc": agg[clf]["auc_mean"],
                    "status": "ok",
                })

            for f in res["folds"]:
                rows_fold.append({"dataset": ds, "method": method, **f})

            print(f"  ->  auc={agg['xgb']['auc_mean']:.4f}  "
                  f"logreg={agg['logreg']['auc_mean']:.4f}  "
                  f"mlp={agg['mlp']['auc_mean']:.4f}  "
                  f"({res['time_select_s']:.0f}s sel)")

        append_rows(OUT_E1, rows_e1)
        append_rows(OUT_E5, rows_e5)
        append_rows(OUT_FOLDS, rows_fold)

    print(f"\nTotal wall time {(time.time() - t_start) / 60:.1f} min")
    for p in (OUT_E1, OUT_E5, OUT_FOLDS):
        if p.exists():
            print(f"  {p.name}: {len(pd.read_csv(p))} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
