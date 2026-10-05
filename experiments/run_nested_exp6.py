#!/usr/bin/env python3
"""
experiments/run_nested_exp6.py -- nested arm of Experiment 6.

The three adaptive selectors (BorutaSHAP, PowerSHAP, SHAP-Select) carry their
own stopping rule.  Experiment 6 honours that rule instead of forcing exactly
k features.  Under nesting the rule fires independently inside every outer
fold, so the subset size now varies across folds -- a source of variability
that the single-shot v2 version could not expose.

Empty selections are recorded, not suppressed: v2 found SHAP-Select returns an
empty set on 2 of 21 datasets, and under nesting a method may return empty on
only a subset of folds, which is a finer-grained version of the same result.

Output: results/exp6_native_nested.csv  (63 rows = 21 x 3)
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
from nested_eval import (append_rows, load_done, make_xgb,  # noqa: E402
                         nested_evaluate)

warnings.filterwarnings("ignore")

RES = Path(config.RESULTS_DIR)
OUT = RES / "exp6_native_nested.csv"

ADAPTIVE = {
    "borutashap": fs.select_borutashap,
    "powershap": fs.select_powershap,
    "shap_select": fs.select_shap_select,
}


def native_selector(fn):
    """Adapt a select_* function to the harness's (X, y, k, seed) contract."""
    def _call(X, y, k, seed):
        return fn(X, y, k, seed, native=True)
    return _call


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    e1 = pd.read_csv(RES / "exp1_performance.csv")
    e1 = e1[e1.status == "ok"].drop_duplicates(["dataset", "method"])
    datasets = args.datasets or sorted(e1.dataset.unique())

    done = load_done(OUT, ["dataset", "method"]) if args.resume else set()
    if done:
        print(f"[resume] skipping {len(done)} completed pairs")

    t_start = time.time()
    counter, total = 0, len(datasets) * len(ADAPTIVE)

    for ds in datasets:
        try:
            X, y, meta = load_dataset(ds)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {ds}: {e}")
            continue
        k = meta["top_k"]
        rows = []

        for method, fn in ADAPTIVE.items():
            counter += 1
            tag = f"[{counter}/{total}] {ds} x {method}"
            if (ds, method) in done:
                print(f"  {tag}  ->  done, skipped")
                continue
            print(f"  {tag}", end="", flush=True)

            base = {
                "dataset": ds, "method": method,
                "n_samples": meta["n_samples"],
                "n_features": meta["n_features"], "k": k,
            }
            try:
                res = nested_evaluate(X, y, native_selector(fn), k, make_xgb)
            except Exception as e:  # noqa: BLE001
                print(f"  ->  failed: {e}")
                rows.append({**base, "status": f"error: {e}"})
                continue

            agg = res["aggregated"]["clf"]
            sizes = res["n_selected"]
            n_empty = len(res["empty_folds"])
            union = sorted({c for s in res["feature_sets"] for c in s})

            rows.append({
                **base,
                "n_native_mean": float(np.mean(sizes)),
                "n_native_min": int(min(sizes)),
                "n_native_max": int(max(sizes)),
                "n_native_std": float(np.std(sizes)),
                "fold_sizes": "|".join(str(n) for n in sizes),
                "n_empty_folds": n_empty,
                "n_union": len(union),
                "union_features": "|".join(union),
                "auc_native": agg["auc_mean"],
                "auc_native_std": agg["auc_std"],
                "f1_native": agg["f1_macro_mean"],
                "acc_native": agg["accuracy_mean"],
                "n_folds_scored": agg["auc_n_folds"],
                "time_select_s": res["time_select_s"],
                "status": "empty" if n_empty == len(sizes) else "ok",
            })
            print(f"  ->  |S|={np.mean(sizes):.1f} "
                  f"[{min(sizes)}-{max(sizes)}]  "
                  f"auc={agg['auc_mean']:.4f}  empty_folds={n_empty}  "
                  f"({res['time_select_s']:.0f}s sel)")

        append_rows(OUT, rows)

    df = pd.read_csv(OUT)
    print(f"\n{OUT.name}: {len(df)} rows "
          f"({(time.time() - t_start) / 60:.1f} min)")
    if len(df) != 63:
        print(f"[warn] expected 63 rows, got {len(df)}")
    print(f"empty-fold events: {int(df.n_empty_folds.fillna(0).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
