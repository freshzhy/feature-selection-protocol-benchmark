#!/usr/bin/env python3
"""
Experiment 6 -- native-threshold ablation for the three adaptive selectors.

Experiments 1-4 force every method to return exactly k features, which for
BorutaSHAP, PowerSHAP and SHAP-Select means discarding the stopping rule that
is part of the method and re-ranking (or padding) by the underlying importance
score. For SHAP-Select the padding step admits features its own significance
test rejected, so its poor showing under the fixed budget has two competing
explanations: the two-stage architecture, or the protocol.

This experiment discriminates between them. Each adaptive method is run at its
native threshold -- no truncation, no padding -- and the resulting subset is
scored with the same XGBoost 5-fold protocol as Experiment 1. If SHAP-Select
still underperforms when its own stopping rule is honoured, the fixed-budget
result is not an artefact of padding.

Usage:
    python3 experiments/exp6_native.py
    python3 experiments/exp6_native.py --datasets diabetes sonar
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config          # noqa: E402
import data_loader as dl   # noqa: E402
import feature_selectors as fs  # noqa: E402

from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import xgboost as xgb  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path("results/exp6_native.csv")
SEED = getattr(config, "RANDOM_SEED", 42)
ADAPTIVE = {
    "borutashap": fs.select_borutashap,
    "powershap": fs.select_powershap,
    "shap_select": fs.select_shap_select,
}


def auc_of(X: pd.DataFrame, y: np.ndarray, cols: list[str]) -> float:
    """Exp. 1's downstream protocol: XGBoost, stratified 5-fold, AUC-ROC."""
    n_cls = len(np.unique(y))
    clf = xgb.XGBClassifier(
        random_state=SEED, missing=np.nan, eval_metric="logloss",
        **({"objective": "multi:softprob", "num_class": n_cls}
           if n_cls > 2 else {}))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(clf, X[cols], y, cv=cv, method="predict_proba")
    return (roc_auc_score(y, proba[:, 1]) if n_cls == 2
            else roc_auc_score(y, proba, multi_class="ovr", average="macro"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()

    e1 = pd.read_csv("results/exp1_performance.csv")
    e1 = e1[e1.status == "ok"].drop_duplicates(["dataset", "method"],
                                               keep="first")
    datasets = args.datasets or sorted(e1.dataset.unique())

    rows: list[dict] = []
    done: set = set()
    if OUT.exists():
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = set(zip(prev.dataset, prev.method))
        print(f"[resume] {len(done)} rows already present")

    for ds in datasets:
        sub = e1[e1.dataset == ds]
        if sub.empty:
            continue
        if all((ds, m) in done for m in ADAPTIVE):
            continue
        try:
            X, y, _ = dl.load_dataset(ds)
        except Exception as exc:                     # noqa: BLE001
            print(f"[fail] {ds}: {exc}")
            continue
        y_s = pd.Series(np.asarray(y), index=X.index)  # selectors call y.nunique()
        y = np.asarray(y)
        k = int(sub.k.iloc[0])

        for m, fn in ADAPTIVE.items():
            if (ds, m) in done:
                continue
            rec = {"dataset": ds, "method": m, "n_samples": len(X),
                   "n_features": X.shape[1], "k": k}
            try:
                t0 = time.time()
                native = fn(X, y_s, k, SEED, native=True)
                rec["time_select_s"] = round(time.time() - t0, 3)
                rec["n_native"] = len(native)
                rec["native_features"] = "|".join(native)
                fixed = str(sub[sub.method == m].selected_features.iloc[0]).split("|")
                rec["n_fixed"] = len(fixed)
                inter = len(set(native) & set(fixed))
                rec["jaccard_native_vs_fixed"] = round(
                    inter / len(set(native) | set(fixed)), 4) if native else 0.0
                rec["auc_fixed"] = round(
                    float(sub[sub.method == m].auc_mean.iloc[0]), 6)
                rec["auc_native"] = (round(auc_of(X, y, native), 6)
                                     if native else float("nan"))
                rec["status"] = "ok" if native else "empty_selection"
            except Exception as exc:                 # noqa: BLE001
                rec["status"] = f"error:{type(exc).__name__}"
                print(f"[fail] {ds}/{m}: {exc}")
            rows.append(rec)
            if rec["status"] == "ok":
                print(f"[ok]   {ds:<22} {m:<12} native={rec['n_native']:>3} "
                      f"(k={k})  AUC {rec['auc_fixed']:.4f} -> "
                      f"{rec['auc_native']:.4f}")
        pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ok = df[df.status == "ok"]
    print(f"\nWrote {OUT}: {len(df)} rows, {len(ok)} ok")
    if len(ok):
        g = ok.groupby("method").agg(
            n_native=("n_native", "mean"), k=("k", "mean"),
            auc_fixed=("auc_fixed", "mean"), auc_native=("auc_native", "mean"))
        g["delta"] = g.auc_native - g.auc_fixed
        print(g.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
