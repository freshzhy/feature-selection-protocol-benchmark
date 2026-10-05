#!/usr/bin/env python3
"""
Experiment 5 -- classifier sensitivity of the Experiment 1 ranking.

Every method in Experiments 1-4 is scored with XGBoost as the downstream
classifier, and six of the nine methods derive their importance scores from an
XGBoost fit.  That circularity is the single largest threat to the paper's
performance ranking: the reviewer objection is that we may only have shown that
XGBoost-derived importances work well when evaluated with XGBoost.

This experiment re-evaluates the *already selected* feature subsets recorded in
results/exp1_performance.csv with two non-tree classifiers, so no feature
selection is repeated and the comparison isolates the effect of the downstream
learner:

    logreg  Logistic regression (L2, standardised features)
    mlp     Single hidden layer (64 units) neural network

Usage:
    python3 experiments/exp5_classifiers.py            # all 21 datasets
    python3 experiments/exp5_classifiers.py --datasets diabetes sonar
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

import config  # noqa: E402
import data_loader as dl  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # noqa: E402

warnings.filterwarnings("ignore")

SRC = Path("results/exp1_performance.csv")
OUT = Path("results/exp5_classifiers.csv")
SEED = getattr(config, "RANDOM_SEED", 42)


def make_clf(kind: str):
    """Both classifiers are standardised; neither is tree-based."""
    if kind == "logreg":
        est = LogisticRegression(max_iter=2000, random_state=SEED)
    elif kind == "mlp":
        est = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                            early_stopping=True, random_state=SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("sc", StandardScaler()), ("clf", est)])


def evaluate(X: pd.DataFrame, y: np.ndarray, kind: str) -> dict:
    """Stratified 5-fold CV, mirroring the Experiment 1 protocol."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    t0 = time.time()
    proba = cross_val_predict(make_clf(kind), X, y, cv=cv, method="predict_proba")
    pred = proba.argmax(axis=1)
    n_cls = len(np.unique(y))
    auc = (roc_auc_score(y, proba[:, 1]) if n_cls == 2
           else roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    return {
        "accuracy": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "auc": auc,
        "time_eval_s": round(time.time() - t0, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--classifiers", nargs="*", default=["logreg", "mlp"])
    args = ap.parse_args()

    e1 = pd.read_csv(SRC)
    e1 = (e1[e1.status == "ok"]
          .drop_duplicates(["dataset", "method"], keep="first"))
    datasets = args.datasets or sorted(e1.dataset.unique())

    done = set()
    rows: list[dict] = []
    if OUT.exists():                      # checkpoint-based resumption
        prev = pd.read_csv(OUT)
        rows = prev.to_dict("records")
        done = set(zip(prev.dataset, prev.method, prev.classifier))
        print(f"[resume] {len(done)} rows already present in {OUT}")

    for ds in datasets:
        sub = e1[e1.dataset == ds]
        if sub.empty:
            print(f"[skip] {ds}: not in {SRC}")
            continue
        try:
            X_all, y, _ = dl.load_dataset(ds)
        except Exception as exc:                       # noqa: BLE001
            print(f"[fail] {ds}: {exc}")
            continue
        y = np.asarray(y)

        for _, r in sub.iterrows():
            feats = [f for f in str(r.selected_features).split("|") if f]
            missing = [f for f in feats if f not in X_all.columns]
            for kind in args.classifiers:
                if (ds, r.method, kind) in done:
                    continue
                rec = {"dataset": ds, "method": r.method, "classifier": kind,
                       "n_samples": int(r.n_samples), "k": int(r.k),
                       "n_selected": len(feats)}
                if missing:
                    rec |= {"status": f"missing_features:{len(missing)}"}
                    rows.append(rec)
                    print(f"[fail] {ds}/{r.method}/{kind}: "
                          f"{len(missing)} features not in frame")
                    continue
                try:
                    rec |= evaluate(X_all[feats], y, kind) | {"status": "ok"}
                    print(f"[ok]   {ds:<22} {r.method:<16} {kind:<7} "
                          f"AUC={rec['auc']:.4f}")
                except Exception as exc:               # noqa: BLE001
                    rec |= {"status": f"error:{type(exc).__name__}"}
                    print(f"[fail] {ds}/{r.method}/{kind}: {exc}")
                rows.append(rec)
            pd.DataFrame(rows).to_csv(OUT, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    ok = df[df.status == "ok"]
    print(f"\nWrote {OUT}: {len(df)} rows, {len(ok)} ok")
    for kind in args.classifiers:
        k = ok[ok.classifier == kind]
        if len(k):
            print(f"\n--- {kind}: mean AUC by method ---")
            print(k.groupby("method").auc.mean().sort_values(ascending=False)
                  .round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
