#!/usr/bin/env python3
"""
experiments/nested_eval.py -- leakage-free (nested) evaluation harness.

Experiments 1, 4 (in-domain), 5 and 6 of the v2 study all share the same
protocol:

    full dataset -> feature selection (sees every label) -> k features
                 -> StratifiedKFold(5) -> mean metric

Because all nine selectors are supervised, label information from every
future test fold reaches the selection step.  That is the "leaky" protocol.
This module implements the alternative condition, in which selection is
confined to each outer training split:

    for each stratified outer fold:
        train split -> selector (train labels only) -> k features
                    -> fit classifier on train[features]
                    -> score on test[features]

Both conditions are deliberate experimental arms of protocol axis P3; this
module produces the nested arm.  The leaky arm is the existing results/*.csv
and is never overwritten.

Design constraints (see CLAUDE_CODE_PROMPT_V3.md sections 4 and 9):
  * the selector interface from feature_selectors.py is used unchanged;
  * the fold structure is byte-identical to the leaky runs
    (StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
     random_state=config.RANDOM_SEED)), so the two arms are paired by
     dataset and by fold;
  * per-fold selected feature sets are persisted, not aggregated away --
    Experiment 7 needs them for fold-to-fold stability;
  * metric aggregation matches exp1_performance.py: the mean over folds of
    the per-fold metric (not a pooled out-of-fold score).
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.metrics import (accuracy_score, f1_score,  # noqa: E402
                             roc_auc_score)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

warnings.filterwarnings("ignore")

SEED = getattr(config, "RANDOM_SEED", 42)
CV_FOLDS = getattr(config, "CV_FOLDS", 5)


# ─────────────────────────────────────────────────────────────────────────────
# classifiers -- exp1's XGBoost plus the two non-tree learners of exp5
# ─────────────────────────────────────────────────────────────────────────────

def make_xgb(n_classes: int):
    """Identical construction to exp1_performance._make_clf."""
    params = dict(config.XGB_PARAMS)
    if n_classes > 2:
        params["objective"] = "multi:softprob"
        params["num_class"] = n_classes
    return XGBClassifier(**params)


def make_logreg(n_classes: int):
    """L2 logistic regression; standardisation stays inside the fold."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])


def make_mlp(n_classes: int):
    """Single 64-unit hidden layer; standardisation stays inside the fold."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                              early_stopping=True, random_state=SEED)),
    ])


CLASSIFIERS: dict[str, Callable[[int], object]] = {
    "xgb": make_xgb,
    "logreg": make_logreg,
    "mlp": make_mlp,
}


# ─────────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────────

def _auc(y_true, proba, n_classes: int, classes) -> float:
    """AUC-ROC, binary or one-vs-rest macro, tolerant of absent classes."""
    if n_classes == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    # A test fold can be missing a class only for very small multiclass sets;
    # restrict the label set to what roc_auc_score can score.
    return float(roc_auc_score(y_true, proba, multi_class="ovr",
                               average="macro", labels=classes))


def _score_fold(clf, X_tr, y_tr, X_te, y_te, n_classes: int,
                classes) -> dict:
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    proba = clf.predict_proba(X_te)
    return {
        "accuracy": float(accuracy_score(y_te, pred)),
        "f1_macro": float(f1_score(y_te, pred, average="macro",
                                   zero_division=0)),
        "auc": _auc(y_te, proba, n_classes, classes),
    }


# ─────────────────────────────────────────────────────────────────────────────
# the harness
# ─────────────────────────────────────────────────────────────────────────────

def nested_evaluate(X: pd.DataFrame,
                    y: pd.Series,
                    selector_fn: Callable,
                    k: int,
                    classifier_fn,
                    n_splits: int = CV_FOLDS,
                    seed: int = SEED) -> dict:
    """
    Leakage-free evaluation of one (dataset, selector) pair.

    Parameters
    ----------
    X, y         : the full dataset, already sanitised by data_loader.
    selector_fn  : callable (X_train, y_train, k, random_state) -> list[str].
                   Use feature_selectors.run_selector via functools.partial,
                   or a bare select_* function for the native-threshold runs.
    k            : feature budget handed to the selector.  Native-threshold
                   selectors ignore it and return their own subset size.
    classifier_fn: either a callable (n_classes) -> estimator, or a mapping
                   name -> such a callable.  A mapping scores every classifier
                   on the *same* per-fold subsets, which is what protocol axis
                   P1 requires under nesting.
    n_splits, seed
                 : must match the leaky arm for the comparison to be paired.

    Returns
    -------
    dict with keys
        folds        : list of per-fold dicts (fold index, train/test sizes,
                       selected features, selection time, per-classifier
                       metrics)
        aggregated   : {classifier: {metric_mean, metric_std, ...}}
        feature_sets : list[list[str]], one per fold, in fold order
        n_selected   : list[int], per-fold subset size
        empty_folds  : list[int], folds where the selector returned nothing
        time_select_s: total selection seconds across folds
    """
    if callable(classifier_fn):
        clf_map: Mapping[str, Callable] = {"clf": classifier_fn}
    else:
        clf_map = dict(classifier_fn)

    y_arr = np.asarray(y)
    classes = np.unique(y_arr)
    n_classes = len(classes)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds: list[dict] = []
    feature_sets: list[list[str]] = []
    empty_folds: list[int] = []
    t_select_total = 0.0

    for fold_i, (tr, te) in enumerate(cv.split(X, y_arr)):
        X_tr_full, X_te_full = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y_arr[tr], y_arr[te]

        # ── selection sees the training split only ────────────────────────
        t0 = time.time()
        selected = list(selector_fn(X_tr_full, pd.Series(y_tr,
                                                         index=X_tr_full.index),
                                    k, seed))
        t_sel = time.time() - t0
        t_select_total += t_sel

        # de-duplicate, preserve order, keep only columns that exist
        seen, cols = set(), []
        for c in selected:
            if c in X.columns and c not in seen:
                seen.add(c)
                cols.append(c)

        feature_sets.append(cols)

        rec = {
            "fold": fold_i,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_selected": len(cols),
            "selected_features": "|".join(cols),
            "time_select_s": round(t_sel, 3),
        }

        if not cols:
            # An empty native-threshold selection is a reportable outcome,
            # not an error: record it and leave the metrics missing.
            empty_folds.append(fold_i)
            for name in clf_map:
                for m in ("accuracy", "f1_macro", "auc"):
                    rec[f"{name}_{m}"] = np.nan
            folds.append(rec)
            continue

        X_tr, X_te = X_tr_full[cols], X_te_full[cols]
        for name, factory in clf_map.items():
            scores = _score_fold(factory(n_classes), X_tr, y_tr, X_te, y_te,
                                 n_classes, classes)
            for m, v in scores.items():
                rec[f"{name}_{m}"] = v

        folds.append(rec)

    # ── aggregate: mean over folds, matching exp1's convention ────────────
    aggregated: dict[str, dict] = {}
    for name in clf_map:
        agg = {}
        for m in ("accuracy", "f1_macro", "auc"):
            vals = np.array([f[f"{name}_{m}"] for f in folds], dtype=float)
            vals = vals[~np.isnan(vals)]
            agg[f"{m}_mean"] = float(vals.mean()) if vals.size else np.nan
            agg[f"{m}_std"] = float(vals.std()) if vals.size else np.nan
            agg[f"{m}_n_folds"] = int(vals.size)
        aggregated[name] = agg

    return {
        "folds": folds,
        "aggregated": aggregated,
        "feature_sets": feature_sets,
        "n_selected": [len(s) for s in feature_sets],
        "empty_folds": empty_folds,
        "time_select_s": round(t_select_total, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# checkpoint / resume helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_done(path: Path, key_cols: list[str]) -> set:
    """Return the set of already-completed key tuples in an output CSV.

    The v2 project had a resume bug that silently dropped a row
    (hr-attrition x shap_select).  Guarding against a repeat, only rows whose
    status is 'ok' or 'empty' count as done; anything else is re-run.
    """
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].isin(["ok", "empty"])]
    return set(map(tuple, df[key_cols].astype(str).to_numpy()))


def append_rows(path: Path, rows: list[dict]) -> None:
    """Append rows to a CSV, writing the header only when the file is new."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)


def assert_row_count(path: Path, expected: int, label: str) -> None:
    """Fail loudly rather than silently shipping a short result file."""
    df = pd.read_csv(path)
    got = len(df.drop_duplicates())
    if got != expected:
        raise SystemExit(
            f"[{label}] row count {got} != expected {expected}. "
            f"The v2 resume bug dropped a row this way; refusing to continue. "
            f"File: {path}")
    print(f"[{label}] row count OK: {got}")


# ─────────────────────────────────────────────────────────────────────────────
# smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from functools import partial
    from data_loader import load_dataset
    import feature_selectors as fs

    print("=" * 72)
    print("nested_eval smoke test -- sonar, chi2 + shap_topk")
    print("=" * 72)

    X, y, meta = load_dataset("sonar")
    k = meta["top_k"]
    print(f"n={meta['n_samples']} p={meta['n_features']} k={k} "
          f"classes={meta['n_classes']}")

    leaky = pd.read_csv(config.RESULTS_DIR / "exp1_performance.csv")
    leaky = leaky[(leaky.dataset == "sonar") & (leaky.status == "ok")]

    for method in ("chi2", "shap_topk"):
        res = nested_evaluate(
            X, y,
            partial(fs.run_selector, method),
            k,
            CLASSIFIERS,
        )
        a = res["aggregated"]["xgb"]
        ref = leaky[leaky.method == method]
        leaky_auc = float(ref.auc_mean.iloc[0]) if len(ref) else float("nan")
        sizes = res["n_selected"]
        print(f"\n{method:12s} nested AUC = {a['auc_mean']:.4f} "
              f"+/- {a['auc_std']:.4f}   leaky AUC = {leaky_auc:.4f}   "
              f"optimism = {leaky_auc - a['auc_mean']:+.4f}")
        print(f"{'':12s} per-fold sizes {sizes}, "
              f"logreg AUC = {res['aggregated']['logreg']['auc_mean']:.4f}, "
              f"mlp AUC = {res['aggregated']['mlp']['auc_mean']:.4f}")
        jac = []
        for i in range(len(res['feature_sets'])):
            for j in range(i + 1, len(res['feature_sets'])):
                a_, b_ = set(res['feature_sets'][i]), set(res['feature_sets'][j])
                jac.append(len(a_ & b_) / len(a_ | b_))
        print(f"{'':12s} mean fold-to-fold Jaccard = {np.mean(jac):.4f}")

    print("\nSmoke test complete.")
