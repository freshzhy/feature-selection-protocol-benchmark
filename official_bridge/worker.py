#!/usr/bin/env python3
"""
official_bridge/worker.py -- runs inside ONE of the three isolated venvs
(venv_powershap / venv_shapselect / venv_boruta) and performs exactly one
selection call using the official library. Talks to the parent process via
two JSON/parquet files so it never needs the parent's package versions.

Usage (invoked by official_bridge/bridge.py, not by hand):
    <venv>/bin/python worker.py <lib> <X.parquet> <y.parquet> <k> <seed> <native:0|1> <out.json>
"""
import sys, json, warnings, traceback
warnings.filterwarnings("ignore")

def main():
    lib, x_path, y_path, k, seed, native, out_path = sys.argv[1:8]
    k, seed, native = int(k), int(seed), bool(int(native))
    import pandas as pd, numpy as np
    X = pd.read_csv(x_path)
    y = pd.read_csv(y_path).iloc[:, 0]
    n_classes = int(y.nunique())
    feature_names = list(X.columns)

    def get_xgb(seed_, n_classes_):
        from xgboost import XGBClassifier
        p = dict(n_estimators=200, max_depth=6, learning_rate=0.1,
                 random_state=seed_, n_jobs=-1, verbosity=0, missing=float("nan"))
        if n_classes_ > 2:
            p["objective"] = "multi:softprob"; p["num_class"] = n_classes_
            p["eval_metric"] = "mlogloss"
        else:
            p["eval_metric"] = "logloss"
        return XGBClassifier(**p)

    result = {"ok": True, "selected": [], "ranked": [], "n_native": None}
    try:
        if lib == "shapselect":
            from sklearn.model_selection import train_test_split
            from shap_select import shap_select
            strat = y if y.value_counts().min() >= 2 else None
            Xtr, Xva, ytr, yva = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=strat)
            clf = get_xgb(seed, n_classes); clf.fit(Xtr, ytr)
            task = "multiclass" if n_classes > 2 else "binary"
            va = Xva.copy(); va["__target__"] = yva.values
            res = shap_select(clf, va, "__target__", feature_names, task=task, threshold=0.05)
            sig = res[res["selected"] == 1].sort_values("t-value", ascending=False)
            result["selected"] = sig["feature name"].tolist()
            result["ranked"] = res.sort_values("t-value", ascending=False)["feature name"].tolist()
            result["n_native"] = len(result["selected"])

        elif lib == "powershap":
            from powershap import PowerShap
            ps = PowerShap(model=get_xgb(seed, n_classes), automatic=True,
                           show_progress=False, verbose=False)
            ps.fit(X, y)
            df = ps._processed_shaps_df
            support = np.asarray(ps.get_support())
            ranked = df.sort_values("impact", ascending=False).index.tolist()
            selected = [f for f in ranked if f in set(np.asarray(feature_names)[support])]
            result["selected"] = selected
            result["ranked"] = [f for f in ranked if f in feature_names]
            result["n_native"] = len(selected)

        else:  # boruta
            from BorutaShap import BorutaShap
            bs = BorutaShap(model=get_xgb(seed, n_classes), importance_measure="shap",
                            classification=True, percentile=100, pvalue=0.05)
            strat = y if n_classes > 1 else None
            bs.fit(X=X, y=y, n_trials=30, random_state=seed, verbose=False,
                   sample=False, stratify=strat)
            accepted = list(bs.accepted)
            # secondary ranking: mean recorded importance across trials (higher = better)
            try:
                means = bs.history_x[accepted].mean(axis=0, skipna=True)
                ranked_accept = means.sort_values(ascending=False).index.tolist()
            except Exception:
                ranked_accept = accepted
            result["selected"] = ranked_accept
            result["ranked"] = ranked_accept
            result["n_native"] = len(accepted)

    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()}

    json.dump(result, open(out_path, "w"))

if __name__ == "__main__":
    main()
