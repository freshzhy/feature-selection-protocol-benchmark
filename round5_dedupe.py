#!/usr/bin/env python3
"""
round5_dedupe.py -- collapse duplicate rows left by two overlapping Round 5
rerun campaigns.

What happened: the first `official_bridge/rerun_all.sh` (started 2026-09-20,
old bridge without a subprocess timeout) was never stopped. Its nested-Exp1
process sat on Madelon x SHAP-Select for ~10 days, *completed it*, appended
the remaining 7 datasets, then ran steps 3-7 again (mostly resuming / skipping).
Meanwhile the second campaign (2026-09-22, timeout bridge) had already
produced every row. Result: exact-duplicate rows in exp1_performance_nested,
exp5_classifier_nested, exp1_folds_nested and exp6_native_nested.

Rule: within each key, prefer a row with status == "ok" over an error row
(this keeps the one genuinely new result, the completed Madelon x SHAP-Select
nested run); among equal-status rows keep the first. Every duplicate group is
asserted numerically identical (time columns excepted) before it is dropped.
Idempotent. Backups: results/*.pre_dedupe_round5.csv.bak
"""
import shutil
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent / "results"
KEYS = {
    "exp1_performance_nested": ["dataset", "method"],
    "exp5_classifier_nested": ["dataset", "method", "classifier"],
    "exp1_folds_nested": ["dataset", "method", "fold"],
    "exp6_native_nested": ["dataset", "method"],
    "exp1_performance": ["dataset", "method"],
    "exp6_native": ["dataset", "method"],
    "exp2_stability": ["dataset", "method"],
    "exp3_noise": ["dataset", "method", "noise_ratio"],
    "exp4_transfer": ["source", "target", "method"],
    "exp4_transfer_nested": ["source", "target", "method"],
}
for name, key in KEYS.items():
    p = RES / f"{name}.csv"
    d = pd.read_csv(p)
    n0 = len(d)
    if not d.duplicated(key).any():
        print(f"{name:26s} {n0} rows, no duplicates")
        continue
    bak = RES / f"{name}.pre_dedupe_round5.csv.bak"
    if not bak.exists():
        shutil.copy(p, bak)
    status_ok = (d["status"] == "ok") if "status" in d else pd.Series(True, index=d.index)
    d["_ok"] = status_ok.astype(int)
    num = [c for c in d.select_dtypes("number").columns
           if "time" not in c and c not in key and c != "_ok"]
    for g, x in d[d.duplicated(key, keep=False)].groupby(key):
        x_ok = x[x._ok == 1]
        if len(x_ok) > 1:
            assert (x_ok[num].round(6).nunique(dropna=False) <= 1).all(), (name, g)
    d = (d.sort_values("_ok", ascending=False, kind="stable")
           .drop_duplicates(key, keep="first")
           .sort_index().drop(columns="_ok"))
    d.to_csv(p, index=False)
    print(f"{name:26s} {n0} -> {len(d)} rows")
