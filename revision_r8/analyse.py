#!/usr/bin/env python3
"""
Analyse the multi-seed runs.  For each seed, only datasets on which all nine methods
completed both arms are used (complete case across methods and evaluators).
Reports per seed: nested leader, top three, top six, and Spearman rho of P1-logistic,
P1-MLP and P3 against that seed's nested-XGBoost ordering; plus the range and median
over seeds, and a check that seed 42 reproduces the primary results.
Writes results/multiseed/multiseed_summary.csv and multiseed_summary.md.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ms_io import read_runs

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "results" / "multiseed"
METHODS = ["shap_topk", "borutashap", "powershap", "shap_select", "mi", "chi2", "rfe", "lasso", "tree_importance"]
d = pd.concat([read_runs(f) for f in sorted((MS / "runs").glob("*.csv"))], ignore_index=True)
d = d.drop_duplicates(["seed", "dataset", "method"], keep="last")
n_env = int(d.status.astype(str).str.contains("venv for", regex=False).sum())
if n_env:
    sys.exit(f"{n_env} rows failed because a library environment was missing; run revision_r8/fix_bridge_venvs.sh and revision_r8/run_parallel.sh first")
missing = 9 * 5 * 8 - len(d)
if missing > 0:
    print(f"WARNING: {missing} of 360 rows not yet run; results below are incomplete")
rows, orders = [], {}
for s, g in d.groupby("seed"):
    ok = g[g.status == "ok"]
    cnt = ok.groupby("dataset").method.nunique()
    ds = sorted(cnt[cnt == len(METHODS)].index)
    ok = ok[ok.dataset.isin(ds)]
    m = {c: ok.groupby("method")[c].mean()[METHODS] for c in ("nested_xgb_auc", "nested_logreg_auc", "nested_mlp_auc", "leaky_xgb_auc")}
    ref = rankdata(-m["nested_xgb_auc"].to_numpy())
    r = lambda c: float(spearmanr(ref, rankdata(-m[c].to_numpy())).statistic)
    order = list(m["nested_xgb_auc"].sort_values(ascending=False).index)
    orders[s] = order
    rows.append({"seed": int(s), "datasets_complete": len(ds), "failed_cells": int((g.status != "ok").sum()),
                 "nested_leader": order[0], "top3": " > ".join(order[:3]), "top6": " > ".join(order[:6]),
                 "leaky_leader": m["leaky_xgb_auc"].idxmax(),
                 "rho_P1_logreg": r("nested_logreg_auc"), "rho_P1_mlp": r("nested_mlp_auc"), "rho_P3": r("leaky_xgb_auc"),
                 "rfe_rank_leaky": int(rankdata(-m["leaky_xgb_auc"].to_numpy())[METHODS.index("rfe")]),
                 "rfe_rank_nested": int(ref[METHODS.index("rfe")])})
res = pd.DataFrame(rows)
summ = {c: (res[c].min(), res[c].median(), res[c].max()) for c in ("rho_P1_logreg", "rho_P1_mlp", "rho_P3")}
pd.set_option("display.width", 250)
print(res.to_string(index=False))
print("\nmin / median / max over seeds:", {k: tuple(round(float(x), 2) for x in v) for k, v in summ.items()})

# seed-42 reproduction check against the primary result files
chk = ""
try:
    n0 = pd.read_csv(ROOT / "results" / "exp1_performance_nested.csv"); n0 = n0[n0.status == "ok"].drop_duplicates(["dataset", "method"])
    l0 = pd.read_csv(ROOT / "results" / "exp1_performance.csv"); l0 = l0[l0.status == "ok"].drop_duplicates(["dataset", "method"])
    s42 = d[(d.seed == 42) & (d.status == "ok")].set_index(["dataset", "method"])
    a = (s42.nested_xgb_auc - n0.set_index(["dataset", "method"]).auc_mean.reindex(s42.index)).abs()
    b = (s42.leaky_xgb_auc - l0.set_index(["dataset", "method"]).auc_mean.reindex(s42.index)).abs()
    chk = f"seed 42 vs primary results: {len(a)} cells, max |diff| nested {a.max():.2e}, leaky {b.max():.2e}"
    print("\n" + chk)
except FileNotFoundError:
    pass
for _f in ("multiseed_summary.csv", "multiseed_summary.md"):
    (MS / _f).unlink(missing_ok=True)   # replace, even if an extracted copy is read-only
res.to_csv(MS / "multiseed_summary.csv", index=False)
with open(MS / "multiseed_summary.md", "w") as f:
    try:
        f.write(res.to_markdown(index=False))
    except ImportError:
        f.write(res.to_string(index=False))
    f.write("\n\nmin / median / max over seeds: " + str({k: tuple(round(float(x), 2) for x in v) for k, v in summ.items()}) + "\n" + chk + "\n")
print(f"\nwritten: {MS / 'multiseed_summary.csv'}")
