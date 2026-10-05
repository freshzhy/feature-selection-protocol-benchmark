#!/usr/bin/env python3
"""
round5_axis_sensitivity.py -- the three protocol axes recomputed on matched
dataset sets, and P2 under three rules for native-threshold failures.

Why: after the official-library rerun, the three adaptive selectors fail on a
few datasets under their native thresholds (PowerSHAP returns an empty set on
blood-transfusion and kc2 in every fold; SHAP-Select times out on Madelon).
exp7_protocol.py averages each method over whatever datasets it succeeded on,
so the nine per-method means in P2 are taken over different dataset sets. This
script makes every axis comparable by fixing the dataset set first.

Output: results/round5_axis_sensitivity.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

RES = Path(__file__).resolve().parent / "results"
M = ["rfe", "lasso", "tree_importance", "chi2", "mi",
     "shap_topk", "borutashap", "powershap", "shap_select"]
AD = ["borutashap", "powershap", "shap_select"]

lk = pd.read_csv(RES / "exp1_performance.csv"); lk = lk[lk.status == "ok"]
ns = pd.read_csv(RES / "exp1_performance_nested.csv"); ns = ns[ns.status == "ok"]
e5 = pd.read_csv(RES / "exp5_classifier_nested.csv"); e5 = e5[e5.status == "ok"]
e6 = pd.read_csv(RES / "exp6_native_nested.csv")

def rk(s): return pd.Series(rankdata(-s.to_numpy()), index=s.index)
def mean(df, ds, col): return df[df.dataset.isin(ds)].groupby("method")[col].mean()[M]

def axes(ds, empty_rule):
    ref = mean(ns, ds, "auc_mean"); r0 = rk(ref)
    out = {}
    for clf in ("logreg", "mlp"):
        r = rk(e5[(e5.classifier == clf) & e5.dataset.isin(ds)]
               .groupby("method").auc.mean()[M])
        out[f"P1_{clf}"] = (spearmanr(r0, r).statistic, (r0 - r).abs().max())
    nat = ref.copy()
    for m in AD:
        g = e6[(e6.method == m) & e6.dataset.isin(ds)].copy()
        if empty_rule == "chance":
            g.loc[g.status == "empty", "auc_native"] = 0.5
        nat[m] = g.auc_native.dropna().mean()
    r = rk(nat)
    out["P2"] = (spearmanr(r0, r).statistic, (r0 - r).abs().max())
    rl = rk(mean(lk, ds, "auc_mean"))
    out["P3"] = (spearmanr(rl, r0).statistic, (rl - r0).abs().max())
    return out

all21 = sorted(ns.dataset.unique())
assert len(all21) == 21 and ns.groupby("method").size().eq(21).all()
no_mad = [d for d in all21 if d != "madelon"]
ok3 = e6[e6.status == "ok"].groupby("dataset").method.nunique()
complete = sorted(ok3[ok3 == 3].index)
SETS = [
    ("A: 20 datasets (Madelon dropped: SHAP-Select native timeout); empty native set = AUC 0.5", no_mad, "chance"),
    ("B: 20 datasets; empty native sets dropped per method", no_mad, "drop"),
    (f"C: {len(complete)} datasets where all three native runs succeed", complete, "drop"),
]
rows = []
for label, ds, rule in SETS:
    for axis, (rho, mx) in axes(ds, rule).items():
        rows.append({"variant": label, "n_datasets": len(ds), "axis": axis,
                     "spearman_rho": round(float(rho), 4), "max_rank_change": float(mx)})
df = pd.DataFrame(rows)
df.to_csv(RES / "round5_axis_sensitivity.csv", index=False)
print(df.pivot(index="axis", columns="variant", values="spearman_rho").to_string())
print("complete-set datasets excluded:", sorted(set(all21) - set(complete)))
