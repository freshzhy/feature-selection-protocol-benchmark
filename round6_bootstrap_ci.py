#!/usr/bin/env python3
"""
round6_bootstrap_ci.py -- bootstrap confidence intervals for the three
protocol-axis Spearman correlations (P1, P2, P3), for their pairwise
differences, and a same-protocol noise baseline.

Resampling unit: the dataset (21 datasets, drawn with replacement).  For each
draw every method's mean AUC is recomputed over the resampled datasets
(multiplicities respected), ranked, and correlated with the reference ranking
exactly as experiments/exp7_protocol.py does.  No selection is re-run.

Definitions mirror exp7_protocol.py:
  reference = nested XGBoost mean AUC per method
  P1 = rho(reference, nested logreg | mlp)   [exp5_classifier_nested.csv]
  P2 = rho(reference, native-threshold AUC)  [exp6_native_nested.csv, status ok;
        only the 3 adaptive methods change; per-method mean over datasets
        where the native run succeeded]
  P3 = rho(leaky, nested)                    [exp1_performance*.csv]
Extra P2 variants from round5_axis_sensitivity.py (A: Madelon dropped, empty
native set scored 0.5; C: datasets where all 3 native runs succeeded).

Noise baseline: split the 21 datasets at random into 10 + 11, rank the nine
methods under the NESTED protocol on each half, and take rho between halves.
This is how far two rankings of the same nine methods under one protocol
drift apart purely because of which datasets were used.

Output: results/round6_bootstrap_ci.json, results/round6_bootstrap_ci.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

SEED, B = 42, 10000
RES = Path(__file__).resolve().parent / "results"
M = ["rfe", "lasso", "tree_importance", "chi2", "mi",
     "shap_topk", "borutashap", "powershap", "shap_select"]
AD = ["borutashap", "powershap", "shap_select"]

lk = pd.read_csv(RES / "exp1_performance.csv"); lk = lk[lk.status == "ok"]
ns = pd.read_csv(RES / "exp1_performance_nested.csv"); ns = ns[ns.status == "ok"]
e5 = pd.read_csv(RES / "exp5_classifier_nested.csv"); e5 = e5[e5.status == "ok"]
e6 = pd.read_csv(RES / "exp6_native_nested.csv")

DS = sorted(ns.dataset.unique())
assert len(DS) == 21
D = {d: i for i, d in enumerate(DS)}


def mat(df, col, extra=None):
    """datasets x methods matrix (NaN where missing)."""
    out = np.full((len(DS), len(M)), np.nan)
    for r in df.itertuples():
        out[D[r.dataset], M.index(r.method)] = getattr(r, col)
    return out


X_ns = mat(ns, "auc_mean")
X_lk = mat(lk, "auc_mean")
X_lr = mat(e5[e5.classifier == "logreg"], "auc")
X_ml = mat(e5[e5.classifier == "mlp"], "auc")
# P2 main: native AUC where status ok, only for adaptive methods
X_nat = np.full_like(X_ns, np.nan)
for r in e6[e6.status == "ok"].itertuples():
    X_nat[D[r.dataset], M.index(r.method)] = r.auc_native
# P2 variant A: empty native set = 0.5, Madelon excluded
X_natA = X_nat.copy()
for r in e6[e6.status == "empty"].itertuples():
    X_natA[D[r.dataset], M.index(r.method)] = 0.5
X_natA[D["madelon"], :] = np.nan
mask_A = np.array([d != "madelon" for d in DS])
# P2 variant C: datasets where all three native runs succeeded
ok3 = e6[e6.status == "ok"].groupby("dataset").method.nunique()
mask_C = np.array([ok3.get(d, 0) == 3 for d in DS])
for X in (X_ns, X_lk, X_lr, X_ml):
    assert not np.isnan(X).any()


def wmean(X, w):
    """weighted column means, ignoring NaN cells (weights renormalised)."""
    ok = ~np.isnan(X)
    num = np.nansum(X * w[:, None], axis=0)
    den = (ok * w[:, None]).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def rk(v):
    return rankdata(-v)


def rho(a, b):
    return spearmanr(a, b).statistic


def axes(w, wA=None, wC=None):
    ref = wmean(X_ns, w)
    r0 = rk(ref)
    out = {"P1_logreg": rho(r0, rk(wmean(X_lr, w))),
           "P1_mlp": rho(r0, rk(wmean(X_ml, w))),
           "P3": rho(rk(wmean(X_lk, w)), r0)}
    nat = ref.copy()
    nm = wmean(X_nat, w)
    for m in AD:
        i = M.index(m)
        nat[i] = nm[i]
    out["P2"] = rho(r0, rk(nat)) if not np.isnan(nat).any() else np.nan
    # variants
    for tag, X, mask in (("P2_A", X_natA, mask_A), ("P2_C", X_nat, mask_C)):
        ww = w * mask
        if ww.sum() == 0:
            out[tag] = np.nan
            continue
        ref_v = wmean(X_ns, ww)
        r0v = rk(ref_v)
        nat_v = ref_v.copy()
        nmv = wmean(X, ww)
        for m in AD:
            i = M.index(m)
            nat_v[i] = nmv[i]
        out[tag] = rho(r0v, rk(nat_v)) if not np.isnan(nat_v).any() else np.nan
    return out


rng = np.random.default_rng(SEED)
point = axes(np.ones(len(DS)))
draws = []
for _ in range(B):
    w = np.bincount(rng.integers(0, len(DS), len(DS)), minlength=len(DS)).astype(float)
    draws.append(axes(w))
bs = pd.DataFrame(draws)
n_invalid = bs.isna().sum().to_dict()
bs["P3_minus_P2"] = bs.P3 - bs.P2
bs["P3_minus_P1_logreg"] = bs.P3 - bs.P1_logreg
bs["P3_minus_P1_mlp"] = bs.P3 - bs.P1_mlp
bs["P2_minus_P1_logreg"] = bs.P2 - bs.P1_logreg
bs["P2_minus_P1_mlp"] = bs.P2 - bs.P1_mlp
bs["P1_mlp_minus_P1_logreg"] = bs.P1_mlp - bs.P1_logreg


def ci(s):
    s = s.dropna()
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


rows = []
for c in bs.columns:
    pt = point.get(c)
    if pt is None:  # differences: point estimate from the two points
        a, b = c.split("_minus_")
        key = lambda s: {"P1_logreg": "P1_logreg", "P1_mlp": "P1_mlp", "P2": "P2", "P3": "P3"}[s]
        pt = point[key(a)] - point[key(b)]
    lo, hi = ci(bs[c])
    rows.append({"quantity": c, "point": pt, "ci_lo": lo, "ci_hi": hi,
                 "boot_mean": float(bs[c].mean()),
                 "frac_gt0": float((bs[c].dropna() > 0).mean()),
                 "n_valid": int(bs[c].notna().sum())})
res = pd.DataFrame(rows)

# ---- same-protocol noise baseline: nested ranking, random 10 / 11 split ----
rng2 = np.random.default_rng(SEED + 1)
half = []
for _ in range(B):
    perm = rng2.permutation(len(DS))
    a, b = np.zeros(len(DS)), np.zeros(len(DS))
    a[perm[:10]] = 1
    b[perm[10:]] = 1
    half.append(rho(rk(wmean(X_ns, a)), rk(wmean(X_ns, b))))
half = np.array(half)
# same split, leaky vs nested on the SAME datasets (P3 on half the data)
p3_half = []
rng3 = np.random.default_rng(SEED + 2)
for _ in range(B):
    perm = rng3.permutation(len(DS))
    a = np.zeros(len(DS)); a[perm[:10]] = 1
    p3_half.append(rho(rk(wmean(X_lk, a)), rk(wmean(X_ns, a))))
p3_half = np.array(p3_half)

noise = {
    "split_half_nested_rho_mean": float(half.mean()),
    "split_half_nested_rho_median": float(np.median(half)),
    "split_half_nested_rho_ci": [float(np.percentile(half, 2.5)), float(np.percentile(half, 97.5))],
    "p3_on_random_10_datasets_mean": float(p3_half.mean()),
    "p3_on_random_10_datasets_ci": [float(np.percentile(p3_half, 2.5)), float(np.percentile(p3_half, 97.5))],
}

# ---- implementation fidelity (Section 9.6): reimplementation -> official ----
oldn = pd.read_csv(RES / "pre_round5_backup" / "exp1_performance_nested.csv")
oldn = oldn[oldn.status == "ok"].drop_duplicates(["dataset", "method"])
assert set(oldn.dataset) == set(DS), "old nested substrate differs"
X_old = mat(oldn, "auc_mean")
assert not np.isnan(X_old).any()
fid_point = rho(rk(wmean(X_old, np.ones(len(DS)))), rk(wmean(X_ns, np.ones(len(DS)))))
rng4 = np.random.default_rng(SEED + 3)
fid = []
for _ in range(B):
    w = np.bincount(rng4.integers(0, len(DS), len(DS)), minlength=len(DS)).astype(float)
    fid.append(rho(rk(wmean(X_old, w)), rk(wmean(X_ns, w))))
fid = np.array(fid)
noise["fidelity_nested_rho"] = float(fid_point)
noise["fidelity_nested_ci"] = [float(np.percentile(fid, 2.5)), float(np.percentile(fid, 97.5))]

# sanity: point estimates must equal exp7_headline.json
H = json.load(open(RES / "exp7_headline.json"))
chk = {"P3": H["rho_p3"], "P2": H["rho_p2"],
       "P1_logreg": H["rho_p1"]["logreg"], "P1_mlp": H["rho_p1"]["mlp"]}
for k, v in chk.items():
    assert abs(point[k] - v) < 1e-9, (k, point[k], v)

out = {"seed": SEED, "B": B, "n_datasets": len(DS), "n_invalid_draws": n_invalid,
       "point": {k: float(v) for k, v in point.items()},
       "table": rows, "noise_baseline": noise,
       "variant_C_datasets": [d for d, m in zip(DS, mask_C) if m]}
(RES / "round6_bootstrap_ci.json").write_text(json.dumps(out, indent=2))
res.to_csv(RES / "round6_bootstrap_ci.csv", index=False)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("invalid draws:", n_invalid)
print(json.dumps(noise, indent=2))
