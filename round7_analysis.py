#!/usr/bin/env python3
"""
round7_analysis.py -- analyses requested by the round-7 (simulated PLOS ONE)
review that need no new feature-selection runs.  Everything is recomputed from
results/*.csv.  Output: results/round7_numbers.json and round7_*.csv.

C1  transfer: ΔF1 against the TARGET-domain baseline.  For pair A->B the
    in-domain score is the nested score of B's own (training-fold-selected)
    subsets on B, and the cross-domain score is A's fixed subset fitted on the
    same B training folds and scored on the same B test folds: both arms use
    StratifiedKFold(5, shuffle=True, random_state=42) on the same y_B, so the
    folds are identical (verified in the code: nested_eval.nested_evaluate and
    exp4_transfer._cv_score).  The run_nested_exp4 output stores that score
    per (source, method); for target B it is the row whose source is B.
C2  P2 on a common support: a dataset enters only if all three adaptive
    methods completed all five folds with a non-empty selection; reference and
    native conditions are both restricted to that set.  Coverage table and two
    sensitivity conventions.
M4  rank aggregation: mean within-dataset rank instead of mean AUC.
M5  Madelon: every axis recomputed on the 20 datasets without Madelon.
M8  balanced two-way sums-of-squares shares (dataset, method, residual).
M9  Holm-adjusted p for the per-method Wilcoxon families.
m7  stability measures compared on the nine method means.
M7  composite ranks with the corrected transfer column, two missing rules.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

RES = Path(__file__).resolve().parent / "results"
M = ["rfe", "lasso", "tree_importance", "chi2", "mi", "shap_topk", "borutashap", "powershap", "shap_select"]
AD = ["borutashap", "powershap", "shap_select"]
B, SEED = 10000, 42
OUT = {}

ns = pd.read_csv(RES / "exp1_performance_nested.csv"); ns = ns[ns.status == "ok"]
lk = pd.read_csv(RES / "exp1_performance.csv"); lk = lk[lk.status == "ok"]
e5 = pd.read_csv(RES / "exp5_classifier_nested.csv"); e5 = e5[e5.status == "ok"]
e6 = pd.read_csv(RES / "exp6_native_nested.csv")
DS = sorted(ns.dataset.unique()); assert len(DS) == 21


def piv(df, col):
    return df.pivot_table(index="dataset", columns="method", values=col).reindex(index=DS, columns=M)


X_ns, X_lk = piv(ns, "auc_mean"), piv(lk, "auc_mean")
X_lr, X_ml = piv(e5[e5.classifier == "logreg"], "auc"), piv(e5[e5.classifier == "mlp"], "auc")


def rk(v): return rankdata(-np.asarray(v, float))
def rho(a, b): return float(spearmanr(a, b).statistic)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


# ============================================================ C1 transfer
tr = pd.read_csv(RES / "exp4_transfer_nested.csv"); tr = tr[tr.status == "ok"].copy()
ind_n = tr.groupby(["source", "method"]).indomain_f1.first()
ind_l = tr.groupby(["source", "method"]).indomain_f1_leaky.first()
tr["d_tgt"] = [ind_n[(t, m)] - c for t, m, c in zip(tr.target, tr.method, tr.cross_f1)]
tr["d_tgt_leaky"] = [ind_l[(t, m)] - c for t, m, c in zip(tr.target, tr.method, tr.cross_f1)]
tab5 = tr.groupby("method").agg(jaccard=("feature_jaccard", "mean"), cross_f1=("cross_f1", "mean"),
                                d_signed=("d_tgt", "mean"), d_abs=("d_tgt", lambda s: s.abs().mean()),
                                d_abs_leaky=("d_tgt_leaky", lambda s: s.abs().mean()),
                                d_abs_srcbase=("delta_f1", lambda s: s.abs().mean()),
                                d_max=("d_tgt", lambda s: s.abs().max()), n=("d_tgt", "size"))
tab5 = tab5.sort_values("d_abs", kind="mergesort")
tab5.to_csv(RES / "round7_transfer.csv")
tr[["source", "target", "method", "feature_jaccard", "cross_f1", "d_tgt", "d_tgt_leaky", "delta_f1"]].to_csv(
    RES / "round7_transfer_pairs.csv", index=False)
ind_min = float(tr.groupby("source").indomain_f1.min().min())
# a pair is an improvement when the transferred subset beats the target's own (d < 0)
OUT["transfer"] = {
    "table": tab5.round(6).to_dict(orient="index"),
    "order": list(tab5.index),
    "n_pairs_positive": int((tr.d_tgt > 0).sum()), "n_pairs": int(len(tr)),
    "max_change_leaky_vs_nested": float((tab5.d_abs - tab5.d_abs_leaky).abs().max()),
    "spread_top6": float(tab5.d_abs.iloc[5] - tab5.d_abs.iloc[0]),
    "max_abs_pair": float(tr.d_tgt.abs().max()),
    "indomain_f1_min": ind_min,
    "frac_abs_below_001": float((tr.d_tgt.abs() < 0.01).mean()),
}
# per-target-dataset view (descriptive; pairs share datasets)
OUT["transfer"]["by_target"] = tr.groupby("target").d_tgt.apply(lambda s: s.abs().mean()).round(4).to_dict()

# ============================================================ C2 P2 coverage and common support
e6["sizes"] = e6.fold_sizes.fillna("").astype(str)
def complete(r):
    if r.status not in ("ok", "empty") or not r.sizes:
        return False
    s = [int(x) for x in r.sizes.split("|")]
    return len(s) == 5 and min(s) > 0
e6["complete"] = e6.apply(complete, axis=1)
cov = []
for m in AD:
    g = e6[e6.method == m].set_index("dataset")
    timeouts = [d for d in DS if d not in g.index or g.loc[d, "status"] not in ("ok", "empty")]
    allempty = [d for d in g.index if g.loc[d, "status"] == "empty"]
    partial = [d for d in g.index if g.loc[d, "status"] == "ok" and not g.loc[d, "complete"]]
    cov.append({"method": m, "complete": int(g.complete.sum()), "partial_empty": partial,
                "all_empty": allempty, "not_finished": timeouts,
                "empty_folds": int(g.n_empty_folds.fillna(0).sum()),
                "folds_attempted": int(5 * (21 - len(timeouts)))})
OUT["p2_coverage"] = cov
common = sorted(set.intersection(*[set(e6[(e6.method == m) & e6.complete].dataset) for m in AD]))
OUT["p2_common_datasets"] = common
X_nat = piv(e6[e6.complete], "auc_native")


def p2_rho(ds, nat_matrix):
    ref = X_ns.loc[ds].mean()
    nat = ref.copy()
    for m in AD: nat[m] = nat_matrix.loc[ds, m].mean()
    return rho(rk(ref), rk(nat)), rk(ref), rk(nat), ref, nat


r_c, rr, rn, ref_c, nat_c = p2_rho(common, X_nat)
# fold-weighted empty = 0.5 on the datasets where every native run finished
fin = sorted(set.intersection(*[set(e6[(e6.method == m) & e6.status.isin(["ok", "empty"])].dataset) for m in AD]))
X_half = X_ns.copy() * np.nan
for r in e6[e6.status.isin(["ok", "empty"])].itertuples():
    s = [int(x) for x in str(r.fold_sizes).split("|")]
    n_emp = sum(1 for x in s if x == 0); n_sc = 5 - n_emp
    a = (0 if n_sc == 0 else r.auc_native * n_sc)
    X_half.loc[r.dataset, r.method] = (a + 0.5 * n_emp) / 5
r_h, _, rn_h, _, _ = p2_rho(fin, X_half)
# old method-specific available cases (as in round 6)
from round6_bootstrap_ci import point as R6  # noqa: E402  (re-runs the round-6 script: deterministic)
OUT["p2"] = {"common_n": len(common), "rho_common": r_c,
             "ranks_ref_common": dict(zip(M, rr.tolist())), "ranks_native_common": dict(zip(M, rn.tolist())),
             "max_change_common": float(np.max(np.abs(rr - rn))),
             "auc_ref_common": ref_c.round(4).to_dict(), "auc_native_common": nat_c.round(4).to_dict(),
             "finished_n": len(fin), "rho_half": r_h, "ranks_native_half": dict(zip(M, rn_h.tolist())),
             "rho_available_case": float(R6["P2"])}
# per-method paired tests on the common support
rows = []
for m in AD:
    a, b = X_ns.loc[common, m], X_nat.loc[common, m]
    w = wilcoxon(b, a)
    rows.append({"method": m, "auc_fixed": a.mean(), "auc_native": b.mean(), "delta": (b - a).mean(),
                 "p": float(w.pvalue),
                 "native_size": float(e6[(e6.method == m) & e6.dataset.isin(common)].n_native_mean.mean())})
p2t = pd.DataFrame(rows); p2t["p_holm"] = holm(p2t.p)
OUT["p2_table"] = p2t.round(6).to_dict(orient="records")

# bootstrap CI for the common-support P2 rho (datasets resampled within the common set)
rng = np.random.default_rng(SEED)
idx = np.array([DS.index(d) for d in common])
bs = []
for _ in range(B):
    w = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
    ref = (X_ns.iloc[idx].to_numpy() * w[:, None]).sum(0) / w.sum()
    nat = ref.copy()
    for m in AD:
        j = M.index(m); nat[j] = (X_nat.iloc[idx][m].to_numpy() * w).sum() / w.sum()
    bs.append(rho(rk(ref), rk(nat)))
OUT["p2"]["ci_common"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

# ============================================================ M4 aggregation sensitivity
def avg_rank_order(X):            # higher = better; mean within-dataset rank
    R = X.rank(axis=1, ascending=False)
    return -R.mean()


def axis_pair(Xref, Xc, ds, agg):
    if agg == "mean":
        return rho(rk(Xref.loc[ds].mean()), rk(Xc.loc[ds].mean()))
    return rho(rk(avg_rank_order(Xref.loc[ds])), rk(avg_rank_order(Xc.loc[ds])))


agg = {}
for a in ("mean", "rank"):
    X_nat_full = X_ns.copy()
    for m in AD: X_nat_full[m] = X_nat[m]
    agg[a] = {"P3": axis_pair(X_ns, X_lk, DS, a), "P1_logreg": axis_pair(X_ns, X_lr, DS, a),
              "P1_mlp": axis_pair(X_ns, X_ml, DS, a), "P2_common": axis_pair(X_ns, X_nat_full, common, a)}
OUT["aggregation"] = agg

# ============================================================ M5 Madelon sensitivity
nm = [d for d in DS if d != "madelon"]
order20 = list(X_ns.loc[nm].mean().sort_values(ascending=False).index)
OUT["madelon_excluded"] = {"P3": axis_pair(X_ns, X_lk, nm, "mean"), "P1_logreg": axis_pair(X_ns, X_lr, nm, "mean"),
                           "P1_mlp": axis_pair(X_ns, X_ml, nm, "mean"), "nested_order_20": order20,
                           "nested_order_21": list(X_ns.mean().sort_values(ascending=False).index),
                           "nested_auc_20": X_ns.loc[nm].mean().round(4).to_dict()}

# ============================================================ M8 variance shares
def ss_shares(X):
    Y = X.to_numpy(); g = Y.mean()
    ss_t = ((Y - g) ** 2).sum()
    ss_d = Y.shape[1] * ((Y.mean(1) - g) ** 2).sum()
    ss_m = Y.shape[0] * ((Y.mean(0) - g) ** 2).sum()
    return {"dataset": ss_d / ss_t, "method": ss_m / ss_t, "residual": (ss_t - ss_d - ss_m) / ss_t}


OUT["variance"] = {"nested": ss_shares(X_ns), "leaky": ss_shares(X_lk)}

# ============================================================ M9 Holm for Table 8
summ = pd.read_csv(RES / "exp7_method_summary.csv", index_col=0)
summ["p_holm"] = holm(summ.wilcoxon_p)
OUT["p3_holm"] = summ[["optimism_mean", "wilcoxon_p", "p_holm"]].round(6).to_dict(orient="index")

# ============================================================ m7 stability measures on 9 means
st = pd.read_csv(RES / "exp7_stability_by_method.csv", index_col=0)
OUT["stability_two_measures_method_level"] = {"rho": rho(st.fold_jaccard, st.bootstrap_jaccard), "n": len(st)}

# ============================================================ M7 composite with corrected transfer
perf = X_ns.mean()
e2 = pd.read_csv(RES / "exp2_stability.csv")
pv = e2.pivot(index="dataset", columns="method", values="jaccard_mean").dropna()
stab = pv.mean()
e3 = pd.read_csv(RES / "exp3_noise.csv"); e3 = e3[(e3.status == "ok") & (e3.dataset != "madelon")]
fdr50 = e3[e3.noise_ratio == 0.5].groupby("method").fdr.mean()
trn = tab5.d_abs
comp = pd.DataFrame({"performance": rankdata(-perf[M]), "stability": rankdata(-stab[M]),
                     "noise": rankdata(fdr50[M]),
                     "transfer": pd.Series(rankdata(trn.reindex([m for m in M if m != "shap_select"])),
                                           index=[m for m in M if m != "shap_select"]).reindex(M)}, index=M)
comp_last = comp.copy(); comp_last.loc["shap_select", "transfer"] = 9
comp["mean_avail"] = comp[["performance", "stability", "noise", "transfer"]].mean(axis=1)
comp["mean_last"] = comp_last.mean(axis=1)
comp = comp.assign(_m=comp.index).sort_values(["mean_last", "_m"], kind="mergesort").drop(columns="_m")  # stable tie-break
comp.to_csv(RES / "round7_composite.csv")
OUT["composite"] = comp.round(3).to_dict(orient="index")
# Pareto dominance (higher perf/stab better, lower fdr/transfer better)
vals = pd.DataFrame({"p": perf[M], "s": stab[M], "n": -fdr50[M], "t": -trn.reindex(M)})
dom = {}
for a in M:
    for b in M:
        if a == b: continue
        va, vb = vals.loc[a], vals.loc[b]
        common_cols = [c for c in vals.columns if not (np.isnan(va[c]) or np.isnan(vb[c]))]
        if all(va[c] >= vb[c] for c in common_cols) and any(va[c] > vb[c] for c in common_cols):
            dom.setdefault(b, []).append(a)
OUT["pareto_dominated_by"] = dom

(RES / "round7_numbers.json").write_text(json.dumps(OUT, indent=1, default=float))
print(json.dumps({k: OUT[k] for k in ("p2", "aggregation", "madelon_excluded", "variance",
                                      "stability_two_measures_method_level", "pareto_dominated_by")},
                 indent=1, default=float)[:6000])
print(tab5.round(4))
print(pd.DataFrame(cov))
print(p2t.round(4))
print(comp.round(2))

# ---- extra: CI for the fold-weighted (empty fold = AUC 0.5) convention, and P1/P3 on the common set
rng = np.random.default_rng(SEED + 7)
idx = np.array([DS.index(d) for d in fin]); bs = []
for _ in range(B):
    w = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
    ref = (X_ns.iloc[idx].to_numpy() * w[:, None]).sum(0) / w.sum(); nat = ref.copy()
    for m in AD:
        j = M.index(m); nat[j] = (X_half.iloc[idx][m].to_numpy() * w).sum() / w.sum()
    bs.append(rho(rk(ref), rk(nat)))
OUT["p2"]["ci_half"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
OUT["on_common14"] = {"P3": axis_pair(X_ns, X_lk, common, "mean"), "P1_logreg": axis_pair(X_ns, X_lr, common, "mean"),
                      "P1_mlp": axis_pair(X_ns, X_ml, common, "mean")}
(RES / "round7_numbers.json").write_text(json.dumps(OUT, indent=1, default=float))
print("ci_half", OUT["p2"]["ci_half"], "on14", OUT["on_common14"])

# ---- extra: 20-dataset analysis under a uniform 30-minute budget (Madelon excluded)
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
nm = [d for d in DS if d != "madelon"]
rng = np.random.default_rng(SEED + 11)
idx = np.array([DS.index(d) for d in nm])
def wm(X, w): return (X.iloc[idx].to_numpy() * w[:, None]).sum(0) / w.sum()
bsd = {"P3": [], "P1_logreg": [], "P1_mlp": []}
for _ in range(B):
    w = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
    r0 = rk(wm(X_ns, w))
    bsd["P3"].append(rho(rk(wm(X_lk, w)), r0)); bsd["P1_logreg"].append(rho(r0, rk(wm(X_lr, w))))
    bsd["P1_mlp"].append(rho(r0, rk(wm(X_ml, w))))
fr = {}
for name, X in (("nested", X_ns), ("leaky", X_lk)):
    Y = X.loc[nm]
    st_ = friedmanchisquare(*[Y[m] for m in M])
    ph = sp.posthoc_nemenyi_friedman(Y.to_numpy())
    sig = int((np.triu(ph.to_numpy(), 1) < 0.05).sum())
    fr[name] = {"chi2": float(st_.statistic), "p": float(st_.pvalue), "n_sig_pairs": sig}
ph = sp.posthoc_nemenyi_friedman(X_ns.loc[nm].to_numpy()); ph.index = ph.columns = M
OUT["budget_uniform_20"] = {
    "datasets": nm,
    "nested_auc": X_ns.loc[nm].mean().round(4).to_dict(), "leaky_auc": X_lk.loc[nm].mean().round(4).to_dict(),
    "nested_rank": dict(zip(M, rk(X_ns.loc[nm].mean()).tolist())), "leaky_rank": dict(zip(M, rk(X_lk.loc[nm].mean()).tolist())),
    "ci": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bsd.items()},
    "friedman": fr, "p_boruta_rfe": float(ph.loc["borutashap", "rfe"]),
}
(RES / "round7_numbers.json").write_text(json.dumps(OUT, indent=1, default=float))
print(json.dumps(OUT["budget_uniform_20"], indent=1, default=float)[:2500])
