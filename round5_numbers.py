#!/usr/bin/env python3
"""
round5_numbers.py -- every number the Round 5 prose quotes that is not
already printed in a table, computed from results/*.csv and written to
results/round5_numbers.json. draft_v4.md cites these; audit_r5.py checks
them. Nothing here is typed by hand.

Run after: round5_fix_madelon_exp3.py, round5_dedupe.py,
experiments/exp7_protocol.py, round5_axis_sensitivity.py,
run_statistics.py, run_statistics_v3.py.
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (friedmanchisquare, kruskal, pearsonr, rankdata,
                         spearmanr, wilcoxon)

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
T = ROOT / "tables"
OLD = R / "pre_round5_backup"
M = ["shap_topk", "borutashap", "powershap", "shap_select",
     "mi", "chi2", "rfe", "lasso", "tree_importance"]
AD = ["borutashap", "powershap", "shap_select"]
OUT = {}


def put(key, val):
    if isinstance(val, (np.floating, np.integer)):
        val = val.item()
    OUT[key] = val


def rk(s, asc=False):
    return pd.Series(rankdata(s.to_numpy() if asc else -s.to_numpy()),
                     index=s.index)


def jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def kendall_w(mat):                       # rows = blocks, cols = methods
    """Kendall's W with the tie correction (Round 6: the earlier version
    omitted the sum-of-T_j term, which understates W when blocks contain
    tied values -- e.g. the many FDR = 0 cells of Section 4.3)."""
    r = np.vstack([rankdata(row) for row in mat])
    n, k = r.shape
    S = ((r.sum(0) - n * (k + 1) / 2) ** 2).sum()
    T = 0.0
    for row in mat:
        _, cnt = np.unique(row, return_counts=True)
        T += float((cnt ** 3 - cnt).sum())
    return 12 * S / (n ** 2 * (k ** 3 - k) - n * T)


def varshare(mat):
    grand = mat.mean()
    re = mat.mean(1) - grand
    ce = mat.mean(0) - grand
    resid = mat - grand - re[:, None] - ce[None, :]
    n, k = mat.shape
    sd, sm, sr = k * (re ** 2).sum(), n * (ce ** 2).sum(), (resid ** 2).sum()
    t = sd + sm + sr
    return sd / t, sm / t


# ── load ────────────────────────────────────────────────────────────────
e1 = pd.read_csv(R / "exp1_performance.csv")
e1n = pd.read_csv(R / "exp1_performance_nested.csv")
e2 = pd.read_csv(R / "exp2_stability.csv")
e3 = pd.read_csv(R / "exp3_noise.csv")
e4 = pd.read_csv(R / "exp4_transfer.csv")
e4n = pd.read_csv(R / "exp4_transfer_nested.csv")
e5 = pd.read_csv(R / "exp5_classifier_nested.csv")
e6n = pd.read_csv(R / "exp6_native_nested.csv")
folds = pd.read_csv(R / "exp1_folds_nested.csv")
head = json.load(open(R / "exp7_headline.json"))
for d in (e1, e1n, e5):
    assert (d.status == "ok").all() and not d.duplicated(
        [c for c in ("dataset", "method", "classifier") if c in d]).any()

# ── Section 3.1: fixed-budget overlap with SHAP Top-K (leaky Exp. 1) ─────
sel = e1.set_index(["dataset", "method"]).selected_features.str.split("|")
for m in ["powershap", "borutashap", "shap_select", "rfe", "tree_importance"]:
    js = [jac(sel[(d, m)], sel[(d, "shap_topk")]) for d in e1.dataset.unique()]
    put(f"overlap_topk_{m}_identical", int(sum(np.isclose(js, 1))))
    put(f"overlap_topk_{m}_jaccard", round(float(np.mean(js)), 3))

# ── padding share: how much of a fixed-k adaptive subset is SHAP Top-K fill
for m in AD:
    g = e6n[e6n.method == m]
    n_fold = np.concatenate([np.array(s.split("|"), float)
                             for s in g.fold_sizes.dropna()])
    k_fold = np.concatenate([[k] * 5 for k in g[g.fold_sizes.notna()].k])
    pad = np.clip(k_fold - n_fold, 0, None) / k_fold
    put(f"pad_share_{m}", round(float(pad.mean()), 3))
    put(f"pad_folds_below_k_{m}", int((n_fold < k_fold).sum()))
    put(f"pad_folds_total_{m}", int(len(n_fold)))
    put(f"native_gt_k_datasets_{m}",
        int((g.n_native_mean > g.k).sum()))
    put(f"native_lt_k_datasets_{m}",
        int((g.n_native_mean < g.k).sum()))

# ── Section 4.1 ─────────────────────────────────────────────────────────
pn = e1n.pivot(index="dataset", columns="method", values="auc_mean")[M]
pl = e1.pivot(index="dataset", columns="method", values="auc_mean")[M]
for lbl, p in (("nested", pn), ("leaky", pl)):
    st, pv = friedmanchisquare(*[p[c] for c in M])
    put(f"friedman_{lbl}_chi2", round(float(st), 2))
    put(f"friedman_{lbl}_p", float(f"{pv:.2g}"))
    mr = pd.DataFrame([rankdata(-r) for r in p.to_numpy()], columns=M).mean()
    put(f"meanrank_{lbl}", {m: round(float(v), 2) for m, v in mr.items()})
    put(f"kendallW_exp1_{lbl}", round(float(kendall_w(-p.to_numpy())), 3))
    sd, sm = varshare(p.to_numpy())
    put(f"varshare_exp1_{lbl}", [round(sd * 100, 1), round(sm * 100, 1)])
for lbl, f in (("nested", "nemenyi_exp1-nested.csv"),
               ("leaky", "nemenyi_exp1-leaky.csv")):
    nm = pd.read_csv(T / f, index_col=0)
    sig = [(a, b, float(nm.loc[a, b])) for a, b in combinations(M, 2)
           if nm.loc[a, b] < 0.05]
    put(f"nemenyi_{lbl}_n_sig", len(sig))
    put(f"nemenyi_{lbl}_sig", sig)
    put(f"nemenyi_{lbl}_boruta_rfe", float(nm.loc["borutashap", "rfe"]))
    put(f"nemenyi_{lbl}_select_vs",
        {m: float(nm.loc["shap_select", m]) for m in M if m != "shap_select"})
put("time_leaky", e1.groupby("method").time_select_s.mean().round(2).to_dict())
put("time_nested", e1n.groupby("method").time_select_s.mean().round(2).to_dict())
put("time_nested_median",
    e1n.groupby("method").time_select_s.median().round(2).to_dict())
put("select_madelon_leaky_h",
    round(float(e1[(e1.method == "shap_select") & (e1.dataset == "madelon")]
                .time_select_s.iloc[0]) / 3600, 1))
put("select_madelon_nested_h",
    round(float(e1n[(e1n.method == "shap_select") & (e1n.dataset == "madelon")]
                .time_select_s.iloc[0]) / 3600, 1))

# ── Section 4.2: stability over the matched set ──────────────────────────
p2 = e2.pivot(index="dataset", columns="method", values="jaccard_mean")[M]
p2m = p2.dropna()
put("exp2_matched_n", len(p2m))
put("exp2_dropped", sorted(set(p2.index) - set(p2m.index)))
mj = p2m.mean()
ms = e2[e2.dataset.isin(p2m.index)].groupby("method").spearman_mean.mean()[M]
put("exp2_jaccard", mj.round(4).to_dict())
put("exp2_rank", rk(mj).astype(int).to_dict())
put("exp2_pearson", round(float(pearsonr(mj, ms)[0]), 3))
w = wilcoxon(p2m["chi2"], p2m["powershap"])
put("exp2_wilcoxon_chi2_powershap_p", round(float(w.pvalue), 3))
st, pv = friedmanchisquare(*[p2m[c] for c in M])
put("friedman_exp2", [round(float(st), 2), float(f"{pv:.2g}")])
nm2 = pd.read_csv(T / "nemenyi_exp2.csv", index_col=0)
put("nemenyi_exp2_sig", [(a, b, float(nm2.loc[a, b])) for a, b in combinations(M, 2)
                         if nm2.loc[a, b] < 0.05])
put("exp2_adult", p2.loc["adult"].round(3).to_dict())
put("exp2_madelon", p2.loc["madelon"].round(3).to_dict())
put("kendallW_exp2", round(float(kendall_w(-p2m.to_numpy())), 3))
put("varshare_exp2", [round(x * 100, 1) for x in varshare(p2m.to_numpy())])
s2 = e2[e2.dataset.isin(p2m.index)].copy()
s2["EJ"] = s2.k / (2 * s2.n_features - s2.k)
s2["Jc"] = (s2.jaccard_mean - s2.EJ) / (1 - s2.EJ)
jc = s2.pivot(index="dataset", columns="method", values="Jc")[M]
put("exp2_corrected", jc.mean().round(3).to_dict())
put("exp2_corrected_top", jc.mean().idxmax())
put("exp2_corrected_rho", round(float(spearmanr(mj, jc.mean())[0]), 3))
top2 = jc.mean().sort_values(ascending=False).index[:2]
put("exp2_corrected_top2", list(top2))
put("exp2_corrected_top2_p",
    round(float(wilcoxon(jc[top2[0]], jc[top2[1]]).pvalue), 3))
put("exp2_boruta_boot_time_mean_s",
    round(float(e2[e2.method == "borutashap"].time_s.mean()), 0))

# ── Section 4.3 ─────────────────────────────────────────────────────────
ok3 = e3[e3.status == "ok"]
nm3 = ok3[ok3.dataset != "madelon"]
fdr = nm3.pivot_table(index="method", columns="noise_ratio", values="fdr")
put("exp3_fdr", {m: [round(fdr.loc[m, r], 4) for r in (0.1, 0.2, 0.5)] for m in M})
p3 = nm3[nm3.noise_ratio == 0.5].pivot(index="dataset", columns="method",
                                       values="fdr")[M]
st, pv = friedmanchisquare(*[p3[c] for c in M])
put("friedman_exp3", [round(float(st), 2), float(f"{pv:.2g}")])
put("exp3_zero_cells", [int((p3 == 0).sum().sum()), int(p3.size)])
put("kendallW_exp3", round(float(kendall_w(p3.to_numpy())), 3))
put("varshare_exp3", [round(x * 100, 1) for x in varshare(p3.to_numpy())])
mad = ok3[(ok3.dataset == "madelon") & (ok3.noise_ratio == 0.5)].set_index("method")
put("madelon_fdr", mad.fdr.round(2).to_dict())
put("madelon_recovered", (mad.tpr * 20).round().astype(int).to_dict())

# ── Section 4.4 ─────────────────────────────────────────────────────────
ok4n = e4n[e4n.status == "ok"]
g4 = ok4n.groupby("method")
d_abs = g4.delta_f1.apply(lambda s: s.abs().mean())
d_leak = g4.delta_f1_leaky.apply(lambda s: s.abs().mean())
put("exp4_absdf1", d_abs.round(4).to_dict())
put("exp4_n_down", int((d_abs < d_leak).sum()))
put("exp4_max_change", round(float((d_abs - d_leak).abs().max()), 4))
put("exp4_min_cross_f1", round(float(g4.cross_f1.mean().min()), 4))
put("exp4_jaccard", g4.feature_jaccard.mean().round(3).to_dict())
p4 = ok4n.assign(a=ok4n.delta_f1.abs()).pivot_table(
    index=["source", "target"], columns="method", values="a")
put("kendallW_exp4", round(float(kendall_w(p4.to_numpy())), 3))
put("varshare_exp4", [round(x * 100, 1) for x in varshare(p4.to_numpy())])
put("exp4_failed", sorted(set(e4n.method) - set(ok4n.method)))
put("exp4_fail_msg", e4n[e4n.status != "ok"].status.str.extract(
    r"(Logistic regression[^.]*)")[0].dropna().unique().tolist())

# ── Section 5 (P1) ──────────────────────────────────────────────────────
xgb = pn.mean()
lr = e5[e5.classifier == "logreg"].groupby("method").auc.mean()[M]
mlp = e5[e5.classifier == "mlp"].groupby("method").auc.mean()[M]
for n_, s in (("xgb", xgb), ("logreg", lr), ("mlp", mlp)):
    put(f"p1_rank_{n_}", rk(s).to_dict())
    put(f"p1_auc_{n_}", s.round(4).to_dict())
put("p1_rho", head["rho_p1"])

# ── Section 6 (P2) ──────────────────────────────────────────────────────
sens = pd.read_csv(R / "round5_axis_sensitivity.csv")
put("sensitivity", sens.to_dict(orient="records"))
put("p2_rho", head["rho_p2"])

# ── Section 7 (P3) ──────────────────────────────────────────────────────
opt = pd.read_csv(R / "exp7_protocol.csv")
put("family_optimism", opt.groupby("family").optimism_auc.mean().round(4).to_dict())
put("shap3_optimism", round(float(opt[opt.method.isin(
    ["shap_topk", "borutashap", "powershap"])].optimism_auc.mean()), 4))
put("kruskal", [round(head["kruskal_H"], 2), round(head["kruskal_p"], 3)])
ds = opt.groupby("dataset").agg(o=("optimism_auc", "mean"),
                                np_=("n_over_p", "first"),
                                kp=("k_over_p", "first"))
r1, pv1 = spearmanr(ds.o, ds.np_)
r2_, pv2 = spearmanr(ds.o, ds.kp)
put("geom_np", [round(float(r1), 3), float(f"{pv1:.2g}")])
put("geom_kp", [round(float(r2_), 3), float(f"{pv2:.2g}")])
put("mi_madelon_optimism",
    round(float(opt[(opt.method == "mi") & (opt.dataset == "madelon")]
                .optimism_auc.iloc[0]), 4))
put("max_optimism", opt.loc[opt.optimism_auc.idxmax(),
                            ["method", "dataset", "optimism_auc"]].tolist())
st7 = pd.read_csv(R / "exp7_stability.csv")
put("fold_vs_boot", {"n": int(st7.dropna().shape[0])})
fb = pd.read_csv(R / "exp7_stability_by_method.csv", index_col=0)
put("fold_jaccard", fb.iloc[:, 0].round(3).to_dict())
rr, pp = spearmanr(st7.dropna().iloc[:, -2], st7.dropna().iloc[:, -1])
put("fold_boot_rho_check", [round(float(rr), 3), float(f"{pp:.2g}")])
put("claims", {k: head[k] for k in head if k.startswith("claim")})

# ── Section 9.1: Pareto over the four dimension values ───────────────────
perf, stab = xgb, mj
noise = p3.mean()
trans = d_abs.reindex(M)
dfp = pd.DataFrame({"perf": perf, "stab": stab, "noise": noise, "trans": trans})
dom = {}
for m in M:
    for o in M:
        if o == m:
            continue
        a, b = dfp.loc[m], dfp.loc[o]
        if pd.isna(a.trans):           # cannot run on transfer: compare on 3
            cols_ge = (b.perf >= a.perf, b.stab >= a.stab, b.noise <= a.noise)
            cols_gt = (b.perf > a.perf, b.stab > a.stab, b.noise < a.noise)
        elif pd.isna(b.trans):
            continue
        else:
            cols_ge = (b.perf >= a.perf, b.stab >= a.stab, b.noise <= a.noise,
                       b.trans <= a.trans)
            cols_gt = (b.perf > a.perf, b.stab > a.stab, b.noise < a.noise,
                       b.trans < a.trans)
        if all(cols_ge) and any(cols_gt):
            dom.setdefault(m, []).append(o)
put("pareto_dominated", dom)
put("pareto_front", [m for m in M if m not in dom])

# ── Section 9.x: implementation fidelity (reimplementation -> official) ──
o1n = pd.read_csv(OLD / "exp1_performance_nested.csv")
o2 = pd.read_csv(OLD / "exp2_stability.csv").drop_duplicates(["dataset", "method"])
o3 = pd.read_csv(OLD / "exp3_noise.csv").drop_duplicates(
    ["dataset", "method", "noise_ratio"])
o4n = pd.read_csv(OLD / "exp4_transfer_nested.csv")
old_auc = o1n.groupby("method").auc_mean.mean()[M]
put("fid_old_auc", old_auc.round(4).to_dict())
put("fid_old_rank", rk(old_auc).astype(int).to_dict())
put("fid_new_rank", rk(xgb).astype(int).to_dict())
put("fid_rho_nested", round(float(spearmanr(rk(old_auc), rk(xgb))[0]), 2))
put("fid_maxmove", float((rk(old_auc) - rk(xgb)).abs().max()))
old_l = pd.read_csv(OLD / "exp1_performance.csv").groupby("method").auc_mean.mean()[M]
put("fid_rho_leaky", round(float(spearmanr(rk(old_l), rk(pl.mean()))[0]), 2))
o2m = o2[o2.dataset.isin(p2m.index)].groupby("method").jaccard_mean.mean()[M]
put("fid_old_stab_matched", o2m.round(3).to_dict())
put("fid_rho_stab", round(float(spearmanr(rk(o2m), rk(mj))[0]), 2))
o3m = o3[(o3.dataset != "madelon") & (o3.noise_ratio == 0.5)].groupby("method").fdr.mean()[M]
put("fid_old_fdr50", o3m.round(4).to_dict())
put("fid_rho_noise", round(float(spearmanr(rk(o3m, True), rk(noise, True))[0]), 2))
put("fid_old_trans", o4n.groupby("method").delta_f1.apply(
    lambda s: s.abs().mean()).round(4).to_dict())
moved = {m: [int(rk(old_auc)[m]), int(rk(xgb)[m])] for m in M}
put("fid_rank_pairs", moved)


# ── Round 6 additions ───────────────────────────────────────────────────
from scipy.stats import hypergeom
# (a) random-selection FDR baseline for Section 4.3 (8 non-Madelon datasets)
base = (e3[e3.status == "ok"].drop_duplicates(["dataset", "noise_ratio"])
        [["dataset", "noise_ratio", "n_original", "n_noise_injected",
          "n_total", "k_used"]].copy())
base["rand_fdr"] = base.n_noise_injected / base.n_total
base["rand_tpr"] = (1 - base.rand_fdr) * base.k_used / base.n_original
nb = base[base.dataset != "madelon"]
put("rand_fdr", {str(r): round(float(v), 6) for r, v in
                 nb.groupby("noise_ratio").rand_fdr.mean().items()})
put("rand_tpr50", round(float(nb[nb.noise_ratio == 0.5].rand_tpr.mean()), 4))
put("rand_fdr_madelon", round(float(base[base.dataset == "madelon"].rand_fdr.iloc[0]), 2))
# (b) chance-level Jaccard: exact expectation for two random k-subsets of p
def exp_jaccard(p, k):
    i = np.arange(0, k + 1)
    pm = hypergeom.pmf(i, p, k, k)
    return float((pm * i / (2 * k - i)).sum())
ds_k = e2.drop_duplicates("dataset").set_index("dataset")[["n_features", "k"]]
ej = {d: exp_jaccard(int(r.n_features), int(r.k)) for d, r in ds_k.iterrows()}
put("chance_jaccard", {d: round(v, 6) for d, v in ej.items()})
put("chance_jaccard_mean9", round(float(np.mean([ej[d] for d in p2m.index])), 6))
put("n_stability_datasets", int(e2.dataset.nunique()))
put("n_stability_matched", int(len(p2m)))
# (c) transfer: signed change under nesting
sg = g4.delta_f1.mean(); sgl = g4.delta_f1_leaky.mean()
put("exp4_signed_down", int((sg < sgl).sum()))
put("exp4_pairs_positive_delta", [int((ok4n.delta_f1 > 0).sum()), int(len(ok4n))])
# (d) bootstrap CIs of the protocol-axis correlations
B6 = json.load(open(R / "round6_bootstrap_ci.json"))
put("boot", {r["quantity"]: [round(r["point"], 3), round(r["ci_lo"], 3),
                              round(r["ci_hi"], 3), round(r["frac_gt0"], 3)]
             for r in B6["table"]})
put("boot_noise", B6["noise_baseline"])
put("boot_B", B6["B"]); put("boot_seed", B6["seed"])

json.dump(OUT, open(R / "round5_numbers.json", "w"), indent=1, default=str)
for k, v in OUT.items():
    print(f"{k}: {v}")
