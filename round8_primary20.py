#!/usr/bin/env python3
"""
round8_primary20.py -- round-8 re-analysis with a uniform 30-minute budget.

Primary dataset set (PRIMARY): the 20 datasets other than Madelon. Madelon is
excluded for every method because the only completed SHAP-Select runs on it
(49.5 h leaky, 237 h nested) exceed the budget that governs every other
experiment and the archived reproduction path.  The 21-dataset results are the
"unconstrained-completion" sensitivity analysis (SENS).

Everything that depends on the classification substrate is recomputed on both
sets: Table 2, Friedman/Nemenyi (+CD diagrams), P1 (Table 6), P3 (Table 8 with
Holm), bootstrap CIs and differences, split-half reference, fidelity rho,
mean-rank aggregation, rank changes (Table 10), variance decomposition, Kendall
W, geometry correlation, composite ranks.  P2 (common support) contains no
Madelon and is unchanged; its available-case variant is recomputed on 20.
Output: results/round8_numbers.json, results/round8_*.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, kruskal, rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
M = ["rfe", "lasso", "tree_importance", "chi2", "mi", "shap_topk", "borutashap", "powershap", "shap_select"]
AD = ["borutashap", "powershap", "shap_select"]
FAM = {"rfe": "wrapper", "lasso": "embedded", "tree_importance": "embedded", "chi2": "filter", "mi": "filter",
       "shap_topk": "SHAP", "borutashap": "SHAP", "powershap": "SHAP", "shap_select": "SHAP"}
B, SEED = 10000, 42

ns = pd.read_csv(RES / "exp1_performance_nested.csv"); ns = ns[ns.status == "ok"]
lk = pd.read_csv(RES / "exp1_performance.csv"); lk = lk[lk.status == "ok"]
e5 = pd.read_csv(RES / "exp5_classifier_nested.csv"); e5 = e5[e5.status == "ok"]
e6 = pd.read_csv(RES / "exp6_native_nested.csv")
old = pd.read_csv(RES / "pre_round5_backup" / "exp1_performance_nested.csv")
old = old[old.status == "ok"].drop_duplicates(["dataset", "method"])
geo = pd.read_csv(RES / "exp7_protocol.csv")
ALL = sorted(ns.dataset.unique()); assert len(ALL) == 21
PRIMARY = [d for d in ALL if d != "madelon"]


def P(df, col): return df.pivot_table(index="dataset", columns="method", values=col).reindex(index=ALL, columns=M)


X = {"ns": P(ns, "auc_mean"), "lk": P(lk, "auc_mean"), "lr": P(e5[e5.classifier == "logreg"], "auc"),
     "ml": P(e5[e5.classifier == "mlp"], "auc"), "old": P(old, "auc_mean"),
     "f1": P(ns, "f1_mean"), "acc": P(ns, "accuracy_mean"), "t_ns": P(ns, "time_select_s"), "t_lk": P(lk, "time_select_s")}
for k_ in ("ns", "lk", "lr", "ml", "old"): assert not X[k_].isna().any().any(), k_


def rk(v): return rankdata(-np.asarray(v, float))
def rho(a, b): return float(spearmanr(a, b).statistic)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * p[j]); adj[j] = min(1.0, run)
    return adj


def kendall_w(mat):
    r = np.vstack([rankdata(row) for row in mat]); n, k = r.shape
    S = ((r.sum(0) - n * (k + 1) / 2) ** 2).sum(); T = 0.0
    for row in mat:
        _, c = np.unique(row, return_counts=True); T += float((c ** 3 - c).sum())
    return 12 * S / (n ** 2 * (k ** 3 - k) - n * T)


def varshare(Y):
    g = Y.mean(); re = Y.mean(1) - g; ce = Y.mean(0) - g
    resid = Y - g - re[:, None] - ce[None, :]; n, k = Y.shape
    sd, sm, sr = k * (re ** 2).sum(), n * (ce ** 2).sum(), (resid ** 2).sum(); t = sd + sm + sr
    return {"dataset": sd / t, "method": sm / t, "residual": sr / t}


def nemenyi(piv):
    stat, p = friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
    pv = sp.posthoc_nemenyi_friedman(piv.to_numpy()); pv.index = pv.columns = piv.columns
    mr = piv.rank(axis=1, ascending=False).mean()
    sig = [(a, b, float(pv.loc[a, b])) for i, a in enumerate(M) for b in M[i + 1:] if pv.loc[a, b] < 0.05]
    return {"chi2": float(stat), "p": float(p), "n_sig": len(sig), "sig": sig, "mean_ranks": mr.round(3).to_dict(),
            "pv": pv.round(4).to_dict()}


def wmean(Xm, w):
    return (Xm * w[:, None]).sum(0) / w.sum()


def analyse(ds, tag):
    o = {}
    S = lambda k_: X[k_].loc[ds]
    ns_, lk_, lr_, ml_, old_ = S("ns"), S("lk"), S("lr"), S("ml"), S("old")
    mean = lambda D: D.mean()
    # ---- Table 2
    t2 = pd.DataFrame({"auc_ns": ns_.mean(), "auc_ns_sd": ns_.std(), "auc_lk": lk_.mean(),
                       "f1": S("f1").mean(), "acc": S("acc").mean(), "t_ns": S("t_ns").mean(),
                       "t_lk": S("t_lk").mean(), "t_ns_median": S("t_ns").median()})
    t2["delta"] = t2.auc_ns - t2.auc_lk
    t2 = t2.sort_values("auc_ns", ascending=False)
    o["table2"] = t2.round(6).to_dict(orient="index"); o["order_nested"] = list(t2.index)
    o["order_leaky"] = list(lk_.mean().sort_values(ascending=False).index)
    # ---- Friedman / Nemenyi
    o["friedman_nested"] = nemenyi(ns_); o["friedman_leaky"] = nemenyi(lk_)
    # ---- P3 table
    opt = lk_ - ns_
    p3 = pd.DataFrame({"family": pd.Series(FAM), "auc_lk": lk_.mean(), "auc_ns": ns_.mean(),
                       "opt_mean": opt.mean(), "opt_sd": opt.std(),
                       "p": [float(wilcoxon(lk_[m], ns_[m]).pvalue) for m in M],
                       "rank_lk": rk(lk_.mean()), "rank_ns": rk(ns_.mean())}, index=M)
    p3["p_holm"] = holm(p3.p)
    o["table8"] = p3.sort_values("opt_mean", ascending=False).round(6).to_dict(orient="index")
    o["kw_means"] = float(kruskal(*[p3.opt_mean[[m for m in M if FAM[m] == f]] for f in ("wrapper", "embedded", "filter", "SHAP")]).pvalue)
    o["n_positive_optimism"] = int((p3.opt_mean > 0).sum())
    # ---- P1 table
    p1 = pd.DataFrame({"xgb": ns_.mean(), "lr": lr_.mean(), "ml": ml_.mean()}, index=M)
    p1["d_lr"] = rk(p1.xgb) - rk(p1.lr); p1["d_ml"] = rk(p1.xgb) - rk(p1.ml)
    o["table6"] = p1.sort_values("xgb", ascending=False).round(6).to_dict(orient="index")
    o["rho_lr_ml"] = rho(rk(p1.lr), rk(p1.ml))
    # ---- axes, bootstrap
    A = {"P1_logreg": (ns_, lr_), "P1_mlp": (ns_, ml_), "P3": (ns_, lk_), "fidelity": (old_, ns_)}
    o["point"] = {k_: rho(rk(a.mean()), rk(b.mean())) for k_, (a, b) in A.items()}
    o["maxchange"] = {k_: float(np.max(np.abs(rk(a.mean()) - rk(b.mean())))) for k_, (a, b) in A.items()}
    o["rank_agg"] = {k_: rho(rk(-a.rank(axis=1, ascending=False).mean()), rk(-b.rank(axis=1, ascending=False).mean()))
                     for k_, (a, b) in A.items()}
    rng = np.random.default_rng(SEED); n = len(ds); bs = {k_: [] for k_ in A}
    arr = {k_: (a.to_numpy(), b.to_numpy()) for k_, (a, b) in A.items()}
    for _ in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for k_, (a, b) in arr.items():
            bs[k_].append(rho(rk(wmean(a, w)), rk(wmean(b, w))))
    bs = {k_: np.array(v) for k_, v in bs.items()}
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    o["ci"] = {k_: pc(v) for k_, v in bs.items()}
    o["diff"] = {}
    for a, b in (("P3", "P1_logreg"), ("P3", "P1_mlp"), ("P1_mlp", "P1_logreg")):
        d = bs[a] - bs[b]
        o["diff"][f"{a}-{b}"] = {"point": o["point"][a] - o["point"][b], "ci": pc(d), "frac_gt0": float((d > 0).mean())}
    # split-half reference on the same set
    rng2 = np.random.default_rng(SEED + 1); half = []; p3h = []
    Y, L = ns_.to_numpy(), lk_.to_numpy(); h = n // 2
    for _ in range(B):
        pm = rng2.permutation(n); a_, b_ = pm[:h], pm[h:]
        half.append(rho(rk(Y[a_].mean(0)), rk(Y[b_].mean(0))))
        p3h.append(rho(rk(L[a_].mean(0)), rk(Y[a_].mean(0))))
    half, p3h = np.array(half), np.array(p3h)
    o["split_half"] = {"halves": [h, n - h], "median": float(np.median(half)), "mean": float(half.mean()), "range95": pc(half),
                       "p3_half_mean": float(p3h.mean()), "p3_half_range95": pc(p3h)}
    # ---- rank changes for Table 10
    o["rank_change"] = {m: {"P1_logreg": float((rk(p1.xgb) - rk(p1.lr))[M.index(m)]),
                            "P1_mlp": float((rk(p1.xgb) - rk(p1.ml))[M.index(m)]),
                            "P3": float((rk(ns_.mean()) - rk(lk_.mean()))[M.index(m)])} for m in M}
    o["ranks"] = {"leaky": dict(zip(M, rk(lk_.mean()).tolist())), "nested": dict(zip(M, rk(ns_.mean()).tolist())),
                  "logreg": dict(zip(M, rk(lr_.mean()).tolist())), "mlp": dict(zip(M, rk(ml_.mean()).tolist()))}
    # ---- variance, Kendall
    o["variance"] = {"nested": varshare(ns_.to_numpy()), "leaky": varshare(lk_.to_numpy())}
    o["kendall_w"] = {"nested": float(kendall_w(-ns_.to_numpy())), "leaky": float(kendall_w(-lk_.to_numpy()))}
    # ---- geometry
    g = geo[geo.dataset.isin(ds)].groupby("dataset").agg(opt=("optimism_auc", "mean"), np_=("n_over_p", "first"), kp=("k_over_p", "first"))
    r1, r2 = spearmanr(g.np_, g.opt), spearmanr(g.kp, g.opt)
    o["geometry"] = {"rho_np": float(r1.statistic), "p_np": float(r1.pvalue), "rho_kp": float(r2.statistic), "p_kp": float(r2.pvalue),
                     "median_opt_np_gt100": float(g[g.np_ > 100].opt.median()), "max_dataset": g.opt.idxmax(), "max_value": float(g.opt.max())}
    o["opt_max_cell"] = {"value": float(opt.to_numpy().max()), "where": [str(x) for x in np.unravel_index(opt.to_numpy().argmax(), opt.shape)]}
    o["opt_max_cell"]["where"] = [ds[int(o["opt_max_cell"]["where"][0])], M[int(o["opt_max_cell"]["where"][1])]]
    # ---- P2 available-case variant on this set
    nat = ns_.mean().copy()
    for m in AD:
        v = e6[(e6.method == m) & (e6.status == "ok") & e6.dataset.isin(ds)].auc_native
        nat[m] = v.mean()
    o["p2_available_case"] = rho(rk(ns_.mean()), rk(nat))
    o["n"] = n
    return o


OUT = {"primary_datasets": PRIMARY, "primary": analyse(PRIMARY, "20"), "sensitivity21": analyse(ALL, "21")}
(RES / "round8_numbers.json").write_text(json.dumps(OUT, indent=1, default=float))
pr = OUT["primary"]
print("order nested 20:", pr["order_nested"]); print("order leaky 20:", pr["order_leaky"])
print("points:", {k: round(v, 3) for k, v in pr["point"].items()}, "ci:", {k: [round(x, 2) for x in v] for k, v in pr["ci"].items()})
print("rank agg:", {k: round(v, 3) for k, v in pr["rank_agg"].items()}, "maxchange", pr["maxchange"])
print("diff:", pr["diff"]); print("split:", pr["split_half"])
print("friedman nested:", round(pr["friedman_nested"]["chi2"], 2), pr["friedman_nested"]["p"], pr["friedman_nested"]["n_sig"], pr["friedman_nested"]["sig"])
print("friedman leaky:", round(pr["friedman_leaky"]["chi2"], 2), pr["friedman_leaky"]["p"], pr["friedman_leaky"]["n_sig"])
print("variance", pr["variance"], "W", pr["kendall_w"]); print("geometry", pr["geometry"]); print("rho lr-ml", pr["rho_lr_ml"])
print("p2 avail", pr["p2_available_case"], "kw", pr["kw_means"], "n_pos", pr["n_positive_optimism"], "maxcell", pr["opt_max_cell"])
print(pd.DataFrame(pr["table8"]).T)
print(pd.DataFrame(pr["table6"]).T.round(4)); print(pd.DataFrame(pr["table2"]).T.round(4))
s21 = OUT["sensitivity21"]; print("21 check:", {k: round(v, 3) for k, v in s21["point"].items()}, s21["ci"])
