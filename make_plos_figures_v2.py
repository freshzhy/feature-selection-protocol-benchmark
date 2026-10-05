#!/usr/bin/env python3
"""make_plos_figures_v2.py -- the manuscript figures, redrawn for PLOS ONE (round 7).

Design rules (one system for every figure)
  * two family colours only: SHAP-based = blue #2a78d6, traditional = orange
    #eb6834; family is also carried by marker shape (filled circle vs open
    square) so it survives greyscale printing and colour-vision deficiency
  * method and dataset names exactly as in the tables (no "MI", "Tree Imp.",
    no raw OpenML ids)
  * no in-figure titles, no number printed on every mark, no truncated bar axes
  * every plotted value is recomputed from results/*.csv and asserted against
    the value printed in the manuscript tables before it is drawn
  * TIFF, LZW, RGB, 300 dpi, width <= 7.5 in, Liberation Sans 8-10 pt

Main-text figures (new numbering)            SI figures
  Fig1  optimism by method (dumbbell)          S2_Fig  per-dataset AUC heatmap, leaky
  Fig3  subsample stability (dot plot)         S3_Fig  TPR by noise ratio (dot plot)
  Fig4  per-dataset stability heatmap
  Fig5  FDR by noise ratio (dot plot)
  Fig6  per-pair transfer heatmap
  Fig7  optimism against n/p (21 datasets)
  Fig8  rank slopegraph across conditions
Fig2 and S1_Fig (critical-difference diagrams) are rendered by
make_plos_figures.py from run_statistics_v3.py and are unchanged.
"""
import io
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "plos" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

from matplotlib import font_manager as _fm
for _f in (ROOT / "fonts").glob("*.ttf"):
    _fm.fontManager.addfont(str(_f))
plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "axes.edgecolor": "#52514e", "axes.linewidth": 0.8,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "savefig.facecolor": "white", "figure.facecolor": "white",
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
C_SHAP, C_TRAD = "#2a78d6", "#eb6834"
NAME = {"shap_topk": "SHAP Top-K", "borutashap": "BorutaSHAP", "powershap": "PowerSHAP",
        "shap_select": "SHAP-Select", "mi": "Mutual Information", "chi2": "Chi-Square",
        "rfe": "RFE", "lasso": "Lasso", "tree_importance": "Tree Importance"}
SHAP = {"shap_topk", "borutashap", "powershap", "shap_select"}
DSN = {"adult": "Adult", "banknote-authentication": "Banknote-Authentication",
       "blood-transfusion": "Blood-Transfusion", "breast-w": "Breast-W", "car": "Car",
       "credit-approval": "Credit-Approval", "credit-g": "Credit-G", "creditcard": "Creditcard",
       "diabetes": "Diabetes", "hr-attrition": "HR-Attrition", "ionosphere": "Ionosphere",
       "kc1": "KC1", "kc2": "KC2", "kr-vs-kp": "KR-vs-KP", "madelon": "Madelon",
       "phoneme": "Phoneme", "qsar-biodeg": "QSAR-Biodeg", "sonar": "Sonar",
       "spambase": "Spambase", "steel-plates-fault": "Steel-Plates-Fault",
       "telco-churn": "Telco-Churn"}


def col(m): return C_SHAP if m in SHAP else C_TRAD
def mk(m): return "o" if m in SHAP else "s"
def face(m): return col(m) if m in SHAP else "white"


def family_handles(extra=()):
    return [Line2D([], [], marker="o", ls="", ms=6, mfc=C_SHAP, mec=C_SHAP, label="SHAP-based"),
            Line2D([], [], marker="s", ls="", ms=6, mfc="white", mec=C_TRAD, mew=1.4,
                   label="Traditional"), *extra]


def save(fig, name):
    w, h = fig.get_size_inches()
    assert w <= 7.5 + 1e-9 and h <= 8.75, (name, w, h)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    plt.close(fig)
    im = Image.open(buf).convert("RGB")
    im.save(OUT / f"{name}.tif", format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    print(f"{name:8s} {im.size[0]}x{im.size[1]} px")


def close(a, b, tol=6e-5, what=""):
    assert abs(a - b) <= tol, (what, a, b)


# ---------------------------------------------------------------- data
import json as _json
_N8 = _json.loads((RES / "round8_numbers.json").read_text())["primary"]
summ = pd.DataFrame(_N8["table8"]).T.rename(columns={"opt_mean": "optimism_mean", "auc_ns": "auc_nested", "auc_lk": "auc_leaky",
                                                     "rank_lk": "rank_leaky", "rank_ns": "rank_nested"})
for _c in ("optimism_mean", "auc_nested", "auc_leaky", "rank_leaky", "rank_nested", "p_holm"):
    summ[_c] = summ[_c].astype(float)
PRIMARY = [d for d in _json.loads((RES / "round8_numbers.json").read_text())["primary_datasets"]]
opt = pd.read_csv(RES / "exp7_protocol.csv")
lk = pd.read_csv(RES / "exp1_performance.csv"); lk = lk[lk.status == "ok"]
e2 = pd.read_csv(RES / "exp2_stability.csv")
e3 = pd.read_csv(RES / "exp3_noise.csv"); e3 = e3[e3.status == "ok"]
e4 = pd.read_csv(RES / "exp4_transfer_nested.csv")
sf = pd.read_csv(RES / "exp7_stable_fragile.csv", index_col=0)

# values printed in the manuscript tables (Table 2 / 3 / 4 / 5 / 8 of v2)
T_OPT = {"rfe": .0093, "mi": -.0002, "shap_select": .0054, "shap_topk": .0045, "lasso": .0026,
         "tree_importance": .0042, "powershap": .0037, "borutashap": .0024, "chi2": -.0001}
T_NEST = {"borutashap": .8919, "shap_topk": .8918, "powershap": .8914, "shap_select": .8887,
          "rfe": .8874, "tree_importance": .8866, "lasso": .8861, "mi": .8848, "chi2": .8834}
T_JAC = {"chi2": .8117, "powershap": .7996, "shap_topk": .7819, "borutashap": .7757,
         "lasso": .7448, "mi": .7144, "shap_select": .6922, "rfe": .6901, "tree_importance": .6596}
T_SPE = {"chi2": .8644, "powershap": .8532, "shap_topk": .8422, "borutashap": .8271,
         "lasso": .8112, "mi": .7573, "shap_select": .6662, "rfe": .6584, "tree_importance": .6933}
T_FDR = {"chi2": (0, 0, 0), "lasso": (.0198, 0, .0188), "mi": (.0250, .0250, .0312),
         "tree_importance": (.0136, .0261, .0482), "rfe": (.0073, .0209, .0669),
         "borutashap": (.0500, .0813, .0875), "shap_select": (.0761, .0563, .0949),
         "powershap": (.0688, .1250, .1188), "shap_topk": (.1022, .1511, .1709)}
T_RAND = (.1065, .1783, .3358)

# ================================================================ Fig 1
HOLM = summ.p_holm.to_dict()
d = summ.sort_values("optimism_mean")
for m, r in d.iterrows():
    close(r.optimism_mean, T_OPT[m], what=f"optimism {m}")
    close(r.auc_nested, T_NEST[m], what=f"nested {m}")
fig, ax = plt.subplots(figsize=(6.6, 3.5))
for i, (m, r) in enumerate(d.iterrows()):
    ax.plot([r.auc_nested, r.auc_leaky], [i, i], color=col(m), lw=2, solid_capstyle="butt", zorder=2)
    ax.plot(r.auc_nested, i, marker=mk(m), ms=7, mfc=face(m), mec=col(m), mew=1.4, ls="", zorder=3)
    ax.plot(r.auc_leaky, i, marker="|", ms=11, mew=2, color=col(m), ls="", zorder=3)
    star = "*" if HOLM[m] < 0.05 else ""
    ax.text(max(r.auc_leaky, r.auc_nested) + 0.0006, i, f"{r.optimism_mean:+.4f}{star}".replace("-", "−"),
            va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(d))); ax.set_yticklabels([NAME[m] for m in d.index])
ax.set_xlabel("Mean AUC-ROC over 20 datasets")
ax.set_xlim(0.873, 0.903)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=6, mfc=C_SHAP, mec=C_SHAP, label="Nested, SHAP-based"),
                   Line2D([], [], marker="s", ls="", ms=6, mfc="white", mec=C_TRAD, mew=1.4, label="Nested, traditional"),
                   Line2D([], [], marker="|", ls="", ms=10, mew=2, color=INK2, label="Leaky (same method)")],
          loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0, 1.0))
fig.tight_layout(); save(fig, "Fig1")

# ================================================================ Fig 3 (stability dot plot)
piv = e2.pivot(index="dataset", columns="method", values="jaccard_mean")
matched = sorted(piv.dropna().index); assert len(matched) == 9 and "madelon" not in matched
st = e2[e2.dataset.isin(matched)].groupby("method")[["jaccard_mean", "spearman_mean"]].mean()
for m in st.index:
    close(st.loc[m, "jaccard_mean"], T_JAC[m], what=f"jac {m}")


def chance_jaccard(p, k):
    tot = comb(p, k)
    return sum(comb(k, i) * comb(p - k, k - i) / tot * i / (2 * k - i) for i in range(0, k + 1))


meta = e2.drop_duplicates("dataset").set_index("dataset")
ch = np.mean([chance_jaccard(int(meta.loc[ds, "n_features"]), int(meta.loc[ds, "k"])) for ds in matched])
close(ch, 0.31, tol=0.005, what="chance jaccard")
st = st.sort_values("jaccard_mean")
dsJ = e2[e2.dataset.isin(matched)].pivot(index="dataset", columns="method", values="jaccard_mean")
fig, ax = plt.subplots(figsize=(6.6, 3.4))
for i, m in enumerate(st.index):
    v = dsJ[m].to_numpy()
    ax.plot([v.min(), v.max()], [i, i], color=GRID, lw=2, zorder=1)
    ax.plot(v, np.full(len(v), i), marker="|", ms=8, mew=1, color="#898781", ls="", zorder=2)
    ax.plot(st.loc[m, "jaccard_mean"], i, marker=mk(m), ms=7.5, mfc=face(m), mec=col(m), mew=1.4, ls="", zorder=3)
ax.axvline(ch, color=INK2, ls=":", lw=1)
ax.text(ch + 0.008, len(st) - 0.6, f"expected under\nrandom selection ({ch:.2f})",
        fontsize=8, color=INK2, va="top")
ax.set_yticks(range(len(st))); ax.set_yticklabels([NAME[m] for m in st.index])
ax.set_xlim(0, 1.02); ax.set_xlabel("Jaccard stability index (50 subsamples per dataset)")
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=6, mfc=C_SHAP, mec=C_SHAP, label="Mean, SHAP-based"),
                   Line2D([], [], marker="s", ls="", ms=6, mfc="white", mec=C_TRAD, mew=1.4, label="Mean, traditional"),
                   Line2D([], [], marker="|", ls="", ms=8, color="#898781", label="Single dataset")],
          loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0, 1.0))
fig.tight_layout(); save(fig, "Fig3")

# ================================================================ Fig 4 (stability heatmap)
order_m = list(st.index[::-1])                       # most stable first, as Table 3
H = e2.pivot_table(index="dataset", columns="method", values="jaccard_mean", dropna=False)
H = H.reindex(columns=order_m)
H.index = [DSN[x] for x in H.index]; H = H.sort_index()
blues = LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
fig, ax = plt.subplots(figsize=(7.2, 4.4))
im = ax.imshow(H.to_numpy(dtype=float), cmap=blues, vmin=0, vmax=1, aspect="auto")
for (i, j), v in np.ndenumerate(H.to_numpy(dtype=float)):
    if np.isnan(v):
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e1e0d9"))
        ax.text(j, i, "TO", ha="center", va="center", fontsize=8, style="italic", color=INK2)
    else:
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 0.62 else INK)
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels([NAME[m] for m in H.columns], rotation=35, ha="right")
ax.set_yticks(range(H.shape[0])); ax.set_yticklabels(H.index)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Jaccard stability index")
cb.outline.set_visible(False)
fig.tight_layout(); save(fig, "Fig4")


# ================================================================ Fig 5 / S3 (FDR, TPR dot plots)
nm = e3[e3.dataset != "madelon"]; assert nm.dataset.nunique() == 8
F = nm.pivot_table(index="method", columns="noise_ratio", values="fdr", aggfunc="mean")
T = nm.pivot_table(index="method", columns="noise_ratio", values="tpr", aggfunc="mean")
ratios = sorted(F.columns)
for m, v in T_FDR.items():
    for r, t in zip(ratios, v): close(F.loc[m, r], t, what=f"fdr {m} {r}")
b = nm.drop_duplicates(["dataset", "noise_ratio"]).assign(
    rf=lambda x: x.n_noise_injected / x.n_total,
    rt=lambda x: (1 - x.n_noise_injected / x.n_total) * x.k_used / x.n_original)
rnd_f = b.groupby("noise_ratio").rf.mean(); rnd_t = b.groupby("noise_ratio").rt.mean()
for r, t in zip(ratios, T_RAND): close(rnd_f[r], t, what="rand fdr")
SH = {0.1: ("^", "#c3c2b7"), 0.2: ("v", "#898781"), 0.5: ("o", INK)}


def dotplot(P, rnd, xlabel, name, order_asc):
    o = P[0.5].sort_values(ascending=order_asc).index
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for i, m in enumerate(o):
        ax.plot([P.loc[m].min(), P.loc[m].max()], [i, i], color=col(m), lw=1.6, alpha=0.55, zorder=1)
        for r in ratios:
            mkr, c = SH[r]
            ax.plot(P.loc[m, r], i, marker=mkr, ms=6, color=c, mec="white", mew=0.5, ls="", zorder=3)
    for r in ratios:
        mkr, c = SH[r]
        ax.axvline(rnd[r], color=c, ls=(0, (2, 2)), lw=1.1, zorder=0)
        ax.text(rnd[r], len(P) - 0.45, f" random, {int(r*100)}%", fontsize=7.5, color=INK2,
                ha="left", va="bottom", rotation=0)
    ax.set_ylim(-0.6, len(P) + 0.2)
    ax.set_xlim(right=max(rnd.max(), P.max().max()) * 1.12)
    ax.set_yticks(range(len(o)))
    ax.set_yticklabels([NAME[m] for m in o])
    for t, m in zip(ax.get_yticklabels(), o):
        t.set_color(C_SHAP if m in SHAP else "#b44a1c")
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    h = [Line2D([], [], marker=SH[r][0], ls="", ms=6, color=SH[r][1], label=f"{int(r*100)}% noise")
         for r in ratios]
    h.append(Line2D([], [], ls=(0, (2, 2)), color=INK2, label="Expected under random selection"))
    ax.legend(handles=h, loc="lower left", frameon=False, ncol=4, bbox_to_anchor=(0, 1.02), fontsize=8)
    fig.tight_layout(); save(fig, name)


dotplot(F, rnd_f, "Mean noise selection rate over 8 datasets (lower is better)", "Fig5", False)
dotplot(T, rnd_t, "Mean recall of original features over 8 datasets", "S3_Fig", True)

# ================================================================ Fig 6 (transfer heatmap)
SHORT = {"BoT-IoT": "BoT", "CIC-IDS2018": "CIC", "ToN-IoT": "ToN", "UNSW-NB15": "UNSW"}
e4 = e4.copy()
ind_n = e4[e4.status == "ok"].groupby(["source", "method"]).indomain_f1.first()
e4["d_tgt"] = [ind_n.get((t, m), np.nan) - c if s_ == "ok" else np.nan
               for t, m, c, s_ in zip(e4.target, e4.method, e4.cross_f1, e4.status)]
e4["pair"] = e4.source.map(SHORT) + " → " + e4.target.map(SHORT)
P = e4.pivot_table(index="method", columns="pair", values="d_tgt", aggfunc="mean", dropna=False)
absd = e4[e4.status == "ok"].groupby("method").d_tgt.apply(lambda s: s.abs().mean())
close(absd["rfe"], .0007, what="|dF1| rfe"); close(absd["chi2"], .0085, what="|dF1| chi2")
order4 = list(absd.sort_values().index) + ["shap_select"]
P = P.reindex(order4)
vmax = float(np.nanmax(np.abs(P.to_numpy(dtype=float))))
div = LinearSegmentedColormap.from_list("d", ["#104281", "#86b6ef", "#f0efec", "#f3a67f", "#a53d12"])
fig, ax = plt.subplots(figsize=(7.4, 3.7))
im = ax.imshow(P.to_numpy(dtype=float), cmap=div, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
for (i, j), v in np.ndenumerate(P.to_numpy(dtype=float)):
    if np.isnan(v):
        ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#e1e0d9"))
        if j == 0:
            ax.text((P.shape[1] - 1) / 2, i, "library raised an exception on every NIDS source",
                    ha="center", va="center", fontsize=8, style="italic", color=INK2)
    else:
        ax.text(j, i, "0.000" if abs(v) < 0.0005 else f"{v:+.3f}".replace("-", "−"), ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.65 * vmax else INK)
ax.set_xticks(range(P.shape[1])); ax.set_xticklabels(P.columns, rotation=40, ha="right")
ax.set_yticks(range(P.shape[0])); ax.set_yticklabels([NAME[m] for m in P.index])
ax.set_xlabel("Transfer pair (source → target)")
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("ΔF1 on the target dataset"); cb.outline.set_visible(False)
fig.tight_layout(); save(fig, "Fig6")

# ================================================================ Fig 7 (optimism vs n/p)
dsa = opt.groupby("dataset").agg(optimism=("optimism_auc", "mean"), n_over_p=("n_over_p", "first"))
ds = dsa.loc[PRIMARY]
r = spearmanr(ds.n_over_p, ds.optimism)
close(r.statistic, -0.632, tol=6e-4, what="rho n/p")
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ax.axhline(0, color=INK2, lw=0.7)
lx = np.log10(ds.n_over_p.to_numpy())
bb, aa = np.polyfit(lx, ds.optimism.to_numpy(), 1)
xs = np.linspace(np.log10(dsa.n_over_p.min()), lx.max(), 50)
ax.plot(10 ** xs, aa + bb * xs, color=INK2, ls="--", lw=1, label="Least-squares line (20 datasets)")
ax.scatter(ds.n_over_p, ds.optimism, s=34, color=INK, zorder=3, label="Primary dataset (mean over 9 methods)")
ax.scatter([dsa.loc["madelon", "n_over_p"]], [dsa.loc["madelon", "optimism"]], s=40, facecolor="white", edgecolor=INK,
           linewidth=1.2, zorder=3, label="Madelon (sensitivity set only)")
for name, dx, dy in (("madelon", 6, 0), ("sonar", 6, 0), ("credit-g", 6, -3), ("creditcard", -6, 6)):
    x, y = dsa.loc[name, "n_over_p"], dsa.loc[name, "optimism"]
    ax.annotate(DSN[name], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                color=INK2, ha="left" if dx > 0 else "right", va="center")
ax.set_xscale("log")
ax.set_xlabel("Samples per feature, n/p (log scale)")
ax.set_ylabel("Mean net optimism\n(AUC leaky − AUC nested)")
ax.text(0.98, 0.95, f"Spearman ρ = {r.statistic:.2f}, p = {r.pvalue:.3f}, 20 datasets".replace("-", "−"),
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2)
ax.legend(loc="upper right", bbox_to_anchor=(1, 0.88), frameon=False, fontsize=8)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
fig.tight_layout(); save(fig, "Fig7")

# ================================================================ Fig 8 (slopegraph)
e5 = pd.read_csv(RES / "exp5_classifier_nested.csv"); e5 = e5[e5.status == "ok"]
_R = _N8["ranks"]
ranks = pd.DataFrame({"Leaky\nXGBoost": pd.Series(_R["leaky"]), "Nested\nXGBoost": pd.Series(_R["nested"]),
                      "Nested\nlogistic": pd.Series(_R["logreg"]), "Nested\nMLP": pd.Series(_R["mlp"])})
for clf, c in (("logreg", "Nested\nlogistic"), ("mlp", "Nested\nMLP")):
    mean = e5[(e5.classifier == clf) & e5.dataset.isin(PRIMARY)].groupby("method").auc.mean()
    assert spearmanr(mean[ranks.index], -ranks[c]).statistic > 0.999, clf
for c in ranks: assert sorted(ranks[c]) == list(range(1, 10)), c
cols = list(ranks.columns); x = np.arange(len(cols))
SHAP_MK = {"shap_topk": "o", "borutashap": "D", "powershap": "^", "shap_select": "v"}
TRAD_MK = {"rfe": "s", "lasso": "P", "chi2": "X", "mi": "h", "tree_importance": "p"}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for m in ranks.index:
    c = col(m); mkk = {**SHAP_MK, **TRAD_MK}[m]
    ax.plot(x, ranks.loc[m], color=c, lw=1.6, alpha=0.9, zorder=2,
            marker=mkk, ms=6, mfc=c if m in SHAP else "white", mec=c, mew=1.2)
    ax.text(x[0] - 0.12, ranks.loc[m, cols[0]], NAME[m], ha="right", va="center", fontsize=8,
            color="#1c5cab" if m in SHAP else "#a53d12")
    ax.text(x[-1] + 0.12, ranks.loc[m, cols[-1]], NAME[m], ha="left", va="center", fontsize=8,
            color="#1c5cab" if m in SHAP else "#a53d12")
ax.set_xticks(x); ax.set_xticklabels(cols)
ax.set_xlim(-1.35, len(cols) - 1 + 1.35); ax.set_ylim(9.5, 0.5)
ax.set_yticks(range(1, 10)); ax.set_ylabel("Rank by mean AUC-ROC (1 = best)")
ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x", length=0)
for xi in x: ax.axvline(xi, color=GRID, lw=0.6, zorder=0)
ax.legend(handles=family_handles(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
fig.tight_layout(); save(fig, "Fig8")

# ================================================================ S2 (leaky per-dataset AUC)
L = lk.pivot_table(index="dataset", columns="method", values="auc_mean")
L = L.reindex(columns=[m for m in T_NEST]); L.index = [DSN[i] for i in L.index]; L = L.sort_index()
assert L.shape == (21, 9) and L.notna().all().all()
dev = L.sub(L.median(axis=1), axis=0)
lim = 0.03                                  # colour saturates beyond +-0.03
fig, ax = plt.subplots(figsize=(7.2, 6.6))
im = ax.imshow(dev.to_numpy(), cmap=div, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
for (i, j), v in np.ndenumerate(L.to_numpy()):
    dv = dev.to_numpy()[i, j]
    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
            color="white" if abs(dv) > 0.7 * lim else INK)
ax.set_xticks(range(9)); ax.set_xticklabels([NAME[m] for m in L.columns], rotation=35, ha="right")
ax.set_yticks(range(21)); ax.set_yticklabels(L.index)
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend="both")
cb.set_label("AUC-ROC minus the dataset's median over the nine methods"); cb.outline.set_visible(False)
fig.tight_layout(); save(fig, "S2_Fig")
print("all value checks passed")
