#!/usr/bin/env python3
"""
experiments/exp7_protocol.py -- Experiment 7: protocol axis P3, and the
cross-axis synthesis over P1, P2 and P3.

Experiment 7 is an analysis layer, not a new selection run.  It pairs the two
protocol conditions over the Experiment 1 substrate -- 21 datasets x 9 methods,
evaluated under the leaky protocol (selection on the full dataset, then
cross-validation) and under the nested protocol (selection confined to each
outer training split) -- and asks how much of the ranking is a property of the
methods rather than of the protocol.

Quantities produced (CLAUDE_CODE_PROMPT_V3.md section 5.2):
  (a) optimism per method, with a paired Wilcoxon signed-rank test
  (b) ranking displacement, leaky vs nested
  (c) optimism by method family, Kruskal-Wallis
  (d) optimism against dataset geometry (n/p, k/p)
  (e) fold-to-fold selection stability, and its relation to Exp 2's
      bootstrap stability
  (f) survival of the two load-bearing claims, with Friedman/Nemenyi
  (g) the three-axis synthesis table and the stable/fragile split

Outputs:
    results/exp7_protocol.csv          per dataset x method optimism
    results/exp7_method_summary.csv    per-method optimism and ranks
    results/exp7_axes.csv              the three-axis synthesis
    results/exp7_stability.csv         fold-to-fold vs bootstrap Jaccard
    tables/nemenyi_exp1_nested.csv     nested post-hoc matrix
"""
from __future__ import annotations

import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import (friedmanchisquare, kruskal, rankdata,  # noqa: E402
                         spearmanr, wilcoxon)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

warnings.filterwarnings("ignore")

RES = Path(config.RESULTS_DIR)
TAB = Path("tables")

FAMILY = {
    "rfe": "wrapper",
    "lasso": "embedded", "tree_importance": "embedded",
    "chi2": "filter", "mi": "filter",
    "shap_topk": "SHAP", "borutashap": "SHAP",
    "powershap": "SHAP", "shap_select": "SHAP",
}
METHODS = list(FAMILY)
ADAPTIVE = ["borutashap", "powershap", "shap_select"]
SHAP_METHODS = [m for m, f in FAMILY.items() if f == "SHAP"]
TRADITIONAL = [m for m, f in FAMILY.items() if f != "SHAP"]


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    leaky = pd.read_csv(RES / "exp1_performance.csv")
    leaky = leaky[leaky.status == "ok"].drop_duplicates(["dataset", "method"])
    nested = pd.read_csv(RES / "exp1_performance_nested.csv")
    nested = nested[nested.status == "ok"].drop_duplicates(["dataset",
                                                            "method"])
    common = sorted(set(leaky.dataset) & set(nested.dataset))
    return (leaky[leaky.dataset.isin(common)],
            nested[nested.dataset.isin(common)])


def _ranks(series: pd.Series) -> pd.Series:
    """Rank 1 = best (highest mean AUC)."""
    return pd.Series(rankdata(-series.to_numpy()), index=series.index)


# ─────────────────────────────────────────────────────────────────────────────
# (a)-(d) optimism
# ─────────────────────────────────────────────────────────────────────────────

def optimism(leaky: pd.DataFrame, nested: pd.DataFrame) -> pd.DataFrame:
    merged = leaky.merge(
        nested, on=["dataset", "method"], suffixes=("_leaky", "_nested"))
    merged["optimism_auc"] = merged.auc_mean_leaky - merged.auc_mean_nested
    merged["optimism_f1"] = merged.f1_mean_leaky - merged.f1_mean_nested
    merged["family"] = merged.method.map(FAMILY)
    merged["n_over_p"] = merged.n_samples_leaky / merged.n_features_leaky
    merged["k_over_p"] = merged.k_leaky / merged.n_features_leaky
    cols = ["dataset", "method", "family", "n_samples_leaky",
            "n_features_leaky", "k_leaky", "n_over_p", "k_over_p",
            "auc_mean_leaky", "auc_mean_nested", "optimism_auc",
            "f1_mean_leaky", "f1_mean_nested", "optimism_f1"]
    out = merged[cols].rename(columns={"n_samples_leaky": "n_samples",
                                       "n_features_leaky": "n_features",
                                       "k_leaky": "k"})
    out.to_csv(RES / "exp7_protocol.csv", index=False)
    return out


def method_summary(opt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in opt.groupby("method"):
        d = g.optimism_auc.to_numpy()
        try:
            stat, p = wilcoxon(g.auc_mean_leaky, g.auc_mean_nested)
        except ValueError:                      # all differences zero
            stat, p = np.nan, 1.0
        rows.append({
            "method": m, "family": FAMILY[m], "n_datasets": len(g),
            "auc_leaky": g.auc_mean_leaky.mean(),
            "auc_nested": g.auc_mean_nested.mean(),
            "optimism_mean": d.mean(), "optimism_std": d.std(ddof=1),
            "optimism_min": d.min(), "optimism_max": d.max(),
            "wilcoxon_stat": stat, "wilcoxon_p": p,
        })
    df = pd.DataFrame(rows).set_index("method")
    df["rank_leaky"] = _ranks(df.auc_leaky)
    df["rank_nested"] = _ranks(df.auc_nested)
    df["rank_change"] = df.rank_leaky - df.rank_nested
    return df.sort_values("optimism_mean", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# (e) fold-to-fold selection stability
# ─────────────────────────────────────────────────────────────────────────────

def fold_stability() -> pd.DataFrame:
    folds = pd.read_csv(RES / "exp1_folds_nested.csv")
    folds["selected_features"] = folds.selected_features.fillna("")
    rows = []
    for (ds, m), g in folds.groupby(["dataset", "method"]):
        sets = [set(s.split("|")) - {""} for s in g.selected_features]
        js = [len(a & b) / len(a | b)
              for a, b in combinations(sets, 2) if (a | b)]
        rows.append({"dataset": ds, "method": m,
                     "fold_jaccard": float(np.mean(js)) if js else np.nan})
    df = pd.DataFrame(rows)

    # exp2_stability.csv carries exact-duplicate re-run rows for a few keys
    # (generate_tables.py dedups the same way); an unguarded merge would
    # silently fan out into duplicate (dataset, method) pairs here.
    boot = pd.read_csv(RES / "exp2_stability.csv").drop_duplicates(
        ["dataset", "method"], keep="first")
    jcol = next((c for c in boot.columns
                 if "jaccard" in c.lower() and "mean" in c.lower()),
                None) or next(c for c in boot.columns
                              if "jaccard" in c.lower())
    boot = boot.rename(columns={jcol: "bootstrap_jaccard"})
    merged = df.merge(boot[["dataset", "method", "bootstrap_jaccard"]],
                      on=["dataset", "method"], how="left")
    merged.to_csv(RES / "exp7_stability.csv", index=False)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# (f) Friedman / Nemenyi on the nested arm
# ─────────────────────────────────────────────────────────────────────────────

def friedman_nested(nested: pd.DataFrame) -> dict:
    pivot = nested.pivot(index="dataset", columns="method", values="auc_mean")
    pivot = pivot[[m for m in METHODS if m in pivot.columns]].dropna()
    stat, p = friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
    mr = pivot.rank(axis=1, ascending=False).mean().sort_values()

    posthoc = sp.posthoc_nemenyi_friedman(pivot.to_numpy())
    posthoc.index = posthoc.columns = pivot.columns
    TAB.mkdir(exist_ok=True)
    posthoc.round(4).to_csv(TAB / "nemenyi_exp1_nested.csv")

    return {"n_datasets": len(pivot), "chi2": float(stat), "p": float(p),
            "mean_ranks": mr, "pivot": pivot, "posthoc": posthoc}


# ─────────────────────────────────────────────────────────────────────────────
# (g) the three protocol axes on one scale
# ─────────────────────────────────────────────────────────────────────────────

def axis_p1(nested: pd.DataFrame) -> dict:
    """Downstream classifier: XGBoost vs logistic regression vs MLP,
    all scoring the same nested per-fold subsets."""
    e5 = pd.read_csv(RES / "exp5_classifier_nested.csv")
    e5 = e5[e5.status == "ok"]
    ref = nested.groupby("method").auc_mean.mean()
    ref = ref[[m for m in METHODS if m in ref.index]]
    r_ref = _ranks(ref)

    out = {"reference_ranks": r_ref, "orderings": {}, "rho": {}}
    for clf, g in e5.groupby("classifier"):
        s = g.groupby("method").auc.mean().reindex(ref.index)
        r = _ranks(s)
        out["orderings"][clf] = r
        out["rho"][clf] = float(spearmanr(r_ref, r).statistic)
    # agreement between the two non-tree learners
    if {"logreg", "mlp"} <= set(out["orderings"]):
        out["rho"]["logreg_vs_mlp"] = float(
            spearmanr(out["orderings"]["logreg"],
                      out["orderings"]["mlp"]).statistic)
    changes = {clf: (r_ref - r) for clf, r in out["orderings"].items()}
    out["changes"] = changes
    return out


def axis_p2(nested: pd.DataFrame) -> dict:
    """Feature budget: fixed k vs each method's native stopping rule.

    Only three of the nine selectors have a native rule, so the six others
    are held at their fixed-k nested AUC.  The caption must say so.
    """
    e6 = pd.read_csv(RES / "exp6_native_nested.csv")
    e6 = e6[e6.status == "ok"]
    ref = nested.groupby("method").auc_mean.mean()
    ref = ref[[m for m in METHODS if m in ref.index]]
    native = ref.copy()
    for m in ADAPTIVE:
        g = e6[e6.method == m]
        if len(g):
            native[m] = g.auc_native.mean()
    r_ref, r_nat = _ranks(ref), _ranks(native)
    return {"reference_ranks": r_ref, "native_ranks": r_nat,
            "rho": float(spearmanr(r_ref, r_nat).statistic),
            "changes": r_ref - r_nat, "native_auc": native,
            "n_adaptive_covered": int(e6.method.nunique())}


def axis_p3(summary: pd.DataFrame) -> dict:
    r_leaky, r_nested = summary.rank_leaky, summary.rank_nested
    # Sign convention shared with P1 and P2: change = rank(nested XGBoost
    # reference) - rank(condition), here condition = leaky.  (Before Round 6
    # this was written leaky - nested, which reversed the sign in Table 12.)
    return {"rho": float(spearmanr(r_leaky, r_nested).statistic),
            "changes": r_nested - r_leaky}


def synthesis(p1: dict, p2: dict, p3: dict) -> pd.DataFrame:
    def worst(ch: pd.Series) -> tuple[str, float]:
        i = ch.abs().idxmax()
        return i, float(ch[i])

    rows = []
    for clf in ("logreg", "mlp"):
        if clf in p1["changes"]:
            m, d = worst(p1["changes"][clf])
            rows.append({"axis": "P1", "condition": f"XGBoost → {clf}",
                         "spearman_rho": p1["rho"][clf],
                         "max_rank_change": abs(d), "method_most_affected": m})
    m, d = worst(p2["changes"])
    rows.append({"axis": "P2", "condition": "fixed k → native threshold",
                 "spearman_rho": p2["rho"], "max_rank_change": abs(d),
                 "method_most_affected": m})
    m, d = worst(p3["changes"])
    rows.append({"axis": "P3", "condition": "leaky → nested",
                 "spearman_rho": p3["rho"], "max_rank_change": abs(d),
                 "method_most_affected": m})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "exp7_axes.csv", index=False)
    return df


def stable_vs_fragile(p1: dict, p2: dict, p3: dict) -> pd.DataFrame:
    cols = {}
    for clf, ch in p1["changes"].items():
        if clf in ("logreg", "mlp"):
            cols[f"P1_{clf}"] = ch
    cols["P2"] = p2["changes"]
    cols["P3"] = p3["changes"]
    df = pd.DataFrame(cols)
    df["max_abs_change"] = df.abs().max(axis=1)
    df["verdict"] = np.where(df.max_abs_change <= 1, "stable", "fragile")
    return df.sort_values("max_abs_change")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    leaky, nested = _load()
    print(f"paired substrate: {nested.dataset.nunique()} datasets x "
          f"{nested.method.nunique()} methods = {len(nested)} rows\n")

    opt = optimism(leaky, nested)
    summ = method_summary(opt)

    print("(a) optimism per method  (AUC_leaky - AUC_nested)")
    print(summ[["family", "auc_leaky", "auc_nested", "optimism_mean",
                "optimism_std", "optimism_min", "optimism_max",
                "wilcoxon_p"]].round(4).to_string())

    print("\n(b) ranking displacement, leaky -> nested")
    print(summ[["auc_leaky", "rank_leaky", "auc_nested", "rank_nested",
                "rank_change"]].round(4).to_string())
    rho_p3 = spearmanr(summ.rank_leaky, summ.rank_nested).statistic
    biggest = summ.rank_change.abs().idxmax()
    print(f"    Spearman rho = {rho_p3:.4f}; largest move "
          f"{biggest} {summ.rank_leaky[biggest]:.0f} -> "
          f"{summ.rank_nested[biggest]:.0f}")

    print("\n(c) optimism by family")
    fam = opt.groupby("family").optimism_auc.agg(["mean", "std", "count"])
    print(fam.round(4).to_string())
    groups = [g.optimism_auc.to_numpy() for _, g in opt.groupby("family")]
    H, p_kw = kruskal(*groups)
    print(f"    Kruskal-Wallis H = {H:.3f}, p = {p_kw:.4g}")

    print("\n(d) optimism vs dataset geometry")
    per_ds = opt.groupby("dataset").agg(
        optimism=("optimism_auc", "mean"),
        n_over_p=("n_over_p", "first"), k_over_p=("k_over_p", "first"))
    for col in ("n_over_p", "k_over_p"):
        r = spearmanr(per_ds[col], per_ds.optimism)
        print(f"    rho(optimism, {col}) = {r.statistic:+.4f}, "
              f"p = {r.pvalue:.4g}")

    print("\n(e) fold-to-fold vs bootstrap stability")
    st = fold_stability()
    by_m = st.groupby("method").agg(
        fold_jaccard=("fold_jaccard", "mean"),
        bootstrap_jaccard=("bootstrap_jaccard", "mean")).sort_values(
            "fold_jaccard", ascending=False)
    print(by_m.round(4).to_string())
    sub = st.dropna(subset=["bootstrap_jaccard"])
    if len(sub) > 3:
        r = spearmanr(sub.fold_jaccard, sub.bootstrap_jaccard)
        print(f"    rho(fold, bootstrap) = {r.statistic:+.4f}, "
              f"p = {r.pvalue:.4g}  (n = {len(sub)} pairs)")

    print("\n(f) load-bearing claims under the nested protocol")
    fr = friedman_nested(nested)
    print(f"    Friedman chi2 = {fr['chi2']:.3f}, p = {fr['p']:.4g}, "
          f"N = {fr['n_datasets']}")
    print("    mean ranks: " + ", ".join(
        f"{m} {v:.2f}" for m, v in fr["mean_ranks"].items()))

    mean_auc = nested.groupby("method").auc_mean.mean()
    best_shap = mean_auc[[m for m in SHAP_METHODS
                          if m != "shap_select"]].idxmax()
    best_trad = mean_auc[TRADITIONAL].idxmax()
    gap = mean_auc[best_shap] - mean_auc[best_trad]
    print(f"    claim 1 -- best SHAP (excl. SHAP-Select) = {best_shap} "
          f"{mean_auc[best_shap]:.4f}; best traditional = {best_trad} "
          f"{mean_auc[best_trad]:.4f}; gap = {gap:+.4f} "
          f"-> {'SURVIVES' if gap <= 0 else 'FAILS'}")

    order = mean_auc.sort_values(ascending=False)
    rank_ss = int(list(order.index).index("shap_select")) + 1
    gap8 = order.iloc[-2] - order.iloc[-1]
    print(f"    claim 2 -- SHAP-Select rank = {rank_ss}/9, gap to 8th "
          f"({order.index[-2]}) = {gap8:.4f} "
          f"-> {'SURVIVES' if rank_ss == 9 else 'FAILS'}")

    print("\n(g) cross-axis synthesis")
    p1, p2 = axis_p1(nested), axis_p2(nested)
    p3 = axis_p3(summ)
    ax = synthesis(p1, p2, p3)
    print(ax.round(4).to_string(index=False))
    print(f"    P1 logreg vs mlp agreement: "
          f"rho = {p1['rho'].get('logreg_vs_mlp', float('nan')):.4f}")

    sf = stable_vs_fragile(p1, p2, p3)
    print("\n    rank stability across all axes (change <= 1 = stable)")
    print(sf.round(2).to_string())

    summ.to_csv(RES / "exp7_method_summary.csv")
    by_m.to_csv(RES / "exp7_stability_by_method.csv")
    sf.to_csv(RES / "exp7_stable_fragile.csv")

    (RES / "exp7_headline.json").write_text(json.dumps({
        "rho_p3": float(rho_p3),
        "rho_p1": {k: float(v) for k, v in p1["rho"].items()},
        "rho_p2": float(p2["rho"]),
        "kruskal_H": float(H), "kruskal_p": float(p_kw),
        "friedman_chi2": fr["chi2"], "friedman_p": fr["p"],
        "claim1_best_shap": best_shap, "claim1_best_trad": best_trad,
        "claim1_gap": float(gap), "claim1_survives": bool(gap <= 0),
        "claim2_rank_shap_select": rank_ss,
        "claim2_gap_to_8th": float(gap8),
        "claim2_survives": bool(rank_ss == 9),
    }, indent=2))
    print(f"\nwrote {RES/'exp7_protocol.csv'}, {RES/'exp7_method_summary.csv'},"
          f" {RES/'exp7_axes.csv'}, {RES/'exp7_headline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
