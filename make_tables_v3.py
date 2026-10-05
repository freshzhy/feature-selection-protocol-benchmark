#!/usr/bin/env python3
"""Round-7 tables (markdown rows) for the revised manuscript, from results/*.csv
and results/round7_numbers.json.  Writes plos/v3/tables/T*.md (rows only)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "tables" / "manuscript"; OUT.mkdir(parents=True, exist_ok=True)
N = json.loads((RES / "round7_numbers.json").read_text())
R6 = json.loads((RES / "round6_bootstrap_ci.json").read_text())
NAME = {"shap_topk": "SHAP Top-K", "borutashap": "BorutaSHAP†", "powershap": "PowerSHAP†",
        "shap_select": "SHAP-Select†", "mi": "Mutual Information", "chi2": "Chi-Square",
        "rfe": "RFE", "lasso": "Lasso", "tree_importance": "Tree Importance"}
FAM = {m: ("SHAP" if m in ("shap_topk", "borutashap", "powershap", "shap_select") else "Traditional") for m in NAME}
NAT = {"borutashap": "BorutaSHAP", "powershap": "PowerSHAP", "shap_select": "SHAP-Select"}


def minus(x, f):
    return (f.format(x)).replace("-", "−")


def pfmt(p):
    return "< 0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def write(name, lines):
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------- Table 3 stability (Jaccard only)
e2 = pd.read_csv(RES / "exp2_stability.csv")
pv = e2.pivot(index="dataset", columns="method", values="jaccard_mean").dropna()
st = pd.DataFrame({"mean": pv.mean(), "sd": pv.std(), "min": pv.min(), "max": pv.max()}).sort_values("mean", ascending=False)
L = ["| Method | Family | Jaccard (mean ± s.d.) | Min | Max |", "|---|---|---|---|---|"]
for i, (m, r) in enumerate(st.iterrows()):
    mean = f"**{r['mean']:.4f}**" if i == 0 else f"{r['mean']:.4f}"
    L.append(f"| {NAME[m]} | {FAM[m]} | {mean} ± {r['sd']:.4f} | {r['min']:.3f} | {r['max']:.3f} |")
write("T3", L)

# ---------------- Table 5 transfer (target baseline)
t = pd.read_csv(RES / "round7_transfer.csv", index_col=0)
L = ["| Method | Family | Feature Jaccard | Cross-domain F1 | ΔF1 (signed) | \\|ΔF1\\| (nested) | \\|ΔF1\\| (leaky) | Max \\|ΔF1\\| |",
     "|---|---|---|---|---|---|---|---|"]
for i, (m, r) in enumerate(t.iterrows()):
    a = f"**{r.d_abs:.4f}**" if i == 0 else f"{r.d_abs:.4f}"
    L.append(f"| {NAME[m]} | {FAM[m]} | {r.jaccard:.4f} | {r.cross_f1:.4f} | {minus(r.d_signed, '{:+.4f}')} | {a} | {r.d_abs_leaky:.4f} | {r.d_max:.4f} |")
L.append("| SHAP-Select† | SHAP | – | – | – | – | – | – |")
write("T5", L)

# ---------------- Table 7 P2 coverage + common-support comparison
cov = {c["method"]: c for c in N["p2_coverage"]}
p2 = {r["method"]: r for r in N["p2_table"]}
e6 = pd.read_csv(RES / "exp6_native_nested.csv")
L = ["| Method | Datasets finished | Datasets with all 5 folds non-empty | Empty folds / folds run | Native size, common set | AUC fixed k | AUC native | Δ | *p* | *p* (Holm) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for m in ("borutashap", "powershap", "shap_select"):
    c, r = cov[m], p2[m]
    fin = 21 - len(c["not_finished"])
    L.append(f"| {NAT[m]} | {fin} / 21 | {c['complete']} / 21 | {c['empty_folds']} / {c['folds_attempted']} | {r['native_size']:.1f} | "
             f"{r['auc_fixed']:.4f} | {r['auc_native']:.4f} | {minus(r['delta'], '{:+.4f}')} | {pfmt(r['p'])} | {pfmt(r['p_holm'])} |")
write("T7", L)

# ---------------- Table 8 P3 with Holm
summ = pd.read_csv(RES / "exp7_method_summary.csv", index_col=0)
H = N["p3_holm"]
L = ["| Method | Family | AUC (leaky) | AUC (nested) | Optimism (mean) | Optimism (s.d.) | *p* | *p* (Holm) | Rank (leaky) | Rank (nested) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for m, r in summ.sort_values("optimism_mean", ascending=False).iterrows():
    L.append(f"| {NAME[m]} | {r.family} | {r.auc_leaky:.4f} | {r.auc_nested:.4f} | {minus(r.optimism_mean, '{:+.4f}')} | {r.optimism_std:.4f} | "
             f"{pfmt(r.wilcoxon_p)} | {pfmt(H[m]['p_holm'])} | {int(r.rank_leaky)} | {int(r.rank_nested)} |")
write("T8", L)

# ---------------- Table 9 axes
pt = R6["point"]; tb = {r["quantity"]: r for r in R6["table"]}
ag = N["aggregation"]; mx = N["madelon_excluded"]; p2n = N["p2"]
ci = lambda q: f"[{tb[q]['ci_lo']:.2f}, {tb[q]['ci_hi']:.2f}]".replace("-", "−")
L = ["| Axis | Manipulation | Datasets | Spearman ρ | 95% bootstrap CI | ρ, mean-rank aggregation | ρ, Madelon excluded | Max rank change |",
     "|---|---|---|---|---|---|---|---|",
     f"| P1 | XGBoost → logistic regression | 21 | {pt['P1_logreg']:.2f} | {ci('P1_logreg')} | {ag['rank']['P1_logreg']:.2f} | {mx['P1_logreg']:.2f} | 6 |",
     f"| P1 | XGBoost → MLP | 21 | {pt['P1_mlp']:.2f} | {ci('P1_mlp')} | {ag['rank']['P1_mlp']:.2f} | {mx['P1_mlp']:.2f} | 4 |",
     f"| P2 | fixed *k* → native rule (common support) | {p2n['common_n']} | {p2n['rho_common']:.2f} | "
     f"[{p2n['ci_common'][0]:.2f}, {p2n['ci_common'][1]:.2f}] | {ag['rank']['P2_common']:.2f} | (Madelon not in set) | {int(p2n['max_change_common'])} |".replace("[-", "[−"),
     f"| P3 | leaky → nested | 21 | {pt['P3']:.2f} | {ci('P3')} | {ag['rank']['P3']:.2f} | {mx['P3']:.2f} | 4 |"]
write("T9", L)

# ---------------- Table 10 rank changes (P2 = common support)
sf = pd.read_csv(RES / "exp7_stable_fragile.csv", index_col=0)
rr, rn = p2n["ranks_ref_common"], p2n["ranks_native_common"]
rows = []
for m in sf.index:
    d = [int(sf.loc[m, "P1_logreg"]), int(sf.loc[m, "P1_mlp"]), int(rr[m] - rn[m]), int(sf.loc[m, "P3"])]
    mxa = max(abs(x) for x in d)
    rows.append((mxa, m, d))
L = ["| Method | Δ P1 (logistic) | Δ P1 (MLP) | Δ P2 (common support) | Δ P3 | Max abs. Δ | Verdict |", "|---|---|---|---|---|---|---|"]
for mxa, m, d in sorted(rows):
    L.append(f"| {NAME[m]} | " + " | ".join(minus(x, '{:+d}') for x in d) + f" | {mxa} | {'stable' if mxa <= 1 else 'fragile'} |")
write("T10", L)
print("stable:", [m for mxa, m, d in rows if mxa <= 1])

# ---------------- Table D (S1) composite
c = pd.read_csv(RES / "round7_composite.csv", index_col=0)
L = ["| Method | Performance (nested) | Stability | Noise selection | Transfer (nested) | Mean rank, SHAP-Select transfer = 9 | Mean rank, available dimensions |",
     "|---|---|---|---|---|---|---|"]
for m, r in c.iterrows():
    tr = "–" if np.isnan(r.transfer) else int(r.transfer)
    L.append(f"| {NAME[m]} | {int(r.performance)} | {int(r.stability)} | {int(r.noise)} | {tr} | {r.mean_last:.2f} | {r.mean_avail:.2f} |")
write("TD", L)

# ---------------- Table C (S1) datasets provenance
src = {"diabetes": ("OpenML", "37"), "credit-g": ("OpenML", "31"), "breast-w": ("OpenML", "15"),
       "credit-approval": ("OpenML", "29"), "kr-vs-kp": ("OpenML", "3"), "banknote-authentication": ("OpenML", "1462"),
       "blood-transfusion": ("OpenML", "1464"), "phoneme": ("OpenML", "1489"), "qsar-biodeg": ("OpenML", "1494"),
       "steel-plates-fault": ("OpenML", "40982"), "car": ("OpenML", "40975"), "adult": ("OpenML", "1590"),
       "kc1": ("OpenML", "1063"), "kc2": ("OpenML", "1067"),
       "madelon": ("UCI", "Madelon (training file)"), "ionosphere": ("UCI", "Ionosphere"),
       "sonar": ("UCI", "Connectionist Bench (Sonar)"), "spambase": ("UCI", "Spambase"),
       "creditcard": ("Kaggle", "Credit Card Fraud Detection"), "telco-churn": ("Kaggle", "Telco Customer Churn"),
       "hr-attrition": ("Kaggle", "IBM HR Analytics Employee Attrition")}
write("TC_src", [json.dumps(src)])
print("ok")
