#!/usr/bin/env python3
"""Round-8 tables that depend on the classification substrate, on the 20-dataset
primary set (results/round8_numbers.json).  T3, T5, T7 (stability, transfer, P2)
do not involve Madelon and are produced by make_tables_v3.py."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for up in (ROOT, ROOT.parent, ROOT.parent.parent):
    if (up / "results" / "round8_numbers.json").exists():
        RES = up / "results"; break
OUT = ROOT / "tables" / "manuscript"; OUT.mkdir(parents=True, exist_ok=True)
N8 = json.loads((RES / "round8_numbers.json").read_text())
N7 = json.loads((RES / "round7_numbers.json").read_text())
pr, s21 = N8["primary"], N8["sensitivity21"]
NAME = {"shap_topk": "SHAP Top-K", "borutashap": "BorutaSHAP†", "powershap": "PowerSHAP†", "shap_select": "SHAP-Select†",
        "mi": "Mutual Information", "chi2": "Chi-Square", "rfe": "RFE", "lasso": "Lasso", "tree_importance": "Tree Importance"}
FAM = {m: ("SHAP" if m in ("shap_topk", "borutashap", "powershap", "shap_select") else "Traditional") for m in NAME}
mi = lambda x, f: f.format(x).replace("-", "−")
pf = lambda p: "< 0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")
def write(n, L): (OUT / f"{n}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
def tfmt(t): return f"{t:.2f}" if t < 1 else (f"{t:.1f}" if t < 100 else f"{t:.0f}")

# Table 2
t2 = pr["table2"]; best = max(t2, key=lambda m: t2[m]["auc_ns"])
L = ["| Method | Family | AUC (nested) | AUC (leaky) | Δ | Macro F1 (nested) | Accuracy (nested) | Selection time, s (nested) |",
     "|---|---|---|---|---|---|---|---|"]
for m, r in t2.items():
    a = f"**{r['auc_ns']:.4f}**" if m == best else f"{r['auc_ns']:.4f}"
    L.append(f"| {NAME[m]} | {FAM[m]} | {a} ± {r['auc_ns_sd']:.4f} | {r['auc_lk']:.4f} | {mi(r['delta'], '{:+.4f}')} | "
             f"{r['f1']:.4f} | {r['acc']:.4f} | {tfmt(r['t_ns'])} |")
write("T2", L)

# Table 6
t6 = pr["table6"]; bl = max(t6, key=lambda m: t6[m]["lr"]); bm = max(t6, key=lambda m: t6[m]["ml"])
L = ["| Method | Family | AUC (XGBoost) | AUC (logistic) | AUC (MLP) | Rank Δ (logistic) | Rank Δ (MLP) |", "|---|---|---|---|---|---|---|"]
for i, (m, r) in enumerate(t6.items()):
    x = f"**{r['xgb']:.4f}**" if i == 0 else f"{r['xgb']:.4f}"
    l = f"**{r['lr']:.4f}**" if m == bl else f"{r['lr']:.4f}"
    ml = f"**{r['ml']:.4f}**" if m == bm else f"{r['ml']:.4f}"
    L.append(f"| {NAME[m]} | {FAM[m]} | {x} | {l} | {ml} | {mi(int(r['d_lr']), '{:+d}')} | {mi(int(r['d_ml']), '{:+d}')} |")
write("T6", L)

# Table 8
L = ["| Method | Family | AUC (leaky) | AUC (nested) | Net optimism (mean) | Net optimism (s.d.) | *p* | *p* (Holm) | Rank (leaky) | Rank (nested) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for m, r in pr["table8"].items():
    L.append(f"| {NAME[m]} | {r['family']} | {r['auc_lk']:.4f} | {r['auc_ns']:.4f} | {mi(r['opt_mean'], '{:+.4f}')} | {r['opt_sd']:.4f} | "
             f"{pf(r['p'])} | {pf(r['p_holm'])} | {int(r['rank_lk'])} | {int(r['rank_ns'])} |")
write("T8", L)

# Table 9
ci = lambda v: f"[{v[0]:.2f}, {v[1]:.2f}]".replace("-", "−")
p2 = N7["p2"]; ag7 = N7["aggregation"]["rank"]
L = ["| Axis | Manipulation | Datasets | Spearman ρ | 95% bootstrap CI | ρ, mean-rank aggregation | ρ, 21 datasets incl. Madelon | Max rank change |",
     "|---|---|---|---|---|---|---|---|",
     f"| P1 | XGBoost → logistic regression | 20 | {pr['point']['P1_logreg']:.2f} | {ci(pr['ci']['P1_logreg'])} | {pr['rank_agg']['P1_logreg']:.2f} | {s21['point']['P1_logreg']:.2f} | {int(pr['maxchange']['P1_logreg'])} |",
     f"| P1 | XGBoost → MLP | 20 | {pr['point']['P1_mlp']:.2f} | {ci(pr['ci']['P1_mlp'])} | {pr['rank_agg']['P1_mlp']:.2f} | {s21['point']['P1_mlp']:.2f} | {int(pr['maxchange']['P1_mlp'])} |",
     f"| P2 | fixed *k* → native rule (common support) | {p2['common_n']} | {p2['rho_common']:.2f} | {ci(p2['ci_common'])} | {ag7['P2_common']:.2f} | (Madelon not in set) | {int(p2['max_change_common'])} |",
     f"| P3 | leaky → nested | 20 | {pr['point']['P3']:.2f} | {ci(pr['ci']['P3'])} | {pr['rank_agg']['P3']:.2f} | {s21['point']['P3']:.2f} | {int(pr['maxchange']['P3'])} |"]
write("T9", L)

# Table 10
rr, rn = p2["ranks_ref_common"], p2["ranks_native_common"]; rows = []
for m, c in pr["rank_change"].items():
    d = [int(c["P1_logreg"]), int(c["P1_mlp"]), int(rr[m] - rn[m]), int(c["P3"])]
    rows.append((max(abs(x) for x in d), m, d))
L = ["| Method | Δ P1 (logistic) | Δ P1 (MLP) | Δ P2 (common support) | Δ P3 | Max abs. Δ | Verdict |", "|---|---|---|---|---|---|---|"]
for mx, m, d in sorted(rows):
    L.append(f"| {NAME[m]} | " + " | ".join(mi(x, '{:+d}') for x in d) + f" | {mx} | {'stable' if mx <= 1 else 'fragile'} |")
write("T10", L)
print("stable:", [m for mx, m, d in rows if mx <= 1])

# Table D (composite) with 20-dataset performance ranks
from scipy.stats import rankdata
c = pd.read_csv(RES / "round7_composite.csv", index_col=0)
perf = pd.Series({m: r["auc_ns"] for m, r in t2.items()})
c["performance"] = pd.Series(rankdata(-perf[c.index]), index=c.index)
c["mean_last"] = c[["performance", "stability", "noise"]].join(c.transfer.fillna(9)).mean(axis=1)
c["mean_avail"] = c[["performance", "stability", "noise", "transfer"]].mean(axis=1)
c = c.assign(_m=c.index).sort_values(["mean_last", "performance", "_m"], kind="mergesort").drop(columns="_m")  # stable tie-break by method id
L = ["| Method | Performance (nested) | Stability | Noise selection | Transfer (nested) | Mean rank, SHAP-Select transfer = 9 | Mean rank, available dimensions |",
     "|---|---|---|---|---|---|---|"]
for m, r in c.iterrows():
    tr = "–" if np.isnan(r.transfer) else int(r.transfer)
    L.append(f"| {NAME[m]} | {int(r.performance)} | {int(r.stability)} | {int(r.noise)} | {tr} | {r.mean_last:.2f} | {r.mean_avail:.2f} |")
write("TD", L)

# Table F (S1): 21-dataset unconstrained-completion sensitivity, classification
t = s21["table2"]
L = ["| Method | AUC nested (21) | AUC leaky (21) | Net optimism (21) | *p* (Holm, 21) | Rank nested (21) | Rank nested (20) |", "|---|---|---|---|---|---|---|"]
for m, r in t.items():
    o8 = s21["table8"][m]
    L.append(f"| {NAME[m]} | {r['auc_ns']:.4f} | {r['auc_lk']:.4f} | {mi(o8['opt_mean'], '{:+.4f}')} | {pf(o8['p_holm'])} | {int(o8['rank_ns'])} | {int(pr['table8'][m]['rank_ns'])} |")
write("TF", L)
print("ok")

# Table G (S1): multi-seed sensitivity, from results/multiseed/multiseed_summary.csv
ms = RES / "multiseed" / "multiseed_summary.csv"
if ms.exists():
    s = pd.read_csv(ms)
    L = ["| Seed | Nested leader | Top three (nested) | ρ P1 (logistic) | ρ P1 (MLP) | ρ P3 | RFE rank, leaky → nested |", "|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        top3 = " > ".join(NAME[x.strip()] for x in r.top3.split(">"))
        L.append(f"| {r.seed} | {NAME[r.nested_leader]} | {top3} | {r.rho_P1_logreg:.2f} | {r.rho_P1_mlp:.2f} | {r.rho_P3:.2f} | {r.rfe_rank_leaky} → {r.rfe_rank_nested} |".replace("| -", "| −"))
    write("TG", L)
