#!/usr/bin/env python3
"""
round5_fix_madelon_exp3.py -- rescore Madelon rows of results/exp3_noise.csv
against the corrected putative relevant set (see madelon_putative_set.py and S1 Appendix).

The Round 5 rerun of Exp. 3 used --resume, so only borutashap / powershap /
shap_select were recomputed (and scored against the corrected 20-feature
cluster). The six other methods' Madelon rows were left from the original run,
scored against the wrong f0..f19 set. Their *selections* do not depend on the
ground truth, and every row stores `selected_features`, so they can be rescored
exactly without re-running selection. Idempotent.
"""
import re, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from metrics import fdr_tpr

src = (ROOT / "experiments/exp3_noise.py").read_text()
T = set(re.findall(r'"(f\d+)"', src[src.index("MADELON_TRUE_FEATURES = {"):
                                     src.index("assert len(MADELON_TRUE_FEATURES)")]))
assert len(T) == 20
p = ROOT / "results/exp3_noise.csv"
d = pd.read_csv(p)
m = (d.dataset == "madelon") & (d.status == "ok")
n_changed = 0
for i in d[m].index:
    f, t = fdr_tpr(d.at[i, "selected_features"].split("|"), T)
    f, t = round(f, 4), round(t, 4)
    if (d.at[i, "fdr"], d.at[i, "tpr"]) != (f, t):
        n_changed += 1
    d.at[i, "fdr"], d.at[i, "tpr"] = f, t
d.to_csv(p, index=False)
print(f"rescored {m.sum()} Madelon rows, {n_changed} changed")
