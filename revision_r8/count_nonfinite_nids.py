#!/usr/bin/env python3
"""Share of infinite and missing feature cells in the four NFv2 NIDS datasets,
before the Inf -> NaN -> 0 sanitisation applied by data_loader.py.

Two populations are reported for each dataset:
  full   : every row of the released CSV files (after dropping the identifier columns)
  sample : the stratified 100,000-row sample actually used in the experiments
Output: counts and proportions (results/nids_nonfinite.csv).  Slow on CSE-CIC-IDS2018."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
import config
from sklearn.model_selection import train_test_split

rows = []
for e in config.NIDS_DATASETS:
    d = config.NIDS_DIR / e["data_dir"]
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in sorted(d.glob("*.csv"))
                    if f.name not in config.NIDS_SKIP_FILES], ignore_index=True)
    df = df.drop(columns=[c for c in config.NIDS_DROP_COLS if c in df.columns]).dropna(subset=[e["target_col"]])
    pops = {"full": df}
    if len(df) > e["sample_n"]:
        _, pops["sample"] = train_test_split(df, test_size=e["sample_n"] / len(df), random_state=config.RANDOM_SEED,
                                             stratify=df[e["target_col"]])
    else:
        pops["sample"] = df
    for name, part in pops.items():
        a = part.drop(columns=[e["target_col"]]).apply(pd.to_numeric, errors="coerce").to_numpy(float)
        n_inf, n_nan = int(np.isinf(a).sum()), int(np.isnan(a).sum())
        rows.append({"dataset": e["name"], "population": name, "rows": a.shape[0], "cells": a.size,
                     "inf_cells": n_inf, "nan_cells": n_nan, "inf_prop": n_inf / a.size, "nan_prop": n_nan / a.size,
                     "rows_affected_prop": float((~np.isfinite(a)).any(axis=1).mean())})
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(Path(__file__).resolve().parents[1] / "results" / "nids_nonfinite.csv", index=False)
