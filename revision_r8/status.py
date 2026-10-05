#!/usr/bin/env python3
"""Progress of the multi-seed check: finished methods per (seed, dataset), errors."""
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ms_io import read_runs
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "multiseed" / "runs"
SEEDS = [42, 43, 44, 45, 46]
DATASETS = ["sonar", "ionosphere", "credit-g", "qsar-biodeg", "kc1", "spambase", "phoneme", "telco-churn"]
files = list(RUNS.glob("*.csv"))
d = pd.concat([read_runs(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=["seed", "dataset", "method", "status"])
tab = d.groupby(["dataset", "seed"]).method.nunique().unstack().reindex(index=DATASETS, columns=SEEDS).fillna(0).astype(int)
print("methods finished (of 9) per dataset x seed:\n", tab.to_string())
print(f"\ntotal {len(d)} / {9 * len(SEEDS) * len(DATASETS)} rows")
err = d[d.status != "ok"]
if len(err):
    print("\nerrors / timeouts:\n", err[["seed", "dataset", "method", "status"]].to_string(index=False))
