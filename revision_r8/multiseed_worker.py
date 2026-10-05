#!/usr/bin/env python3
"""
revision_r8/multiseed_worker.py -- one worker of the multi-seed sensitivity check
(review item M2).  Start several workers in parallel with run_parallel.sh; they
share a job queue of (seed, dataset) units through lock files.

Pre-specified design (fixed before any run):
  seeds     42 (replication of the primary run), 43, 44, 45, 46
  datasets  sonar, ionosphere, credit-g, qsar-biodeg, kc1, spambase, phoneme, telco-churn
  methods   all nine; official libraries via official_bridge (30-min budget per call)
For every seed, fold assignment, every classifier and every selector receive that
seed, and BOTH protocols are run:
  nested : selection inside each of the 5 outer training folds; scored by XGBoost,
           logistic regression and the MLP (P1)
  leaky  : one selection on the full dataset, the subset scored by XGBoost on the
           same 5 folds (P3)
One CSV per (seed, dataset) in results/multiseed/runs/; one row per method.
Re-running skips methods already recorded, so an interrupted run can be restarted.
"""
import argparse, os, sys, time, json
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
os.chdir(ROOT)

SEEDS = [42, 43, 44, 45, 46]
DATASETS = ["sonar", "ionosphere", "credit-g", "qsar-biodeg", "kc1", "spambase", "phoneme", "telco-churn"]
METHODS = ["shap_topk", "borutashap", "powershap", "shap_select", "mi", "chi2", "rfe", "lasso", "tree_importance"]
OUT = ROOT / "results" / "multiseed"
RUNS, CLAIMS = OUT / "runs", OUT / "claims"
for d in (RUNS, CLAIMS): d.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
ap.add_argument("--datasets", nargs="+", default=DATASETS)
args = ap.parse_args()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import config  # noqa: E402
import feature_selectors as fs  # noqa: E402
import nested_eval as ne  # noqa: E402
from data_loader import load_dataset  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ms_io import read_runs, append_row  # noqa: E402

# biggest datasets first so that the parallel workers finish at about the same time
ORDER = {d: i for i, d in enumerate(["spambase", "telco-churn", "phoneme", "sonar", "qsar-biodeg", "kc1", "credit-g", "ionosphere"])}
JOBS = sorted([(s, d) for s in args.seeds for d in args.datasets], key=lambda j: (ORDER.get(j[1], 99), j[0]))


def pid_alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def claim(job):
    """Atomically claim a job; take over a claim whose owner process is dead."""
    path = CLAIMS / f"seed{job[0]}_{job[1]}.claim"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "worker": args.worker, "t": time.time()}).encode()); os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = json.loads(path.read_text())["pid"]
            except Exception:  # noqa: BLE001
                return False
            if pid_alive(owner):
                return False
            path.unlink(missing_ok=True)
    return False


def run_job(seed, ds):
    csv = RUNS / f"seed{seed}_{ds}.csv"
    done = set(read_runs(csv).method) if csv.exists() else set()
    if done >= set(METHODS):
        return
    config.RANDOM_SEED = seed; config.XGB_PARAMS["random_state"] = seed; ne.SEED = seed
    X, y, meta = load_dataset(ds)
    k = meta["top_k"]; y_arr = np.asarray(y); classes = np.unique(y_arr); n_classes = len(classes)
    for method in METHODS:
        if method in done:
            continue
        row = {"seed": seed, "dataset": ds, "method": method, "k": k}
        t0 = time.time()
        try:
            res = ne.nested_evaluate(X, y, partial(fs.run_selector, method), k, ne.CLASSIFIERS, seed=seed)
            if res["empty_folds"]:
                raise RuntimeError(f"empty selection in folds {res['empty_folds']}")
            agg = res["aggregated"]
            for c in ("xgb", "logreg", "mlp"):
                row[f"nested_{c}_auc"] = agg[c]["auc_mean"]
            row["nested_select_s"] = round(res["time_select_s"], 1)
            sel = list(fs.run_selector(method, X, y, k, seed))
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
            aucs = [ne._score_fold(ne.make_xgb(n_classes), X.iloc[tr][sel], y_arr[tr], X.iloc[te][sel], y_arr[te],
                                   n_classes, classes)["auc"] for tr, te in cv.split(X, y_arr)]
            row["leaky_xgb_auc"] = float(np.mean(aucs))
            row["status"] = "ok"
        except Exception as e:  # noqa: BLE001
            row["status"] = "error: " + str(e).splitlines()[0][:200]
        row["time_s"] = round(time.time() - t0, 1)
        append_row(csv, row)
        print(time.strftime("%H:%M:%S"), f"[w{args.worker}]", seed, ds, method, row["status"], row["time_s"], "s", flush=True)


for job in JOBS:
    if claim(job):
        print(time.strftime("%H:%M:%S"), f"[w{args.worker}] start", job, flush=True)
        run_job(*job)
print(time.strftime("%H:%M:%S"), f"[w{args.worker}] no jobs left", flush=True)
