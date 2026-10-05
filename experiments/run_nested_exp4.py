#!/usr/bin/env python3
"""
experiments/run_nested_exp4.py -- nested in-domain arm of Experiment 4.

Only the in-domain baseline of the transfer experiment is affected by
selection-evaluation coupling.  The cross-domain arm is already leakage-free:
features are chosen on the source using source labels, then the model is
trained and tested on the target, so no target label ever reaches selection.
The in-domain baseline, by contrast, selects on all of the source and then
cross-validates on the source.

Because the in-domain score depends only on (source, method) and not on the
target, 4 sources x 9 methods = 36 nested evaluations expand to the 108 rows
of the transfer table.

Output: results/exp4_transfer_nested.csv  (108 rows when all 9 methods succeed)

Round 5 fix: per-(source, method) try/except (official shap-select raises
on NIDS data), and results are MERGED into the existing CSV instead of
overwriting it, so `--methods a b c` no longer wipes the other methods.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from functools import partial
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import feature_selectors as fs  # noqa: E402
from data_loader import load_dataset  # noqa: E402
from exp4_transfer import sanitize  # noqa: E402
from nested_eval import make_xgb, nested_evaluate  # noqa: E402

warnings.filterwarnings("ignore")

RES = Path(config.RESULTS_DIR)
OUT = RES / "exp4_transfer_nested.csv"
NIDS_NAMES = [e["name"] for e in config.NIDS_DATASETS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=None)
    args = ap.parse_args()

    methods = args.methods or list(fs.ALL_METHODS)
    leaky = pd.read_csv(RES / "exp4_transfer.csv")
    leaky = leaky[leaky.status == "ok"]

    # Reproduce the leaky run's common-feature space and budget exactly.
    data, all_k = {}, []
    for name in NIDS_NAMES:
        X, y, meta = load_dataset(name)
        data[name] = (X, y, meta)
        all_k.append(meta["top_k"])

    common = None
    for name in NIDS_NAMES:
        cols = set(data[name][0].columns)
        common = cols if common is None else (common & cols)
    common = sorted(common)
    k = min(min(all_k), len(common) // 2, config.TOP_K_MAX)
    print(f"common features {len(common)}, k = {k}")

    k_leaky = int(leaky.k.iloc[0])
    if k != k_leaky or len(common) != int(leaky.n_common.iloc[0]):
        raise SystemExit(
            f"budget/common-feature mismatch vs the leaky arm: "
            f"k {k} vs {k_leaky}, common {len(common)} vs "
            f"{int(leaky.n_common.iloc[0])}")

    # ── nested in-domain score per (source, method) ───────────────────────
    indomain: dict[tuple[str, str], dict] = {}
    t_start = time.time()
    for src in NIDS_NAMES:
        X_src, y_src, _ = data[src]
        X_c = sanitize(X_src[common])
        for method in methods:
            print(f"  {src} x {method}", end="", flush=True)
            try:
                res = nested_evaluate(X_c, y_src,
                                      partial(fs.run_selector, method),
                                      k, make_xgb)
            except Exception as e:  # noqa: BLE001
                msg = str(e).splitlines()[0][:300]
                print(f"  ->  failed: {msg}")
                indomain[(src, method)] = {"error": msg}
                continue
            agg = res["aggregated"]["clf"]
            indomain[(src, method)] = {
                "indomain_f1_nested": agg["f1_macro_mean"],
                "indomain_auc_nested": agg["auc_mean"],
                "indomain_f1_std_nested": agg["f1_macro_std"],
                "fold_sizes": "|".join(str(n) for n in res["n_selected"]),
                "fold_features": ";".join("|".join(s)
                                          for s in res["feature_sets"]),
            }
            print(f"  ->  F1={agg['f1_macro_mean']:.4f}  "
                  f"AUC={agg['auc_mean']:.4f}  "
                  f"({res['time_select_s']:.0f}s sel)")

    # ── expand to the 108 transfer rows ───────────────────────────────────
    rows = []
    for src, tgt in permutations(NIDS_NAMES, 2):
        for method in methods:
            ref = leaky[(leaky.source == src) & (leaky.target == tgt)
                        & (leaky.method == method)]
            if ref.empty and "error" not in indomain[(src, method)]:
                print(f"  [warn] no leaky row for {src}->{tgt} x {method}")
                continue
            nd = indomain[(src, method)]
            if "error" in nd:
                rows.append({"source": src, "target": tgt, "method": method,
                             "n_common": len(common), "k": k,
                             "status": f"error: {nd['error']}"})
                continue
            r = ref.iloc[0]
            rows.append({
                "source": src, "target": tgt, "method": method,
                "n_common": len(common), "k": k,
                "feature_jaccard": r.feature_jaccard,
                "indomain_f1_leaky": r.indomain_f1,
                "indomain_auc_leaky": r.indomain_auc,
                "indomain_f1": nd["indomain_f1_nested"],
                "indomain_auc": nd["indomain_auc_nested"],
                # the cross-domain arm is unchanged: it was never leaky
                "cross_f1": r.cross_f1,
                "cross_auc": r.cross_auc,
                "n_transferred": r.n_transferred,
                "delta_f1": round(nd["indomain_f1_nested"] - r.cross_f1, 4),
                "delta_auc": round(nd["indomain_auc_nested"] - r.cross_auc, 4),
                "delta_f1_leaky": r.delta_f1,
                "fold_sizes": nd["fold_sizes"],
                "status": "ok",
            })

    df_new = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # MERGE, never overwrite: keep other methods' rows already in the file,
    # replace only the rows of the methods rerun here.
    if OUT.exists():
        old = pd.read_csv(OUT)
        old = old[~old.method.isin(methods)]
        df = pd.concat([old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(OUT, index=False)
    print(f"\n{OUT.name}: {len(df)} rows total, {len(df_new)} written now "
          f"({(time.time() - t_start) / 60:.1f} min)")
    print(df.groupby("method").status.apply(
        lambda s: f"{(s == 'ok').sum()} ok / {len(s)}").to_string())

    ok = df[df.status == "ok"]
    print("\nmean |dF1| by method (nested vs leaky):")
    for m, g in ok.groupby("method"):
        print(f"  {m:16s} {g.delta_f1.abs().mean():.4f}   "
              f"{g.delta_f1_leaky.abs().mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
