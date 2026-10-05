#!/usr/bin/env python3
"""
run_statistics.py — Friedman + Nemenyi significance testing for
"SHAP-based Feature Selection Revisited".

For each of three experiments, the nine methods are ranked within every
dataset, a Friedman test is run over the resulting rank matrix, and — if the
Friedman test is significant — a Nemenyi post-hoc test identifies which
pairwise differences are significant:

    Experiment 1  AUC-ROC,   21 datasets x 9 methods  (higher is better)
    Experiment 2  Jaccard,   10 datasets x 9 methods  (higher is better)
    Experiment 3  FDR@50%,    8 datasets x 9 methods  (lower is better,
                                                        Madelon excluded)

Outputs:
    figures/fig11_cd_diagram_exp1.pdf   critical-difference diagram (Exp. 1)
    tables/nemenyi_exp{1,2,3}.csv       full pairwise p-value matrices
    stdout                              summary used to write Sec. 8.9

Duplicate re-run rows in exp2_stability.csv / exp3_noise.csv are removed
before analysis, matching generate_tables.py.

Usage:  python3 run_statistics.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, rankdata, studentized_range

from generate_tables import METHOD_LABEL

RESULTS_DIR = "results"
FIG_DIR = "figures"
TAB_DIR = "tables"
ALPHA = 0.05


def mean_ranks(pivot: pd.DataFrame, higher_is_better: bool) -> pd.Series:
    """Rank methods within each dataset (rank 1 = best), then average."""
    sign = -1.0 if higher_is_better else 1.0
    ranks = np.vstack([rankdata(sign * row) for row in pivot.to_numpy()])
    return pd.Series(ranks.mean(axis=0), index=pivot.columns).sort_values()


def critical_difference(k: int, n: int, alpha: float = ALPHA) -> float:
    """Nemenyi critical difference for k methods over n datasets."""
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


def cliques(order: list[str], ranks: pd.Series, cd: float) -> list[tuple[int, int]]:
    """Maximal groups of rank-adjacent methods spanning at most CD."""
    out: list[tuple[int, int]] = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and ranks[order[j + 1]] - ranks[order[i]] <= cd:
            j += 1
        if j > i:
            out.append((i, j))
    # Drop groups fully contained in another.
    return [g for g in out
            if not any(g != h and h[0] <= g[0] and g[1] <= h[1] for h in out)]


def analyse(name: str, pivot: pd.DataFrame, higher_is_better: bool,
            metric: str) -> tuple[pd.Series, pd.DataFrame, float]:
    n, k = pivot.shape
    print(f"\n{'=' * 72}\n{name}: {n} datasets x {k} methods ({metric})\n{'=' * 72}")

    stat, p = friedmanchisquare(*[pivot[c].to_numpy() for c in pivot.columns])
    df = k - 1
    print(f"Friedman chi2({df}) = {stat:.3f}, p = {p:.3e} "
          f"({'significant' if p < ALPHA else 'NOT significant'} at α={ALPHA})")

    mr = mean_ranks(pivot, higher_is_better)
    print("\nMean ranks (1 = best):")
    for m, r in mr.items():
        print(f"  {METHOD_LABEL[m]:<20s} {r:.3f}")

    cd = critical_difference(k, n)
    print(f"\nNemenyi critical difference (α={ALPHA}): CD = {cd:.3f}")

    if p >= ALPHA:
        return mr, pd.DataFrame(), cd

    pvals = sp.posthoc_nemenyi_friedman(
        pivot.to_numpy() if higher_is_better else -pivot.to_numpy())
    pvals.index = pvals.columns = pivot.columns
    os.makedirs(TAB_DIR, exist_ok=True)
    pvals.round(4).to_csv(os.path.join(TAB_DIR, f"nemenyi_{name.lower().replace(' ', '')}.csv"))

    order = list(mr.index)
    sig = [(a, b, pvals.loc[a, b]) for i, a in enumerate(order)
           for b in order[i + 1:] if pvals.loc[a, b] < ALPHA]
    print(f"\nSignificant pairwise differences ({len(sig)} of "
          f"{k * (k - 1) // 2}):")
    for a, b, pv in sorted(sig, key=lambda x: x[2]):
        print(f"  {METHOD_LABEL[a]:<20s} vs {METHOD_LABEL[b]:<20s} p = {pv:.4f}")

    grps = cliques(order, mr, cd)
    print("\nGroups not significantly different (CD-connected):")
    for a, b in grps:
        print("  {" + ", ".join(METHOD_LABEL[m] for m in order[a:b + 1]) + "}")
    return mr, pvals, cd


def cd_diagram(mr: pd.Series, cd: float, n: int, path: str, title: str) -> None:
    """Demšar-style critical-difference diagram."""
    order = list(mr.index)
    k = len(order)
    lo, hi = 1, k
    grps = cliques(order, mr, cd)

    n_side = (k + 1) // 2
    axis_y, row_h, top = 0.92, 0.072, 0.82
    y_min = top - row_h * n_side - 0.04

    fig_h = 1.4 + 0.30 * n_side + 0.14 * len(grps)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    ax.set_xlim(hi + 0.5, lo - 0.5)          # reversed: best (rank 1) on right
    ax.set_ylim(y_min, 1.18)                 # headroom for the CD bar
    ax.axis("off")
    ax.plot([lo, hi], [axis_y, axis_y], "k-", lw=1.2)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [axis_y, axis_y + 0.028], "k-", lw=1.2)
        ax.text(t, axis_y + 0.05, str(t), ha="center", va="bottom", fontsize=9)

    # Critical-difference bar, drawn at the poor-rank (display-left) end so it
    # does not collide with the title.
    bar_y = axis_y + 0.135
    ax.plot([hi - cd, hi], [bar_y, bar_y], "k-", lw=2.2)
    for x in (hi - cd, hi):
        ax.plot([x, x], [bar_y - 0.02, bar_y + 0.02], "k-", lw=2.2)
    ax.text(hi - cd / 2, bar_y + 0.035, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=9)

    for i, m in enumerate(order):
        r = mr[m]
        right_side = i < n_side          # best ranks listed on the right
        idx = i if right_side else k - 1 - i
        y = top - row_h * (idx + 1)
        x_end = lo - 0.35 if right_side else hi + 0.35
        ax.plot([r, r], [axis_y, y], "k-", lw=0.9)
        ax.plot([r, x_end], [y, y], "k-", lw=0.9)
        # The x-axis is reversed, so display-left corresponds to larger data
        # values: anchor each label just outside its leader line and let it run
        # away from the plot.
        ax.text(x_end + (-0.08 if right_side else 0.08), y,
                f"{METHOD_LABEL[m]} ({r:.2f})",
                ha="left" if right_side else "right",
                va="center", fontsize=9.5)

    # Cliques: thick bars joining methods that are not significantly different.
    y = axis_y - 0.045
    for a, b in grps:
        ax.plot([mr[order[a]] - 0.045, mr[order[b]] + 0.045], [y, y],
                "k-", lw=4.0, solid_capstyle="butt")
        y -= 0.032

    ax.set_title(title, fontsize=11, pad=26)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {path}")


def main() -> None:
    e1 = pd.read_csv(f"{RESULTS_DIR}/exp1_performance.csv")
    e2 = (pd.read_csv(f"{RESULTS_DIR}/exp2_stability.csv")
            .drop_duplicates(["dataset", "method"], keep="first"))
    e3 = (pd.read_csv(f"{RESULTS_DIR}/exp3_noise.csv")
            .drop_duplicates(["dataset", "method", "noise_ratio"], keep="first"))

    p1 = e1.pivot(index="dataset", columns="method", values="auc_mean")
    p2 = e2.pivot(index="dataset", columns="method", values="jaccard_mean")
    e3n = e3[(e3.dataset != "madelon") & (e3.noise_ratio == 0.5)]
    p3 = e3n.pivot(index="dataset", columns="method", values="fdr")

    # Round 5: official SHAP-Select times out on Madelon in Exp. 2 (all 50
    # subsampling draws). Friedman needs complete blocks, so drop that dataset
    # for Exp. 2 and say so. Exp. 1 (leaky) is complete; Exp. 3 already
    # excludes Madelon.
    n2 = len(p2)
    p2 = p2.dropna()
    if len(p2) < n2:
        print(f"[Exp2] dropped {n2 - len(p2)} incomplete dataset(s): "
              f"Friedman over {len(p2)} datasets")
    for name, p in [("Exp1", p1), ("Exp2", p2), ("Exp3", p3)]:
        assert not p.isna().any().any(), f"{name}: pivot contains NaN"
        assert p.shape[1] == 9, f"{name}: expected 9 methods, got {p.shape[1]}"

    mr1, _, cd1 = analyse("Exp1", p1, True, "AUC-ROC")
    analyse("Exp2", p2, True, "Jaccard stability")
    analyse("Exp3", p3, False, "FDR@50% (Madelon excluded)")

    cd_diagram(mr1, cd1, len(p1),
               f"{FIG_DIR}/fig11_cd_diagram_exp1.pdf",
               "Critical-difference diagram — Experiment 1 (AUC-ROC, "
               "21 datasets)")


if __name__ == "__main__":
    main()
