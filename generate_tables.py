#!/usr/bin/env python3
"""
generate_tables.py — Table generation for
"SHAP-based Feature Selection Revisited: A Comprehensive Benchmark on
 Stability, Noise Robustness, and Cross-Dataset Transferability"

Reads the raw experiment CSVs in results/ and emits, for each table, both a
booktabs LaTeX file (tables/tableN.tex) and a Markdown fragment
(tables/tableN.md) for insertion into the manuscript.

Table numbering (sequential by order of first appearance in the manuscript;
every displayed table carries a number and a caption, as the target journals
require):
    Table 1  Gap comparison vs. prior work        (Sec. 2.6, literature table)
    Table 2  Dataset characteristics              (Sec. 3.2.1)
    Table 3  Summary of experimental scale        (Sec. 3.5)
    Table 4  Experiment 1 -- classification       (Sec. 4.1)
    Table 5  Experiment 2 -- stability            (Sec. 5.1)
    Table 6  Experiment 3 -- noise robustness     (Sec. 6.1)
    Table 7  Experiment 4 -- transferability      (Sec. 7.1)
    Table 8  Comprehensive four-dimension ranking (Sec. 8.1)
Table 9  Exp. 5 -- classifier sensitivity     (Sec. 8.10)
Table 10 Exp. 6 -- native-threshold ablation  (Sec. 8.11)

IMPORTANT DATA NOTES:
  * exp2_stability.csv and exp3_noise.csv contain exact-duplicate re-run rows
    for a few (dataset, method[, noise_ratio]) keys. They are deduplicated
    here. The shipped exp2_summary.csv / exp3_summary.csv do NOT deduplicate
    and therefore double-count those datasets; this script recomputes all
    aggregates from the raw per-dataset rows instead.
  * Experiment 3 aggregates EXCLUDE Madelon by default (reported separately).
  * Experiment 4 ranks on |delta_f1| but the table reports signed delta_f1.

Usage:  python3 generate_tables.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

RESULTS_DIR = "results"
OUT_DIR = "tables"

# Canonical display order / labels for the nine methods.
METHOD_LABEL = {
    "shap_topk": "SHAP Top-K",
    "borutashap": "BorutaSHAP",
    "powershap": "PowerSHAP",
    "shap_select": "SHAP-Select",
    "rfe": "RFE",
    "chi2": "Chi-Square",
    "lasso": "Lasso",
    "mi": "Mutual Information",
    "tree_importance": "Tree Importance",
}
SHAP_METHODS = {"shap_topk", "borutashap", "powershap", "shap_select"}

# Human-readable dataset names and application domains (Table 2).
DATASET_LABEL = {
    "adult": "Adult",
    "banknote-authentication": "Banknote-Authentication",
    "blood-transfusion": "Blood-Transfusion",
    "breast-w": "Breast-W",
    "car": "Car",
    "credit-approval": "Credit-Approval",
    "credit-g": "Credit-G",
    "creditcard": "Creditcard",
    "diabetes": "Diabetes",
    "hr-attrition": "HR-Attrition",
    "ionosphere": "Ionosphere",
    "kc1": "KC1",
    "kc2": "KC2",
    "kr-vs-kp": "KR-vs-KP",
    "madelon": "Madelon",
    "phoneme": "Phoneme",
    "qsar-biodeg": "QSAR-Biodeg",
    "sonar": "Sonar",
    "spambase": "Spambase",
    "steel-plates-fault": "Steel-Plates-Fault",
    "telco-churn": "Telco-Churn",
}
DATASET_DOMAIN = {
    "adult": "Census / social",
    "banknote-authentication": "Image forensics",
    "blood-transfusion": "Medical",
    "breast-w": "Medical",
    "car": "Consumer / automotive",
    "credit-approval": "Finance",
    "credit-g": "Finance",
    "creditcard": "Finance (fraud)",
    "diabetes": "Medical",
    "hr-attrition": "Business / HR",
    "ionosphere": "Radar signal",
    "kc1": "Software defect",
    "kc2": "Software defect",
    "kr-vs-kp": "Game (chess)",
    "madelon": "Synthetic",
    "phoneme": "Speech",
    "qsar-biodeg": "Chemistry",
    "sonar": "Sonar signal",
    "spambase": "Text / email",
    "steel-plates-fault": "Manufacturing",
    "telco-churn": "Business / telecom",
}

# Table 8 must reproduce the ranking quoted in Discussion Sec. 8.1.
EXPECTED_RANKS = {
    "borutashap": (4, 1, 5, 1, 2.75),
    "rfe": (1, 5, 6, 2, 3.50),
    "chi2": (7, 2, 1, 7, 4.25),
    "powershap": (3, 4, 7, 4, 4.50),
    "shap_topk": (2, 3, 8, 5, 4.50),
    "tree_importance": (5, 8, 4, 3, 5.00),
    "lasso": (6, 6, 2, 8, 5.50),
    "mi": (8, 7, 3, 6, 6.00),
    "shap_select": (9, 9, 9, 9, 9.00),
}

DISCREPANCIES: list[str] = []


def note(msg: str) -> None:
    """Record a data discrepancy and echo it to stderr so it cannot be missed."""
    DISCREPANCIES.append(msg)
    print(f"  [DISCREPANCY] {msg}", file=sys.stderr)


# --------------------------------------------------------------------------
# Loading + integrity checks
# --------------------------------------------------------------------------

def load_data():
    e1 = pd.read_csv(os.path.join(RESULTS_DIR, "exp1_performance.csv"))
    e2 = pd.read_csv(os.path.join(RESULTS_DIR, "exp2_stability.csv"))
    e3 = pd.read_csv(os.path.join(RESULTS_DIR, "exp3_noise.csv"))
    e4 = pd.read_csv(os.path.join(RESULTS_DIR, "exp4_transfer.csv"))

    # -- Experiment 1: expect a clean 21 x 9 grid -------------------------
    g1 = e1.groupby(["dataset", "method"]).size()
    if not (g1 == 1).all():
        note(f"exp1: {(g1 != 1).sum()} duplicated (dataset, method) groups")
    assert e1.auc_mean.notna().all(), "exp1 contains NaN AUC values"
    print(f"  exp1: {len(e1)} rows, {e1.dataset.nunique()} datasets, "
          f"{e1.method.nunique()} methods")

    # -- Experiment 2: deduplicate exact re-run rows ----------------------
    dup2 = e2.duplicated(["dataset", "method"], keep=False)
    if dup2.any():
        for keys, grp in e2[dup2].groupby(["dataset", "method"]):
            identical = (grp.jaccard_mean.nunique() == 1
                         and grp.spearman_mean.nunique() == 1)
            if not identical:
                note(f"exp2: conflicting duplicate values for {keys}")
        note(f"exp2: {int(dup2.sum())} duplicated rows across "
             f"{e2[dup2].groupby(['dataset', 'method']).ngroups} "
             f"(dataset, method) keys -- deduplicated (keep=first). The shipped "
             f"exp2_summary.csv does not deduplicate and therefore "
             f"double-counts these datasets.")
    e2 = e2.drop_duplicates(["dataset", "method"], keep="first")
    n_per = e2.groupby("method").dataset.nunique().unique()
    assert len(n_per) == 1, f"exp2: uneven dataset coverage per method: {n_per}"
    print(f"  exp2: {len(e2)} rows after dedup, {n_per[0]} datasets/method")

    # -- Experiment 3: deduplicate exact re-run rows ----------------------
    key3 = ["dataset", "method", "noise_ratio"]
    dup3 = e3.duplicated(key3, keep=False)
    if dup3.any():
        for keys, grp in e3[dup3].groupby(key3):
            if grp.fdr.nunique() != 1 or grp.tpr.nunique() != 1:
                note(f"exp3: conflicting duplicate values for {keys}")
        note(f"exp3: {int(dup3.sum())} duplicated rows across "
             f"{e3[dup3].groupby(key3).ngroups} "
             f"(dataset, method, noise_ratio) keys -- deduplicated "
             f"(keep=first). The shipped exp3_summary.csv does not "
             f"deduplicate and therefore double-counts these datasets.")
    e3 = e3.drop_duplicates(key3, keep="first")
    n_per3 = e3.groupby(["method", "noise_ratio"]).size().unique()
    assert len(n_per3) == 1, f"exp3: uneven coverage per method/ratio: {n_per3}"
    print(f"  exp3: {len(e3)} rows after dedup, {n_per3[0]} datasets/cell")

    # -- Experiment 4: must already be de-duplicated (108 rows) -----------
    g4 = e4.groupby(["source", "target", "method"]).size()
    bad = g4[g4 != 1]
    if len(bad):
        raise AssertionError(
            "exp4_transfer.csv is NOT de-duplicated -- "
            f"{len(bad)} (source, target, method) groups have != 1 row:\n{bad}")
    if len(e4) != 108:
        raise AssertionError(f"exp4_transfer.csv has {len(e4)} rows, expected 108")
    print(f"  exp4: {len(e4)} rows, all (source, target, method) groups unique")

    return e1, e2, e3, e4


# --------------------------------------------------------------------------
# Emitters
# --------------------------------------------------------------------------

def write_tex(num: int, caption: str, label: str, colspec: str,
              header: list[str], rows: list[list[str]],
              notes: str = "", midrules: set[int] | None = None) -> None:
    """Write tables/tableN.tex in booktabs style (threeparttable-compatible)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    midrules = midrules or set()
    # pdfLaTeX has no U+00B1; captions and notes bypass tex_bold, so convert here.
    caption, notes = caption.replace("±", r"$\pm$"), notes.replace("±", r"$\pm$")

    tex = [r"\begin{table}[htbp]", r"\centering",
           r"\begin{threeparttable}",
           f"\\caption{{{caption}}}",
           f"\\label{{{label}}}",
           f"\\begin{{tabular}}{{{colspec}}}", r"\toprule",
           " & ".join(header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        tex.append(" & ".join(r) + r" \\")
        if i in midrules:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        tex += [r"\begin{tablenotes}[flushleft]\footnotesize",
                f"\\item {notes}", r"\end{tablenotes}"]
    tex += [r"\end{threeparttable}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, f"table{num}.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")


def bold(values: list[float], idx: int, fmt: str | None = None,
         best: str = "max") -> str:
    """Format values[idx], wrapping in bold markers if it is the best."""
    v = values[idx]
    target = max(values) if best == "max" else min(values)
    s = fmt_time(v) if fmt is None else fmt.format(v)
    return f"**{s}**" if np.isclose(v, target) else s


def fmt_time(v: float) -> str:
    """Timings span four orders of magnitude; keep three significant digits."""
    return f"{v:.3f}" if v < 1 else f"{v:.2f}"


def tex_bold(s: str) -> str:
    """Convert markdown **bold** / *italic* markers to LaTeX commands."""
    while "**" in s:
        s = s.replace("**", r"\textbf{", 1).replace("**", "}", 1)
    while "*" in s:
        s = s.replace("*", r"\emph{", 1).replace("*", "}", 1)
    # Keep the .tex files compilable under pdfLaTeX, which has no
    # U+00B1 / U+00D7 / U+2192.
    return (s.replace("±", r"$\pm$").replace("×", r"$\times$")
             .replace("→", r"$\rightarrow$"))


# LaTeX fragments used in captions/headers, and their Markdown equivalents.
# Longest / most specific patterns first.
_TEX_TO_MD = [
    (r"$\checkmark$", "✓"),
    (r"$\times$", "✗"),
    # Pipes must be backslash-escaped or they split Markdown table cells.
    (r"$|\Delta$F1$|$", r"\|ΔF1\|"),
    (r"$k = \min(20, \max(3, \lfloor p/2 \rfloor))$",
     "k = min(20, max(3, ⌊p/2⌋))"),
    # Underscores would otherwise be read as Markdown emphasis.
    (r"F1$_{\mathrm{in\text{-}domain}}$", "F1(in-domain)"),
    (r"F1$_{\mathrm{cross\text{-}domain}}$", "F1(cross-domain)"),
    (r"$\Delta$F1", "ΔF1"),
    (r"$\rho$", "ρ"),
    (r"\%", "%"),
    (r"\&", "&"),
    (r"$n$", "n"), (r"$p$", "p"), (r"$k$", "k"),
    ("--", "–"),  # LaTeX en-dash; must run last
]


def to_md(s: str) -> str:
    """Render a LaTeX-flavoured caption/header cell as plain Markdown."""
    for tex, md in _TEX_TO_MD:
        s = s.replace(tex, md)
    return s


def emit(num, caption, label, colspec, header, rows, notes="", midrules=None):
    """Emit tables/tableN.tex (LaTeX) and tables/tableN.md (Markdown)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- LaTeX: markdown bold markers become \textbf{} ---
    write_tex(num, caption, label, colspec, header,
              [[tex_bold(c) for c in r] for r in rows], notes, midrules)

    # --- Markdown: LaTeX math/escapes become plain text ---
    md_header = [to_md(h) for h in header]
    md = [f"**Table {num}.** {to_md(caption)}", "",
          "| " + " | ".join(md_header) + " |",
          "|" + "|".join(["---"] * len(md_header)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(to_md(c) for c in r) + " |")
    if notes:
        md += ["", f"*Note: {to_md(notes)}*"]
    with open(os.path.join(OUT_DIR, f"table{num}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  wrote tables/table{num}.tex and tables/table{num}.md")


# --------------------------------------------------------------------------
# Table 1 -- Gap comparison against prior work (literature table)
# --------------------------------------------------------------------------

def table1_gap():
    header = ["Study", "Unified multi-method comparison", "Stability",
              "Noise robustness", "Cross-dataset transfer",
              "Computational profiling"]
    Y, N, P = r"$\checkmark$", r"$\times$", "partial"
    rows = [
        ["Kraev et al. [4], SHAP-Select", P, N, N, N, P],
        ["Verhaeghe et al. [3], PowerSHAP", P, N, P, N, Y],
        ["Kılıç \\& Şengül [24], NIDS transfer", N, N, N, Y, N],
        ["Li et al. [9], FS survey", Y, N, N, N, P],
        ["Nogueira et al. [21], stability", N, Y, N, N, N],
        # Sec. 2.6 calls this the closest methodological precedent, so omitting
        # it from the gap table would read as curation. Listing it costs three
        # of the five columns and leaves the two that carry this paper's
        # increment -- noise robustness and cross-dataset transfer -- untouched.
        ["Bommert et al. [29], filter benchmark", Y, Y, N, N, Y],
        ["**This work**", Y, Y, Y, Y, Y],
    ]
    caption = (r"Coverage of the five evaluation dimensions in prior work and "
               r"in this benchmark. $\checkmark$ = addressed; partial = "
               r"addressed for a single method or on a narrow set of datasets; "
               r"$\times$ = not addressed.")
    emit(1, caption, "tab:gap", "lccccc", header, rows)


# --------------------------------------------------------------------------
# Table 2 -- Dataset characteristics
# --------------------------------------------------------------------------

def table2_datasets(e1):
    ds = (e1.drop_duplicates("dataset")
            [["dataset", "n_samples", "n_features", "n_classes", "k"]]
            .sort_values("n_samples"))
    assert len(ds) == 21, f"expected 21 datasets, got {len(ds)}"
    header = ["Dataset", "Domain", "$n$", "$p$", "Classes", "$k$"]
    rows = [[DATASET_LABEL[r.dataset], DATASET_DOMAIN[r.dataset],
             f"{r.n_samples:,}", str(r.n_features), str(r.n_classes), str(r.k)]
            for r in ds.itertuples()]
    caption = ("Characteristics of the 21 general benchmark datasets "
               "(Experiments 1--3), sorted by sample size. $n$ = samples, "
               "$p$ = original features, $k$ = number of features each method "
               "is required to select, $k = \\min(20, \\max(3, "
               "\\lfloor p/2 \\rfloor))$.")
    emit(2, caption, "tab:datasets", "llrrrr", header, rows)


# --------------------------------------------------------------------------
# Table 3 -- Summary of experimental scale
#
# Sec. 3.5 of draft_v1 carried this as an unnumbered, uncaptioned Markdown
# table. It is generated here so that (a) it gets a number and a caption like
# every other table, and (b) its counts are recomputed from the CSVs rather
# than typed by hand.
# --------------------------------------------------------------------------

def table3_scale(e1, e2, e3, e4, e5=None):
    n_ds1 = e1.dataset.nunique()
    n_ds2 = e2.dataset.nunique()
    n_ds3 = e3.dataset.nunique()
    n_ratios = e3.noise_ratio.nunique()
    n_pairs = e4.groupby(["source", "target"]).ngroups
    n_nids = len(set(e4.source) | set(e4.target))
    n_m = e1.method.nunique()
    assert (e2.method.nunique() == e3.method.nunique()
            == e4.method.nunique() == n_m), "method count differs across exps"

    header = ["Experiment", "Datasets", "Methods", "Runs", "Key metrics"]
    rows = [
        ["Exp. 1: Classification performance", f"{n_ds1} (general)",
         str(n_m), f"{len(e1):,}", "AUC-ROC, macro F1, accuracy"],
        ["Exp. 2: Stability", f"{n_ds2} (general)", str(n_m),
         f"{len(e2):,} × 50 bootstraps",
         "Jaccard, Spearman $\\rho$"],
        ["Exp. 3: Noise robustness",
         f"{n_ds3} (general) × {n_ratios} ratios", str(n_m),
         f"{len(e3):,}", "FDR, TPR"],
        ["Exp. 4: Cross-dataset transfer",
         f"{n_nids} (NIDS) × {n_pairs} pairs", str(n_m),
         f"{len(e4):,}", "$\\Delta$F1, cross-domain F1"],
    ]
    n_sel = len(e1) + len(e2) + len(e3) + len(e4)
    tail = ""
    if e5 is not None:
        n_clf = e5.classifier.nunique()
        rows.append(
            ["Exp. 5: Classifier sensitivity",
             # A literal × here, not $\times$: _TEX_TO_MD maps the macro to ✗.
             f"{e5.dataset.nunique()} (general) × {n_clf} classifiers",
             str(n_m), f"{len(e5):,}", "AUC-ROC"])
        tail = (f" Experiment 5 adds {len(e5):,} evaluations, counted "
                "separately because they re-score subsets already selected in "
                "Experiment 1 rather than performing new selection.")
        _p6 = os.path.join("results", "exp6_native.csv")
        if os.path.exists(_p6):
            e6 = pd.read_csv(_p6)
            rows.append(
                ["Exp. 6: Native thresholds",
                 f"{e6.dataset.nunique()} (general)",
                 f"{e6.method.nunique()} (adaptive)", f"{len(e6):,}",
                 "AUC-ROC, selection size"])
            tail += (f" Experiment 6 repeats the selection step for the three "
                     f"adaptive methods at their own thresholds "
                     f"({len(e6):,} runs), so it is likewise listed apart from "
                     "the common-budget total.")
    caption = ("Summary of the experimental scale. A run is one "
               "method--dataset combination (one method--dataset--noise-ratio "
               "combination in Experiment 3, one method--transfer-pair "
               "combination in Experiment 4). Experiments 1--4 perform feature "
               f"selection and comprise {n_sel:,} such runs in total, of which "
               "the Experiment 2 rows each expand into 50 bootstrap "
               "resamples." + tail)
    emit(3, caption, "tab:scale", "llccl", header, rows)
    return n_sel


# --------------------------------------------------------------------------
# Table 4 -- Experiment 1: classification performance
# --------------------------------------------------------------------------

def table4_exp1(e1):
    agg = e1.groupby("method").agg(
        auc_m=("auc_mean", "mean"), auc_s=("auc_mean", "std"),
        f1_m=("f1_mean", "mean"), f1_s=("f1_mean", "std"),
        acc_m=("accuracy_mean", "mean"), acc_s=("accuracy_mean", "std"),
        t=("time_select_s", "mean"),
    ).sort_values("auc_m", ascending=False)

    aucs = list(agg.auc_m); f1s = list(agg.f1_m)
    accs = list(agg.acc_m); ts = list(agg.t)
    header = ["Method", "Family", "AUC-ROC", "Macro F1", "Accuracy",
              "Select. time (s)"]
    rows = []
    for i, (m, r) in enumerate(agg.iterrows()):
        fam = "SHAP" if m in SHAP_METHODS else "Traditional"
        rows.append([
            METHOD_LABEL[m], fam,
            f"{bold(aucs, i, '{:.4f}')} ± {r.auc_s:.4f}",
            f"{bold(f1s, i, '{:.4f}')} ± {r.f1_s:.4f}",
            f"{bold(accs, i, '{:.4f}')} ± {r.acc_s:.4f}",
            bold(ts, i, None, best="min"),
        ])
    caption = ("Experiment 1: downstream classification performance across the "
               "21 general benchmark datasets, sorted by mean AUC-ROC. Values "
               "are mean ± standard deviation across datasets; each per-dataset "
               "value is itself a stratified 5-fold cross-validation mean. "
               "Selection time is the mean wall-clock time of a single feature "
               "selection call per dataset. Best value per column in bold.")
    notes = ("Selection time is per single call. BorutaSHAP's cost in "
             "Experiment 2, where it is invoked 50 times per dataset under the "
             "bootstrap protocol, is approximately 2,400 s per dataset and is "
             "not comparable to the figure above.")
    emit(4, caption, "tab:exp1", "llcccr", header, rows, notes)
    return agg


# --------------------------------------------------------------------------
# Table 5 -- Experiment 2: stability
# --------------------------------------------------------------------------

def table5_exp2(e2):
    agg = e2.groupby("method").agg(
        jac_m=("jaccard_mean", "mean"), jac_s=("jaccard_mean", "std"),
        sp_m=("spearman_mean", "mean"), sp_s=("spearman_mean", "std"),
        n=("dataset", "nunique"),
    ).sort_values("jac_m", ascending=False)

    jacs = list(agg.jac_m); sps = list(agg.sp_m)
    header = ["Method", "Family", "Jaccard", "Spearman $\\rho$"]
    rows = []
    for i, (m, r) in enumerate(agg.iterrows()):
        fam = "SHAP" if m in SHAP_METHODS else "Traditional"
        rows.append([
            METHOD_LABEL[m], fam,
            f"{bold(jacs, i, '{:.4f}')} ± {r.jac_s:.4f}",
            f"{bold(sps, i, '{:.4f}')} ± {r.sp_s:.4f}",
        ])
    r_pearson = np.corrcoef(agg.jac_m, agg.sp_m)[0, 1]
    caption = ("Experiment 2: feature subset stability across 10 datasets under "
               "50-iteration bootstrap resampling (1,225 pairwise comparisons "
               "per method--dataset pair), sorted by mean Jaccard index. Values "
               "are mean ± standard deviation across datasets. Best value per "
               "column in bold.")
    notes = (f"The two stability metrics are strongly correlated across methods "
             f"(Pearson r = {r_pearson:.3f}).")
    emit(5, caption, "tab:exp2", "llcc", header, rows, notes)
    return agg


# --------------------------------------------------------------------------
# Table 6 -- Experiment 3: noise robustness
# --------------------------------------------------------------------------

def table6_exp3(e3):
    excl = e3[e3.dataset != "madelon"]
    n_ds = excl.dataset.nunique()
    fdr = excl.pivot_table(index="method", columns="noise_ratio",
                           values="fdr", aggfunc="mean")
    tpr = excl.pivot_table(index="method", columns="noise_ratio",
                           values="tpr", aggfunc="mean")
    agg = pd.DataFrame({
        "fdr10": fdr[0.1], "fdr20": fdr[0.2], "fdr50": fdr[0.5],
        "tpr50": tpr[0.5],
    }).sort_values("fdr50")

    c10 = list(agg.fdr10); c20 = list(agg.fdr20)
    c50 = list(agg.fdr50); t50 = list(agg.tpr50)
    header = ["Method", "Family", "FDR@10\\%", "FDR@20\\%", "FDR@50\\%",
              "TPR@50\\%"]
    rows = []
    for i, (m, r) in enumerate(agg.iterrows()):
        fam = "SHAP" if m in SHAP_METHODS else "Traditional"
        rows.append([
            METHOD_LABEL[m], fam,
            bold(c10, i, "{:.4f}", best="min"),
            bold(c20, i, "{:.4f}", best="min"),
            bold(c50, i, "{:.4f}", best="min"),
            bold(t50, i, "{:.4f}", best="max"),
        ])

    # Madelon reported separately as an extreme case.
    mad = e3[e3.dataset == "madelon"]
    mf = mad.groupby("noise_ratio").fdr.mean()
    mt = mad.groupby("noise_ratio").tpr.mean()
    mad50 = mad[mad.noise_ratio == 0.5]
    rows.append([
        "*Madelon (all 9 methods)*", "--",
        f"{mf[0.1]:.4f}", f"{mf[0.2]:.4f}", f"{mf[0.5]:.4f}", f"{mt[0.5]:.4f}",
    ])
    midrules = {len(rows) - 2}

    caption = (f"Experiment 3: noise robustness under 10\\%, 20\\% and 50\\% "
               f"injected N(0,1) noise features. FDR = fraction of selected "
               f"features that are injected noise (lower is better); TPR = "
               f"fraction of original features recovered (higher is better). "
               f"Main rows are means over the {n_ds} datasets EXCLUDING Madelon, "
               f"sorted by FDR@50\\%. Best value per column in bold.")
    notes = (f"Madelon is excluded from the main rows and reported separately "
             f"because its 20 relevant features among 500 make it an extreme "
             f"outlier: at 50\\% noise every method scores FDR "
             f"{mad50.fdr.min():.2f}--{mad50.fdr.max():.2f} and TPR "
             f"{mad50.tpr.min():.2f}--{mad50.tpr.max():.2f}. The Madelon row "
             f"gives the mean across all nine methods. Including Madelon, "
             f"Chi-Square's FDR@50\\% rises from 0.000 to "
             f"{e3[(e3.method == 'chi2') & (e3.noise_ratio == 0.5)].fdr.mean():.3f}, "
             f"driven entirely by that single dataset. The TPR column carries "
             f"no information beyond FDR---once a method returns its full "
             f"budget, TPR = (1 - FDR) $\\times k/p$, where $p$ counts the "
             f"original features (Section 3.3.3)---and is retained only for "
             f"comparability with "
             f"the noise-robustness literature, which conventionally reports "
             f"both.")
    emit(6, caption, "tab:exp3", "llcccc", header, rows, notes, midrules)
    return agg


# --------------------------------------------------------------------------
# Table 7 -- Experiment 4: cross-dataset transferability
# --------------------------------------------------------------------------

def table7_exp4(e4):
    e4 = e4.copy()
    e4["abs_delta_f1"] = e4.delta_f1.abs()
    agg = e4.groupby("method").agg(
        signed=("delta_f1", "mean"),
        absd=("abs_delta_f1", "mean"),
        absd_sd=("abs_delta_f1", "std"),
        jac=("feature_jaccard", "mean"),
        cross=("cross_f1", "mean"),
    ).sort_values("absd")

    signed = list(agg.signed); absd = list(agg.absd)
    jacs = list(agg.jac); cross = list(agg.cross)
    header = ["Method", "Family", "Mean $\\Delta$F1 (signed)",
              "Mean $|\\Delta$F1$|$", "SD $|\\Delta$F1$|$",
              "Feature Jaccard", "Cross-domain F1"]
    rows = []
    for i, (m, r) in enumerate(agg.iterrows()):
        fam = "SHAP" if m in SHAP_METHODS else "Traditional"
        rows.append([
            METHOD_LABEL[m], fam,
            f"{r.signed:+.4f}",
            bold(absd, i, "{:.4f}", best="min"),
            f"{r.absd_sd:.4f}",
            bold(jacs, i, "{:.4f}", best="max"),
            bold(cross, i, "{:.4f}", best="max"),
        ])
    caption = ("Experiment 4: cross-dataset transferability across the 12 "
               "ordered NIDS transfer pairs, sorted by mean $|\\Delta$F1$|$. "
               "$\\Delta$F1 = F1$_{\\mathrm{in\\text{-}domain}}$ - "
               "F1$_{\\mathrm{cross\\text{-}domain}}$, so positive values mean "
               "cross-domain performance is worse. The signed mean shows "
               "transfer direction; $|\\Delta$F1$|$ is used for ranking because "
               "it measures deviation magnitude rather than allowing positive "
               "and negative deviations to cancel. Best value per ranked column "
               "in bold.")
    notes = ("Feature Jaccard is the overlap between the source-selected and "
             "target-selected feature sets. Note that the highest feature "
             "overlap (Mutual Information) does not yield the best transfer "
             "performance.")
    emit(7, caption, "tab:exp4", "llcccrr", header, rows, notes)
    return agg


# --------------------------------------------------------------------------
# Table 8 -- Comprehensive four-dimension ranking
# --------------------------------------------------------------------------

def table8_ranking(a1, a2, a3, a4):
    perf = a1.auc_m.rank(ascending=False, method="min").astype(int)
    stab = a2.jac_m.rank(ascending=False, method="min").astype(int)
    noise = a3.fdr50.rank(ascending=True, method="min").astype(int)
    trans = a4.absd.rank(ascending=True, method="min").astype(int)

    rank = pd.DataFrame({"perf": perf, "stab": stab,
                         "noise": noise, "trans": trans})
    rank["mean"] = rank.mean(axis=1)
    rank = rank.sort_values(["mean", "perf"])

    # Verify against the ranking quoted in Discussion Sec. 8.1.
    for m, (p, s, n, t, mu) in EXPECTED_RANKS.items():
        got = rank.loc[m]
        if (int(got.perf), int(got.stab), int(got.noise), int(got.trans)) != (p, s, n, t) \
           or not np.isclose(got["mean"], mu):
            note(f"Table 8: {m} computed ranks "
                 f"({int(got.perf)},{int(got.stab)},{int(got.noise)},"
                 f"{int(got.trans)}) mean {got['mean']:.2f} != specified "
                 f"({p},{s},{n},{t}) mean {mu:.2f}")
    if not DISCREPANCIES or all("Table 8" not in d for d in DISCREPANCIES):
        print("  Table 8 reproduces the Sec. 8.1 ranking exactly ✓")

    header = ["Method", "Family", "Performance", "Stability", "Noise",
              "Transfer", "Mean rank"]
    means = list(rank["mean"])
    rows = []
    for i, (m, r) in enumerate(rank.iterrows()):
        fam = "SHAP" if m in SHAP_METHODS else "Traditional"
        rows.append([
            METHOD_LABEL[m], fam, str(int(r.perf)), str(int(r.stab)),
            str(int(r.noise)), str(int(r.trans)),
            bold(means, i, "{:.2f}", best="min"),
        ])
    caption = ("Comprehensive ranking of the nine methods across the four "
               "evaluation dimensions (1 = best, 9 = worst), sorted by mean "
               "rank. Performance is ranked by mean AUC-ROC (Table 4), "
               "stability by mean Jaccard index (Table 5), noise robustness by "
               "FDR@50\\% excluding Madelon (Table 6), and transferability by "
               "mean $|\\Delta$F1$|$ (Table 7). No method ranks in the top "
               "three on all four dimensions. The performance column is "
               "obtained with XGBoost as the downstream classifier and is not "
               "invariant to that choice; Table 9 and Section 8.10 report how "
               "far it moves under two non-tree classifiers. The four columns "
               "are also estimated from different numbers of blocks---21 "
               "datasets for performance, 10 for stability, 8 for noise "
               "robustness and 12 transfer pairs---so they are not equally "
               "precise, and the mean of ranks does not weight them for that.")
    emit(8, caption, "tab:ranking", "llccccc", header, rows, caption_notes(rank))
    return rank


def caption_notes(rank):
    top = rank.index[0]
    return (f"{METHOD_LABEL[top]} attains the best mean rank "
            f"({rank['mean'].iloc[0]:.2f}) yet ranks only "
            f"{int(rank.perf.iloc[0])}th in classification performance, "
            f"illustrating the performance--stability--robustness trilemma.")


def table9_classifiers(e1):
    """Table 9 -- Experiment 5: sensitivity of the Exp. 1 ranking to the
    downstream classifier.

    Six of the nine methods derive their importance scores from an XGBoost fit
    and are then scored with XGBoost, which makes the Experiment 1 ranking
    partly circular. This table re-scores the feature subsets already selected
    in Experiment 1 with two non-tree classifiers, so the selection step is
    never repeated and only the downstream learner changes.
    """
    src = os.path.join("results", "exp5_classifiers.csv")
    if not os.path.exists(src):
        note("Table 9 skipped: results/exp5_classifiers.csv not found "
             "(run experiments/exp5_classifiers.py)")
        return None
    e5 = pd.read_csv(src)
    e5 = e5[e5.status == "ok"]
    if e5.dataset.nunique() != e1.dataset.nunique():
        note(f"Table 9: exp5 covers {e5.dataset.nunique()} datasets, "
             f"exp1 covers {e1.dataset.nunique()}")

    xgb = e1.groupby("method").auc_mean.mean()
    cols = {"xgboost": xgb}
    for c in ("logreg", "mlp"):
        cols[c] = e5[e5.classifier == c].groupby("method").auc.mean()
    t = pd.DataFrame(cols)
    for c in t.columns:
        t[c + "_rk"] = t[c].rank(ascending=False, method="min").astype(int)
    t = t.sort_values("xgboost", ascending=False)

    rho_lr = spearmanr(t.xgboost_rk, t.logreg_rk)[0]
    rho_ml = spearmanr(t.xgboost_rk, t.mlp_rk)[0]
    rho_nn = spearmanr(t.logreg_rk, t.mlp_rk)[0]

    header = ["Method", "Family", "XGBoost AUC", "Rank",
              "Logistic AUC", "Rank", "MLP AUC", "Rank"]
    rows = []
    for i, (m, r) in enumerate(t.iterrows()):
        fam = "SHAP" if m in SHAP_METHODS else "Traditional"
        rows.append([
            METHOD_LABEL[m], fam,
            bold(list(t.xgboost), i, "{:.4f}", best="max"), str(int(r.xgboost_rk)),
            bold(list(t.logreg), i, "{:.4f}", best="max"), str(int(r.logreg_rk)),
            bold(list(t.mlp), i, "{:.4f}", best="max"), str(int(r.mlp_rk)),
        ])
    caption = ("Experiment 5: sensitivity of the Experiment 1 ranking to the "
               "downstream classifier. Each method's Experiment 1 feature "
               "subset is re-scored, without re-running selection, under "
               "stratified 5-fold cross-validation with two non-tree "
               "classifiers: $L_2$-regularised logistic regression and a "
               "64-unit single-hidden-layer network, both on standardised "
               "features. Sorted by XGBoost AUC; best value per AUC column in "
               "bold.")
    notes = (f"Spearman rank correlation with the XGBoost ordering is "
             f"{rho_lr:.2f} for logistic regression and {rho_ml:.2f} for the "
             f"MLP, whereas the two non-tree classifiers agree with each other "
             f"at {rho_nn:.2f}. Lasso rises from 6th under XGBoost to 1st "
             f"under both; SHAP-Select ranks last under all three.")
    emit(9, caption, "tab:classifiers", "llcccccc", header, rows, notes)
    return t


def table10_native(e1):
    """Table 10 -- Experiment 6: the three adaptive selectors at their own
    stopping rules.

    Experiments 1-4 force every method to return exactly k features, which
    discards the stopping rule that is part of BorutaSHAP, PowerSHAP and
    SHAP-Select. This table reports what each selects when that rule is
    honoured, and what the resulting subset scores under the Experiment 1
    downstream protocol.
    """
    src = os.path.join("results", "exp6_native.csv")
    if not os.path.exists(src):
        note("Table 10 skipped: results/exp6_native.csv not found")
        return None
    e6 = pd.read_csv(src)
    ok = e6[e6.status == "ok"]

    header = ["Method", "Datasets", "Native size", "Budget $k$",
              "AUC (fixed $k$)", "AUC (native)", "$\\Delta$", "$p$"]
    rows = []
    order = ["borutashap", "powershap", "shap_select"]
    for m in order:
        t = ok[ok.method == m]
        w = wilcoxon(t.auc_native, t.auc_fixed)
        delta = t.auc_native.mean() - t.auc_fixed.mean()
        rows.append([
            METHOD_LABEL[m], f"{len(t)}",
            f"{t.n_native.mean():.1f}", f"{t.k.mean():.1f}",
            f"{t.auc_fixed.mean():.4f}", f"{t.auc_native.mean():.4f}",
            ("**" if w.pvalue < 0.05 else "") + f"{delta:+.4f}"
            + ("**" if w.pvalue < 0.05 else ""),
            ("**" if w.pvalue < 0.05 else "") + f"{w.pvalue:.4f}"
            + ("**" if w.pvalue < 0.05 else "")])

    n_empty = int((e6.status == "empty_selection").sum())
    caption = ("Experiment 6: the three adaptive selectors evaluated at their "
               "own stopping rules rather than at the common budget $k$. "
               "Native size is the mean number of features in the unmodified "
               "selection; the "
               "subset is then scored under the Experiment 1 protocol "
               "(XGBoost, stratified 5-fold, AUC-ROC), so only the selection "
               "step differs. $\\Delta$ is the paired mean difference and "
               "$p$ comes from a Wilcoxon signed-rank test over the datasets "
               "in that row; significant changes are shown in bold.")
    notes = (f"SHAP-Select returns an empty set on {n_empty} of the 21 "
             f"datasets, which the fixed budget concealed by padding; those "
             f"rows are excluded from its averages, so its comparison is over "
             f"19 datasets rather than 21. Its native selections range from 2 "
             f"to 123 features against a mean budget of 11.4, and exceed the "
             f"budget on 15 of 19 datasets: under the common budget it was "
             f"predominantly truncated, not padded.")
    emit(10, caption, "tab:native", "lccccccc", header, rows, notes)
    return ok


# --------------------------------------------------------------------------

def main():
    print("Loading and validating experiment data ...")
    e1, e2, e3, e4 = load_data()
    print("\nGenerating tables ...")
    table1_gap()
    table2_datasets(e1)
    e5 = None
    _p5 = os.path.join('results', 'exp5_classifiers.csv')
    if os.path.exists(_p5):
        e5 = pd.read_csv(_p5)
        e5 = e5[e5.status == 'ok']
    table3_scale(e1, e2, e3, e4, e5)
    a1 = table4_exp1(e1)
    a2 = table5_exp2(e2)
    a3 = table6_exp3(e3)
    a4 = table7_exp4(e4)
    table8_ranking(a1, a2, a3, a4)
    table9_classifiers(e1)
    table10_native(e1)

    print("\n" + "=" * 70)
    if DISCREPANCIES:
        print(f"{len(DISCREPANCIES)} data discrepancy note(s) recorded:")
        for d in DISCREPANCIES:
            print(f"  - {d}")
    else:
        print("No data discrepancies detected.")
    print("=" * 70)


if __name__ == "__main__":
    main()
