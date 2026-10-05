#!/usr/bin/env python3
"""
run_statistics_v3.py -- regenerate the Exp. 1 CD diagram from the nested
arm, and keep the leaky CD diagram as a supplementary panel since it differs
interestingly (RFE 1st -> 4th).

Reuses mean_ranks / critical_difference / cliques / cd_diagram / analyse from
run_statistics.py unchanged. Experiments 2 and 3 are not re-run (protocol axis
P3 does not apply to them; see build_draft_v3.SEC_331_PROTOCOLS) but are
re-verified against their source CSVs here for completeness.

Outputs:
    figures/fig11_cd_diagram_exp1.pdf          nested arm (primary, v3)
    figures/fig11b_cd_diagram_exp1_leaky.pdf   leaky arm (supplementary)
    tables/nemenyi_exp1_nested.csv             already written by exp7_protocol
"""
from __future__ import annotations

import pandas as pd

from run_statistics import (FIG_DIR, analyse, cd_diagram,
                            critical_difference, mean_ranks)

RESULTS_DIR = "results"


def main() -> None:
    nested = pd.read_csv(f"{RESULTS_DIR}/exp1_performance_nested.csv")
    nested = nested[nested.status == "ok"].drop_duplicates(["dataset",
                                                            "method"])
    leaky = pd.read_csv(f"{RESULTS_DIR}/exp1_performance.csv")
    leaky = leaky[leaky.status == "ok"].drop_duplicates(["dataset", "method"])

    p_nested = nested.pivot(index="dataset", columns="method",
                            values="auc_mean")
    p_leaky = leaky.pivot(index="dataset", columns="method",
                          values="auc_mean")
    # round 8: uniform 30-minute budget -> Madelon excluded for every method
    p_nested = p_nested.drop(index="madelon"); p_leaky = p_leaky.drop(index="madelon")
    assert p_nested.shape == (20, 9), p_nested.shape
    assert p_leaky.shape == (20, 9), p_leaky.shape
    assert not p_nested.isna().any().any()

    print("### Nested Exp. 1 (primary, v3) ###")
    mr_n, _, cd_n = analyse("Exp1-nested-20", p_nested, True, "AUC-ROC (nested)")
    cd_diagram(mr_n, cd_n, len(p_nested),
              f"{FIG_DIR}/cd_nested_20.pdf",
              "Critical-difference diagram — Experiment 1, nested "
              "protocol (AUC-ROC, 20 datasets)")

    print("\n### Leaky Exp. 1 (supplementary) ###")
    mr_l, _, cd_l = analyse("Exp1-leaky-20", p_leaky, True, "AUC-ROC (leaky)")
    cd_diagram(mr_l, cd_l, len(p_leaky),
              f"{FIG_DIR}/cd_leaky_20.pdf",
              "Critical-difference diagram — Experiment 1, leaky "
              "protocol, supplementary (AUC-ROC, 20 datasets)")

    print("\nRank changes leaky -> nested:")
    both = pd.DataFrame({"leaky": mr_l, "nested": mr_n})
    both["shift"] = both.nested - both.leaky
    print(both.sort_values("nested").round(3).to_string())


if __name__ == "__main__":
    main()
