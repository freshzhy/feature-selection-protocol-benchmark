#!/usr/bin/env bash
# Recompute every statistic, table and figure of the manuscript from results/*.csv
# without re-running any feature selection.  pip install -r requirements-analysis.txt
set -euo pipefail
cd "$(dirname "$0")"
python3 experiments/exp7_protocol.py        # P3 summaries (21 datasets; used by later scripts)
python3 round5_axis_sensitivity.py           # P2 variants on fixed dataset sets
python3 run_statistics.py                    # Friedman / Nemenyi on all experiments
python3 run_statistics_v3.py                 # 21-dataset nested/leaky Nemenyi (sensitivity)
python3 round6_bootstrap_ci.py               # 21-dataset bootstrap CIs (sensitivity)
python3 round5_numbers.py                    # stability / noise / fidelity numbers
python3 round7_analysis.py                   # transfer target baseline, P2 coverage and conventions
python3 round8_primary20.py                  # 20-dataset primary analysis (Table 2, 6, 8-10, CIs, variance ...)
rm -f results/multiseed/multiseed_summary.csv results/multiseed/multiseed_summary.md   # regenerated below
if ls results/multiseed/runs/*.csv >/dev/null 2>&1; then python3 revision_r8/analyse.py; fi   # S1 Table F source
python3 make_tables_v3.py                    # Tables 3, 5, 7
python3 make_tables_v4.py                    # Tables 2, 6, 8, 9, 10, S1 Tables E, F, G
python3 make_plos_figures.py                 # Fig 2, S1 Fig (20-dataset CD diagrams)
python3 make_plos_figures_v2.py              # Fig 1, 3-8, S2 Fig, S3 Fig (values asserted)
echo "light reproduction finished"
