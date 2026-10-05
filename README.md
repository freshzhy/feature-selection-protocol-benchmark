# Replication package

**How much does the evaluation protocol reshape a feature selection ranking? Downstream evaluator, feature budget and nested selection compared on nine methods**

This deposit corresponds to the submitted version of the manuscript. Every result was
produced with the authors' **official libraries** for BorutaSHAP (`BorutaShap` 1.0.17),
PowerSHAP (`powershap` 0.1.0.1) and SHAP-Select (`shap-select` 0.1.2), called unmodified
through `official_bridge/`. The results of the earlier reimplementation-based version are
kept only for the comparison reported in the paper (`results/pre_round5_backup/`).

**Primary dataset set.** All classification analyses (Tables 2, 6, 8–10, Figs 1, 2, 7, 8)
use the 20 datasets other than Madelon, so that every primary result can be regenerated
under the 30-minute per-call budget. The 21-dataset results (including the unconstrained
49.5 h / 237 h SHAP-Select runs on Madelon) are a sensitivity analysis (S1 Appendix, Table E).

## Light path: recompute every statistic, table and figure (≈2 min)

No feature selection is re-run. Python 3.11 or 3.12:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-analysis.txt
./reproduce_light.sh
```

Outputs: `results/round*_numbers.json`, `results/round7_*.csv`, `tables/manuscript/*.md`,
`plos/figures/*.tif`. Fonts are bundled in `fonts/` (Liberation Sans, SIL OFL 1.1) and
registered by the figure scripts. `make_plos_figures_v2.py` asserts every plotted value
against the value printed in the manuscript tables. See `AUDIT.md` for the clean-directory
check of this path.

## Full path: re-run the experiments (days on one workstation)

1. Obtain the datasets (`DATASETS.md`) and place them under `dataset/`.
2. Main environment (Python 3.12): `pip install -r requirements-experiments.txt`.
3. The three isolated library environments: follow `official_bridge/README.md`
   (exact versions from `official_bridge/locks/`, installed with `--no-deps`).
4. Run, in order:

```bash
python3 experiments/exp1_performance.py      # leaky classification arm
python3 experiments/run_nested_exp1.py       # nested arm + P1 evaluators
python3 experiments/exp2_stability.py
python3 experiments/exp3_noise.py && python3 round5_fix_madelon_exp3.py
python3 experiments/exp4_transfer.py && python3 experiments/run_nested_exp4.py
python3 experiments/run_nested_exp6.py       # P2 native stopping rules
python3 round5_dedupe.py                     # removes exact duplicate rows left by overlapping reruns
bash revision_r8/setup_env.sh && bash revision_r8/run_parallel.sh   # multi-seed check (seeds 42-46, 8 datasets; parallel workers)
python3 revision_r8/analyse.py
./reproduce_light.sh
```

Every library call has a 30-minute wall-clock budget (enforced in `official_bridge/bridge.py`).
Seed 42 is used throughout except in the multi-seed check (`revision_r8/`), which uses seeds 42–46.
The stored multi-seed runs (`results/multiseed/runs/`, 360 rows, 0 failures) were produced on a
macOS workstation with six workers × three threads; the seed-42 rows reproduce the stored nested
and leaky XGBoost AUC of all 72 method–dataset cells exactly.

## What the result files contain

| File | One row per | Selected features stored |
|---|---|---|
| `exp1_performance.csv` (leaky) | dataset × method | the selected subset |
| `exp1_performance_nested.csv` | dataset × method | union of the five fold subsets |
| `exp1_folds_nested.csv` | dataset × method × fold | each fold's subset; XGBoost, logistic and MLP scores |
| `exp2_stability.csv` | dataset × method | none (summary statistics of 50 subsamples only) |
| `exp3_noise.csv` | dataset × method × noise ratio | the selected subset (Madelon repeated for each ratio) |
| `exp4_transfer.csv` | pair × method | source- and target-selected subsets (leaky) |
| `exp4_transfer_nested.csv` | pair × method | fold sizes only; `delta_f1` here is the earlier source-dataset baseline, superseded by `round7_transfer_pairs.csv` (`d_tgt`) |
| `exp5_classifier_nested.csv` | dataset × method × evaluator | none (see `exp1_folds_nested.csv`) |
| `exp6_native_nested.csv` | dataset × method | union of the native subsets and per-fold sizes |
| `multiseed/runs/seed*_*.csv` | seed × dataset × method | none (AUC of both arms and three evaluators) |

The `spearman_mean` column of `exp2_stability.csv` is **not interpretable** as rank stability
(the selectors return top-*k* lists whose internal order is not comparable across methods,
and RFE returns an unordered set); it is retained only for provenance and is not used.

## Correspondence with the manuscript

| Manuscript item | Script → output |
|---|---|
| Table 2, Table 6, Table 8, Table 9, Table 10, Fig 1, Fig 7, Fig 8 | `round8_primary20.py` → `round8_numbers.json`; `make_tables_v4.py` |
| Fig 2, S1 Fig | `run_statistics_v4.py` via `make_plos_figures.py` |
| Table 3, Fig 3, Fig 4 | `results/exp2_stability.csv`; `make_tables_v3.py` |
| Table 4, Fig 5, S3 Fig | `results/exp3_noise.csv` |
| Table 5, Fig 6 | `round7_analysis.py` → `round7_transfer*.csv` |
| Table 7, S1 Table D | `round7_analysis.py` (`p2_*` in `round7_numbers.json`) |
| S1 Table E, S1 Table G | `make_tables_v4.py` |
| S1 Table F (multi-seed) | `revision_r8/analyse.py` → `results/multiseed/multiseed_summary.csv`, `.md` |

## Other scripts

`madelon_putative_set.py` (Madelon's 20-feature putative relevant set), `revision_r8/count_nonfinite_nids.py`
(share of infinite/missing NIDS cells, full files and the 100,000-row samples; output `results/nids_nonfinite.csv`: zero in all four datasets and both populations),
`finalize_release.sh` (writes `CHECKSUMS.sha256` and `RELEASE.txt`, which records the content commit hash; run once before archiving).

## Licences

Code: MIT (`LICENSE-CODE`). Results, tables and figures: CC BY 4.0 (`LICENSE-DATA`).
Fonts: SIL OFL 1.1 (`fonts/`). Datasets are third-party and are not redistributed.
