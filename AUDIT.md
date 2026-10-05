# Reproduction audit (light path)

**Criterion.** The light path is considered reproduced when every statistic agrees with the
stored value to within floating-point tolerance (relative difference ≤ 1e−12), every table
of the manuscript is identical as printed (same rounded values, same row order), and every
figure is visually and numerically unchanged (`make_plos_figures_v2.py` asserts each plotted
value against the manuscript tables). Byte- or pixel-identity is *not* claimed across
platforms: on other operating systems or BLAS/font builds, JSON/CSV files can differ in the
last floating-point digit (about 1e−16) and anti-aliased figures in single pixel channels by 1.

**Audit, 2026-10-05.** The deposit was zipped, extracted into an empty directory, all
generated outputs (tables/manuscript/, plos/figures/, results/round8_numbers.json) were
deleted, `results/multiseed/multiseed_summary.csv` was made read-only (0444) to test
overwriting, and `./reproduce_light.sh` was run in a new virtual environment created from
`requirements-analysis.txt` only (Linux x86-64, CPython 3.11.15). Wall time 3 min 36 s;
exit status 0; all value assertions passed.

| Output | Result on this platform | Criterion |
|---|---|---|
| results/round5–round8 JSON, exp7_headline.json, exp7_method_summary.csv, round5–round7 CSV | byte-identical | met |
| results/multiseed/multiseed_summary.csv and .md (from the 360 stored runs) | byte-identical | met |
| tables/manuscript/T2, T3, T5–T10, TD, TF, TG | byte-identical | met |
| plos/figures/Fig1–Fig8, S1_Fig–S3_Fig | pixel-identical (max channel difference 0) | met |

An independent re-run on another platform reported floating-point tails of about 1e−16 in
a few JSON/CSV files, a different row order among tied values in one table, and single-pixel
channel differences of 1 in Fig 7 and Fig 8, with all numbers and printed tables unchanged.
Since then, ties are broken by method identifier with a stable sort, and `tabulate` (used for
the multi-seed Markdown summary) was added to `requirements-analysis.txt`.

**Full path.** The multi-seed check (`revision_r8/`, 360 runs re-running feature selection
with the official libraries) was executed on the original macOS workstation; its seed-42
rows reproduce the stored nested and leaky XGBoost AUC of all 72 method–dataset cells
exactly. The other full-path experiments were not repeated for this audit.

**Release identifiers.** `finalize_release.sh` writes `CHECKSUMS.sha256` and `RELEASE.txt`
(content commit hash and the SHA-256 of the checksum file). The Zenodo DOI identifies the
archived version.
