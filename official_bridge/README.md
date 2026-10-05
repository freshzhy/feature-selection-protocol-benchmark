# official_bridge/

Calls the OFFICIAL `powershap`, `shap-select`, and `BorutaShap` libraries
from `feature_selectors.py`, each in its own isolated virtual environment,
because their pinned dependencies (numpy/shap/scipy versions) are mutually
incompatible and also incompatible with the main project environment
(shap 0.52 / numpy 2.x). The manuscript's S1 Appendix, section 'Reimplementations and official libraries',
explains why the earlier inline reimplementations were replaced.

## Layout
- `bridge.py`   — called from the main environment; picks the right venv,
                  writes X/y to temp CSVs, invokes `worker.py` as a
                  subprocess, reads back the JSON result.
- `worker.py`   — runs *inside* one of the three venvs below; never imported
                  directly by the main environment (its own numpy/shap
                  versions would conflict).
- `venvs/venv_powershap/`   — shap 0.45.1, numpy<2, scikit-learn 1.6.1, xgboost 2.1.4
- `venvs/venv_shapselect/`  — the project's own shap 0.52 / numpy 2.x stack
                               (this library never actually needed isolation;
                               it is here for a uniform interface only)
- `venvs/venv_boruta/`      — shap 0.44.1, scipy<1.12, numpy<2, xgboost 2.1.4

## Rebuilding the three environments (exact versions)

`locks/venv_*.txt` list every package installed in each environment used for the
reported runs (Python 3.12). Install them exactly, without dependency resolution,
because the recorded sets are not jointly satisfiable by a strict resolver (for
example, `contourpy` 1.4.0 declares numpy>=2 but the BorutaShap and powershap
environments use numpy 1.26.4; contourpy is only used by matplotlib plotting and is
never imported by the selectors):

```bash
for v in boruta powershap shapselect; do
  uv venv --python 3.12 official_bridge/venvs/venv_$v
  grep -v '^#' official_bridge/locks/venv_$v.txt | sed 's/_/-/g' > /tmp/lock_$v.txt
  uv pip install --no-deps --python official_bridge/venvs/venv_$v/bin/python -r /tmp/lock_$v.txt
done
```

The imports and versions of BorutaShap 1.0.17, powershap 0.1.0.1 and shap-select 0.1.2
installed this way were verified on Linux x86-64. The multi-seed runs were made on the
original macOS workstation with the existing environments, whose interpreter links were
re-pointed by `revision_r8/fix_bridge_venvs.sh` (packages unchanged); the seed-42 rerun
reproduced the stored nested and leaky AUC values of all 72 method–dataset cells exactly.

## Verified official-vs-reimplementation agreement (6 datasets, Round 5)
| Method | Mean Jaccard | Notes |
|---|---|---|
| SHAP-Select | 0.195 | Old reimplementation crashed (silently, via bare `except`) on both multiclass datasets tested |
| PowerSHAP | 0.592 | Old reimplementation systematically over-selected (~2x the official count) |
| BorutaSHAP | 0.735 | Closest of the three; old reimplementation was directionally faithful |
