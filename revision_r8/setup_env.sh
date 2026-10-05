#!/usr/bin/env bash
# Create the main environment for the multi-seed runs (.venv_ms, Python 3.12) with the
# package versions used in the paper. The three official-library environments in
# official_bridge/venvs/ are used as they are.
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v uv >/dev/null; then
  uv venv --python 3.12 .venv_ms
  uv pip install --python .venv_ms/bin/python -r revision_r8/requirements-experiments.txt
else
  python3.12 -m venv .venv_ms
  .venv_ms/bin/pip install -r revision_r8/requirements-experiments.txt
fi
bash revision_r8/fix_bridge_venvs.sh
.venv_ms/bin/python -c "import numpy, pandas, sklearn, xgboost, shap; print('main env ok:', numpy.__version__, pandas.__version__, sklearn.__version__, xgboost.__version__, shap.__version__)"
