#!/usr/bin/env bash
# Repair the three official-library environments in official_bridge/venvs/ when their
# interpreter link is broken (they point to /Library/Frameworks/Python.framework/.../3.12,
# which is no longer installed). The installed packages are kept unchanged; only the
# Python 3.12 interpreter the environment runs on is re-pointed (uv-managed CPython 3.12).
set -euo pipefail
cd "$(dirname "$0")/.."
PY312=""
if command -v uv >/dev/null; then
  PY312=$(uv python find 3.12 2>/dev/null || true)
  [ -n "$PY312" ] || { uv python install 3.12; PY312=$(uv python find 3.12); }
fi
[ -n "$PY312" ] || PY312=$(command -v python3.12 || true)
[ -n "$PY312" ] || { echo "no Python 3.12 found (install uv or python3.12)"; exit 1; }
REAL=$("$PY312" -c "import os,sys; print(os.path.realpath(sys.executable))")
HOME_DIR=$(dirname "$REAL")
echo "Python 3.12 used for the library environments: $REAL"
for v in venv_boruta venv_powershap venv_shapselect; do
  case $v in venv_boruta) MODN=BorutaShap;; venv_powershap) MODN=powershap;; *) MODN=shap_select;; esac
  d=official_bridge/venvs/$v
  if "$d/bin/python" -c "import sys" 2>/dev/null; then
    echo "$v: interpreter ok"
  else
    echo "$v: interpreter missing -> re-pointing to $REAL"
    ln -sf "$REAL" "$d/bin/python"
    sed -i '' "s#^home = .*#home = $HOME_DIR#" "$d/pyvenv.cfg" 2>/dev/null || sed -i "s#^home = .*#home = $HOME_DIR#" "$d/pyvenv.cfg"
  fi
  "$d/bin/python" -c "import $MODN, shap, xgboost, numpy; print('  $v ok: $MODN imported; shap', shap.__version__, '| xgboost', xgboost.__version__, '| numpy', numpy.__version__)"
done
