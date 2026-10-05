#!/usr/bin/env python3
"""Run before the workers: (1) the three official-library environments must start and
import their library; (2) run files are rewritten with the fixed schema, and rows that
failed only because a library environment was missing are removed so they are re-run."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ms_io import repair_all

ok = True
for v, mod in (("venv_boruta", "BorutaShap"), ("venv_powershap", "powershap"), ("venv_shapselect", "shap_select")):
    py = ROOT / "official_bridge" / "venvs" / v / "bin" / "python"
    r = subprocess.run([str(py), "-c", f"import {mod}"], capture_output=True, text=True) if py.exists() else None
    if r is None or r.returncode != 0:
        ok = False
        print(f"[preflight] {v}: cannot start or import {mod}. Run: bash revision_r8/fix_bridge_venvs.sh")
    else:
        print(f"[preflight] {v}: ok")
if not ok:
    sys.exit(1)
runs = ROOT / "results" / "multiseed" / "runs"
if runs.exists():
    n = repair_all(runs)
    print(f"[preflight] run files repaired; {n} rows with a missing library environment removed (will be re-run)")
