"""
official_bridge/bridge.py -- calls the OFFICIAL powershap / shap-select /
BorutaShap libraries from the main environment, each in its own isolated
venv (they pin mutually incompatible numpy/shap/scipy versions and cannot
share one interpreter). See official_bridge/README.md for why this exists.

Public entry point: call_official(lib, X, y, k, seed, native=False)
    lib      : "powershap" | "shapselect" | "boruta"
    X, y     : pd.DataFrame / pd.Series, numeric, as loaded by data_loader
    k        : int, the fixed budget (used only for the non-native return)
    seed     : int
    native   : if True, return exactly the method's own accepted set,
               unpadded and untruncated (mirrors the existing native=True
               contract in feature_selectors.py).
Returns: list[str] of selected feature names.
Raises: RuntimeError with the worker's traceback on failure -- never
        silently returns an empty list the way the old reimplementation's
        bare `except Exception` did.

Round 5 fix: added TIMEOUT_S (default 1800s = 30 min) to prevent hangs
on computationally infeasible combinations (e.g. Madelon 500-feature ×
shap_select). TimeoutExpired is caught and raised as RuntimeError so
callers' existing try/except handles it the same as any other failure.
"""
import json, subprocess, tempfile, os, pathlib

HERE = pathlib.Path(__file__).resolve().parent
VENVS = {
    "powershap":  HERE / "venvs" / "venv_powershap" / "bin" / "python",
    "shapselect": HERE / "venvs" / "venv_shapselect" / "bin" / "python",
    "boruta":     HERE / "venvs" / "venv_boruta" / "bin" / "python",
}
WORKER = HERE / "worker.py"

# Maximum wall-clock seconds for a single worker subprocess.
# 1800s (30 min) is generous -- most calls finish in <60s; Madelon×shap_select
# is the known outlier that never converges within any reasonable budget.
TIMEOUT_S = int(os.environ.get("BRIDGE_TIMEOUT", 1800))


def _topk_cols(feature_names, scores, k):
    feature_names = list(feature_names)
    k = min(k, len(feature_names))
    import numpy as np
    idx = np.argsort(scores)[::-1][:k]
    return [feature_names[i] for i in idx]


def call_official_raw(lib, X, y, seed):
    """Runs the worker and returns its full JSON result dict."""
    py = VENVS[lib]
    if not py.exists():
        raise RuntimeError(f"official_bridge: venv for '{lib}' not found at {py}")
    with tempfile.TemporaryDirectory() as td:
        x_path = os.path.join(td, "X.csv")
        y_path = os.path.join(td, "y.csv")
        out_path = os.path.join(td, "out.json")
        X.to_csv(x_path, index=False)
        y.to_frame(name="target").to_csv(y_path, index=False)
        cmd = [str(py), str(WORKER), lib, x_path, y_path, "0", str(seed), "0", out_path]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"official_bridge[{lib}]: worker timed out after {TIMEOUT_S}s. "
                f"This combination is computationally infeasible within the "
                f"time budget. Set BRIDGE_TIMEOUT env var to override.")
        if not os.path.exists(out_path):
            raise RuntimeError(
                f"official_bridge[{lib}]: worker produced no output.\n"
                f"stdout(tail): {proc.stdout[-2000:]}\nstderr(tail): {proc.stderr[-2000:]}")
        result = json.load(open(out_path))
    if not result.get("ok", False):
        raise RuntimeError(
            f"official_bridge[{lib}] selection failed: {result.get('error')}\n"
            f"{result.get('traceback', '')}")
    return result


def call_official(lib, X, y, k, seed, native=False, pad_fn=None):
    """
    Mirrors the existing native/non-native contract used throughout
    feature_selectors.py:
      native=True  -> the method's own accepted set, as-is.
      native=False -> truncate to k (by the library's own ranking) if the
                       accepted set is >= k, else pad with `pad_fn` (a
                       callable(X_remaining, y, n_needed) -> list[str],
                       typically select_shap_topk) up to k.
    """
    result = call_official_raw(lib, X, y, seed)
    accepted = result["selected"]
    ranked = result.get("ranked") or accepted
    if native:
        return list(accepted)
    if len(accepted) >= k:
        # keep the library's own ranking order, take the first k of the accepted set
        rank_pos = {f: i for i, f in enumerate(ranked)}
        ordered = sorted(accepted, key=lambda f: rank_pos.get(f, len(ranked)))
        return ordered[:k]
    remaining = [c for c in X.columns if c not in accepted]
    if remaining and pad_fn is not None:
        extra = pad_fn(X[remaining], y, k - len(accepted))
        return list(accepted) + list(extra)
    return list(accepted)
