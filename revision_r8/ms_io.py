"""Shared I/O for the multi-seed check: fixed column schema and a tolerant reader
that also repairs files written by the first version of the worker (whose error
rows had fewer columns than the header)."""
import csv
from pathlib import Path
import pandas as pd

COLS = ["seed", "dataset", "method", "k", "nested_xgb_auc", "nested_logreg_auc", "nested_mlp_auc",
        "nested_select_s", "leaky_xgb_auc", "status", "time_s"]
ERR_COLS = ["seed", "dataset", "method", "k", "status", "time_s"]     # layout of old error rows
ENV_ERROR = "venv for"                                                # bridge could not find a library env


def read_runs(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f); header = next(r, None)
        for rec in r:
            if not rec:
                continue
            if len(rec) == len(header):
                d = dict(zip(header, rec))
            elif len(rec) == len(ERR_COLS):
                d = dict(zip(ERR_COLS, rec))
            else:
                continue
            rows.append({c: d.get(c, "") for c in COLS})
    df = pd.DataFrame(rows, columns=COLS)
    for c in COLS:
        if c not in ("dataset", "method", "status"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def write_runs(path: Path, df: pd.DataFrame) -> None:
    df[COLS].to_csv(path, index=False)


def append_row(path: Path, row: dict) -> None:
    pd.DataFrame([{c: row.get(c) for c in COLS}])[COLS].to_csv(path, mode="a", header=not path.exists(), index=False)


def repair_all(runs_dir: Path, drop_env_errors: bool = True) -> int:
    """Rewrite every run file with the fixed schema; drop rows that failed only because a
    library environment was missing (so they are re-run). Returns the number of rows dropped."""
    dropped = 0
    for p in sorted(runs_dir.glob("*.csv")):
        df = read_runs(p)
        if drop_env_errors:
            bad = df.status.astype(str).str.contains(ENV_ERROR, regex=False)
            dropped += int(bad.sum()); df = df[~bad]
        df = df.drop_duplicates(["seed", "dataset", "method"], keep="last")
        write_runs(p, df)
    return dropped
