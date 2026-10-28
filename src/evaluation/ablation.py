"""Tiny experiment logger: every run of the pipeline appends one row to reports/ablation_table.csv.
Re-running a config name overwrites its row, so the table always reflects the latest result."""
from pathlib import Path
import pandas as pd

COLS = ["config", "enhancement", "restoration", "segmentation", "features", "classifier"]


def log_result(path, config, params: dict, metrics: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"config": config, **params, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}}
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[old["config"] != config]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def load_results(path):
    return pd.read_csv(path) if Path(path).exists() else pd.DataFrame(columns=COLS)
