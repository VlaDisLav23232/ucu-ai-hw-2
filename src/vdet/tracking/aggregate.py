"""Aggregate `experiments/runs/*/summary.json` and `experiments/lb.csv` into `experiments/results.csv` (§6.5, U0.7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from vdet.constants import CLASS_NAMES
from vdet.paths import P
from vdet.utils import load_json

RESULTS_COLUMNS: tuple[str, ...] = (
    "run_id",
    "unit",
    "family",
    "exp",
    "parent_run",
    "hypothesis",
    "cache",
    "imgsz",
    "k",
    "epochs_done",
    "gpu_h",
    "gpu_type",
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR500",
    "fps",
    *(f"ap_{name}" for name in CLASS_NAMES),
    "lb_public",
    "commit",
    "date",
)


def aggregate_runs(
    experiments_dir: str | Path | None = None,
    out_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Scan all `experiments/runs/*/summary.json`, join `lb.csv`, and write `experiments/results.csv`."""
    exp_root = Path(experiments_dir) if experiments_dir is not None else P.experiments
    runs_dir = exp_root / "runs"
    lb_csv = exp_root / "lb.csv"
    dest_csv = Path(out_csv) if out_csv is not None else (exp_root / "results.csv")

    lb_map: dict[str, Any] = {}
    if lb_csv.exists():
        lb_df = pd.read_csv(lb_csv)
        if not lb_df.empty and "run_id" in lb_df.columns and "lb_public" in lb_df.columns:
            for row in lb_df.itertuples(index=False):
                lb_map[str(row.run_id)] = row.lb_public

    rows: list[dict[str, Any]] = []
    if runs_dir.exists():
        for summary_file in sorted(runs_dir.glob("*/summary.json")):
            data = load_json(summary_file)
            run_id = str(data.get("run_id", summary_file.parent.name))
            record = {col: data.get(col, "") for col in RESULTS_COLUMNS}
            record["run_id"] = run_id
            if run_id in lb_map and pd.notna(lb_map[run_id]) and str(lb_map[run_id]) != "":
                record["lb_public"] = lb_map[run_id]
            rows.append(record)

    if rows:
        df = pd.DataFrame(rows, columns=list(RESULTS_COLUMNS))
        df = df.sort_values(["date", "run_id"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=list(RESULTS_COLUMNS))

    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_csv, index=False)
    return df

