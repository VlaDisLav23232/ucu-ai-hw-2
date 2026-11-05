"""RunLogger: creates and manages local experiment run directories (§6.4, §9, U0.7)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vdet.constants import CLASS_NAMES, DEFAULT_SEED
from vdet.paths import P
from vdet.utils import get_env_metadata, load_json, save_json, save_yaml

METRICS_COLUMNS: tuple[str, ...] = (
    "epoch",
    "gpu_time_h",
    "lr",
    "train_loss",
    "loss_box",
    "loss_cls",
    "loss_dfl",
    "val_AP",
    "val_AP50",
    "val_APs",
)


def make_run_id(family: str, exp: str, timestamp: datetime | None = None) -> str:
    """Format canonical run ID `<family>-<exp>-<YYYYMMDD-HHMM>` (§6.4)."""
    ts = (timestamp or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M")
    return f"{family}-{exp}-{ts}"


class RunLogger:
    """Local experiment logger writing config.yaml, env.json, metrics.csv, notes.md, and summary.json."""

    def __init__(
        self,
        run_id: str,
        config: dict[str, Any] | None = None,
        runs_root: Path | None = None,
        seed: int = DEFAULT_SEED,
        hypothesis: str = "",
        parent_run: str = "",
        unit: str = "",
        family: str = "",
        exp: str = "",
    ) -> None:
        self.run_id = run_id
        self.runs_root = runs_root or P.exp_runs_dir
        self.run_dir = self.runs_root / run_id
        self.plots_dir = self.run_dir / "plots"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}
        self.seed = seed
        self.hypothesis = hypothesis or str(self.config.get("hypothesis", ""))
        self.parent_run = parent_run or str(self.config.get("parent_run", ""))
        self.unit = unit or str(self.config.get("unit", ""))
        self.family = family or str(self.config.get("family", run_id.split("-")[0]))
        self.exp = exp or str(
            self.config.get("exp", run_id.split("-")[1] if "-" in run_id else "e0")
        )

        # Write config.yaml & env.json at start
        save_yaml(self.config, self.run_dir / "config.yaml")
        self.env_meta = get_env_metadata(seed=self.seed)
        save_json(self.env_meta, self.run_dir / "env.json")

        # Initialize notes.md if not present
        notes_path = self.run_dir / "notes.md"
        if not notes_path.exists():
            self.write_notes(
                hypothesis=self.hypothesis or "Baseline / experimental run.",
                change_vs_parent=f"Parent: {self.parent_run or 'none'}",
                outcome="In progress.",
            )

    @property
    def metrics_csv_path(self) -> Path:
        return self.run_dir / "metrics.csv"

    @property
    def metrics_path(self) -> Path:
        return self.metrics_csv_path

    @property
    def summary_json_path(self) -> Path:
        return self.run_dir / "summary.json"

    @property
    def eval_val_json_path(self) -> Path:
        return self.run_dir / "eval_val.json"

    def write_notes(
        self,
        hypothesis: str,
        change_vs_parent: str = "",
        outcome: str = "",
    ) -> Path:
        """Write concise 2-5 line factual notes.md (§6.4)."""
        lines = [
            f"# Run `{self.run_id}`",
            f"- **Hypothesis:** {hypothesis}",
            f"- **Change vs parent (`{self.parent_run or 'none'}`):** {change_vs_parent or 'N/A'}",
            f"- **Outcome:** {outcome or 'Completed.'}",
        ]
        p = self.run_dir / "notes.md"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

    def log_epoch(self, row: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Append one epoch row to metrics.csv (§6.4)."""
        merged = dict(row or {})
        merged.update(kwargs)
        file_exists = self.metrics_csv_path.exists()
        out_row = {col: merged.get(col, "") for col in METRICS_COLUMNS}
        with self.metrics_csv_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(METRICS_COLUMNS))
            if not file_exists:
                writer.writeheader()
            writer.writerow(out_row)

    def finalize(
        self,
        eval_metrics: dict[str, Any] | None = None,
        speed_metrics: dict[str, Any] | None = None,
        epochs_done: int | None = None,
        gpu_h: float | None = None,
        checkpoint_uri: str = "",
        outcome_note: str | None = None,
        extra_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write eval_val.json (if given), speed.json (if given), and summary.json (§6.4, §6.5)."""
        if eval_metrics is not None:
            save_json(eval_metrics, self.eval_val_json_path)
        elif self.eval_val_json_path.exists():
            eval_metrics = load_json(self.eval_val_json_path)
        else:
            eval_metrics = {}

        speed_path = self.run_dir / "speed.json"
        if speed_metrics is not None:
            save_json(speed_metrics, speed_path)
        elif speed_path.exists():
            speed_metrics = load_json(speed_path)
        else:
            speed_metrics = {}

        # Infer epochs_done and gpu_h from metrics.csv if not explicitly passed
        if (epochs_done is None or gpu_h is None) and self.metrics_csv_path.exists():
            with self.metrics_csv_path.open("r", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows:
                if epochs_done is None:
                    try:
                        epochs_done = int(float(rows[-1]["epoch"]))
                    except (KeyError, ValueError):
                        epochs_done = len(rows)
                if gpu_h is None:
                    try:
                        gpu_h = round(float(rows[-1]["gpu_time_h"]), 4)
                    except (KeyError, ValueError):
                        gpu_h = 0.0

        data_cfg = self.config.get("data", {}) if isinstance(self.config.get("data"), dict) else {}
        input_cfg = (
            self.config.get("input", {}) if isinstance(self.config.get("input"), dict) else {}
        )

        cache_name = str(
            self.config.get("cache", data_cfg.get("cache_name", "k5_blur_L1920"))
        )
        imgsz = int(self.config.get("imgsz", input_cfg.get("imgsz", 1280)))
        step_k = int(self.config.get("k", data_cfg.get("frame_step_k", 5)))
        gpu_type = str(self.env_meta.get("gpu", {}).get("device_name", "cpu"))
        commit_short = str(self.env_meta.get("git", {}).get("commit_short", "unknown"))
        date_str = str(self.env_meta.get("timestamp_utc", ""))[:10]

        summary: dict[str, Any] = {
            "run_id": self.run_id,
            "unit": self.unit,
            "family": self.family,
            "exp": self.exp,
            "parent_run": self.parent_run,
            "hypothesis": self.hypothesis,
            "cache": cache_name,
            "imgsz": imgsz,
            "k": step_k,
            "epochs_done": int(epochs_done or 0),
            "gpu_h": round(float(gpu_h or 0.0), 4),
            "gpu_type": gpu_type,
            "AP": eval_metrics.get("AP", ""),
            "AP50": eval_metrics.get("AP50", ""),
            "AP75": eval_metrics.get("AP75", ""),
            "APs": eval_metrics.get("APs", ""),
            "APm": eval_metrics.get("APm", ""),
            "APl": eval_metrics.get("APl", ""),
            "AR500": eval_metrics.get("AR500", ""),
            "fps": speed_metrics.get("fps", ""),
            **{f"ap_{name}": eval_metrics.get(f"ap_{name}", "") for name in CLASS_NAMES},
            "checkpoint_uri": checkpoint_uri,
            "commit": commit_short,
            "date": date_str,
        }
        if extra_summary:
            summary.update(extra_summary)

        save_json(summary, self.summary_json_path)
        if outcome_note:
            self.write_notes(
                hypothesis=self.hypothesis or "Experimental run.",
                change_vs_parent=f"Parent: {self.parent_run or 'none'}",
                outcome=outcome_note,
            )
        return summary

