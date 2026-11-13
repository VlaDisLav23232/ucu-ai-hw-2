"""Diagnostics: summary figures, TIDE error breakdown, failure galleries and report tables."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vdet.eval.errors import (
    compute_tide_breakdown,
    plot_failure_gallery,
    plot_tide_breakdown,
)
from vdet.infer.predictors import load_predictions_parquet
from vdet.paths import P
from vdet.tracking.aggregate import aggregate_runs
from vdet.tracking.plots import (
    curves,
    map_vs_fps,
    map_vs_gpu_time,
    per_class_ap,
    size_breakdown,
)
from vdet.utils import save_json, timer

EXCLUDED_EXPS = ("smoke", "u12_blur", "u12_drop")


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-aggregate experiments/runs/*/summary.json; return (all runs, evaluated non-smoke runs)."""
    aggregate_runs()
    df = pd.read_csv(P.results_csv)
    eval_df = df[df["AP"].notna() & (~df["exp"].isin(list(EXCLUDED_EXPS)))].copy()
    return df, eval_df


def make_summary_figures() -> None:
    """Summary figures: size_breakdown, per_class_ap, map_vs_fps, curves_<family>, map_vs_gpu_time_all."""
    df, eval_df = load_results()
    with timer("Summary figures (size_breakdown, map_vs_fps, per_class_ap, curves)"):
        if not eval_df.empty:
            size_breakdown(eval_df, P.images / "size_breakdown.png")
            per_class_ap(eval_df, P.images / "per_class_ap.png")
            fps_df = eval_df[eval_df["fps"].notna() & (pd.to_numeric(eval_df["fps"], errors="coerce") > 0)]
            if not fps_df.empty:
                map_vs_fps(fps_df, P.images / "map_vs_fps.png")

        # Curves per family & map_vs_gpu_time_all.png
        for fam in ["yolo", "rtdetr", "faster_rcnn", "retinanet"]:
            fam_runs = {}
            for rid in df[df["family"] == fam]["run_id"].tolist():
                if "smoke" in str(rid) or "u12" in str(rid):
                    continue
                mp = P.exp_runs_dir / str(rid) / "metrics.csv"
                if mp.exists():
                    fam_runs[str(rid)] = pd.read_csv(mp)
            if fam_runs:
                curves(fam_runs, P.images / f"curves_{fam}.png")

        ultra_curves = {}
        for rid in ["yolo-e0", "yolo-e1", "yolo-e2", "rtdetr-e0", "rtdetr-e1", "rtdetr-e2"]:
            mp = P.exp_runs_dir / rid / "metrics.csv"
            if mp.exists():
                ultra_curves[rid] = pd.read_csv(mp)
        if len(ultra_curves) >= 2:
            map_vs_gpu_time(ultra_curves, P.images / "map_vs_gpu_time_all.png")



def make_tide_and_failure_figures() -> dict[str, Any]:
    """TIDE breakdown + failure gallery for the best run of each family."""
    tide_summary: dict[str, object] = {}
    for fam_tag, candidate_rids in [
        ("yolo", ["yolo-e1", "yolo-e0"]),
        ("rtdetr", ["rtdetr-e0", "rtdetr-e1"]),
        ("faster_rcnn", ["frcnn-e2", "frcnn-e1", "frcnn-e0"]),
        ("retinanet", ["retina-e2", "retina-e1", "retina-e0"]),
        ("pipeline", ["pipeline-final", "ensemble-wbf"]),
    ]:
        chosen_parquet = None
        chosen_rid = None
        for rid in candidate_rids:
            for cand in [
                P.runs / rid / "preds_val.parquet",
                P.runs / "pipeline-final" / "preds_val_final.parquet" if rid == "pipeline-final" else P.runs / rid / "preds_val.parquet",
                P.exp_runs_dir / rid / "preds_val_slim.parquet",
            ]:
                if cand.exists():
                    chosen_parquet = cand
                    chosen_rid = rid
                    break
            if chosen_parquet is not None:
                break

        if chosen_parquet is None or chosen_rid is None:
            continue

        with timer(f"TIDE & failure gallery ({fam_tag}: {chosen_rid})"):
            preds_df = load_predictions_parquet(chosen_parquet)
            tide_res = compute_tide_breakdown(preds_df, split="val", conf_thresh=0.20, top_k_per_frame=100)
            tide_summary[chosen_rid] = tide_res
            plot_tide_breakdown(tide_res, f"{chosen_rid} ({fam_tag})", P.images / f"tide_{fam_tag}.png")
            plot_failure_gallery(preds_df, f"{chosen_rid} ({fam_tag})", P.images / f"failures_{fam_tag}.png")

    if tide_summary:
        save_json(tide_summary, P.experiments / "tide_breakdown_summary.json")

    return tide_summary


def write_report_tables() -> str:
    """Write experiments/report_tables.md (markdown table of all evaluated runs) and return it."""
    _, eval_df = load_results()
    # 4. Write Copy-Paste-Ready Markdown Tables for the Report (experiments/report_tables.md)
    cols = [
        "run_id",
        "family",
        "exp",
        "imgsz",
        "epochs_done",
        "gpu_h",
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR500",
        "fps",
    ]
    present_cols = [c for c in cols if c in eval_df.columns]
    header = "| " + " | ".join(present_cols) + " |"
    sep = "| " + " | ".join(["---"] * len(present_cols)) + " |"
    table_rows = [header, sep]
    for _, row in eval_df[present_cols].iterrows():
        cells = []
        for c in present_cols:
            val = row[c]
            if pd.isna(val) or val == "":
                cells.append("-")
            elif isinstance(val, float):
                cells.append(f"{val:.4f}" if c not in {"gpu_h", "fps"} else f"{val:.2f}")
            else:
                cells.append(str(val))
        table_rows.append("| " + " | ".join(cells) + " |")

    md_lines = [
        "# Master Experimental Results Table (Auto-Generated from `experiments/results.csv`)",
        "",
        *table_rows,
        "",
    ]
    report_md_path = P.experiments / "report_tables.md"
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved report-ready Markdown summary table to {report_md_path}")
    return "\n".join(md_lines)
