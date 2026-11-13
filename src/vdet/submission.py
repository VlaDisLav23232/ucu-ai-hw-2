"""Kaggle submission formatter, parser, and strict validator."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vdet.constants import MAX_SUBMISSION_DETS, NUM_CLASSES
from vdet.infer.predictors import empty_predictions_df
from vdet.paths import P


def _resolve_test_images_csv(test_images_csv: str | Path | None = None) -> Path:
    """Robustly resolve test_images.csv across root, notebooks/, and data/ directories."""
    if test_images_csv is not None:
        candidate = Path(test_images_csv)
        if candidate.exists():
            return candidate
    for candidate in (
        P.test_images_csv,
        P.root / "data" / "test_images.csv",
        P.root / "test_images.csv",
        Path("data/test_images.csv"),
        Path("../data/test_images.csv"),
        Path("test_images.csv"),
        Path("../test_images.csv"),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "test_images.csv could not be found in data/ or repository root."
    )


def to_csv(
    preds_df: pd.DataFrame,
    out_csv: str | Path,
    test_images_csv: str | Path | None = None,
    max_dets: int = MAX_SUBMISSION_DETS,
    decimals: int = 2,
) -> Path:
    """Write a prediction table to a competition submission CSV.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Prediction table with `image_key, cls, score, x, y, w, h` in original-image pixels.
    out_csv : str | Path
        Destination CSV file path.
    test_images_csv : str | Path | None
        Reference `test_images.csv` defining row order and per-frame `(width, height)`.
    max_dets : int
        Maximum detections per image (default 500).
    decimals : int
        Decimal precision for `x, y, w, h` (default 2).
    """
    ref_csv = _resolve_test_images_csv(test_images_csv)
    ref_df = pd.read_csv(ref_csv)

    preds_by_key: dict[str, pd.DataFrame] = {}
    if preds_df is not None and not preds_df.empty:
        preds_by_key = {str(k): g for k, g in preds_df.groupby("image_key", sort=False)}

    rows: list[dict[str, str]] = []
    coord_fmt = f"{{:.{decimals}f}}"

    for ref_row in ref_df.itertuples(index=False):
        img_id = str(ref_row.image_id)
        img_w = float(ref_row.width)
        img_h = float(ref_row.height)

        sub = preds_by_key.get(img_id)
        if sub is None or sub.empty:
            rows.append({"image_id": img_id, "PredictionString": ""})
            continue

        # Sort descending by score
        sub_sorted = sub.sort_values("score", ascending=False)
        groups: list[str] = []

        for det in sub_sorted.itertuples(index=False):
            cls_id = int(det.cls)
            score = float(det.score)
            if not (0 <= cls_id < NUM_CLASSES):
                continue
            if not (math.isfinite(score) and 0.0 < score <= 1.0):
                continue

            bx = float(det.x)
            by = float(det.y)
            bw = float(det.w)
            bh = float(det.h)
            if not (math.isfinite(bx) and math.isfinite(by) and math.isfinite(bw) and math.isfinite(bh)):
                continue

            # Clip to [0, img_w] x [0, img_h]
            x1 = min(img_w, max(0.0, bx))
            y1 = min(img_h, max(0.0, by))
            x2 = min(img_w, max(0.0, bx + bw))
            y2 = min(img_h, max(0.0, by + bh))
            cw = x2 - x1
            ch = y2 - y1

            rx = round(x1, decimals)
            ry = round(y1, decimals)
            rw = round(cw, decimals)
            rh = round(ch, decimals)
            if rw <= 0.0 or rh <= 0.0:
                continue
            # Ensure rounded box stays within image bounds
            if rx + rw > img_w:
                rw = round(max(0.0, img_w - rx), decimals)
            if ry + rh > img_h:
                rh = round(max(0.0, img_h - ry), decimals)
            if rw <= 0.0 or rh <= 0.0:
                continue

            s_str = f"{min(1.0, max(1e-6, score)):.4f}".rstrip("0").rstrip(".")
            if s_str in {"", "0"}:
                s_str = "0.0001"
            x_str = coord_fmt.format(rx)
            y_str = coord_fmt.format(ry)
            w_str = coord_fmt.format(rw)
            h_str = coord_fmt.format(rh)

            groups.append(f"{cls_id} {s_str} {x_str} {y_str} {w_str} {h_str}")
            if len(groups) >= max_dets:
                break

        rows.append({"image_id": img_id, "PredictionString": " ".join(groups)})

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def from_csv(sub_csv: str | Path) -> pd.DataFrame:
    """Parse a Kaggle submission CSV into a prediction DataFrame."""
    df = pd.read_csv(sub_csv, dtype=str, keep_default_na=False)
    records: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        img_id = str(row.image_id)
        pred_str = str(row.PredictionString).strip()
        if not pred_str:
            continue
        tokens = pred_str.split()
        if len(tokens) % 6 != 0:
            raise ValueError(
                f"Row '{img_id}' has {len(tokens)} tokens in PredictionString (not divisible by 6)."
            )
        for i in range(0, len(tokens), 6):
            records.append(
                {
                    "image_key": img_id,
                    "cls": np.int8(int(tokens[i])),
                    "score": np.float32(float(tokens[i + 1])),
                    "x": np.float32(float(tokens[i + 2])),
                    "y": np.float32(float(tokens[i + 3])),
                    "w": np.float32(float(tokens[i + 4])),
                    "h": np.float32(float(tokens[i + 5])),
                }
            )
    if not records:
        return empty_predictions_df()
    return pd.DataFrame(records)


def validate(
    sub_csv: str | Path,
    test_images_csv: str | Path | None = None,
    max_dets: int = MAX_SUBMISSION_DETS,
    bounds_tol_px: float = 1.0,
) -> dict[str, Any]:
    """Strictly validate a submission CSV against all and contract rules.

    Raises `ValueError` with a descriptive message if any check fails.
    """
    csv_path = Path(sub_csv)
    if not csv_path.exists():
        raise ValueError(f"Submission file does not exist: {csv_path}")

    ref_path = _resolve_test_images_csv(test_images_csv)
    ref_df = pd.read_csv(ref_path)
    sub_df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    # 1. Columns check
    expected_cols = ["image_id", "PredictionString"]
    if list(sub_df.columns) != expected_cols:
        raise ValueError(
            f"Invalid columns {list(sub_df.columns)}; expected exact columns {expected_cols}."
        )

    # 2. Row count & ID set / ordering check
    if len(sub_df) != len(ref_df):
        raise ValueError(
            f"Row count mismatch: submission has {len(sub_df)} rows, expected {len(ref_df)}."
        )

    ref_ids = ref_df["image_id"].astype(str).tolist()
    sub_ids = sub_df["image_id"].astype(str).tolist()
    if sub_ids != ref_ids:
        if set(sub_ids) != set(ref_ids):
            missing = list(set(ref_ids) - set(sub_ids))[:5]
            extra = list(set(sub_ids) - set(ref_ids))[:5]
            raise ValueError(f"image_id set mismatch: missing={missing}, extra={extra}")
        raise ValueError("image_id ordering does not match test_images.csv row order.")

    # 3. Per-row prediction format, class range, score range, count, finite values, bounds
    total_dets = 0
    non_empty_rows = 0
    widths = ref_df["width"].to_numpy(dtype=np.float64)
    heights = ref_df["height"].to_numpy(dtype=np.float64)
    pred_strings = sub_df["PredictionString"].tolist()

    for row_idx, (img_id, pred_str, img_w, img_h) in enumerate(
        zip(sub_ids, pred_strings, widths, heights)
    ):
        s = str(pred_str).strip()
        if not s:
            continue
        if "nan" in s.lower() or "inf" in s.lower():
            raise ValueError(f"Row {row_idx} ({img_id}) contains NaN or Inf token in PredictionString.")

        tokens = s.split()
        if len(tokens) % 6 != 0:
            raise ValueError(
                f"Row {row_idx} ({img_id}) has {len(tokens)} tokens; must be a multiple of 6."
            )
        n_dets = len(tokens) // 6
        if n_dets > max_dets:
            raise ValueError(
                f"Row {row_idx} ({img_id}) has {n_dets} detections; exceeds max_dets={max_dets}."
            )

        non_empty_rows += 1
        total_dets += n_dets

        for g in range(n_dets):
            base = g * 6
            cls_tok = tokens[base]
            try:
                cls_val = int(cls_tok)
                if str(cls_val) != cls_tok:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_idx} ({img_id}), det {g}: non-integer class token '{cls_tok}'."
                ) from exc

            if not (0 <= cls_val < NUM_CLASSES):
                raise ValueError(
                    f"Row {row_idx} ({img_id}), det {g}: class {cls_val} out of valid range 0..{NUM_CLASSES - 1}."
                )

            try:
                score = float(tokens[base + 1])
                bx = float(tokens[base + 2])
                by = float(tokens[base + 3])
                bw = float(tokens[base + 4])
                bh = float(tokens[base + 5])
            except ValueError as exc:
                raise ValueError(
                    f"Row {row_idx} ({img_id}), det {g}: non-numeric score or bbox coordinate."
                ) from exc

            if not (
                math.isfinite(score)
                and math.isfinite(bx)
                and math.isfinite(by)
                and math.isfinite(bw)
                and math.isfinite(bh)
            ):
                raise ValueError(f"Row {row_idx} ({img_id}), det {g}: non-finite number detected.")

            if not (0.0 < score <= 1.0):
                raise ValueError(
                    f"Row {row_idx} ({img_id}), det {g}: score {score} outside valid range (0, 1]."
                )

            if bw <= 0.0 or bh <= 0.0:
                raise ValueError(
                    f"Row {row_idx} ({img_id}), det {g}: non-positive box size w={bw}, h={bh}."
                )

            if (
                bx < -bounds_tol_px
                or by < -bounds_tol_px
                or (bx + bw) > (img_w + bounds_tol_px)
                or (by + bh) > (img_h + bounds_tol_px)
            ):
                raise ValueError(
                    f"Row {row_idx} ({img_id}), det {g}: box [{bx}, {by}, {bw}, {bh}] out of image bounds "
                    f"({img_w}x{img_h}, tol={bounds_tol_px}px)."
                )

    return {
        "valid": True,
        "rows": len(sub_df),
        "non_empty_rows": non_empty_rows,
        "total_detections": total_dets,
        "mean_dets_per_frame": round(total_dets / max(1, len(sub_df)), 2),
    }

