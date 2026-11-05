"""Camera-motion compensation via ORB + RANSAC homography between consecutive drone frames (§10 U4.6)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from vdet.paths import P


def estimate_frame_homography(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """Estimate `3x3` homography mapping `prev -> curr` in full-resolution pixel coordinates."""
    orb = cv2.ORB_create(nfeatures=400)
    kp1, des1 = orb.detectAndCompute(gray_prev, None)
    kp2, des2 = orb.detectAndCompute(gray_curr, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.eye(3, dtype=np.float64)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    if len(matches) < 8:
        return np.eye(3, dtype=np.float64)

    matches = sorted(matches, key=lambda m: m.distance)[:120]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H_small, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    if H_small is None or not np.all(np.isfinite(H_small)):
        return np.eye(3, dtype=np.float64)

    # Convert homography from downscaled coordinates back to original coordinates: H = S^-1 * H_small * S
    S = np.diag([scale_x, scale_y, 1.0])
    S_inv = np.diag([1.0 / max(scale_x, 1e-6), 1.0 / max(scale_y, 1e-6), 1.0])
    H_full = S_inv @ H_small.astype(np.float64) @ S
    # Reject implausible degenerate homographies
    if abs(H_full[2, 2]) < 1e-6:
        return np.eye(3, dtype=np.float64)
    H_full /= H_full[2, 2]
    if abs(np.linalg.det(H_full[:2, :2]) - 1.0) > 0.35:
        return np.eye(3, dtype=np.float64)
    return H_full


def warp_boxes_xywh(
    boxes_xywh: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    """Warp `(N, 4)` `(x, y, w, h)` boxes through a `3x3` homography `H` and return enclosing `(x, y, w, h)`."""
    if len(boxes_xywh) == 0 or np.allclose(H, np.eye(3)):
        return boxes_xywh.copy()

    x1 = boxes_xywh[:, 0]
    y1 = boxes_xywh[:, 1]
    x2 = x1 + boxes_xywh[:, 2]
    y2 = y1 + boxes_xywh[:, 3]

    # 4 corners per box: (N, 4, 2)
    corners = np.stack(
        [
            np.stack([x1, y1], axis=-1),
            np.stack([x2, y1], axis=-1),
            np.stack([x2, y2], axis=-1),
            np.stack([x1, y2], axis=-1),
        ],
        axis=1,
    ).astype(np.float32)

    warped = cv2.perspectiveTransform(corners, H.astype(np.float64))
    wx1 = warped[:, :, 0].min(axis=1)
    wy1 = warped[:, :, 1].min(axis=1)
    wx2 = warped[:, :, 0].max(axis=1)
    wy2 = warped[:, :, 1].max(axis=1)

    return np.stack([wx1, wy1, np.maximum(1.0, wx2 - wx1), np.maximum(1.0, wy2 - wy1)], axis=1)


def compute_clip_homographies(
    clip_manifest_df: pd.DataFrame,
    data_root: Path | None = None,
    max_side: int = 640,
) -> dict[tuple[int, int], np.ndarray]:
    """Estimate consecutive frame-to-frame homographies `(frame_t, frame_t+1) -> H_3x3` for a clip."""
    d_root = data_root or P.data
    df_work = clip_manifest_df.copy()
    if "frame_idx" not in df_work.columns and "frame" in df_work.columns:
        df_work["frame_idx"] = df_work["frame"]
    rows = df_work.sort_values("frame_idx").reset_index(drop=True)
    homographies: dict[tuple[int, int], np.ndarray] = {}
    prev_gray: np.ndarray | None = None
    prev_f: int | None = None
    prev_sx, prev_sy = 1.0, 1.0

    for r in rows.itertuples(index=False):
        bgr = cv2.imread((d_root / str(r.path)).as_posix(), cv2.IMREAD_GRAYSCALE)
        if bgr is None:
            prev_gray = None
            prev_f = None
            continue
        h, w = bgr.shape[:2]
        scale = min(1.0, float(max_side) / float(max(h, w)))
        small_w = max(32, int(round(w * scale)))
        small_h = max(32, int(round(h * scale)))
        gray = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
        sx = small_w / float(w)
        sy = small_h / float(h)

        f_idx = int(r.frame_idx)
        if prev_gray is not None and prev_f is not None and gray.shape == prev_gray.shape:
            homographies[(prev_f, f_idx)] = estimate_frame_homography(prev_gray, gray, scale_x=sx, scale_y=sy)
        prev_gray = gray
        prev_f = f_idx
        prev_sx, prev_sy = sx, sy

    return homographies

