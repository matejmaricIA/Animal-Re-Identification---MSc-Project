"""Geometric verification visualisations."""
from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from . import matching


def verify_and_draw(
    img1: np.ndarray,
    kp1: np.ndarray,
    img2: np.ndarray,
    kp2: np.ndarray,
    matches: Iterable[cv2.DMatch],
    ransac_thresh: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Compute RANSAC verification and draw inliers/outliers.

    Returns
    -------
    inlier_vis, outlier_vis, combined_vis, metadata
    """
    matches = list(matches)
    if len(matches) < 4:
        return img1.copy(), img2.copy(), np.zeros_like(img1), {"n_inliers": 0, "n_outliers": len(matches)}

    pts1 = np.float32([kp1[m.queryIdx] for m in matches])
    pts2 = np.float32([kp2[m.trainIdx] for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    inliers = [m for m, msk in zip(matches, mask.ravel()) if msk]
    outliers = [m for m, msk in zip(matches, mask.ravel()) if not msk]

    inlier_vis, _ = matching.draw_matches(img1, kp1, img2, kp2, inliers)
    outlier_vis, _ = matching.draw_matches(img1, kp1, img2, kp2, outliers)

    # Combined visualisation
    h1, w1 = img1.shape[:2]
    combined = inlier_vis.copy()
    for m, msk in zip(matches, mask.ravel()):
        p1 = tuple(np.round(kp1[m.queryIdx]).astype(int))
        p2 = tuple(np.round(kp2[m.trainIdx]).astype(int))
        colour = (0, 255, 0) if msk else (0, 0, 255)
        cv2.line(combined, p1, (p2[0] + w1, p2[1]), colour, 1, cv2.LINE_AA)

    if H is not None:
        h, w = img1.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, H)
        cv2.polylines(combined, [np.int32(projected)], True, (255, 0, 0), 2, cv2.LINE_AA, shift=0, offset=(w1, 0))

    metadata = {"n_inliers": len(inliers), "n_outliers": len(outliers)}
    return inlier_vis, outlier_vis, combined, metadata
