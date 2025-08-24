"""Visualisation of feature matches."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import cv2
import numpy as np



def draw_matches(
    img1: np.ndarray,
    kp1: np.ndarray,
    img2: np.ndarray,
    kp2: np.ndarray,
    matches: Iterable[cv2.DMatch],
    max_matches: int = 50,
) -> Tuple[np.ndarray, dict]:
    """Draw matches between two images.

    Parameters
    ----------
    img1, img2 : np.ndarray
        Input images (BGR).
    kp1, kp2 : np.ndarray
        Keypoint coordinates ``(N, 2)`` corresponding to ``img1`` and
        ``img2``.
    matches : Iterable[cv2.DMatch]
        Iterable of matches, typically from :class:`cv2.DMatch`.
    max_matches : int, optional
        Only the ``max_matches`` best (smallest distance) are drawn.
    """
    matches = sorted(matches, key=lambda m: m.distance)[:max_matches]

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    offset = w1
    matched1 = set()
    matched2 = set()

    for m in matches:
        p1 = tuple(np.round(kp1[m.queryIdx]).astype(int))
        p2 = tuple(np.round(kp2[m.trainIdx]).astype(int))
        matched1.add(m.queryIdx)
        matched2.add(m.trainIdx)
        cv2.line(vis, p1, (p2[0] + offset, p2[1]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, p1, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(vis, (p2[0] + offset, p2[1]), 3, (0, 255, 0), -1, cv2.LINE_AA)

    for i, p in enumerate(kp1):
        if i not in matched1:
            cv2.circle(vis, tuple(np.round(p).astype(int)), 3, (0, 0, 255), -1, cv2.LINE_AA)
    for i, p in enumerate(kp2):
        if i not in matched2:
            cv2.circle(vis, (int(p[0]) + offset, int(p[1])), 3, (0, 0, 255), -1, cv2.LINE_AA)

    metadata = {
        "n_matches": len(matches),
        "n_unmatched1": len(kp1) - len(matched1),
        "n_unmatched2": len(kp2) - len(matched2),
        "caption": f"{len(matches)} matches",
    }
    return vis, metadata
