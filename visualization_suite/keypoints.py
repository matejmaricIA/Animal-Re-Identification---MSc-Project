"""Utilities for visualising local keypoints."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np



def _iter_keypoints(kps: Iterable) -> Iterable:
    for kp in kps:
        if isinstance(kp, cv2.KeyPoint):
            yield kp
        else:
            x, y = kp[:2]
            size = float(kp[2]) if len(kp) > 2 else 3.0
            angle = float(kp[3]) if len(kp) > 3 else 0.0
            yield cv2.KeyPoint(x, y, size, angle)


def draw_keypoints(image: np.ndarray, kps, color=(0, 255, 0), diameter: int = 3):
    """Draw keypoints and orientations on ``image``.

    Parameters
    ----------
    image : np.ndarray
        Input image in BGR format.
    kps : Iterable
        Sequence of ``cv2.KeyPoint`` or arrays of ``(x, y[, size, angle])``.
    color : tuple, optional
        BGR colour for drawing the keypoints.
    diameter : int, optional
        Minimum diameter for drawn circles.

    Returns
    -------
    image : np.ndarray
        Visualisation of the keypoints.
    metadata : dict
        Contains a caption describing the drawing.
    """
    vis = image.copy()
    kps = list(_iter_keypoints(kps))
    for kp in kps:
        radius = max(int(kp.size / 2), diameter)
        center = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        cv2.circle(vis, center, radius, color, 1, cv2.LINE_AA)
        if kp.angle != -1:
            ang = np.deg2rad(kp.angle)
            pt2 = (
                int(round(kp.pt[0] + radius * np.cos(ang))),
                int(round(kp.pt[1] + radius * np.sin(ang))),
            )
            cv2.line(vis, center, pt2, color, 1, cv2.LINE_AA)
    caption = f"{len(kps)} keypoints"
    return vis, {"caption": caption}
