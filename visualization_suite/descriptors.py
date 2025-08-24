"""Descriptor visualisation helpers."""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt

from . import io, style


def visualize_descriptor(desc: np.ndarray, binary: bool = False) -> Tuple[np.ndarray, str]:
    """Visualise a single descriptor as a heatmap.

    Parameters
    ----------
    desc : np.ndarray
        Descriptor vector.  It will be reshaped to ``(8, 16)`` for display.
    binary : bool, optional
        If ``True`` an additional radial plot is produced for binary
        descriptors (e.g. ORB/BRIEF).
    """
    style.set_style()
    fig, ax = plt.subplots(1, 2 if binary else 1, figsize=(4 if not binary else 8, 4))
    if not isinstance(ax, np.ndarray):
        ax = np.array([ax])

    heat = desc.reshape(8, 16)
    im = ax[0].imshow(heat, cmap="viridis")
    ax[0].set_title("Heatmap")
    ax[0].axis("off")
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    if binary:
        bits = desc.ravel() > 0
        theta = np.linspace(0, 2 * math.pi, bits.size, endpoint=False)
        ax[1] = plt.subplot(1, 2, 2, projection='polar')
        ax[1].bar(theta, bits.astype(int), width=2 * math.pi / bits.size)
        ax[1].set_title("Binary")
        ax[1].set_rticks([])
        ax[1].set_xticks([])

    fig.tight_layout()
    image = io.fig_to_image(fig)
    plt.close(fig)
    caption = "Descriptor visualisation"
    return image, caption
