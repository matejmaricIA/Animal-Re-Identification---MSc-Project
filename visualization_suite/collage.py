"""Image collage helpers."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from . import io, style


def make_grid(images: Sequence[np.ndarray], titles: Sequence[str] | None = None, cols: int = 2,
              figsize: Tuple[int, int] | None = None):
    """Arrange images into a simple grid using matplotlib."""
    style.set_style()
    n = len(images)
    cols = max(cols, 1)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize or (cols * 4, rows * 4))
    axes = np.atleast_2d(axes)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.imshow(io.bgr_to_rgb(img))
        ax.axis('off')
        if titles and idx < len(titles):
            ax.set_title(titles[idx])

    for ax in axes.flat[n:]:
        ax.axis('off')

    fig.tight_layout()
    grid_img = io.fig_to_image(fig)
    plt.close(fig)
    return grid_img, {"rows": rows, "cols": cols}
