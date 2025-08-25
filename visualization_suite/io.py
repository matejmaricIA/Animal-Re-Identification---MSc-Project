"""I/O utilities for the :mod:`visualization_suite` package.

All images are loaded in BGR colour order (OpenCV default).  Helper
functions are provided to convert to RGB when displaying with
:mod:`matplotlib`.
"""
from __future__ import annotations

import cv2
import h5py
import numpy as np
from typing import Dict, Sequence


def load_image(path: str) -> np.ndarray:
    """Load an image from ``path`` in BGR format.

    Parameters
    ----------
    path: str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Image in BGR colour space.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_image(path: str, image: np.ndarray) -> None:
    """Save ``image`` (BGR) to ``path`` using :func:`cv2.imwrite`."""
    cv2.imwrite(path, image)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for matplotlib display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def fig_to_image(fig) -> np.ndarray:
    """Convert a matplotlib figure to a BGR ``numpy.ndarray``."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def load_keypoints_h5(h5_path: str, ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load keypoints for ``ids`` from an HDF5 file.

    Parameters
    ----------
    h5_path: str
        Path to the HDF5 file.  Typical values are provided in
        :mod:`constants` such as ``SAVE_TRAIN_DESCRIPTORS_PATH``.
    ids: Sequence[str]
        Identifiers to retrieve from the file.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of identifier to array of keypoints.
    """
    data: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        # Determine zero-padding length from any existing key (if numeric)
        pad_len = None
        keys = list(f.keys())
        if keys:
            sample_key = keys[0]
            if sample_key.isdigit():
                pad_len = len(sample_key)

        for image_id in ids:
            key = str(image_id)
            if key not in f and pad_len and key.isdigit():
                key = key.zfill(pad_len)
            if key in f:
                # Store under the original id (as string) so callers can use
                # their provided identifier regardless of padding.
                data[str(image_id)] = np.array(f[key])
    return data


def load_descriptors_h5(h5_path: str, ids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load descriptors for ``ids`` from an HDF5 file."""
    data: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as f:
        pad_len = None
        keys = list(f.keys())
        if keys:
            sample_key = keys[0]
            if sample_key.isdigit():
                pad_len = len(sample_key)

        for image_id in ids:
            key = str(image_id)
            if key not in f and pad_len and key.isdigit():
                key = key.zfill(pad_len)
            if key in f:
                data[str(image_id)] = np.array(f[key])
    return data
