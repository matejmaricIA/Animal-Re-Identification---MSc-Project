"""Provided-mask contracts shared by CzechLynx preparation and evaluation."""
from __future__ import annotations

import numpy as np
from PIL import Image

from reid_demo.benchmark import file_digest, fingerprint


MASKED_PREPROCESSING = {
    "version": "provided_mask_gray128_png_v1",
    "background_rgb": [128, 128, 128],
    "image_format": "PNG", "mask_values": [0, 255],
}
MASK_KEYPOINT_POLICY = "inside_crop_nearest_pixel_floor_plus_half_v1"


def local_input_hash(row, *, masked=False):
    if not masked:
        return row["image_sha256"]
    if not all(isinstance(row.get(key), str) and row[key] for key in ("mask_path", "mask_sha256")):
        raise ValueError("Masked images require mask_path and mask_sha256.")
    if file_digest(row["mask_path"]) != row["mask_sha256"]:
        raise ValueError(f"Mask changed since preparation: {row['image_id']}")
    return fingerprint({"image": row["image_sha256"], "mask": row["mask_sha256"]})


def load_foreground_mask(row):
    """Require a checksummed binary mask at the exact extracted-image resolution."""
    local_input_hash(row, masked=True)
    with Image.open(row["mask_path"]) as image:
        if image.mode != "L":
            raise ValueError("Prepared animal masks must be grayscale binary PNGs.")
        mask = np.asarray(image).copy()
    with Image.open(row["path"]) as image:
        if mask.shape != (image.height, image.width):
            raise ValueError(f"Mask/image dimensions differ: {row['image_id']}")
    if not np.isin(mask, [0, 255]).all() or not np.any(mask):
        raise ValueError(f"Invalid or empty binary mask: {row['image_id']}")
    return mask != 0


def filter_foreground_features(descriptors, keypoints, mask):
    """Keep matching descriptor/keypoint rows, including correctly shaped empties."""
    descriptors, keypoints = np.asarray(descriptors), np.asarray(keypoints)
    if (descriptors.ndim != 2 or keypoints.shape != (len(descriptors), 2) or
            mask.ndim != 2 or not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all()):
        raise ValueError("Invalid local features or animal mask.")
    height, width = mask.shape
    in_bounds = ((keypoints[:, 0] >= 0) & (keypoints[:, 0] <= width - 1) &
                 (keypoints[:, 1] >= 0) & (keypoints[:, 1] <= height - 1))
    keep = np.zeros(len(keypoints), dtype=bool)
    positions = np.floor(keypoints[in_bounds] + 0.5).astype(np.int64)
    keep[in_bounds] = mask[positions[:, 1], positions[:, 0]] != 0
    return descriptors[keep], keypoints[keep]
