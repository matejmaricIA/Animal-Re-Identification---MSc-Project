import cv2
import numpy as np
from pathlib import Path


def segment_animal(image_path):
    """Return a binary mask for the animal in the image.

    If a mask with suffix ``_mask`` exists next to the image, it will be used.
    Otherwise a simple Otsu threshold is applied as a fallback."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    mask_path = Path(image_path).with_name(Path(image_path).stem + "_mask.png")
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            return mask

    # Fallback segmentation using Otsu thresholding
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return mask


def fourier_descriptors(mask, num_coeffs=32):
    """Compute basic Fourier descriptors from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros(num_coeffs * 2, dtype=np.float32)

    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim < 2:
        return np.zeros(num_coeffs * 2, dtype=np.float32)

    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    ft = np.fft.fft(complex_contour)
    descriptors = ft[1 : num_coeffs + 1]
    desc = np.concatenate([descriptors.real, descriptors.imag]).astype(np.float32)
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc /= norm
    return desc


def compute_shape_descriptors(image_paths, num_coeffs=32):
    """Compute shape descriptors for a list of image paths."""
    descriptors = {}
    for path in image_paths:
        img_id = Path(path).stem
        mask = segment_animal(path)
        if mask is None:
            continue
        descriptors[img_id] = fourier_descriptors(mask, num_coeffs=num_coeffs)
    return descriptors


def standardize(descriptor_dict, mean=None, std=None):
    """Standardize a descriptor dictionary."""
    ids = list(descriptor_dict.keys())
    if not ids:
        return descriptor_dict, mean, std
    mat = np.stack([descriptor_dict[i] for i in ids])
    if mean is None:
        mean = mat.mean(axis=0)
    if std is None:
        std = mat.std(axis=0) + 1e-6
    mat = (mat - mean) / std
    return dict(zip(ids, mat)), mean, std