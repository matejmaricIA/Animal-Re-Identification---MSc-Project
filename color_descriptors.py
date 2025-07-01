import cv2
import numpy as np
from pathlib import Path
from scipy.stats import skew


def hsv_histogram(image, bins=(8, 8, 4)):
    """Compute a global HSV histogram and L1-normalize it."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist


def lab_moments(image):
    """Compute mean, variance and skewness for each Lab channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    moments = []
    for i in range(3):
        channel = lab[:, :, i].flatten().astype(np.float32)
        moments.append(channel.mean())
        moments.append(channel.var())
        moments.append(skew(channel))
    return np.array(moments, dtype=np.float32)


def compute_color_descriptors(image_paths):
    """Return dict mapping image_id to concatenated HSV histogram and Lab moments."""
    descriptors = {}
    for path in image_paths:
        img_id = Path(path).stem
        img = cv2.imread(path)
        if img is None:
            continue
        hist = hsv_histogram(img)
        moments = lab_moments(img)
        desc = np.concatenate([hist, moments]).astype(np.float32)
        descriptors[img_id] = desc
    return descriptors


def standardize(descriptor_dict, mean=None, std=None):
    ids = list(descriptor_dict.keys())
    mat = np.stack([descriptor_dict[i] for i in ids])
    if mean is None:
        mean = mat.mean(axis=0)
    if std is None:
        std = mat.std(axis=0) + 1e-6
    mat = (mat - mean) / std
    return dict(zip(ids, mat)), mean, std


def normalize_hsv(descriptor_dict, target_norm=1.0):
    for k, v in descriptor_dict.items():
        hist = v[:256]
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist * (target_norm / norm)
            v[:256] = hist
        descriptor_dict[k] = v
    return descriptor_dict