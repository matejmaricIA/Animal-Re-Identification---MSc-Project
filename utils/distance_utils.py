# distance_utils.py  – single source of truth for Fisher distance
import numpy as np

def fisher_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    1 − cosine similarity of two Fisher vectors.
    Both  vectors are L2-normalised inside this function so the result is
    guaranteed in [0, 2].  Down-stream code usually scales this to [0, 1].
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return 1.0 - float(np.dot(v1, v2))
