import cv2
import numpy as np
from functools import lru_cache
from scipy.spatial.distance import cdist
import h5py
from constants import (
    RATIO_THRESHOLD, 
    INLIER_THRESHOLD, 
    MIN_MATCHES, 
    MIN_INLIERS,
    NORMALIZED_THRESHOLD_DIVISOR,
    ALPHA,
)
from utils.distance_utils import fisher_distance

from lightglue_singleton import get_lightglue
import sys

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

try:
    from lightglue import LightGlue
    _LIGHTGLUE_AVAILABLE = _TORCH_AVAILABLE
except Exception:
    _LIGHTGLUE_AVAILABLE = False

try:
    from kornia.feature import LoFTR as KorniaLoFTR
    _LOFTR_AVAILABLE = _TORCH_AVAILABLE
except Exception:
    _LOFTR_AVAILABLE = False
    
# Fallbacks    
_FD_MIN, _FD_90, _I90 = 0.0, 1.0, 50 

def set_dataset_calibration(fd_min, fd_90, I90):
    """Called once per dataset by calibration.py"""
    global _FD_MIN, _FD_90, _I90
    _FD_MIN, _FD_90, _I90 = fd_min, fd_90, I90

def _scale_fd(fd):
    return np.clip((fd - _FD_MIN) / (_FD_90 - _FD_MIN + 1e-9), 0.0, 1.0)

def _norm_inliers(n):
    return min(n / _I90, 1.0)

def load_keypoints(keypoints_file):
    """Load keypoints from HDF5 file"""
    data = {}
    with h5py.File(keypoints_file, 'r') as f:
        for key in f.keys():
            data[key] = np.array(f[key])
    return data

def _ensure_xy(kp: np.ndarray) -> np.ndarray:
    """Return kp as (N,2) float32 regardless of what we loaded from disk."""
    kp = np.asarray(kp)

    if kp.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # 1) drop any singleton dimensions (e.g. ``(1, N, 2)`` -> ``(N, 2)``)
    kp = kp.squeeze()

    # 2) handle 1‑D arrays representing a single keypoint
    if kp.ndim == 1:
        if kp.shape[0] >= 2:
            kp = kp.reshape(-1, 2)
        else:
            return np.empty((0, 2), dtype=np.float32)

    # 3) flatten arrays with extra trailing dimensions (e.g. ``(N, 2, 2)``)
    elif kp.ndim > 2:
        kp = kp.reshape(-1, kp.shape[-1])

    # 4) keep only ``x`` and ``y`` columns if more are present
    if kp.shape[1] > 2:
        kp = kp[:, :2]
    elif kp.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float32)

    return kp.astype(np.float32)
    

def normalize_coordinates(coords):
    """Normalize coordinates to zero mean and max distance 1"""
    if len(coords) == 0:
        return coords
    
    # Center around zero
    coords_centered = coords - np.mean(coords, axis=0)
    
    # Scale to max distance 1
    distances = np.linalg.norm(coords_centered, axis=1)
    max_distance = np.max(distances)
    
    if max_distance > 0:
        coords_normalized = coords_centered / max_distance
    else:
        coords_normalized = coords_centered
    
    return coords_normalized

# Old, deprecated way to match features, safe to delete.
def match_features_by_descriptors(desc1, desc2, kp1, kp2, ratio_threshold=RATIO_THRESHOLD):
    """Match features based on descriptor similarity"""
    if len(desc1) == 0 or len(desc2) == 0:
        return [], [], []
    
    # Compute cosine distances
    desc1_norm = desc1 / np.linalg.norm(desc1, axis=1, keepdims=True)
    desc2_norm = desc2 / np.linalg.norm(desc2, axis=1, keepdims=True)
    
    distances = cdist(desc1_norm, desc2_norm, metric='cosine')
    
    matches = []
    matched_kp1 = []
    matched_kp2 = []
    
    for i in range(len(desc1)):
        # Find two nearest neighbors
        sorted_indices = np.argsort(distances[i])
        
        if len(sorted_indices) >= 2:
            best_dist = distances[i, sorted_indices[0]]
            second_best_dist = distances[i, sorted_indices[1]]
            
            # Lowe's ratio test
            if best_dist < ratio_threshold * second_best_dist:
                j = sorted_indices[0]
                matches.append((i, j, best_dist))
                matched_kp1.append(kp1[i])
                matched_kp2.append(kp2[j])
    
    return matches, np.array(matched_kp1), np.array(matched_kp2)

def match_features_lightglue(desc1, desc2, kp1, kp2, method='disk'):
    """Match features using LightGlue if available."""
    if not _LIGHTGLUE_AVAILABLE:
        return [], np.empty((0, 2)), np.empty((0, 2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = get_lightglue(features=method)

    kp1 = _ensure_xy(kp1)
    kp2 = _ensure_xy(kp2)

    # Guard against empty keypoints/descriptors
    if kp1.shape[0] == 0 or kp2.shape[0] == 0 or desc1.size == 0 or desc2.size == 0:
        return [], np.empty((0, 2)), np.empty((0, 2))

    kpts0 = torch.from_numpy(kp1).float().unsqueeze(0).to(device)
    desc0 = torch.from_numpy(desc1).float().unsqueeze(0).to(device)
    kpts1 = torch.from_numpy(kp2).float().unsqueeze(0).to(device)
    desc1 = torch.from_numpy(desc2).float().unsqueeze(0).to(device)

    # Does this head expect scale & orientation?
    #need_so = (method == 'doghardnet')
    
    with torch.inference_mode():
        #data = {
        #    'image0': {
        #        'keypoints': torch.from_numpy(kp1).float().unsqueeze(0).to(device),
        #        'descriptors': torch.from_numpy(desc1).float().unsqueeze(0).to(device),
        #    },
        #    'image1': {
        #        'keypoints': torch.from_numpy(kp2).float().unsqueeze(0).to(device),
        #        'descriptors': torch.from_numpy(desc2).float().unsqueeze(0).to(device),
        #    },
        #}
        data = {
        'image0': {'keypoints': kpts0, 'descriptors': desc0},
        'image1': {'keypoints': kpts1, 'descriptors': desc1},
    }
        #if need_so:                                   # fill with safe defaults
        #    for tag, kpts in (('0', kpts0), ('1', kpts1)):
        #        data[f'image{tag}']['scales'] = torch.ones (1, len(kpts), device=device)
        #        data[f'image{tag}']['oris']   = torch.zeros(1, len(kpts), device=device)
        #print(data)
        #sys.exit()

        out = matcher(data)
        if len(out['matches']) == 0:
            return [], np.empty((0, 2)), np.empty((0, 2))
        matches_arr = out['matches'][0].cpu().numpy()
        matched_kp1 = kp1[matches_arr[:, 0]] if len(matches_arr) else np.empty((0, 2))
        matched_kp2 = kp2[matches_arr[:, 1]] if len(matches_arr) else np.empty((0, 2))
        return matches_arr, matched_kp1, matched_kp2


@lru_cache(maxsize=None)
def get_loftr(pretrained: str = "outdoor"):
    if not _LOFTR_AVAILABLE:
        raise RuntimeError("LoFTR is not available. Install kornia with LoFTR support.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = KorniaLoFTR(pretrained=pretrained)
    except TypeError:
        try:
            model = KorniaLoFTR(pretrained=True)
        except TypeError:
            model = KorniaLoFTR()
    return model.to(device).eval()


def _load_grayscale_image(image):
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image)
        return img
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Unsupported image input for LoFTR")


def _to_loftr_tensor(image: np.ndarray):
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    tensor = torch.from_numpy(img).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.mean(dim=0, keepdim=True).unsqueeze(0)
    else:
        raise ValueError("Unsupported image shape for LoFTR")
    return tensor


def match_features_loftr(image0, image1, pretrained: str = "outdoor"):
    if not _LOFTR_AVAILABLE:
        return [], np.empty((0, 2)), np.empty((0, 2))

    img0 = _load_grayscale_image(image0)
    img1 = _load_grayscale_image(image1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = get_loftr(pretrained=pretrained)

    input_dict = {
        "image0": _to_loftr_tensor(img0).to(device),
        "image1": _to_loftr_tensor(img1).to(device),
    }

    with torch.inference_mode():
        output = matcher(input_dict)

    keypoints0 = output.get("keypoints0", torch.empty((0, 2), device=device)).detach().cpu().numpy().astype(np.float32)
    keypoints1 = output.get("keypoints1", torch.empty((0, 2), device=device)).detach().cpu().numpy().astype(np.float32)

    if keypoints0.size == 0 or keypoints1.size == 0:
        return [], np.empty((0, 2)), np.empty((0, 2))

    num_matches = min(keypoints0.shape[0], keypoints1.shape[0])
    keypoints0 = keypoints0[:num_matches]
    keypoints1 = keypoints1[:num_matches]
    matches_arr = np.stack([np.arange(num_matches), np.arange(num_matches)], axis=1).astype(np.int32)

    return matches_arr, keypoints0, keypoints1



def geometric_verification_ransac(kp1, kp2, inlier_threshold=INLIER_THRESHOLD, min_matches=MIN_MATCHES, gv_method = 'RANSAC'):
    """Apply RANSAC for geometric verification"""
    if len(kp1) < min_matches or len(kp2) < min_matches:
        return 0, None
    
    # Normalize coordinates
    kp1_norm = normalize_coordinates(kp1)
    kp2_norm = normalize_coordinates(kp2)
    
    try:
        # Choose method based on configuration
        if gv_method == "MAGSAC":
            H, mask = cv2.findHomography(
                kp1_norm.reshape(-1, 1, 2),
                kp2_norm.reshape(-1, 1, 2),
                cv2.USAC_MAGSAC,
                ransacReprojThreshold=inlier_threshold / NORMALIZED_THRESHOLD_DIVISOR,
                maxIters=10000,          
                confidence=0.999           
            )
        else:  # Default to RANSAC
            H, mask = cv2.findHomography(
                kp1_norm.reshape(-1, 1, 2),
                kp2_norm.reshape(-1, 1, 2),
                cv2.RANSAC,
                ransacReprojThreshold=inlier_threshold / NORMALIZED_THRESHOLD_DIVISOR
            )
        
        
        if H is not None and mask is not None:
            n_inliers = np.sum(mask.ravel())
            return n_inliers, H
        else:
            return 0, None
            
    except cv2.error:
        return 0, None
    
def compute_geometric_similarity(query_desc, query_kp, db_desc, db_kp,
                                feature_distance, min_inliers=MIN_INLIERS,
                                use_lightglue: bool = False, method: str = 'disk', alpha: float = ALPHA,
                                gv_matcher: str | None = None,
                                image0=None, image1=None,
                                loftr_pretrained: str = "outdoor"):
    """Compute geometric similarity and combine it with a base feature distance.

    The ``feature_distance`` argument can be any distance measure where lower
    values indicate higher similarity.  It was originally designed for Fisher
    vector distances but generalises to other descriptor combinations as well.
    """
    
    # Match features
    #matches, matched_kp1, matched_kp2 = match_features_by_descriptors(
    #    query_desc, db_desc, query_kp, db_kp
    #)
    
    matcher = gv_matcher.lower() if isinstance(gv_matcher, str) else None
    if matcher is None:
        matcher = "lightglue" if use_lightglue else "ratio"

    if matcher == "loftr":
        if not _LOFTR_AVAILABLE:
            raise RuntimeError("LoFTR matcher requested but not available.")
        if image0 is None or image1 is None:
            raise ValueError("LoFTR matcher requires image0 and image1 inputs.")
        matches, matched_kp1, matched_kp2 = match_features_loftr(
            image0, image1, pretrained=loftr_pretrained
        )
    elif matcher == "lightglue" and _LIGHTGLUE_AVAILABLE:
        matches, matched_kp1, matched_kp2 = match_features_lightglue(
            query_desc, db_desc, query_kp, db_kp, method=method
        )
    else:
        matches, matched_kp1, matched_kp2 = match_features_by_descriptors(
            query_desc, db_desc, query_kp, db_kp
        )   
        
    fisher_scaled = _scale_fd(feature_distance)
    
    if len(matches) < min_inliers:
        # Heavy penalty for insufficient matches
        #return fisher_scaled * INSUFFICIENT_MATCHES_PENALTY, 0
        return 1.0, 0 

    # Geometric verification
    n_inliers, homography = geometric_verification_ransac(
        matched_kp1, matched_kp2
    )
    
    if n_inliers < min_inliers:
        # Heavy penalty for poor geometric consistency
        #return feature_distance * POOR_GEOMETRY_PENALTY,
        return min(1.0, fisher_scaled + 0.5), n_inliers
    geo_score = 1 - _norm_inliers(n_inliers)   # 0 → perfect, 1 → bad

    # Combine base distance with geometric penalty
    final_distance = alpha * fisher_scaled + (1 - alpha) * geo_score

    # Optional reranking formula: d_C = (d_L)^n (disabled by default)
    #feature_distance_clamped = np.clip(feature_distance, FISHER_DISTANCE_MIN_CLAMP, FISHER_DISTANCE_MAX_CLAMP)
    #final_distance = np.power(feature_distance_clamped, effective_inliers)

    return final_distance, n_inliers


def compute_local_evidence(
    query_desc,
    query_kp,
    db_desc,
    db_kp,
    *,
    local_mu: float = 0.5,
    use_lightglue: bool = False,
    method: str = "disk",
    gv_matcher: str | None = None,
    image0=None,
    image1=None,
    loftr_pretrained: str = "outdoor",
):

    if query_desc is None or db_desc is None or query_kp is None or db_kp is None:
        return 0, 0

    matcher = gv_matcher.lower() if isinstance(gv_matcher, str) else None
    if matcher is None:
        matcher = "lightglue" if use_lightglue else "ratio"

    n_conf_matches = 0

    if matcher == "loftr":
        if not _LOFTR_AVAILABLE:
            return 0, 0
        if image0 is None or image1 is None:
            raise ValueError("LoFTR local evidence requires image0 and image1 inputs.")
        matches_arr, matched_kp1, matched_kp2 = match_features_loftr(
            image0, image1, pretrained=loftr_pretrained
        )
        # Confident match counts are not supported for LoFTR in this codebase.
        n_conf_matches = 0
    elif matcher == "lightglue" and _LIGHTGLUE_AVAILABLE:
        matches_arr, matched_kp1, matched_kp2 = match_features_lightglue(
            query_desc, db_desc, query_kp, db_kp, method=method
        )
        if matches_arr is not None and len(matches_arr) > 0:
            qd = np.asarray(query_desc, dtype=np.float32)
            dd = np.asarray(db_desc, dtype=np.float32)
            qd = qd / (np.linalg.norm(qd, axis=1, keepdims=True) + 1e-9)
            dd = dd / (np.linalg.norm(dd, axis=1, keepdims=True) + 1e-9)
            idx0 = matches_arr[:, 0].astype(np.int64)
            idx1 = matches_arr[:, 1].astype(np.int64)
            sims = np.sum(qd[idx0] * dd[idx1], axis=1)
            n_conf_matches = int(np.sum(sims >= float(local_mu)))
    else:
        matches, matched_kp1, matched_kp2 = match_features_by_descriptors(
            query_desc, db_desc, query_kp, db_kp
        )
        if matches:
            # matches are (i, j, cosine_distance); similarity := 1 - distance.
            sims = np.array([1.0 - float(m[2]) for m in matches], dtype=np.float32)
            n_conf_matches = int(np.sum(sims >= float(local_mu)))

    n_inliers, _ = geometric_verification_ransac(matched_kp1, matched_kp2)
    return int(n_inliers), int(n_conf_matches)
