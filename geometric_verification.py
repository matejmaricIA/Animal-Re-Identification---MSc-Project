import cv2
import numpy as np
from scipy.spatial.distance import cdist
from constants import (
    RATIO_THRESHOLD, 
    INLIER_THRESHOLD, 
    MIN_MATCHES, 
    MIN_INLIERS,
    INSUFFICIENT_MATCHES_PENALTY,
    POOR_GEOMETRY_PENALTY,
    FISHER_DISTANCE_MIN_CLAMP,
    FISHER_DISTANCE_MAX_CLAMP,
    NORMALIZED_THRESHOLD_DIVISOR, MAX_INLIERS_FOR_SCALING,
    ALPHA
)
from utils.distance_utils import fisher_distance

from lightglue_singleton import get_lightglue
import sys

try:
    from lightglue import LightGlue
    import torch
    _LIGHTGLUE_AVAILABLE = True
except Exception:
    _LIGHTGLUE_AVAILABLE = False
    
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

    # 1) drop any singleton dimensions that snuck in (e.g. (1,N,2) → (N,2))
    kp = kp.squeeze()

    # 2) if we still have >2 dims (e.g. (N,2,2)) flatten everything but the last
    if kp.ndim == 3:
        kp = kp.reshape(-1, kp.shape[-1])

    # 3) Some extractors store extra info (scale, ori, score...) – keep x & y only
    if kp.shape[1] > 2:
        kp = kp[:, :2]

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



def geometric_verification_ransac(kp1, kp2, inlier_threshold=INLIER_THRESHOLD, min_matches=MIN_MATCHES):
    """Apply RANSAC for geometric verification"""
    if len(kp1) < min_matches or len(kp2) < min_matches:
        return 0, None
    
    # Normalize coordinates
    kp1_norm = normalize_coordinates(kp1)
    kp2_norm = normalize_coordinates(kp2)
    
    try:
        # Find homography using RANSAC
        H, mask = cv2.findHomography(
            kp1_norm.reshape(-1, 1, 2),
            kp2_norm.reshape(-1, 1, 2),
            cv2.RANSAC,
            ransacReprojThreshold=inlier_threshold / NORMALIZED_THRESHOLD_DIVISOR
        )
        #H, mask = cv2.findHomography(
        #    kp1_norm.reshape(-1,1,2),
        #    kp2_norm.reshape(-1,1,2),
        #    cv2.USAC_MAGSAC,
        #    ransacReprojThreshold=inlier_threshold / NORMALIZED_THRESHOLD_DIVISOR,
        #    maxIters=10000,          
        #    confidence=0.999           
        #)
        
        
        if H is not None and mask is not None:
            n_inliers = np.sum(mask.ravel())
            return n_inliers, H
        else:
            return 0, None
            
    except cv2.error:
        return 0, None
    
def compute_geometric_similarity(query_desc, query_kp, db_desc, db_kp,
                                fisher_distance, min_inliers=MIN_INLIERS,
                                use_lightglue=False, method = 'disk', alpha = ALPHA):
    """Compute geometric similarity and combine with Fisher distance"""
    
    # Match features
    #matches, matched_kp1, matched_kp2 = match_features_by_descriptors(
    #    query_desc, db_desc, query_kp, db_kp
    #)
    
    if use_lightglue and _LIGHTGLUE_AVAILABLE:
        
        matches, matched_kp1, matched_kp2 = match_features_lightglue(
            query_desc, db_desc, query_kp, db_kp, method = method
        )
    else:
        matches, matched_kp1, matched_kp2 = match_features_by_descriptors(
            query_desc, db_desc, query_kp, db_kp
        )   
        
    fisher_scaled = _scale_fd(fisher_distance)
    
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
        #return fisher_distance * POOR_GEOMETRY_PENALTY, 
        return min(1.0, fisher_scaled + 0.5), n_inliers
    
    
    geo_score = 1 - _norm_inliers(n_inliers)   # 0 → perfect, 1 → bad
    
    # Normalize inliers to reasonable range (0-1)
    #normalized_inliers = min(n_inliers / 50.0, 1.0)  # Assume max 50 reasonable inliers
    
    # Combine: 40% Fisher distance + 60% geometric penalty
    #alpha = 0.35
    #final_distance = alpha * fisher_distance + (1 - alpha) * (1 - normalized_inliers)
    final_distance = alpha * fisher_scaled + (1 - alpha) * geo_score
    
    # Apply reranking formula: d_C = (d_L)^n
    # Ensure fisher_distance is in [0,1] range
    #fisher_distance_clamped = np.clip(fisher_distance, FISHER_DISTANCE_MIN_CLAMP, FISHER_DISTANCE_MAX_CLAMP)
    #final_distance = np.power(fisher_distance_clamped, effective_inliers)
    
    return final_distance, n_inliers