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
    NORMALIZED_THRESHOLD_DIVISOR, MAX_INLIERS_FOR_SCALING
)

from lightglue_singleton import get_lightglue

try:
    from lightglue import LightGlue
    import torch
    _LIGHTGLUE_AVAILABLE = True
except Exception:
    _LIGHTGLUE_AVAILABLE = False

def load_keypoints(keypoints_file):
    """Load keypoints from HDF5 file"""
    data = {}
    with h5py.File(keypoints_file, 'r') as f:
        for key in f.keys():
            data[key] = np.array(f[key])
    return data

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

def match_features_lightglue(desc1, desc2, kp1, kp2):
    """Match features using LightGlue if available."""
    if not _LIGHTGLUE_AVAILABLE:
        return [], np.empty((0, 2)), np.empty((0, 2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matcher = get_lightglue(features)
    with torch.inference_mode():
        data = {
            'image0': {
                'keypoints': torch.from_numpy(kp1).float().unsqueeze(0).to(device),
                'descriptors': torch.from_numpy(desc1).float().unsqueeze(0).to(device),
            },
            'image1': {
                'keypoints': torch.from_numpy(kp2).float().unsqueeze(0).to(device),
                'descriptors': torch.from_numpy(desc2).float().unsqueeze(0).to(device),
            },
        }
        out = matcher(data)
        if len(out['matches']) == 0:
            return [], np.empty((0, 2)), np.empty((0, 2))
        matches_arr = out['matches'][0].cpu().numpy()
        matched_kp1 = kp1[matches_arr[:, 0]] if len(matches_arr) else np.empty((0, 2))
        matched_kp2 = kp2[matches_arr[:, 1]] if len(matches_arr) else np.empty((0, 2))
        return matches_arr, matched_kp1, matched_kp2

def geometric_verification_ransac_debug(kp1, kp2, inlier_threshold=INLIER_THRESHOLD, 
                                       min_matches=MIN_MATCHES):
    """Debug version to see what's happening"""
    if len(kp1) < min_matches or len(kp2) < min_matches:
        return 0, None
    
    kp1_norm = normalize_coordinates(kp1)
    kp2_norm = normalize_coordinates(kp2)
    
    actual_threshold = inlier_threshold / NORMALIZED_THRESHOLD_DIVISOR
    print(f"DEBUG: Using RANSAC threshold: {actual_threshold:.6f}")
    print(f"DEBUG: Input keypoints: {len(kp1)} vs {len(kp2)}")
    
    try:
        H, mask = cv2.findHomography(
            kp1_norm.reshape(-1, 1, 2),
            kp2_norm.reshape(-1, 1, 2),
            cv2.RANSAC,
            ransacReprojThreshold=actual_threshold
        )
        
        if H is not None and mask is not None:
            n_inliers = np.sum(mask.ravel())
            inlier_percentage = (n_inliers / len(kp1)) * 100
            print(f"DEBUG: Found {n_inliers} inliers ({inlier_percentage:.1f}%)")
            
            # Sanity check - if >50% are inliers, something is wrong
            if inlier_percentage > 50:
                print(f"WARNING: Suspiciously high inlier percentage!")
            
            return n_inliers, H
        else:
            return 0, None
            
    except cv2.error as e:
        print(f"DEBUG: RANSAC failed: {e}")
        return 0, None


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
                                use_lightglue=False):
    """Compute geometric similarity and combine with Fisher distance"""
    
    # Match features
    #matches, matched_kp1, matched_kp2 = match_features_by_descriptors(
    #    query_desc, db_desc, query_kp, db_kp
    #)
    
    if use_lightglue and _LIGHTGLUE_AVAILABLE:
        
        matches, matched_kp1, matched_kp2 = match_features_lightglue(
            query_desc, db_desc, query_kp, db_kp
        )
    else:
        matches, matched_kp1, matched_kp2 = match_features_by_descriptors(
            query_desc, db_desc, query_kp, db_kp
        )
    
    if len(matches) < min_inliers:
        # Heavy penalty for insufficient matches
        return fisher_distance * INSUFFICIENT_MATCHES_PENALTY, 0
    
    
    # Geometric verification
    n_inliers, homography = geometric_verification_ransac(
        matched_kp1, matched_kp2
    )
    
    if n_inliers < min_inliers:
        # Heavy penalty for poor geometric consistency
        return fisher_distance * POOR_GEOMETRY_PENALTY, n_inliers
    
    
    # Normalize inliers to reasonable range (0-1)
    normalized_inliers = min(n_inliers / 50.0, 1.0)  # Assume max 50 reasonable inliers
    
    # Combine: 40% Fisher distance + 60% geometric penalty
    alpha = 0.4
    final_distance = alpha * fisher_distance + (1 - alpha) * (1 - normalized_inliers)
    
    # Apply reranking formula: d_C = (d_L)^n
    # Ensure fisher_distance is in [0,1] range
    #fisher_distance_clamped = np.clip(fisher_distance, FISHER_DISTANCE_MIN_CLAMP, FISHER_DISTANCE_MAX_CLAMP)
    #final_distance = np.power(fisher_distance_clamped, effective_inliers)
    
    return final_distance, n_inliers