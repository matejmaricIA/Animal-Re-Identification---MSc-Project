import cv2
import numpy as np
from scipy.spatial.distance import cdist

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

def match_features_by_descriptors(desc1, desc2, kp1, kp2, ratio_threshold=0.8):
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

def geometric_verification_ransac(kp1, kp2, inlier_threshold=10.0, min_matches=4):
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
            ransacReprojThreshold=inlier_threshold / 100.0  # Normalized threshold
        )
        
        if H is not None and mask is not None:
            n_inliers = np.sum(mask.ravel())
            return n_inliers, H
        else:
            return 0, None
            
    except cv2.error:
        return 0, None
    
def compute_geometric_similarity(query_desc, query_kp, db_desc, db_kp, 
                                fisher_distance, min_inliers=4):
    """Compute geometric similarity and combine with Fisher distance"""
    
    # Match features
    matches, matched_kp1, matched_kp2 = match_features_by_descriptors(
        query_desc, db_desc, query_kp, db_kp
    )
    
    if len(matches) < min_inliers:
        # Heavy penalty for insufficient matches
        return fisher_distance * 10.0, 0
    
    # Geometric verification
    n_inliers, homography = geometric_verification_ransac(
        matched_kp1, matched_kp2, inlier_threshold=10.0
    )
    
    if n_inliers < min_inliers:
        # Heavy penalty for poor geometric consistency
        return fisher_distance * 5.0, n_inliers
    
    # Apply reranking formula: d_C = (d_L)^n
    # Ensure fisher_distance is in [0,1] range
    fisher_distance_clamped = np.clip(fisher_distance, 0.01, 1.0)
    final_distance = np.power(fisher_distance_clamped, n_inliers)
    
    return final_distance, n_inliers