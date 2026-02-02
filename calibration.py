# calibration.py (new file)

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import PchipInterpolator
from typing import Dict, Tuple, List
import pickle
import random

class ScoreCalibrator:
    """Calibrate raw similarity scores to P(same individual)."""
    
    def __init__(self, method: str = 'isotonic_pchip'):
        """
        method: 'isotonic', 'isotonic_pchip', or 'logistic'
        """
        self.method = method
        self.calibrator = None
        self.pchip = None  # For tie-breaking interpolation
        
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        scores: raw similarity values (higher = more similar)
        labels: 0/1 for different/same individual
        """
        scores = np.asarray(scores).flatten()
        labels = np.asarray(labels).flatten()
        
        if self.method == 'logistic':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(scores.reshape(-1, 1), labels)
        else:
            # Isotonic regression (monotone)
            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds='clip'
            )
            self.calibrator.fit(scores, labels)
            
            if self.method == 'isotonic_pchip':
                # PCHIP interpolation for tie-breaking (WildFusion's trick)
                unique_scores = np.unique(scores)
                unique_probs = self.calibrator.predict(unique_scores)
                # Add small noise to break ties while preserving monotonicity
                self.pchip = PchipInterpolator(unique_scores, unique_probs)
    
    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated P(same individual)."""
        scores = np.asarray(scores).flatten()
        
        if self.method == 'logistic':
            return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic_pchip' and self.pchip is not None:
            return np.clip(self.pchip(scores), 0.0, 1.0)
        else:
            return self.calibrator.predict(scores)
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'method': self.method, 'calibrator': self.calibrator, 'pchip': self.pchip}, f)
    
    @classmethod
    def load(cls, path: str) -> 'ScoreCalibrator':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        cal = cls(method=data['method'])
        cal.calibrator = data['calibrator']
        cal.pchip = data.get('pchip')
        return cal


def build_calibration_pairs(
    train_labels: Dict[str, str],
    cal_size: int = 50,
    max_negatives_per_query: int = 100,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Build calibration pairs from training labels.
    
    Returns:
        query_ids, db_ids, labels (1=same, 0=different)
    """
    import random
    random.seed(seed)
    
    image_ids = list(train_labels.keys())
    
    # Sample calibration query images
    cal_q_ids = random.sample(image_ids, min(cal_size, len(image_ids)))
    
    query_ids, db_ids, pair_labels = [], [], []
    
    for q_id in cal_q_ids:
        q_identity = train_labels[q_id]
        
        # Positives: same identity (excluding self)
        positives = [i for i in image_ids if train_labels[i] == q_identity and i != q_id]
        
        # Negatives: different identity
        negatives = [i for i in image_ids if train_labels[i] != q_identity]
        negatives = random.sample(negatives, min(max_negatives_per_query, len(negatives)))
        
        for p_id in positives:
            query_ids.append(q_id)
            db_ids.append(p_id)
            pair_labels.append(1)
        
        for n_id in negatives:
            query_ids.append(q_id)
            db_ids.append(n_id)
            pair_labels.append(0)
    
    return query_ids, db_ids, pair_labels

def build_calibration_pairs_stratified(train_labels, global_emb, cal_size=50, 
                                        shortlist_size=300, n_negatives=50):
    """
    Build calibration pairs that match inference distribution:
    - Positives: all same-identity pairs
    - Negatives: sampled from top-K global shortlist (hard negatives)
    """
    query_ids, db_ids, pair_labels = [], [], []
    
    # Precompute global embeddings matrix
    all_ids = list(global_emb.keys())
    emb_matrix = np.stack([global_emb[i] for i in all_ids])
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)
    
    cal_q_ids = random.sample(all_ids, min(cal_size, len(all_ids)))
    
    for q_id in cal_q_ids:
        q_identity = train_labels[q_id]
        q_emb = global_emb[q_id] / (np.linalg.norm(global_emb[q_id]) + 1e-9)
        
        # Positives (same identity)
        positives = [i for i in all_ids if train_labels[i] == q_identity and i != q_id]
        for p_id in positives:
            query_ids.append(q_id)
            db_ids.append(p_id)
            pair_labels.append(1)
        
        # Negatives: sample from global shortlist (hard negatives!)
        global_sims = np.dot(emb_matrix, q_emb)
        shortlist_idx = np.argsort(global_sims)[::-1][:shortlist_size]
        
        # Filter to different identities
        hard_negatives = [
            all_ids[i] for i in shortlist_idx 
            if train_labels[all_ids[i]] != q_identity
        ]
        
        # Sample negatives
        selected_negs = random.sample(
            hard_negatives, 
            min(n_negatives, len(hard_negatives))
        )
        
        for n_id in selected_negs:
            query_ids.append(q_id)
            db_ids.append(n_id)
            pair_labels.append(0)
    
    return query_ids, db_ids, pair_labels



def compute_calibration_scores(
    query_ids: List[str],
    db_ids: List[str],
    global_emb: Dict[str, np.ndarray] = None,
    fisher_vectors: Dict[str, np.ndarray] = None,
    keypoints: Dict[str, np.ndarray] = None,
    descriptors: Dict[str, np.ndarray] = None,
    use_lightglue: bool = True,
    method: str = 'disk',
) -> Dict[str, np.ndarray]:
    """
    Compute raw scores for calibration pairs.
    
    Returns dict with keys: 's_global', 's_fisher', 's_gv' (n_inliers)
    """
    from geometric_verification import compute_geometric_similarity
    from utils.distance_utils import fisher_distance
    from tqdm import tqdm
    
    scores = {
        's_global': [],
        's_fisher': [],
        's_gv': [],  # n_inliers
    }
    
    for q_id, d_id in tqdm(zip(query_ids, db_ids), total=len(query_ids), desc="Computing calibration scores"):
        # Global similarity (cosine)
        if global_emb is not None and q_id in global_emb and d_id in global_emb:
            q_emb = global_emb[q_id] / (np.linalg.norm(global_emb[q_id]) + 1e-9)
            d_emb = global_emb[d_id] / (np.linalg.norm(global_emb[d_id]) + 1e-9)
            s_global = np.dot(q_emb, d_emb)
        else:
            s_global = 0.0
        scores['s_global'].append(s_global)
        
        # Fisher similarity (cosine, or 1 - distance)
        if fisher_vectors is not None and q_id in fisher_vectors and d_id in fisher_vectors:
            q_fv = fisher_vectors[q_id] / (np.linalg.norm(fisher_vectors[q_id]) + 1e-9)
            d_fv = fisher_vectors[d_id] / (np.linalg.norm(fisher_vectors[d_id]) + 1e-9)
            s_fisher = np.dot(q_fv, d_fv)
        else:
            s_fisher = 0.0
        scores['s_fisher'].append(s_fisher)
        
        # GV: n_inliers
        if keypoints is not None and descriptors is not None:
            q_kp = keypoints.get(q_id)
            d_kp = keypoints.get(d_id)
            q_desc = descriptors.get(q_id)
            d_desc = descriptors.get(d_id)
            
            if all(x is not None for x in [q_kp, d_kp, q_desc, d_desc]):
                # Use dummy fisher distance for GV (we only need n_inliers)
                fd = 0.5
                _, n_inliers = compute_geometric_similarity(
                    q_desc, q_kp, d_desc, d_kp, fd,
                    use_lightglue=use_lightglue, method=method
                )
            else:
                n_inliers = 0
        else:
            n_inliers = 0
        scores['s_gv'].append(n_inliers)
    
    return {k: np.array(v) for k, v in scores.items()}