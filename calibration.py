# calibration.py (new file)

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import PchipInterpolator
from typing import Any, Dict, Tuple, List
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


class FusionCalibrator:
    """Learn how to fuse multiple calibrated match probabilities into P(match).

    Intended use: fuse Tier-2 probability with GV probability.
    Input features are typically logits: [logit(p_tier2), logit(p_gv)].
    """

    def __init__(self, max_iter: int = 1000):
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).flatten()
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return self.model.predict_proba(X)[:, 1]

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def intercept_(self):
        return self.model.intercept_


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
    rng = random.Random(seed)

    image_ids = sorted(str(k) for k in train_labels.keys())

    # Sample calibration query images
    cal_q_ids = rng.sample(image_ids, min(cal_size, len(image_ids)))
    
    query_ids, db_ids, pair_labels = [], [], []
    
    for q_id in cal_q_ids:
        q_identity = train_labels[q_id]
        
        # Positives: same identity (excluding self)
        positives = [i for i in image_ids if train_labels[i] == q_identity and i != q_id]
        
        # Negatives: different identity
        negatives = [i for i in image_ids if train_labels[i] != q_identity]
        negatives = rng.sample(negatives, min(max_negatives_per_query, len(negatives)))
        
        for p_id in positives:
            query_ids.append(q_id)
            db_ids.append(p_id)
            pair_labels.append(1)
        
        for n_id in negatives:
            query_ids.append(q_id)
            db_ids.append(n_id)
            pair_labels.append(0)
    
    return query_ids, db_ids, pair_labels

def build_calibration_pairs_stratified(
    train_labels,
    global_emb,
    cal_size=50,
    shortlist_size=300,
    n_negatives=100,
    seed: int = 42,
):
    """
    Build calibration pairs that match inference distribution:
    - Positives: all same-identity pairs
    - Negatives: sampled from top-K global shortlist (hard negatives)
    """
    query_ids, db_ids, pair_labels = [], [], []
    
    # Precompute global embeddings matrix
    all_ids = sorted(str(k) for k in global_emb.keys())
    emb_matrix = np.stack([global_emb[i] for i in all_ids])
    emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)
    
    rng = random.Random(seed)
    cal_q_ids = rng.sample(all_ids, min(cal_size, len(all_ids)))
    
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
        shortlist_idx = np.argsort(-global_sims, kind="mergesort")[:shortlist_size]
        
        # Filter to different identities
        hard_negatives = [
            all_ids[i] for i in shortlist_idx 
            if train_labels[all_ids[i]] != q_identity
        ]
        
        # Sample negatives
        selected_negs = rng.sample(
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


def train_count_calibrators_gt(
    train_labels: Dict[str, str],
    image_ids: List[str],
    *,
    global_emb: Dict[str, np.ndarray] | None = None,
    fisher_vectors: Dict[str, np.ndarray] | None = None,
    keypoints: Dict[str, np.ndarray] | None = None,
    descriptors: Dict[str, np.ndarray] | None = None,
    local_evidence: str = "inliers",
    local_mu: float = 0.5,
    target_pairs: int = 500,
    shortlist_size: int = 300,
    n_negatives_per_query: int = 50,
    calibration_method: str = "isotonic_pchip",
    use_lightglue: bool = False,
    method: str = "disk",
    gv_matcher: str | None = None,
    seed: int = 42,
) -> Tuple[Dict[str, "ScoreCalibrator"], Dict[str, Any]]:
    """Train per-signal score calibrators for counting (GT simulation).

    The goal is to map raw similarity signals to probabilities P(same|signal),
    enabling WildFusion-style late fusion for sampling proposals in HITL-NIS.

    Returns
    -------
    calibrators : dict
        Keys among {'global','fisher','local'}.
    info : dict
        Basic diagnostics about the calibration set.
    """

    local_evidence = str(local_evidence).lower().strip()
    if local_evidence not in {"inliers", "conf_matches"}:
        raise ValueError("local_evidence must be 'inliers' or 'conf_matches'")

    if target_pairs <= 0:
        raise ValueError("target_pairs must be > 0")

    if n_negatives_per_query <= 0:
        raise ValueError("n_negatives_per_query must be > 0")

    rng = random.Random(seed)

    # Sample enough calibration queries so that negatives alone roughly hit the budget.
    cal_size = max(1, int(np.ceil(float(target_pairs) / float(n_negatives_per_query))))

    # Build calibration pairs (hard negatives from global shortlist when possible).
    try:
        if global_emb:
            query_ids, db_ids, pair_labels = build_calibration_pairs_stratified(
                train_labels,
                global_emb,
                cal_size=cal_size,
                shortlist_size=shortlist_size,
                n_negatives=n_negatives_per_query,
                seed=seed,
            )
        else:
            raise ValueError("global_emb missing; cannot build stratified pairs")
    except Exception:
        # Fall back to random pairs (usually very imbalanced, but still usable on small sets).
        query_ids, db_ids, pair_labels = build_calibration_pairs(
            train_labels,
            cal_size=cal_size,
            max_negatives_per_query=n_negatives_per_query,
            seed=seed,
        )

    # Cap to the requested budget (keep order for reproducibility).
    if len(pair_labels) > target_pairs:
        query_ids = query_ids[:target_pairs]
        db_ids = db_ids[:target_pairs]
        pair_labels = pair_labels[:target_pairs]

    y = np.asarray(pair_labels, dtype=np.int64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    calibrators: Dict[str, ScoreCalibrator] = {}
    info: Dict[str, Any] = {
        "pairs": int(len(y)),
        "pos": n_pos,
        "neg": n_neg,
        "pos_rate": float(n_pos / max(1, len(y))),
        "cal_size_queries": int(cal_size),
        "n_negatives_per_query": int(n_negatives_per_query),
        "shortlist_size": int(shortlist_size),
        "local_evidence": local_evidence,
        "local_mu": float(local_mu),
        "calibration_method": calibration_method,
    }

    if len(set(y.tolist())) < 2:
        raise ValueError(
            "Calibration set needs both positive and negative pairs. "
            f"Got labels={sorted(set(y.tolist()))}; increase target_pairs or shortlist_size."
        )

    # Global calibration
    if global_emb:
        s_global = []
        for q_id, d_id in zip(query_ids, db_ids):
            q = global_emb.get(q_id)
            d = global_emb.get(d_id)
            if q is None or d is None:
                s_global.append(0.0)
                continue
            q = q / (np.linalg.norm(q) + 1e-9)
            d = d / (np.linalg.norm(d) + 1e-9)
            s_global.append(float(np.dot(q, d)))
        s_global = np.asarray(s_global, dtype=np.float32)
        cal = ScoreCalibrator(method=calibration_method)
        cal.fit(s_global, y)
        calibrators["global"] = cal

    # Fisher calibration
    if fisher_vectors:
        s_fisher = []
        for q_id, d_id in zip(query_ids, db_ids):
            q = fisher_vectors.get(q_id)
            d = fisher_vectors.get(d_id)
            if q is None or d is None:
                s_fisher.append(0.0)
                continue
            q = q / (np.linalg.norm(q) + 1e-9)
            d = d / (np.linalg.norm(d) + 1e-9)
            s_fisher.append(float(np.dot(q, d)))
        s_fisher = np.asarray(s_fisher, dtype=np.float32)
        cal = ScoreCalibrator(method=calibration_method)
        cal.fit(s_fisher, y)
        calibrators["fisher"] = cal

    # Local calibration (count-based signal → log1p)
    if keypoints is None or descriptors is None:
        info["local_skipped"] = True
    else:
        from geometric_verification import compute_local_evidence

        local_signals = []
        for q_id, d_id in zip(query_ids, db_ids):
            q_kp = keypoints.get(q_id)
            d_kp = keypoints.get(d_id)
            q_desc = descriptors.get(q_id)
            d_desc = descriptors.get(d_id)
            n_inliers, n_conf = compute_local_evidence(
                q_desc,
                q_kp,
                d_desc,
                d_kp,
                local_mu=local_mu,
                use_lightglue=use_lightglue,
                method=method,
                gv_matcher=gv_matcher,
            )
            count = int(n_inliers) if local_evidence == "inliers" else int(n_conf)
            local_signals.append(float(np.log1p(max(0, count))))
        local_signals = np.asarray(local_signals, dtype=np.float32)

        cal = ScoreCalibrator(method=calibration_method)
        cal.fit(local_signals, y)
        calibrators["local"] = cal

    return calibrators, info
