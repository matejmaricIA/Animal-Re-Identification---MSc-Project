import numpy as np
from typing import Dict
from calibration import ScoreCalibrator
from calibration import build_calibration_pairs, build_calibration_pairs_stratified, compute_calibration_scores
from geometric_verification import compute_geometric_similarity
from tqdm import tqdm

def train_calibrators_two_stage(
    train_labels, global_emb, fisher_vectors,
    keypoints, descriptors,
    cal_size=50, shortlist_size=300,
    use_lightglue=True, method='disk',
    calibration_method='isotonic_pchip',
):
    """Calibrate global and fisher scores for Tier-2 ranking."""
    
    # === Stage 1: Global calibration (all pairs) ===
    query_ids_global, db_ids_global, labels_global = build_calibration_pairs(
        train_labels, cal_size=cal_size, max_negatives_per_query=100
    )
    
    # Compute global scores
    s_global = []
    for q_id, d_id in zip(query_ids_global, db_ids_global):
        q_emb = global_emb[q_id] / (np.linalg.norm(global_emb[q_id]) + 1e-9)
        d_emb = global_emb[d_id] / (np.linalg.norm(global_emb[d_id]) + 1e-9)
        s_global.append(np.dot(q_emb, d_emb))
    
    cal_global = ScoreCalibrator(method=calibration_method)
    cal_global.fit(np.array(s_global), np.array(labels_global))

    print(
        f"[Calibration] Global pairs={len(s_global)} "
        f"raw(mean={np.mean(s_global):.4f}, std={np.std(s_global):.4f}, "
        f"min={np.min(s_global):.4f}, max={np.max(s_global):.4f})"
    )

    # === Stage 2: Fisher calibration (same pairs) ===
    s_fisher = []
    for q_id, d_id in zip(query_ids_global, db_ids_global):
        if q_id in fisher_vectors and d_id in fisher_vectors:
            q_fv = fisher_vectors[q_id] / (np.linalg.norm(fisher_vectors[q_id]) + 1e-9)
            d_fv = fisher_vectors[d_id] / (np.linalg.norm(fisher_vectors[d_id]) + 1e-9)
            s_fisher.append(np.dot(q_fv, d_fv))
        else:
            s_fisher.append(0.0)

    cal_fisher = ScoreCalibrator(method=calibration_method)
    cal_fisher.fit(np.array(s_fisher), np.array(labels_global))

    print(
        f"[Calibration] Fisher pairs={len(s_fisher)} "
        f"raw(mean={np.mean(s_fisher):.4f}, std={np.std(s_fisher):.4f}, "
        f"min={np.min(s_fisher):.4f}, max={np.max(s_fisher):.4f})"
    )

    return {'global': cal_global, 'fisher': cal_fisher}


def train_calibrators(
    train_labels: Dict[str, str],
    global_emb: Dict[str, np.ndarray],
    fisher_vectors: Dict[str, np.ndarray],
    keypoints: Dict[str, np.ndarray],
    descriptors: Dict[str, np.ndarray],
    cal_size: int = 50,
    calibration_method: str = 'isotonic_pchip',
    use_lightglue: bool = True,
    method: str = 'disk',
) -> Dict[str, ScoreCalibrator]:
    """Train calibrators for late fusion."""
    
    
    # Build calibration pairs
    query_ids, db_ids, pair_labels = build_calibration_pairs(
        train_labels, cal_size=cal_size
    )
    pair_labels = np.array(pair_labels)
    
    # Compute raw scores
    scores = compute_calibration_scores(
        query_ids, db_ids,
        global_emb=global_emb,
        fisher_vectors=fisher_vectors,
        keypoints=keypoints,
        descriptors=descriptors,
        use_lightglue=use_lightglue,
        method=method,
    )
    
    # Fit calibrators
    calibrators = {}
    
    cal_global = ScoreCalibrator(method=calibration_method)
    cal_global.fit(scores['s_global'], pair_labels)
    calibrators['global'] = cal_global
    
    cal_fisher = ScoreCalibrator(method=calibration_method)
    cal_fisher.fit(scores['s_fisher'], pair_labels)
    calibrators['fisher'] = cal_fisher
    
    cal_gv = ScoreCalibrator(method=calibration_method)
    cal_gv.fit(scores['s_gv'], pair_labels)
    calibrators['gv'] = cal_gv
    
    return calibrators
