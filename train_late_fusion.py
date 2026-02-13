import numpy as np
from typing import Dict, List
from calibration import ScoreCalibrator
from calibration import (
    #build_calibration_pairs,
    #build_calibration_pairs_stratified,
    build_calibration_pairs_ids,
    compute_calibration_scores,
)
from geometric_verification import compute_geometric_similarity
from tqdm import tqdm

def train_calibrators_two_stage(
    train_labels,
    global_emb,
    fisher_vectors,
    keypoints,
    descriptors,
    calib_ids: int = 10,
    use_lightglue: bool = True,
    method: str = "disk",
    calibration_method: str = "isotonic_pchip",
    fusion_signals: List[str] | None = None,
    gv_matcher: str | None = None,
) -> Dict[str, ScoreCalibrator]:
    """Train calibrators for late fusion (Tier-2 + optional GV fusion).

    - Tier-2 signals (global/fisher) are calibrated as probabilities and then
      combined by averaging probabilities.
    - GV is calibrated as a probability from a GV evidence signal
      (currently log1p(n_inliers)) and fused downstream.

    Calibration pairs follow a WildFusion-style scheme: sample ``calib_ids``
    identities with >=2 images, pick 2 images per identity (query/database),
    and use the full cross product (pairs ~= ``calib_ids²``).
    """

    if fusion_signals is None:
        fusion_signals = ["global", "fisher"]

    # Build WildFusion-style calibration pairs (~calib_ids^2).
    query_ids, db_ids, pair_labels = build_calibration_pairs_ids(
        train_labels, calib_ids=int(calib_ids), seed=42
    )
    if not pair_labels or len(set(pair_labels)) < 2:
        print(
            "[Calibration] Skipping calibration: need >=2 identities with >=2 images "
            "to generate both positive and negative pairs."
        )
        return {}
    pair_labels = np.asarray(pair_labels)

    calibrators: Dict[str, ScoreCalibrator] = {}
    
    # === Stage 1: Global calibration ===
    if "global" in fusion_signals and global_emb:
        s_global = []
        for q_id, d_id in zip(query_ids, db_ids):
            if q_id in global_emb and d_id in global_emb:
                q_emb = global_emb[q_id] / (np.linalg.norm(global_emb[q_id]) + 1e-9)
                d_emb = global_emb[d_id] / (np.linalg.norm(global_emb[d_id]) + 1e-9)
                s_global.append(float(np.dot(q_emb, d_emb)))
            else:
                s_global.append(0.0)
        s_global = np.asarray(s_global)

        cal_global = ScoreCalibrator(method=calibration_method)
        cal_global.fit(s_global, pair_labels)
        calibrators["global"] = cal_global

        print(
            f"[Calibration] Global pairs={len(s_global)} "
            f"raw(mean={np.mean(s_global):.4f}, std={np.std(s_global):.4f}, "
            f"min={np.min(s_global):.4f}, max={np.max(s_global):.4f})"
        )

    # === Stage 2: Fisher calibration (same pairs) ===
    if "fisher" in fusion_signals and fisher_vectors:
        s_fisher = []
        for q_id, d_id in zip(query_ids, db_ids):
            if q_id in fisher_vectors and d_id in fisher_vectors:
                q_fv = fisher_vectors[q_id] / (np.linalg.norm(fisher_vectors[q_id]) + 1e-9)
                d_fv = fisher_vectors[d_id] / (np.linalg.norm(fisher_vectors[d_id]) + 1e-9)
                s_fisher.append(float(np.dot(q_fv, d_fv)))
            else:
                s_fisher.append(0.0)
        s_fisher = np.asarray(s_fisher)

        cal_fisher = ScoreCalibrator(method=calibration_method)
        cal_fisher.fit(s_fisher, pair_labels)
        calibrators["fisher"] = cal_fisher

        print(
            f"[Calibration] Fisher pairs={len(s_fisher)} "
            f"raw(mean={np.mean(s_fisher):.4f}, std={np.std(s_fisher):.4f}, "
            f"min={np.min(s_fisher):.4f}, max={np.max(s_fisher):.4f})"
        )

    # === Stage 3: GV calibration (subset, expensive) ===
    if "gv" in fusion_signals:
        have_kp_desc = bool(keypoints) and bool(descriptors)
        if not have_kp_desc:
            print("[Calibration] GV skipped: keypoints/descriptors not available.")
        else:
            pos_idx: List[int] = []
            neg_idx: List[int] = []
            for i, (q_id, d_id, y) in enumerate(zip(query_ids, db_ids, pair_labels)):
                if (
                    q_id not in keypoints
                    or d_id not in keypoints
                    or q_id not in descriptors
                    or d_id not in descriptors
                ):
                    continue
                if int(y) == 1:
                    pos_idx.append(i)
                else:
                    neg_idx.append(i)

            total_available = len(pos_idx) + len(neg_idx)
            max_gv_pairs = min(total_available, max(200, 20 * int(calib_ids)))
            if total_available == 0 or max_gv_pairs <= 0:
                print("[Calibration] GV skipped: no valid pairs with keypoints/descriptors.")
            else:
                rng = np.random.default_rng(42)
                n_pos = min(len(pos_idx), max_gv_pairs // 2)
                n_neg = min(len(neg_idx), max_gv_pairs // 2)

                selected: List[int] = []
                if n_pos:
                    selected.extend(rng.choice(pos_idx, size=n_pos, replace=False).tolist())
                if n_neg:
                    selected.extend(rng.choice(neg_idx, size=n_neg, replace=False).tolist())

                remaining = max_gv_pairs - len(selected)
                if remaining > 0:
                    selected_set = set(selected)
                    remaining_pool = [i for i in (pos_idx + neg_idx) if i not in selected_set]
                    if remaining_pool:
                        extra_n = min(remaining, len(remaining_pool))
                        selected.extend(rng.choice(remaining_pool, size=extra_n, replace=False).tolist())

                gv_scores = []
                gv_labels = []
                gv_pairs: List[tuple[str, str]] = []
                for i in tqdm(selected, desc="Calibrating GV", leave=False):
                    q_id = query_ids[i]
                    d_id = db_ids[i]
                    y = int(pair_labels[i])
                    q_kp = keypoints.get(q_id)
                    d_kp = keypoints.get(d_id)
                    q_desc = descriptors.get(q_id)
                    d_desc = descriptors.get(d_id)
                    if q_kp is None or d_kp is None or q_desc is None or d_desc is None:
                        continue
                    _, n_inliers = compute_geometric_similarity(
                        q_desc,
                        q_kp,
                        d_desc,
                        d_kp,
                        0.5,
                        use_lightglue=use_lightglue,
                        method=method,
                        gv_matcher=gv_matcher,
                    )
                    gv_scores.append(float(np.log1p(max(0, int(n_inliers)))))
                    gv_labels.append(y)
                    gv_pairs.append((q_id, d_id))

                if len(set(gv_labels)) < 2:
                    print(
                        "[Calibration] GV skipped: need both positive and negative pairs "
                        f"(got labels={sorted(set(gv_labels))})."
                    )
                else:
                    gv_scores = np.asarray(gv_scores)
                    gv_labels_arr = np.asarray(gv_labels)

                    cal_gv = ScoreCalibrator(method=calibration_method)
                    cal_gv.fit(gv_scores, gv_labels_arr)
                    calibrators["gv"] = cal_gv

                    print(
                        f"[Calibration] GV pairs={len(gv_scores)} "
                        f"signal=log1p(inliers) "
                        f"raw(mean={np.mean(gv_scores):.4f}, std={np.std(gv_scores):.4f}, "
                        f"min={np.min(gv_scores):.4f}, max={np.max(gv_scores):.4f})"
                    )
    return calibrators


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
    #query_ids, db_ids, pair_labels = build_calibration_pairs(
    #    train_labels, cal_size=cal_size
    #)
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
