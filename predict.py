import numpy as np
from constants import *
import pandas as pd
from geometric_verification import compute_geometric_similarity
import time
from tqdm import tqdm
from utils import distance_utils
import csv
from datetime import datetime
import os
from visualization_suite import (
    io as vis_io,
    matching as vis_matching,
    geometric_verification as vis_gv,
    classification as vis_classification,
)
from typing import Dict, List
from calibration import ScoreCalibrator


DEBUG_LOG_PATH = "data/logs/logs_debug.csv"


def _append_debug_summary_csv(row: dict, path: str = DEBUG_LOG_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    fieldnames = list(row.keys())
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def classify_test_images_with_geometric_verification(
    test_fisher_vectors,
    train_fisher_vectors,
    test_keypoints,
    train_keypoints,
    test_descriptors,
    train_descriptors,
    train_labels,
    top_n=5,
    geometric_candidates=GEOMETRIC_CANDIDATES,
    use_lightglue=False,
    method="disk",
    alpha=ALPHA,
    min_inliers=MIN_INLIERS,
    inlier_threshold=INLIER_THRESHOLD,
    gv_matcher: str | None = None,
    image_paths: Dict[str, str] | None = None,
    visualize: bool = False,
    image_root: str | None = None,
    train_kp_h5: str | None = None,
    train_desc_h5: str | None = None,
    test_kp_h5: str | None = None,
    test_desc_h5: str | None = None,
    vis_output_dir: str | None = None,
):
    """Efficient geometric verification with two-stage filtering."""

    matcher = gv_matcher.lower() if isinstance(gv_matcher, str) else None
    if matcher == "loftr" and not image_paths:
        raise ValueError("LoFTR matcher requires image_paths mapping.")

    predictions = {}
    train_vectors = np.stack(list(train_fisher_vectors.values()))
    train_image_ids = list(train_fisher_vectors.keys())
    train_class_labels = np.array([train_labels[image_id] for image_id in train_image_ids])

    train_vectors_normalized = train_vectors / np.linalg.norm(train_vectors, axis=1, keepdims=True)
    
    total_test_images = len(test_fisher_vectors)
    print(f"\n=== Starting Geometric Verification ===")
    print(f"Total test images: {total_test_images}")
    print(f"Total training images: {len(train_fisher_vectors)}")
    print(f"Geometric candidates per query: {geometric_candidates}")
    print(f"=" * 50)
    
    # Track overall statistics  
    total_geometric_verifications = 0
    total_inliers_found = 0
    images_with_keypoints = 0
    images_without_keypoints = 0
    
    start_time = time.time()
    
    # Use tqdm for progress bar
    for i, (test_image_id, test_fisher_vector) in enumerate(tqdm(
        test_fisher_vectors.items(), 
        desc="Processing test images", 
        total=total_test_images,
        unit="img"
    )):
        
        print(f"\n[{i+1}/{total_test_images}] Processing: {test_image_id}")
        
        # Stage 1: Fisher Vector similarity (fast)
        stage1_start = time.time()
        test_fisher_vector = test_fisher_vector / np.linalg.norm(test_fisher_vector)
        #train_vectors_normalized = train_vectors / np.linalg.norm(train_vectors, axis=1, keepdims=True)
        similarities = np.dot(train_vectors_normalized, test_fisher_vector)
        
        # Get top candidates based on Fisher similarity
        top_indices = np.argsort(similarities)[::-1][:geometric_candidates]
        stage1_time = time.time() - stage1_start
        
        print(f"  ✓ Stage 1 (Fisher similarity): {stage1_time:.3f}s")
        print(f"    Best Fisher similarity: {similarities[top_indices[0]]:.4f}")
        
        if test_image_id in test_keypoints and test_image_id in test_descriptors:
            images_with_keypoints += 1
            
            # Stage 2: Geometric verification (slow, but only on top candidates)
            stage2_start = time.time()
            query_kp = test_keypoints.get(test_image_id, np.array([]))
            query_desc = test_descriptors.get(test_image_id, np.array([]))
            
            print(f"  → Stage 2 (Geometric verification): {len(query_kp)} query keypoints")
            
            final_scores = []
            successful_verifications = 0
            
            # Only verify geometric consistency for top candidates
            for j, idx in enumerate(top_indices):
                train_image_id = train_image_ids[idx]
                #fisher_distance = 1.0 - similarities[idx]
                fisher_distance = distance_utils.fisher_distance(test_fisher_vector, train_vectors[idx])
                combined_distance = fisher_distance
                
                train_kp = train_keypoints.get(train_image_id, np.array([]))
                train_desc = train_descriptors.get(train_image_id, np.array([]))
                
                # Show progress for geometric verification
                if j % 5 == 0 or j == len(top_indices) - 1:
                    print(f"    Verifying candidate {j+1}/{len(top_indices)}: {train_image_id}")
                
                final_distance, n_inliers = compute_geometric_similarity(
                    query_desc, query_kp, train_desc, train_kp, combined_distance,
                    use_lightglue=use_lightglue, method=method, alpha=alpha, min_inliers=min_inliers,
                    gv_matcher=matcher,
                    image0=image_paths.get(test_image_id) if image_paths else None,
                    image1=image_paths.get(train_image_id) if image_paths else None,
                )
                
                total_geometric_verifications += 1
                if n_inliers > 0:
                    successful_verifications += 1
                    total_inliers_found += n_inliers
                
                final_scores.append({
                    'distance': final_distance,
                    'fisher_distance': fisher_distance,
                    'n_inliers': n_inliers,
                    'class_label': train_class_labels[idx],
                    'train_image_id': train_image_id
                })
            
            # Sort by final distance and get top matches
            final_scores.sort(key=lambda x: x['distance'])
            top_n_matches = [(1.0 - score['distance'], score['class_label']) for score in final_scores[:top_n]]
            predicted_class = final_scores[0]['class_label']
            
            stage2_time = time.time() - stage2_start
            
            print(f"  ✓ Stage 2 completed: {stage2_time:.3f}s")
            print(f"    Successful verifications: {successful_verifications}/{geometric_candidates}")
            print(f"    Best match: {predicted_class} (inliers: {final_scores[0]['n_inliers']}, final_dist: {final_scores[0]['distance']:.4f})")
            
        else:
            images_without_keypoints += 1
            # Fallback to Fisher Vector only
            top_n_matches = [(similarities[i], train_class_labels[i]) for i in top_indices[:top_n]]
            predicted_class = top_n_matches[0][1]
            
            print(f"  ⚠ No keypoints available - using Fisher Vector only")
            print(f"    Predicted class: {predicted_class}")
        
        # Save the prediction and top-N matches
        predictions[test_image_id] = {
            'predicted_class': predicted_class,
            'top_n': top_n_matches
        }

        if visualize:
            if not all([image_root, train_kp_h5, train_desc_h5, test_kp_h5, test_desc_h5]):
                raise ValueError("Visualization requires image root and HDF5 paths")
            scores_subset = final_scores[:top_n] if 'final_scores' in locals() else []
            candidate_ids = [s['train_image_id'] for s in scores_subset]
            query_img = vis_io.load_image(f"{image_root}/{test_image_id}.jpg")
            candidate_imgs = [vis_io.load_image(f"{image_root}/{cid}.jpg") for cid in candidate_ids]
            q_kp = vis_io.load_keypoints_h5(test_kp_h5, [test_image_id]).get(test_image_id, np.empty((0,2)))
            q_desc = vis_io.load_descriptors_h5(test_desc_h5, [test_image_id]).get(test_image_id, np.empty((0,0)))
            train_kps = vis_io.load_keypoints_h5(train_kp_h5, candidate_ids)
            train_descs = vis_io.load_descriptors_h5(train_desc_h5, candidate_ids)
            match_info = []
            for cid, score in zip(candidate_ids, scores_subset):
                match_info.append({
                    'train_id': cid,
                    'score': 1.0 - score['distance'],
                    'n_inliers': score['n_inliers'],
                    'query_kp': q_kp,
                    'train_kp': train_kps.get(cid, np.empty((0,2))),
                    'query_desc': q_desc,
                    'train_desc': train_descs.get(cid, np.empty((0,0))),
                })
            if candidate_ids:
                vis_img, caption = vis_classification.visualize_top_matches(
                    query_img, candidate_imgs, match_info, top_k=top_n
                )
                if vis_output_dir:
                    os.makedirs(vis_output_dir, exist_ok=True)
                    vis_io.save_image(f"{vis_output_dir}/{test_image_id}.png", vis_img)
        
        # Show running statistics every 10 images
        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_time_per_image = elapsed_time / (i + 1)
            estimated_remaining = avg_time_per_image * (total_test_images - i - 1)
            
            print(f"\n--- Progress Update ---")
            print(f"Processed: {i+1}/{total_test_images} images")
            print(f"Average time per image: {avg_time_per_image:.2f}s")
            print(f"Estimated remaining time: {estimated_remaining/60:.1f} minutes")
            print(f"Images with keypoints: {images_with_keypoints}")
            print(f"Images without keypoints: {images_without_keypoints}")
            if total_geometric_verifications > 0:
                print(f"Average inliers per verification: {total_inliers_found/total_geometric_verifications:.1f}")
            print("-" * 22)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n=== Geometric Verification Complete ===")
    print(f"Total processing time: {total_time/60:.2f} minutes")
    print(f"Average time per image: {total_time/total_test_images:.2f}s")
    print(f"Images processed with keypoints: {images_with_keypoints}/{total_test_images}")
    print(f"Images processed without keypoints: {images_without_keypoints}/{total_test_images}")
    print(f"Total geometric verifications performed: {total_geometric_verifications}")
    if total_geometric_verifications > 0:
        print(f"Average inliers found: {total_inliers_found/total_geometric_verifications:.2f}")
        print(f"Success rate (verifications with inliers): {(total_geometric_verifications - total_inliers_found == 0)/total_geometric_verifications*100:.1f}%")
    print(f"=" * 40)
    
    return predictions



def classify_test_images(
    test_fisher_vectors,
    train_fisher_vectors,
    train_labels,
    top_n=5,
):

    predictions = {}

    # Stack train Fisher Vectors and labels for efficient comparison
    train_vectors = np.stack(list(train_fisher_vectors.values()))
    train_image_ids = list(train_fisher_vectors.keys())
    train_class_labels = np.array([train_labels[image_id] for image_id in train_image_ids])

    for test_image_id, test_fisher_vector in test_fisher_vectors.items():
        # Normalize Fisher Vectors for cosine similarity
        test_fisher_vector = test_fisher_vector / np.linalg.norm(test_fisher_vector)
        train_vectors_normalized = train_vectors / np.linalg.norm(train_vectors, axis=1, keepdims=True)

        # Compute cosine similarity
        similarities = np.dot(train_vectors_normalized, test_fisher_vector)

        # Sort similarities in descending order
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = sorted_indices[:top_n]

        # Get top-N class labels and similarities
        top_n_matches = [(similarities[i], train_class_labels[i]) for i in top_indices]

        # Predicted class is the class of the most similar train image
        predicted_class = top_n_matches[0][1]

        # Save the prediction and top-N matches
        predictions[test_image_id] = {
            "predicted_class": predicted_class,
            "top_n": top_n_matches,
        }

    return predictions

def _prepare_matrix(embeddings: Dict[str, np.ndarray]):
    """Return ids and L2-normalized matrix for fast cosine similarity."""
    if not embeddings:
        return [], None
    ids = list(embeddings.keys())
    matrix = np.stack([embeddings[i] for i in ids]).astype(np.float32)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return ids, matrix

def retrieve_candidates_union(
    test_id: str,
    test_global_emb: Dict[str, np.ndarray],
    train_global_ids: List[str],
    train_global_matrix: np.ndarray | None,
    test_fisher: Dict[str, np.ndarray],
    train_fisher_ids: List[str],
    train_fisher_matrix: np.ndarray | None,
    union_candidates: int,
):
    """Tier 1: Union of top-N global and top-N Fisher candidates."""
    global_ids = []
    fisher_ids = []
    global_sims = None
    fisher_sims = None

    if train_global_matrix is not None and test_id in test_global_emb:
        test_emb = test_global_emb[test_id]
        test_emb = test_emb / (np.linalg.norm(test_emb) + 1e-9)
        global_sims = np.dot(train_global_matrix, test_emb)
        top_idx = np.argsort(global_sims)[::-1][:union_candidates]
        global_ids = [train_global_ids[i] for i in top_idx]

    if train_fisher_matrix is not None and test_id in test_fisher:
        test_fv = test_fisher[test_id]
        test_fv = test_fv / (np.linalg.norm(test_fv) + 1e-9)
        fisher_sims = np.dot(train_fisher_matrix, test_fv)
        top_idx = np.argsort(fisher_sims)[::-1][:union_candidates]
        fisher_ids = [train_fisher_ids[i] for i in top_idx]

    union_ids = list(dict.fromkeys(global_ids + fisher_ids))
    return union_ids, global_sims, fisher_sims

def rank_by_local_score(
    union_ids: List[str],
    global_sims: np.ndarray | None,
    fisher_sims: np.ndarray | None,
    train_global_index: Dict[str, int],
    train_fisher_index: Dict[str, int],
    calibrators: Dict[str, 'ScoreCalibrator'],
    local_rank_candidates: int,
):
    """Tier 2: Rank union candidates by calibrated global + fisher."""
    candidates = []
    for train_id in union_ids:
        s_global = 0.0
        s_fisher = 0.0
        if global_sims is not None:
            idx = train_global_index.get(train_id)
            if idx is not None:
                s_global = float(global_sims[idx])
        if fisher_sims is not None:
            idx = train_fisher_index.get(train_id)
            if idx is not None:
                s_fisher = float(fisher_sims[idx])

        scores = []
        if calibrators is not None:
            if 'global' in calibrators:
                scores.append(float(calibrators['global'].predict_proba([s_global])[0]))
            if 'fisher' in calibrators:
                scores.append(float(calibrators['fisher'].predict_proba([s_fisher])[0]))
        if scores:
            tier2_score = float(np.mean(scores))
        else:
            tier2_score = float(np.mean([s_global, s_fisher]))

        candidates.append({
            "train_id": train_id,
            "local_score": s_fisher,
            "tier2_score": tier2_score,
        })
    candidates.sort(key=lambda x: x["tier2_score"], reverse=True)
    if local_rank_candidates <= 0:
        return candidates
    return candidates[:local_rank_candidates]

def verify_candidates(
    test_id: str,
    candidates: List[Dict[str, float]],
    test_keypoints: Dict[str, np.ndarray],
    train_keypoints: Dict[str, np.ndarray],
    test_descriptors: Dict[str, np.ndarray],
    train_descriptors: Dict[str, np.ndarray],
    test_fisher: Dict[str, np.ndarray],
    train_fisher: Dict[str, np.ndarray],
    train_labels: Dict[str, str],
    use_lightglue: bool,
    method: str,
    gv_matcher: str | None = None,
    image_paths: Dict[str, str] | None = None,
):
    """Tier 3: Geometric verification on ranked candidates."""
    matcher = gv_matcher.lower() if isinstance(gv_matcher, str) else None
    if matcher == "loftr" and not image_paths:
        raise ValueError("LoFTR matcher requires image_paths mapping.")
    q_kp = test_keypoints.get(test_id)
    q_desc = test_descriptors.get(test_id)
    have_query = q_kp is not None and q_desc is not None
    results = []
    for cand in candidates:
        train_id = cand["train_id"]
        n_inliers = 0
        if have_query:
            d_kp = train_keypoints.get(train_id)
            d_desc = train_descriptors.get(train_id)
            if d_kp is not None and d_desc is not None:
                if test_id in test_fisher and train_id in train_fisher:
                    fisher_distance = distance_utils.fisher_distance(
                        test_fisher[test_id], train_fisher[train_id]
                    )
                else:
                    fisher_distance = 1.0
                _, n_inliers = compute_geometric_similarity(
                    q_desc, q_kp, d_desc, d_kp, fisher_distance,
                    use_lightglue=use_lightglue, method=method,
                    gv_matcher=matcher,
                    image0=image_paths.get(test_id) if image_paths else None,
                    image1=image_paths.get(train_id) if image_paths else None,
                )
        results.append({
            "train_id": train_id,
            "label": train_labels[train_id],
            "local_score": cand["local_score"],
            "tier2_score": cand.get("tier2_score", cand["local_score"]),
            "n_inliers": n_inliers,
        })
    return results

def classify_test_images_late_fusion(
    test_global_emb: Dict[str, np.ndarray],
    train_global_emb: Dict[str, np.ndarray],
    test_fisher: Dict[str, np.ndarray],
    train_fisher: Dict[str, np.ndarray],
    test_keypoints: Dict[str, np.ndarray],
    train_keypoints: Dict[str, np.ndarray],
    test_descriptors: Dict[str, np.ndarray],
    train_descriptors: Dict[str, np.ndarray],
    train_labels: Dict[str, str],
    calibrators: Dict[str, 'ScoreCalibrator'],  # {'global': cal, 'fisher': cal, 'gv': cal}
    top_n: int = 5,
    shortlist_size: int = UNION_CANDIDATES,
    use_lightglue: bool = True,
    method: str = 'disk',
    gv_matcher: str | None = None,
    image_paths: Dict[str, str] | None = None,
    fusion_signals: List[str] = ['global', 'gv'],  # or ['global', 'fisher', 'gv']
    test_labels: Dict[str, str] | None = None,
    debug: bool = False,
    dataset_name: str | None = None,
    calibration_method: str | None = None,
    calib_size: int | None = None,
):
    """
    3-Tier Funnel classification (Retrieve → Local Rank → Geometric Verify).
    """
    from tqdm import tqdm
    
    predictions = {}
    print("Tier-2 ranking uses calibrated global+fisher (mean of probabilities).")

    debug_enabled = bool(debug and test_labels is not None)
    if debug and not debug_enabled:
        print("[DEBUG] --debug enabled but test_labels not provided; skipping GT diagnostics.")

    debug_total = 0
    debug_union_hit = 0
    debug_local_hit = 0
    debug_verified_hit = 0
    debug_mispreds = 0
    debug_tier2_mispreds = 0
    debug_gv_changed = 0
    debug_gv_helped = 0
    debug_gv_hurt = 0
    debug_inliers_override_losses = 0
    debug_missing_gt = 0

    train_global_ids, train_global_matrix = _prepare_matrix(train_global_emb)
    train_fisher_ids, train_fisher_matrix = _prepare_matrix(train_fisher)
    train_global_index = {img_id: i for i, img_id in enumerate(train_global_ids)}
    train_fisher_index = {img_id: i for i, img_id in enumerate(train_fisher_ids)}

    test_ids = list(test_global_emb.keys()) if test_global_emb else []
    if test_fisher:
        seen = set(test_ids)
        for tid in test_fisher.keys():
            if tid not in seen:
                test_ids.append(tid)
                seen.add(tid)

    for test_id in tqdm(test_ids, desc="Late fusion classification"):
        union_ids, global_sims, fisher_sims = retrieve_candidates_union(
            test_id,
            test_global_emb,
            train_global_ids,
            train_global_matrix,
            test_fisher,
            train_fisher_ids,
            train_fisher_matrix,
            shortlist_size,
        )

        local_ranked = rank_by_local_score(
            union_ids,
            global_sims,
            fisher_sims,
            train_global_index,
            train_fisher_index,
            calibrators,
            LOCAL_RANK_CANDIDATES,
        )

        print(
            f"Funnel: Union {len(union_ids)} → Local {len(local_ranked)} → "
            f"Geometric Verification {len(local_ranked)}"
        )

        verified = verify_candidates(
            test_id,
            local_ranked,
            test_keypoints,
            train_keypoints,
            test_descriptors,
            train_descriptors,
            test_fisher,
            train_fisher,
            train_labels,
            use_lightglue,
            method,
            gv_matcher,
            image_paths,
        )

        verified.sort(
            key=lambda x: (
                (
                    int(x.get("n_inliers", 0))
                    if int(x.get("n_inliers", 0)) >= MIN_INLIERS
                    else 0
                ),
                float(x.get("tier2_score", 0.0)),
            ),
            reverse=True,
        )

        if verified:
            top_n_matches = [(c["n_inliers"], c["label"]) for c in verified[:top_n]]
            predicted_class = verified[0]["label"]
        else:
            top_n_matches = []
            predicted_class = None

        predictions[test_id] = {
            "predicted_class": predicted_class,
            "top_n": top_n_matches,
        }

        if debug_enabled:
            gt_label = test_labels.get(test_id) if test_labels is not None else None
            if gt_label is None:
                debug_missing_gt += 1
                continue

            debug_total += 1

            tier2_pred_label = None
            tier2_pred_tier2 = None
            if local_ranked:
                tier2_train_id = local_ranked[0].get("train_id")
                tier2_pred_label = train_labels.get(tier2_train_id)
                tier2_pred_tier2 = float(local_ranked[0].get("tier2_score", 0.0))

            union_hit = any(train_labels.get(tid) == gt_label for tid in union_ids)
            local_hit = any(train_labels.get(cand["train_id"]) == gt_label for cand in local_ranked)
            verified_hit = any(cand.get("label") == gt_label for cand in verified)

            if union_hit:
                debug_union_hit += 1
            if local_hit:
                debug_local_hit += 1
            if verified_hit:
                debug_verified_hit += 1

            pred_label = predicted_class
            tier2_pred = tier2_pred_label

            mispred = pred_label != gt_label
            if mispred:
                debug_mispreds += 1

            tier2_mispred = tier2_pred != gt_label
            if tier2_mispred:
                debug_tier2_mispreds += 1

            if tier2_pred != pred_label:
                debug_gv_changed += 1

            if tier2_mispred and not mispred:
                debug_gv_helped += 1
            elif not tier2_mispred and mispred:
                debug_gv_hurt += 1

            pred_inliers = None
            pred_tier2 = None
            gt_best_inliers = None
            gt_best_tier2 = None
            pred_effective_inliers = None
            gt_best_effective_inliers = None
            inliers_override = False

            if verified:
                pred_candidate = verified[0]
                pred_inliers = int(pred_candidate.get("n_inliers", 0))
                pred_tier2 = float(pred_candidate.get("tier2_score", 0.0))
                pred_effective_inliers = (
                    pred_inliers if pred_inliers >= MIN_INLIERS else 0
                )

                gt_candidates = [c for c in verified if c.get("label") == gt_label]
                if gt_candidates:
                    gt_best = max(gt_candidates, key=lambda x: float(x.get("tier2_score", 0.0)))
                    gt_best_inliers = int(gt_best.get("n_inliers", 0))
                    gt_best_tier2 = float(gt_best.get("tier2_score", 0.0))
                    gt_best_effective_inliers = (
                        gt_best_inliers if gt_best_inliers >= MIN_INLIERS else 0
                    )
                    eps = 1e-12
                    if (
                        mispred
                        and gt_best_tier2 > (pred_tier2 + eps)
                        and pred_effective_inliers > gt_best_effective_inliers
                    ):
                        inliers_override = True

            if inliers_override:
                debug_inliers_override_losses += 1

            pred_inliers_s = "NA" if pred_inliers is None else str(pred_inliers)
            pred_tier2_s = "NA" if pred_tier2 is None else f"{pred_tier2:.4f}"
            gt_best_inliers_s = "NA" if gt_best_inliers is None else str(gt_best_inliers)
            gt_best_tier2_s = "NA" if gt_best_tier2 is None else f"{gt_best_tier2:.4f}"
            pred_eff_inliers_s = (
                "NA" if pred_effective_inliers is None else str(pred_effective_inliers)
            )
            gt_best_eff_inliers_s = (
                "NA"
                if gt_best_effective_inliers is None
                else str(gt_best_effective_inliers)
            )
            tier2_pred_s = "NA" if tier2_pred is None else str(tier2_pred)
            tier2_pred_tier2_s = (
                "NA" if tier2_pred_tier2 is None else f"{tier2_pred_tier2:.4f}"
            )

            print(
                "[DEBUG] "
                f"{test_id} "
                f"gt={gt_label} pred={pred_label} tier2_pred={tier2_pred_s} tier2_pred_tier2={tier2_pred_tier2_s} "
                f"union_hit={union_hit} local_hit={local_hit} verified_hit={verified_hit} "
                f"inliers_override={inliers_override} "
                f"pred_inliers={pred_inliers_s} pred_eff_inliers={pred_eff_inliers_s} pred_tier2={pred_tier2_s} "
                f"gt_best_inliers={gt_best_inliers_s} gt_best_eff_inliers={gt_best_eff_inliers_s} gt_best_tier2={gt_best_tier2_s}"
            )
    
    if debug_enabled:
        if debug_total:
            union_rate = debug_union_hit / debug_total
            local_rate = debug_local_hit / debug_total
            verified_rate = debug_verified_hit / debug_total
            print(
                "[DEBUG] Summary: "
                f"queries={debug_total} "
                f"union_hit={union_rate:.4f} "
                f"local_hit={local_rate:.4f} "
                f"verified_hit={verified_rate:.4f} "
                f"mispreds={debug_mispreds} "
                f"tier2_mispreds={debug_tier2_mispreds} "
                f"gv_changed={debug_gv_changed} "
                f"gv_helped={debug_gv_helped} "
                f"gv_hurt={debug_gv_hurt} "
                f"inliers_override_losses={debug_inliers_override_losses}"
            )
            _append_debug_summary_csv(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "dataset": dataset_name or "",
                    "fusion_mode": "late",
                    "fusion_signals": " ".join(fusion_signals or []),
                    "calibration_method": calibration_method or "",
                    "calib_size": "" if calib_size is None else int(calib_size),
                    "queries": int(debug_total),
                    "union_hit_rate": float(union_rate),
                    "local_hit_rate": float(local_rate),
                    "verified_hit_rate": float(verified_rate),
                    "mispreds": int(debug_mispreds),
                    "tier2_mispreds": int(debug_tier2_mispreds),
                    "gv_changed": int(debug_gv_changed),
                    "gv_helped": int(debug_gv_helped),
                    "gv_hurt": int(debug_gv_hurt),
                    "inliers_override_losses": int(debug_inliers_override_losses),
                    "missing_gt": int(debug_missing_gt),
                    "shortlist_size": int(shortlist_size),
                    "local_rank_candidates": int(LOCAL_RANK_CANDIDATES),
                    "top_n": int(top_n),
                    "min_inliers_threshold": int(MIN_INLIERS),
                }
            )
        if debug_missing_gt:
            print(f"[DEBUG] Missing GT labels for {debug_missing_gt} queries; skipped GT diagnostics for those.")

    return predictions

# This is deprecated and is not used anymore.
def predict(pred_fisher_vectors, db_fisher_vectors, dataset_name, threshold = 0.4):
    df = pd.read_csv(DATAFRAME_PATH.format(dataset_name))
    class_labels = dict(zip(df['image_id'], df['identity']))
    db_vectors = np.stack(list(db_fisher_vectors.values()))
    train_image_ids = list(db_fisher_vectors.keys())
    db_class_labels = np.array([class_labels[image_id] for image_id in train_image_ids])

    for test_image_id, test_fisher_vector in pred_fisher_vectors.items():
        test_fisher_vector = test_fisher_vector / np.linalg.norm(test_fisher_vector)
        train_vectors_normalized = db_vectors / np.linalg.norm(db_vectors, axis=1, keepdims=True)

        similarities = np.dot(train_vectors_normalized, test_fisher_vector)
        sorted_indices = np.argsort(similarities)[::-1]
        #print(max(similarities))
        top_indices = sorted_indices[:1]
        top_match = [(similarities[i], db_class_labels[i]) for i in top_indices][0]

        print(f"Top match is {top_match[1]} with similarity score of: {top_match[0]}")

        if top_match[1] >= threshold:
            print(f"Determine class is: {db_class_labels[top_indices]}")
        else:
            print("Unknown class.")
        
