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
    fusion_signals: List[str] | None = None,
):
    """Tier 2: Rank union candidates by calibrated Tier-2 signals."""
    active_signals = set(fusion_signals or ["global", "fisher"])
    candidates = []
    for train_id in union_ids:
        scores = []
        raw_scores = []
        s_global = 0.0
        if "global" in active_signals and global_sims is not None:
            idx = train_global_index.get(train_id)
            if idx is not None:
                s_global = float(global_sims[idx])
                raw_scores.append(s_global)
                if calibrators is not None and "global" in calibrators:
                    scores.append(float(calibrators["global"].predict_proba([s_global])[0]))

        s_fisher = 0.0
        if "fisher" in active_signals and fisher_sims is not None:
            idx = train_fisher_index.get(train_id)
            if idx is not None:
                s_fisher = float(fisher_sims[idx])
                raw_scores.append(s_fisher)
                if calibrators is not None and "fisher" in calibrators:
                    scores.append(float(calibrators["fisher"].predict_proba([s_fisher])[0]))

        if scores:
            tier2_score = float(np.mean(scores))
        else:
            if raw_scores:
                # Fallback: map raw similarities to a probability-like score in [0, 1].
                # This keeps Tier-3 reranking stable when calibration is unavailable.
                tier2_score = float(np.mean(np.clip(raw_scores, 0.0, 1.0)))
            else:
                tier2_score = 0.0

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


def classify_single_image_late_fusion(
    test_id: str,
    test_global_emb: Dict[str, np.ndarray],
    train_global_emb: Dict[str, np.ndarray],
    test_fisher: Dict[str, np.ndarray],
    train_fisher: Dict[str, np.ndarray],
    test_keypoints: Dict[str, np.ndarray],
    train_keypoints: Dict[str, np.ndarray],
    test_descriptors: Dict[str, np.ndarray],
    train_descriptors: Dict[str, np.ndarray],
    train_labels: Dict[str, str],
    calibrators: Dict[str, "ScoreCalibrator"],
    shortlist_size: int = UNION_CANDIDATES,
    local_rank_candidates: int = LOCAL_RANK_CANDIDATES,
    use_lightglue: bool = True,
    method: str = "disk",
    gv_matcher: str | None = None,
    image_paths: Dict[str, str] | None = None,
    fusion_signals: List[str] | None = None,
    top_n: int = 5,
) -> Dict[str, object]:
    """Run the same late-fusion funnel as batch classification for one query image."""
    train_global_ids, train_global_matrix = _prepare_matrix(train_global_emb)
    train_fisher_ids, train_fisher_matrix = _prepare_matrix(train_fisher)
    train_global_index = {img_id: i for i, img_id in enumerate(train_global_ids)}
    train_fisher_index = {img_id: i for i, img_id in enumerate(train_fisher_ids)}

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

    global_ranked = []
    if global_sims is not None:
        top_idx = np.argsort(global_sims)[::-1][:shortlist_size]
        for idx in top_idx:
            train_id = train_global_ids[int(idx)]
            global_ranked.append(
                {
                    "train_id": train_id,
                    "score": float(global_sims[int(idx)]),
                    "label": train_labels.get(train_id),
                }
            )

    fisher_ranked = []
    if fisher_sims is not None:
        top_idx = np.argsort(fisher_sims)[::-1][:shortlist_size]
        for idx in top_idx:
            train_id = train_fisher_ids[int(idx)]
            fisher_ranked.append(
                {
                    "train_id": train_id,
                    "score": float(fisher_sims[int(idx)]),
                    "label": train_labels.get(train_id),
                }
            )

    tier2_ranked = rank_by_local_score(
        union_ids,
        global_sims,
        fisher_sims,
        train_global_index,
        train_fisher_index,
        calibrators,
        local_rank_candidates,
        fusion_signals=fusion_signals,
    )
    for cand in tier2_ranked:
        cand["label"] = train_labels.get(cand["train_id"])

    use_gv = bool(fusion_signals and "gv" in set(fusion_signals))
    cal_gv = calibrators.get("gv") if calibrators is not None else None
    if use_gv:
        tier3_ranked = verify_candidates(
            test_id,
            tier2_ranked,
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
        for cand in tier3_ranked:
            tier2_score = cand.get("tier2_score", 0.0)
            if not (0.0 <= float(tier2_score) <= 1.0):
                cand["_power_score"] = float("-inf")
                cand["_fused_logit"] = float("-inf")
                continue
            p_tier2 = float(tier2_score)
            d_l = float(np.clip(1.0 - p_tier2, 1e-12, 1.0))
            n_inliers = int(cand.get("n_inliers", 0))
            cand["_power_score"] = float(-max(0, n_inliers) * np.log(d_l))
            cand["_fused_logit"] = cand["_power_score"]
            if cal_gv is not None:
                gv_signal = float(np.log1p(max(0, n_inliers)))
                try:
                    cand["_p_gv"] = float(cal_gv.predict_proba([gv_signal])[0])
                except Exception:
                    pass

        tier3_ranked.sort(
            key=lambda x: (
                float(x.get("_power_score", float("-inf"))),
                float(x.get("tier2_score", 0.0)),
                int(x.get("n_inliers", 0)),
            ),
            reverse=True,
        )
    else:
        tier3_ranked = [
            {
                "train_id": cand["train_id"],
                "label": train_labels.get(cand["train_id"]),
                "local_score": float(cand.get("local_score", 0.0)),
                "tier2_score": float(cand.get("tier2_score", cand.get("local_score", 0.0))),
                "n_inliers": 0,
                "_fused_logit": float(cand.get("tier2_score", cand.get("local_score", 0.0))),
            }
            for cand in tier2_ranked
        ]

    if tier3_ranked:
        predicted_class = tier3_ranked[0].get("label")
        top_n_matches = [
            (int(c.get("n_inliers", 0)), str(c.get("label"))) for c in tier3_ranked[:top_n]
        ]
    else:
        predicted_class = None
        top_n_matches = []

    return {
        "test_id": test_id,
        "predicted_class": predicted_class,
        "top_n": top_n_matches,
        "global_ranked": global_ranked,
        "fisher_ranked": fisher_ranked,
        "union_ids": union_ids,
        "tier2_ranked": tier2_ranked,
        "tier3_ranked": tier3_ranked,
    }

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
    calib_ids: int | None = None,
):
    """
    3-Tier Funnel classification (Retrieve → Local Rank → Geometric Verify).
    """
    from tqdm import tqdm
    
    predictions = {}

    tier2_signals = [s for s in (fusion_signals or []) if s in {"global", "fisher"}]
    if not tier2_signals:
        tier2_signals = ["global", "fisher"]
    print(
        "Tier-2 ranking uses calibrated "
        + "+".join(tier2_signals)
        + " (mean of probabilities)."
    )
    if fusion_signals and "gv" in fusion_signals:
        print("Tier-3 reranking uses GV power-rerank on Tier-2 distance (d=(1-p_tier2)^n).")

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
            fusion_signals=fusion_signals,
        )

        use_gv = bool(fusion_signals and "gv" in fusion_signals)
        print(
            f"Funnel: Union {len(union_ids)} → Local {len(local_ranked)} → "
            f"Geometric Verification {len(local_ranked) if use_gv else 0}"
        )
        cal_gv = calibrators.get("gv") if calibrators is not None else None

        if use_gv:
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
            # Paper-style reranking: d_C = (d_L)^n.
            # Here we treat Tier-2 as a match probability, so d_L := 1 - p_tier2 ∈ [0,1].
            # We rank by -log(d_C) = -n*log(d_L), which avoids underflow for large n.
            for cand in verified:
                tier2_score = cand.get("tier2_score", 0.0)
                if not (0.0 <= float(tier2_score) <= 1.0):
                    cand["_power_score"] = float("-inf")
                    continue
                p_tier2 = float(tier2_score)
                d_l = float(np.clip(1.0 - p_tier2, 1e-12, 1.0))
                n_inliers = int(cand.get("n_inliers", 0))
                cand["_power_score"] = float(-max(0, n_inliers) * np.log(d_l))
                cand["_fused_logit"] = cand["_power_score"]

                if cal_gv is not None:
                    gv_signal = float(np.log1p(max(0, n_inliers)))
                    try:
                        cand["_p_gv"] = float(cal_gv.predict_proba([gv_signal])[0])
                    except Exception:
                        pass

            verified.sort(
                key=lambda x: (
                    float(x.get("_power_score", float("-inf"))),
                    float(x.get("tier2_score", 0.0)),
                    int(x.get("n_inliers", 0)),
                ),
                reverse=True,
            )
        else:
            verified = [
                {
                    "train_id": cand["train_id"],
                    "label": train_labels[cand["train_id"]],
                    "local_score": cand["local_score"],
                    "tier2_score": cand.get("tier2_score", cand["local_score"]),
                    "n_inliers": 0,
                }
                for cand in local_ranked
            ]
            # No GV stage: preserve Tier-2 ranking while keeping debug output keys consistent.
            for cand in verified:
                cand["_fused_logit"] = float(cand.get("tier2_score", 0.0))

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
            pred_p_gv = None
            gt_best_p_gv = None
            pred_fused = None
            gt_best_fused = None
            inliers_override = False

            if verified:
                pred_candidate = verified[0]
                pred_inliers = int(pred_candidate.get("n_inliers", 0))
                pred_tier2 = float(pred_candidate.get("tier2_score", 0.0))
                pred_fused = float(pred_candidate.get("_fused_logit", 0.0))
                if use_gv and cal_gv is not None:
                    pred_p_gv = float(pred_candidate.get("_p_gv", 0.5))

                gt_candidates = [c for c in verified if c.get("label") == gt_label]
                if gt_candidates:
                    gt_best = max(gt_candidates, key=lambda x: float(x.get("tier2_score", 0.0)))
                    gt_best_inliers = int(gt_best.get("n_inliers", 0))
                    gt_best_tier2 = float(gt_best.get("tier2_score", 0.0))
                    gt_best_fused = float(gt_best.get("_fused_logit", 0.0))
                    if use_gv and cal_gv is not None:
                        gt_best_p_gv = float(gt_best.get("_p_gv", 0.5))
                    eps = 1e-12
                    if (
                        use_gv
                        and cal_gv is not None
                        and mispred
                        and gt_best_tier2 > (pred_tier2 + eps)
                        and pred_fused > (gt_best_fused + eps)
                    ):
                        inliers_override = True

            if inliers_override:
                debug_inliers_override_losses += 1

            pred_inliers_s = "NA" if pred_inliers is None else str(pred_inliers)
            pred_tier2_s = "NA" if pred_tier2 is None else f"{pred_tier2:.4f}"
            gt_best_inliers_s = "NA" if gt_best_inliers is None else str(gt_best_inliers)
            gt_best_tier2_s = "NA" if gt_best_tier2 is None else f"{gt_best_tier2:.4f}"
            pred_p_gv_s = "NA" if pred_p_gv is None else f"{pred_p_gv:.4f}"
            gt_best_p_gv_s = "NA" if gt_best_p_gv is None else f"{gt_best_p_gv:.4f}"
            pred_fused_s = "NA" if pred_fused is None else f"{pred_fused:.4f}"
            gt_best_fused_s = "NA" if gt_best_fused is None else f"{gt_best_fused:.4f}"
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
                f"pred_inliers={pred_inliers_s} pred_tier2={pred_tier2_s} pred_p_gv={pred_p_gv_s} pred_fused={pred_fused_s} "
                f"gt_best_inliers={gt_best_inliers_s} gt_best_tier2={gt_best_tier2_s} gt_best_p_gv={gt_best_p_gv_s} gt_best_fused={gt_best_fused_s}"
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
                    "calib_ids": "" if calib_ids is None else int(calib_ids),
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
        
