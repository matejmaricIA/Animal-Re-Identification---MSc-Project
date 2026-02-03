import numpy as np
from typing import Dict, Tuple, Sequence, Optional, Union, Any

from geometric_verification import compute_geometric_similarity
from utils.distance_utils import fisher_distance
from constants import MIN_INLIERS


def cosine_similarity_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
    arr = np.vstack(vectors)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm[norm == 0] = 1
    arr = arr / norm
    sim = arr @ arr.T
    np.fill_diagonal(sim, 0.0)
    sim[sim < 0] = 0.0
    return sim


def nested_importance_sampling(
    fisher_vectors: Dict[str, np.ndarray],
    labels: Dict[str, int],
    keypoints: Optional[Dict[str, np.ndarray]] = None,
    descriptors: Optional[Dict[str, np.ndarray]] = None,
    *,
    use_geometric: bool = True,
    use_lightglue: bool = False,
    method: str = "disk",
    gv_matcher: Optional[str] = None,
    image_paths: Optional[Dict[str, str]] = None,
    gv_threshold: float = 0.75,
    n_vertices: int = 100,
    n_neighbors: int = 10,
    label_error_rate: float = 0.0,
    return_stats: bool = False,
    automated_mode: bool = False,
    seed: Optional[int] = None,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
    """Nested Importance Sampling with *gated* human‑(label) feedback and
    **built‑in bookkeeping** for debugging / ablation studies.

    Parameters
    ----------
    fisher_vectors : Dict[str, np.ndarray]
        Image‑id → Fisher vector.
    labels : Dict[str, int]
        Image‑id → ground‑truth individual ID (ignored if automated_mode=True).
    keypoints / descriptors : Dicts or ``None``
        Required for geometric verification.
    use_geometric : bool, default=True
        Whether to apply the GV gate.
    use_lightglue, method, gv_threshold : see original implementation.
    n_vertices, n_neighbors : int
        Outer / inner sample sizes.
    label_error_rate : float, default=0.0
        Probability of flipping the (otherwise perfect) label—simulates human
        slips.
    return_stats : bool, default=False
        If *True*, the function also returns a dict with:
            * ``total_pairs``   – all neighbour pairs considered
            * ``gv_attempts``   – pairs that had enough data to run GV
            * ``gv_passes``     – pairs that passed the GV gate
            * ``label_queries`` – how many times we actually looked at a label
            * ``matches``       – how many of those queries were positive (same ID)
    automated_mode : bool, default=False
        If True, uses only geometric verification without human labels.
        Faster but potentially less accurate than human-in-the-loop mode.
    seed : int or ``None``, default=None
        Random seed for reproducible vertex and neighbour sampling.

    Returns
    -------
    mean_est, stderr_est : floats
        Population size estimate and its standard error.
    stats : dict  (only if ``return_stats=True``)
        See *return_stats* description.
    """    
    matcher = gv_matcher.lower() if isinstance(gv_matcher, str) else None
    if matcher == "loftr" and not image_paths:
        raise ValueError("LoFTR matcher requires image_paths mapping.")


    # 1. Build similarity graph & vertex proposal distribution
    image_ids = list(fisher_vectors.keys())
    vectors = [fisher_vectors[i] for i in image_ids]
    sim = cosine_similarity_matrix(vectors)

    degrees = sim.sum(axis=1)
    Q = 1.0 / (1.0 + degrees)
    Q /= Q.sum()

    rng = np.random.default_rng(seed)
    population_estimates = []

    total_pairs: int = 0
    gv_attempts: int = 0
    gv_passes: int = 0
    label_queries: int = 0
    positive_matches: int = 0

    # 2.   Outer vertex loop                                             #
    outer_vertices = rng.choice(len(image_ids), size=min(n_vertices, len(image_ids)), replace=False, p=Q)
    for u_idx in outer_vertices:
        q = sim[u_idx].copy()
        if q.sum() == 0.0:
            q = np.ones_like(q)
        q[q == 0.0] = 1e-9
        q /= q.sum()

        neighbors = rng.choice(len(image_ids), size=min(n_neighbors, len(image_ids)), replace=False, p=q)

        fb_list = []

        # 3. Inner neighbour loop with GV gate  
        for v_idx in neighbors:
            total_pairs += 1

            u_id = image_ids[u_idx]
            v_id = image_ids[v_idx]
            match = 0  # default feedback

            if use_geometric and keypoints is not None and descriptors is not None:
                desc_u = descriptors.get(u_id)
                desc_v = descriptors.get(v_id)
                kp_u = keypoints.get(u_id)
                kp_v = keypoints.get(v_id)

                # only run GV if we have everything we need
                if all(item is not None for item in (desc_u, desc_v, kp_u, kp_v)):
                    gv_attempts += 1

                    fd = fisher_distance(fisher_vectors[u_id], fisher_vectors[v_id])
                    dist, n_inliers = compute_geometric_similarity(
                        desc_u, kp_u, desc_v, kp_v, fd,
                        use_lightglue=use_lightglue, method=method,
                        gv_matcher=matcher,
                        image0=image_paths.get(u_id) if image_paths else None,
                        image1=image_paths.get(v_id) if image_paths else None,
                    )

                    gv_pass = (dist < gv_threshold) and (n_inliers >= MIN_INLIERS)
                    if gv_pass:
                        gv_passes += 1
                        
                        if automated_mode:
                            # Pure geometric verification with confidence weighting
                            confidence = min(1.0, n_inliers / 20.0)  # Normalize inliers to confidence
                            geometric_quality = 1.0 - dist  # Higher quality = lower distance
                            
                            # Combine geometric confidence with Fisher similarity
                            overall_confidence = 0.6 * geometric_quality + 0.4 * confidence
                            
                            # Use probabilistic matching instead of hard binary
                            match = overall_confidence if overall_confidence > 0.5 else 0
                            if match > 0:
                                positive_matches += 1
                        else:
                            # ask the human (label) only now
                            label_queries += 1
                            match = 1 if labels.get(u_id) == labels.get(v_id) else 0
                            if match:
                                positive_matches += 1
                            # simulate annotation mistake
                            if label_error_rate > 0.0 and rng.random() < label_error_rate:
                                match = 1 - match
                # else: skip GV → match stays 0
            else:
                # Legacy path: direct label query, no gate
                if automated_mode:
                    # In automated mode without GV, fall back to Fisher vector similarity
                    fd = fisher_distance(fisher_vectors[u_id], fisher_vectors[v_id])
                    match = 1 if fd < gv_threshold else 0  # Use similarity threshold
                    if match:
                        positive_matches += 1
                else:
                    label_queries += 1
                    match = 1 if labels.get(u_id) == labels.get(v_id) else 0
                    if match:
                        positive_matches += 1
                    if label_error_rate > 0.0 and rng.random() < label_error_rate:
                        match = 1 - match

            fb_list.append(match)

        feedback = np.asarray(fb_list, dtype=np.float32)

        # -------------------------------------------------------------- #
        # 4.   Importance‑weighted degree estimate                      #
        # -------------------------------------------------------------- #
        denom = q[neighbors]
        denom[denom == 0.0] = 1e-9
        d_u = np.sum(feedback / denom) / n_neighbors
        population_estimates.append((1.0 / Q[u_idx]) * (1.0 / (1.0 + d_u)))

    # Aggregate population estimates
    estimates = np.asarray(population_estimates, dtype=np.float32)
    mean_est = float(estimates.mean())
    stderr_est = float(estimates.std(ddof=1) / np.sqrt(len(estimates)))

    if return_stats:
        stats = {
            "total_pairs": total_pairs,
            "gv_attempts": gv_attempts,
            "gv_passes": gv_passes,
            "label_queries": label_queries,
            "matches": positive_matches,
        }
        return mean_est, stderr_est, stats
    else:
        return mean_est, stderr_est




"""def nested_importance_sampling(
    fisher_vectors: Dict[str, np.ndarray],
    labels: Dict[str, int],
    keypoints: Optional[Dict[str, np.ndarray]] = None,
    descriptors: Optional[Dict[str, np.ndarray]] = None,
    use_geometric: bool = False,
    use_lightglue: bool = False,
    method: str = "disk",
    gv_threshold: float = 0.5,
    n_vertices: int = 100,
    n_neighbors: int = 10,
    label_error_rate = 0.0
) -> Tuple[float, float]:
    image_ids = list(fisher_vectors.keys())
    vectors = [fisher_vectors[i] for i in image_ids]
    sim = cosine_similarity_matrix(vectors)

    degrees = sim.sum(axis=1)
    #Q = np.ones(len(image_ids))
    #Q = Q / Q.sum()
    Q = 1.0 / (1.0 + degrees)
    Q = Q / Q.sum()

    rng = np.random.default_rng()
    population_estimates = []

    for u_idx in rng.choice(len(image_ids), size=min(n_vertices, len(image_ids)), replace=False, p=Q):
        q = sim[u_idx]
        if q.sum() == 0:
            q = np.ones_like(q)
        q = q / q.sum()
        neighbors = rng.choice(len(image_ids), size=min(n_neighbors, len(image_ids)), replace=False, p=q)

        if use_geometric and keypoints is not None and descriptors is not None:
            fb_list = []
            for v in neighbors:
                u_id = image_ids[u_idx]
                v_id = image_ids[v]
                desc_u = descriptors.get(u_id)
                desc_v = descriptors.get(v_id)
                kp_u = keypoints.get(u_id)
                kp_v = keypoints.get(v_id)
                if desc_u is None or desc_v is None or kp_u is None or kp_v is None:
                    print(f"Missing data for {u_id} or {v_id}, skipping geometric verification.")
                    match = labels.get(u_id) == labels.get(v_id)
                else:
                    fd = fisher_distance(fisher_vectors[u_id], fisher_vectors[v_id])
                    dist, n_inliers = compute_geometric_similarity(
                        desc_u, kp_u, desc_v, kp_v, fd,
                        use_lightglue=use_lightglue, method=method,
                    )
                    match = dist < gv_threshold and n_inliers >= MIN_INLIERS
                fb_list.append(1 if match else 0)
            feedback = np.array(fb_list)
        else:
            feedback = np.array([
                1 if labels.get(image_ids[u_idx]) == labels.get(image_ids[v]) else 0
                for v in neighbors
            ])

        denom = q[neighbors]
        denom[denom == 0] = 1e-9
        d_u = np.sum(feedback / denom) / n_neighbors
        #print(np.mean(feedback))
        population_estimates.append((1.0 / Q[u_idx]) * (1.0 / (1.0 + d_u)))

    estimates = np.array(population_estimates)
    return estimates.mean(), estimates.std(ddof=1) / np.sqrt(len(estimates))"""
