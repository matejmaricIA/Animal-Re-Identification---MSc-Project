import numpy as np
from typing import Dict, Tuple, Sequence, Optional, Union, Any

from geometric_verification import compute_geometric_similarity
from utils.distance_utils import fisher_distance
from constants import MIN_INLIERS


def gv_score(n_inliers: int, dist: float, fd: float) -> float:
    """
    Monotone 'matchiness' score in ~[0,1]:
    - higher with more inliers
    - higher when geometric distance is smaller
    - higher when Fisher distance is smaller
    """
    n_inliers = max(int(n_inliers), 0)
    dist = float(dist)
    fd = float(fd)

    t_inl  = np.tanh(n_inliers / 20.0)       # 0..~1
    t_dist = 1.0 - np.clip(dist, 0.0, 1.0)   # 1 when dist~0
    t_fd   = 1.0 - np.tanh(fd / 5.0)         # 1 when Fisher small

    return float(np.clip(0.45*t_dist + 0.35*t_inl + 0.20*t_fd, 0.0, 1.0))


def accept_prob(score: float, a: float, tau: float, pi_floor: float) -> float:
    """Logistic acceptance with floor to bound weights."""
    pi = 1.0 / (1.0 + np.exp(-a * (score - tau)))
    return float(max(pi_floor, min(1.0, pi)))


def pick_tau_from_pilot(scores: np.ndarray, a: float, p_target: float) -> float:
    """Choose τ so that mean σ(a(s-τ)) ≈ p_target via bisection."""
    if scores.size == 0:
        return 0.5
    lo, hi = float(np.min(scores))-1e-6, float(np.max(scores))+1e-6
    for _ in range(30):
        mid = 0.5*(lo+hi)
        acc = 1.0 / (1.0 + np.exp(-a*(scores - mid)))
        if float(acc.mean()) > p_target:
            lo = mid  # too many accepted → raise τ
        else:
            hi = mid
    return 0.5*(lo+hi)


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
    gv_threshold: float = 0.75,
    n_vertices: int = 100,
    n_neighbors: int = 10,
    label_error_rate: float = 0.0,
    return_stats: bool = False,
    automated_mode: bool = False,
    seed: Optional[int] = None,
    randomized_gate: bool = False,
    pi_target: float = 0.80,
    pi_slope: float = 8.0,
    pi_floor: float = 0.05,
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
    randomized_gate : bool, default=False
        Use randomized acceptance with known inclusion probabilities and
        Horvitz–Thompson weighting.
    pi_target : float, default=0.80
        Target acceptance rate for randomized gate.
    pi_slope : float, default=8.0
        Logistic slope ``a`` for inclusion probability.
    pi_floor : float, default=0.05
        Lower bound on inclusion probabilities to cap weights.

    Returns
    -------
    mean_est, stderr_est : floats
        Population size estimate and its standard error.
    stats : dict  (only if ``return_stats=True``)
        See *return_stats* description.
    """


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

    tau = 0.5  # default midpoint; used only if randomized_gate is True

    if randomized_gate and keypoints is not None and descriptors is not None:
        rng_local = np.random.default_rng(seed if seed is not None else 123)
        pilot_scores = []
        # Small pilot: ~min(2000, n_vertices*n_neighbors) pairs
        n_pilot = int(min(2000, max(1, n_vertices) * max(1, n_neighbors)))
        # Sample anchors ~ Q
        n_anchors = max(1, n_pilot // max(1, n_neighbors))
        anchor_idxs = rng_local.choice(len(image_ids), size=min(len(image_ids), n_anchors), replace=False, p=Q)

        for u_idx in anchor_idxs:
            # neighbor proposal for this anchor
            q_vec = sim[u_idx].copy()
            if q_vec.sum() <= 0:
                q_vec = np.ones_like(q_vec, dtype=float)
            q_vec[q_vec <= 0] = 1e-12
            q_vec /= q_vec.sum()

            # sample neighbors ~ q
            k = max(1, n_pilot // max(1, len(anchor_idxs)))
            v_idx_sample = rng_local.choice(len(image_ids), size=min(len(image_ids), k), replace=False, p=q_vec)

            u_id = image_ids[u_idx]
            for v_idx in v_idx_sample:
                v_id = image_ids[v_idx]
                desc_u = descriptors.get(u_id); desc_v = descriptors.get(v_id)
                kp_u   = keypoints.get(u_id);   kp_v   = keypoints.get(v_id)
                if not all(x is not None for x in (desc_u, desc_v, kp_u, kp_v)):
                    continue
                fd = fisher_distance(fisher_vectors[u_id], fisher_vectors[v_id])
                dist, n_inliers = compute_geometric_similarity(
                    desc_u, kp_u, desc_v, kp_v, fd, use_lightglue=use_lightglue, method=method
                )
                pilot_scores.append(gv_score(n_inliers, dist, fd))

        if len(pilot_scores) >= 50:
            tau = pick_tau_from_pilot(np.asarray(pilot_scores, dtype=float), a=pi_slope, p_target=pi_target)

    # 2.   Outer vertex loop                                             #
    outer_vertices = rng.choice(len(image_ids), size=min(n_vertices, len(image_ids)), replace=False, p=Q)
    for u_idx in outer_vertices:
        q = sim[u_idx].copy()
        if q.sum() == 0.0:
            q = np.ones_like(q)
        q[q == 0.0] = 1e-9
        q /= q.sum()

        neighbors = rng.choice(len(image_ids), size=min(n_neighbors, len(image_ids)), replace=False, p=q)

        fb_list = []  # contributions for this anchor u; each term is already HT-weighted

        # 3. Inner neighbour loop with GV gate
        for v_idx in neighbors:
            total_pairs += 1

            u_id = image_ids[u_idx]; v_id = image_ids[v_idx]

            if use_geometric and keypoints is not None and descriptors is not None:
                desc_u = descriptors.get(u_id); desc_v = descriptors.get(v_id)
                kp_u   = keypoints.get(u_id);   kp_v   = keypoints.get(v_id)

                if all(item is not None for item in (desc_u, desc_v, kp_u, kp_v)):
                    gv_attempts += 1

                    fd = fisher_distance(fisher_vectors[u_id], fisher_vectors[v_id])
                    dist, n_inliers = compute_geometric_similarity(
                        desc_u, kp_u, desc_v, kp_v, fd, use_lightglue=use_lightglue, method=method
                    )

                    if randomized_gate:
                        s  = gv_score(n_inliers, dist, fd)
                        pi = accept_prob(s, a=pi_slope, tau=tau, pi_floor=pi_floor)

                        if rng.random() < pi:
                            gv_passes += 1
                            if automated_mode:
                                yhat = float(np.clip(
                                    0.6*(1.0 - dist) + 0.3*np.tanh(n_inliers/20.0) + 0.1*(1.0 - np.tanh(fd/5.0)),
                                    0.0, 1.0
                                ))
                                if yhat > 0.5:
                                    positive_matches += 1
                            else:
                                label_queries += 1
                                yhat = 1.0 if labels.get(u_id) == labels.get(v_id) else 0.0
                                if label_error_rate > 0.0 and rng.random() < label_error_rate:
                                    yhat = 1.0 - yhat
                                if yhat == 1.0:
                                    positive_matches += 1
                            q_v = float(max(q[v_idx], 1e-12))
                            fb_list.append(yhat / (q_v * pi))
                        else:
                            fb_list.append(0.0)
                    else:
                        gv_pass = (dist < gv_threshold) and (n_inliers >= MIN_INLIERS)
                        if gv_pass:
                            gv_passes += 1
                            if automated_mode:
                                yhat = float(np.clip(
                                    0.6*(1.0 - dist) + 0.4*np.tanh(n_inliers/20.0), 0.0, 1.0
                                ))
                                if yhat > 0.5:
                                    positive_matches += 1
                                fb_list.append(yhat / float(max(q[v_idx], 1e-12)))
                            else:
                                label_queries += 1
                                y = 1.0 if labels.get(u_id) == labels.get(v_id) else 0.0
                                if label_error_rate > 0.0 and rng.random() < label_error_rate:
                                    y = 1.0 - y
                                if y == 1.0:
                                    positive_matches += 1
                                fb_list.append(y / float(max(q[v_idx], 1e-12)))
                        else:
                            fb_list.append(0.0)
                else:
                    fb_list.append(0.0)
            else:
                if automated_mode:
                    fd = fisher_distance(fisher_vectors[u_id], fisher_vectors[v_id])
                    yhat = 1.0 if fd < gv_threshold else 0.0
                    if yhat > 0.5:
                        positive_matches += 1
                    fb_list.append(yhat / float(max(q[v_idx], 1e-12)))
                else:
                    label_queries += 1
                    y = 1.0 if labels.get(u_id) == labels.get(v_id) else 0.0
                    if label_error_rate > 0.0 and rng.random() < label_error_rate:
                        y = 1.0 - y
                    if y == 1.0:
                        positive_matches += 1
                    fb_list.append(y / float(max(q[v_idx], 1e-12)))

        contribs = np.asarray(fb_list, dtype=np.float32)

        # -------------------------------------------------------------- #
        # 4.   Importance‑weighted degree estimate                      #
        # -------------------------------------------------------------- #
        d_u = float(contribs.sum()) / n_neighbors
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