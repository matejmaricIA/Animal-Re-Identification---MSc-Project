import numpy as np
from typing import Dict, Tuple, Sequence, Optional

from geometric_verification import compute_geometric_similarity
from utils.distance_utils import fisher_distance
from constants import MIN_INLIERS


#def cosine_similarity_matrix(vectors: Sequence[np.ndarray]) -> np.ndarray:
#    arr = np.vstack(vectors)
#    norm = np.linalg.norm(arr, axis=1, keepdims=True)
#    norm[norm == 0] = 1
#    arr = arr / norm
#    sim = arr @ arr.T
#    np.fill_diagonal(sim, 0.0)
#    sim[sim < 0] = 0.0
#    return sim

def _weight_matrix(vectors: Sequence[np.ndarray], tau: float = 0.5) -> np.ndarray:
    """Return un-normalised exponential weights w_uv.

    Parameters
    ----------
    vectors : Sequence[np.ndarray]
        List of Fisher vectors.
    tau : float, optional
        Temperature parameter controlling weight sharpness.
    """
    arr = np.vstack(vectors)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    arr = arr / norm
    cos = arr @ arr.T
    # Exponential weights; diagonal set to 0
    w = np.exp(cos / tau)
    np.fill_diagonal(w, 0.0)
    return w

"""
def cosine_similarity_matrix(vectors, tau=0.6):
    arr  = np.vstack(vectors).astype(np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)

    # exponentiated cosine / τ
    logits = (arr @ arr.T) / tau
    np.fill_diagonal(logits, -np.inf)          # forbid self-loops
    W = np.exp(logits)                         # unnormalised weights
    W[np.isinf(W)] = 0.0                       # safety; shouldn’t happen

    return W

"""

def nested_importance_sampling(
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
    tau: float = 0.5,
    neighbor_ratio: Optional[float] = None,
    tolerance: float = 0.0,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """Estimate population size using Nested Importance Sampling.

    Parameters
    ----------
    fisher_vectors : dict
        Mapping from image_id to Fisher vector.
    labels : dict
        Mapping from image_id to ground truth identity label.
    keypoints : dict, optional
        Mapping from image_id to 2D keypoints.
    descriptors : dict, optional
        Mapping from image_id to local descriptors.
    use_geometric : bool
        Whether to use geometric verification when generating feedback.
    use_lightglue : bool
        Use LightGlue for feature matching during geometric verification.
    method : str
        Feature extraction method name (affects LightGlue matching).
    gv_threshold : float
        Threshold on geometric verification distance for a positive match.
    n_vertices : int
        Number of vertices (images) to sample.
    n_neighbors : int
        Number of neighbour comparisons for each sampled vertex.

    Returns
    -------
    tuple
        (population_estimate, standard_error)
    """
    image_ids = list(fisher_vectors.keys())
    vectors = [fisher_vectors[i] for i in image_ids]
    weights = _weight_matrix(vectors, tau=tau)

    degrees = weights.sum(axis=1)
    degrees[degrees == 0] = 1.0
    Q = 1.0 / (1.0 + degrees)
    Q = Q / Q.sum()

    if neighbor_ratio is not None:
        n_neighbors = max(1, int(round(neighbor_ratio * n_neighbors)))

    rng = np.random.default_rng(random_state)
    cache: Dict[Tuple[int, int], int] = {}
    population_estimates = []

    # stratified seed sampling: draw roughly equally from degree terciles
    q1, q2 = np.quantile(degrees, [1/3, 2/3])
    low = np.where(degrees <= q1)[0]
    mid = np.where((degrees > q1) & (degrees <= q2))[0]
    high = np.where(degrees > q2)[0]
    per_group = int(np.ceil(n_vertices / 3))
    selected = []
    for group in (low, mid, high):
        if len(selected) >= n_vertices:
            break
        if len(group) == 0:
            continue
        probs = Q[group]
        probs = probs / probs.sum()
        n_sel = min(per_group, len(group), n_vertices - len(selected))
        chosen = rng.choice(group, size=n_sel, replace=False, p=probs)
        selected.extend(chosen.tolist())

    if len(selected) < n_vertices:
        # fill remaining randomly according to Q
        remaining = n_vertices - len(selected)
        extras = rng.choice(len(image_ids), size=remaining, replace=False, p=Q)
        selected.extend(extras.tolist())

    for u_idx in selected:
        w_u = weights[u_idx]
        if w_u.sum() == 0:
            w_u = np.ones_like(w_u)
        q = w_u / w_u.sum()
        neighbors = rng.choice(len(image_ids), size=min(n_neighbors, len(image_ids)), replace=False, p=q)

        fb_list = []
        for v in neighbors:
            pair = (min(u_idx, v), max(u_idx, v))
            cached = cache.get(pair)
            if cached is not None:
                fb_list.append(cached)
                continue
            u_id = image_ids[u_idx]
            v_id = image_ids[v]
            if use_geometric and keypoints is not None and descriptors is not None:
                desc_u = descriptors.get(u_id)
                desc_v = descriptors.get(v_id)
                kp_u = keypoints.get(u_id)
                kp_v = keypoints.get(v_id)
                if desc_u is None or desc_v is None or kp_u is None or kp_v is None:
                    match = labels.get(u_id) == labels.get(v_id)
                else:
                    fd = fisher_distance(fisher_vectors[u_id], fisher_vectors[v_id])
                    dist, n_inliers = compute_geometric_similarity(
                        desc_u, kp_u, desc_v, kp_v, fd,
                        use_lightglue=use_lightglue, method=method,
                    )
                    match = dist < gv_threshold and n_inliers >= MIN_INLIERS
            else:
                match = labels.get(u_id) == labels.get(v_id)
            value = 1 if match else 0
            cache[pair] = value
            fb_list.append(value)
        feedback = np.array(fb_list)

        denom = q[neighbors]
        denom[denom == 0] = 1e-9
        d_u = np.sum(feedback / denom) / n_neighbors
        population_estimates.append((1.0 / Q[u_idx]) * (1.0 / (1.0 + d_u)))

        if tolerance > 0 and len(population_estimates) >= 5:
            hw = 1.96 * np.std(population_estimates, ddof=1) / np.sqrt(len(population_estimates))
            if hw <= tolerance:
                break

    estimates = np.array(population_estimates)

    if len(estimates) == 0:
        return 0.0, 0.0

    # median-of-means aggregation
    g = int(np.sqrt(len(estimates))) or 1
    groups = [estimates[i:i+g] for i in range(0, len(estimates), g)]
    group_means = [np.mean(gp) for gp in groups]
    estimate = float(np.median(group_means))
    se = np.std(estimates, ddof=1) / np.sqrt(len(estimates))
    ci_half = 1.96 * se
    return estimate, ci_half

"""
def nested_importance_sampling(
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
) -> Tuple[float, float]:
    image_ids = list(fisher_vectors.keys())
    vectors = [fisher_vectors[i] for i in image_ids]
    sim = cosine_similarity_matrix(vectors)

    degrees = sim.sum(axis=1)
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
        print(np.mean(feedback))
        population_estimates.append((1.0 / Q[u_idx]) * (1.0 / (1.0 + d_u)))

    estimates = np.array(population_estimates)
    return estimates.mean(), estimates.std(ddof=1) / np.sqrt(len(estimates))"""