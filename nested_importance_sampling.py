import numpy as np
from typing import Dict, Tuple, Sequence, Optional

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
    return estimates.mean(), estimates.std(ddof=1) / np.sqrt(len(estimates))