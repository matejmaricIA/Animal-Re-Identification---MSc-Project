import numpy as np
from typing import Dict, Tuple, Sequence


def cosine_similarity_matrix(vectors):
    """Compute cosine similarity matrix for a list of vectors."""
    arr = np.vstack(vectors)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm[norm == 0] = 1
    arr = arr / norm
    sim = arr @ arr.T
    np.fill_diagonal(sim, 0.0)
    sim[sim < 0] = 0.0
    return sim


def nested_importance_sampling(fisher_vectors, labels, n_vertices = 100, n_neighbors = 10):
    """Estimate population size using Nested Importance Sampling.

    Parameters
    ----------
    fisher_vectors : dict
        Mapping from image_id to Fisher vector.
    labels : dict
        Mapping from image_id to ground truth identity label.
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

        feedback = np.array([
            1 if labels.get(image_ids[u_idx]) == labels.get(image_ids[v]) else 0
            for v in neighbors
        ])
        denom = q[neighbors]
        denom[denom == 0] = 1e-9
        d_u = np.sum(feedback / denom) / n_neighbors
        population_estimates.append((1.0 / Q[u_idx]) * (1.0 / (1.0 + d_u)))

    estimates = np.array(population_estimates)
    return estimates.mean(), estimates.std(ddof=1) / np.sqrt(len(estimates))