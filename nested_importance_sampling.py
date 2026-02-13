from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Sequence, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class CountCalibrators:
    """Per-signal calibrators that map raw similarities to probabilities."""

    global_cal: Any | None = None
    fisher_cal: Any | None = None


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms).astype(np.float32)
    return matrix / norms


def _stack_vectors(
    vectors: Dict[str, np.ndarray] | None,
    image_ids: Sequence[str],
) -> np.ndarray:
    """Stack vectors into a (N,D) float32 matrix in ``image_ids`` order."""

    if not vectors:
        return np.empty((len(image_ids), 0), dtype=np.float32)

    first = next(iter(vectors.values()))
    dim = int(np.asarray(first).reshape(-1).shape[0])
    mat = np.zeros((len(image_ids), dim), dtype=np.float32)
    for idx, image_id in enumerate(image_ids):
        vec = vectors.get(image_id)
        if vec is None:
            continue
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.shape[0] != dim:
            raise ValueError(
                f"Vector dim mismatch for '{image_id}': expected {dim}, got {arr.shape[0]}"
            )
        mat[idx] = arr
    return _l2_normalize_rows(mat)


def _safe_prob_from_raw_similarity(sim: np.ndarray) -> np.ndarray:
    """Fallback mapping for raw cosine similarities to [0, 1] weights."""

    sim = np.asarray(sim, dtype=np.float32)
    return np.clip(sim, 0.0, 1.0)


def nested_importance_sampling(
    global_embeddings: Dict[str, np.ndarray] | None,
    fisher_vectors: Dict[str, np.ndarray] | None,
    image_ids: Sequence[str],
    *,
    oracle: Callable[[str, str], int],
    proposal_mode: str = "calibrated",
    calibrators: CountCalibrators | None = None,
    n_vertices: int = 100,
    n_neighbors: int = 10,
    label_error_rate: float = 0.0,
    confirm_same_votes: int = 1,
    seed: int | None = None,
    return_stats: bool = False,
    q_eps: float = 1e-12,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
    """Estimate population size with HITL Nested Importance Sampling (NIS).

    """

    proposal_mode = str(proposal_mode).lower().strip()
    if proposal_mode not in {"calibrated", "power"}:
        raise ValueError("proposal_mode must be 'calibrated' or 'power'")

    confirm_same_votes = int(confirm_same_votes)
    if confirm_same_votes < 1:
        raise ValueError("confirm_same_votes must be >= 1")

    if n_vertices <= 0 or n_neighbors <= 0:
        raise ValueError("n_vertices and n_neighbors must be > 0")

    if not (0.0 <= float(label_error_rate) <= 1.0):
        raise ValueError("label_error_rate must be in [0, 1]")

    rng = np.random.default_rng(seed)

    image_ids = [str(i) for i in image_ids]
    n_images = len(image_ids)
    if n_images < 2:
        raise ValueError("Need at least 2 images for population estimation.")

    global_matrix = _stack_vectors(global_embeddings, image_ids)
    fisher_matrix = _stack_vectors(fisher_vectors, image_ids)

    have_global = global_matrix.shape[1] > 0
    have_fisher = fisher_matrix.shape[1] > 0
    if not have_global and not have_fisher:
        raise ValueError("At least one of global_embeddings or fisher_vectors must be provided.")

    Q = np.full(n_images, 1.0 / float(n_images), dtype=np.float64)

    oracle_calls = 0
    unique_oracle_pairs: set[tuple[int, int]] = set()
    confirm_triggered = 0
    confirm_extra_votes = 0
    contributions: list[float] = []

    for _ in range(n_vertices):
        u_idx = int(rng.choice(n_images, p=Q))
        u_id = image_ids[u_idx]

        raw_global = None
        raw_fisher = None
        if have_global:
            raw_global = global_matrix @ global_matrix[u_idx]
        if have_fisher:
            raw_fisher = fisher_matrix @ fisher_matrix[u_idx]

        p_global = None
        if raw_global is not None:
            if calibrators is not None and calibrators.global_cal is not None:
                p_global = calibrators.global_cal.predict_proba(raw_global)
            else:
                p_global = _safe_prob_from_raw_similarity(raw_global)

        p_fisher = None
        if raw_fisher is not None:
            if calibrators is not None and calibrators.fisher_cal is not None:
                p_fisher = calibrators.fisher_cal.predict_proba(raw_fisher)
            else:
                p_fisher = _safe_prob_from_raw_similarity(raw_fisher)

        probs = []
        if p_global is not None:
            probs.append(p_global)
        if p_fisher is not None:
            probs.append(p_fisher)
        if not probs:
            raise RuntimeError("Internal error: no base proposal signals available.")

        p_base = np.mean(np.stack(probs, axis=0), axis=0).astype(np.float64)
        p_base = np.clip(p_base, 0.0, 1.0)
        p_base[u_idx] = 0.0

        if proposal_mode == "power":
            # Power trick: stronger emphasis on high-match probabilities while
            # keeping non-zero support through q_eps.
            proposal_scores = -np.log(np.clip(1.0 - p_base, 1e-12, 1.0))
        else:
            proposal_scores = p_base

        base_weights = proposal_scores + float(q_eps)
        base_weights[u_idx] = 0.0
        base_sum = float(np.sum(base_weights))
        if not np.isfinite(base_sum) or base_sum <= 0.0:
            base_weights = np.ones(n_images, dtype=np.float64)
            base_weights[u_idx] = 0.0
            base_sum = float(np.sum(base_weights))
        q_base = base_weights / base_sum

        d_hat_accum = 0.0

        def oracle_vote(left_id: str, right_id: str) -> int:
            nonlocal oracle_calls
            y = int(oracle(left_id, right_id))
            if label_error_rate > 0.0 and float(rng.random()) < float(label_error_rate):
                y = 1 - y
            oracle_calls += 1
            return int(y)

        for _ in range(n_neighbors):
            v_idx = int(rng.choice(n_images, p=q_base))
            if v_idx == u_idx:
                v_idx = int(rng.integers(0, n_images - 1))
                if v_idx >= u_idx:
                    v_idx += 1

            q_u_v = float(q_base[v_idx])
            if q_u_v <= 0.0 or not np.isfinite(q_u_v):
                q_u_v = float(max(float(q_eps), float(q_base[v_idx])))

            v_id = image_ids[v_idx]
            a, b = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
            unique_oracle_pairs.add((a, b))
            y1 = oracle_vote(u_id, v_id)

            y = y1
            if confirm_same_votes > 1 and y1 == 1:
                confirm_triggered += 1
                y_confirmed = True
                for _ in range(confirm_same_votes - 1):
                    y_next = oracle_vote(u_id, v_id)
                    confirm_extra_votes += 1
                    if y_next != 1:
                        y_confirmed = False
                        break
                y = int(y_confirmed)

            d_hat_accum += float(y) / q_u_v

        d_hat = d_hat_accum / float(n_neighbors)
        contrib = (1.0 / float(Q[u_idx])) * (1.0 / (1.0 + d_hat))
        contributions.append(float(contrib))

    contrib_arr = np.asarray(contributions, dtype=np.float64)
    estimate = float(np.mean(contrib_arr))
    stderr = (
        float(np.std(contrib_arr, ddof=1) / np.sqrt(len(contrib_arr)))
        if len(contrib_arr) > 1
        else 0.0
    )

    if not return_stats:
        return estimate, stderr

    stats: Dict[str, Any] = {
        "n_images": int(n_images),
        "n_vertices": int(n_vertices),
        "n_neighbors": int(n_neighbors),
        "oracle_calls": int(oracle_calls),
        "unique_oracle_pairs": int(len(unique_oracle_pairs)),
        "proposal_mode": proposal_mode,
        "confirm_same_votes": int(confirm_same_votes),
        "confirm_triggered": int(confirm_triggered),
        "confirm_extra_votes": int(confirm_extra_votes),
    }
    return estimate, stderr, stats
