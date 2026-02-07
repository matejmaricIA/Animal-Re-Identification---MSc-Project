from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from geometric_verification import compute_local_evidence


@dataclass(frozen=True)
class CountCalibrators:
    """Holds optional per-signal calibrators used to map raw scores to probabilities.

    Each calibrator is expected to expose a ``predict_proba(scores)`` method that
    returns calibrated probabilities in [0, 1].
    """

    global_cal: Any | None = None
    fisher_cal: Any | None = None
    local_cal: Any | None = None


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
    """Stack vectors into a (N,D) float32 matrix in ``image_ids`` order.

    Missing vectors are filled with zeros (cosine similarities become 0).
    """

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
    # Cosine similarity is in [-1, 1]; clip negatives for non-negative weights.
    return np.clip(sim, 0.0, 1.0)


def nested_importance_sampling(
    global_embeddings: Dict[str, np.ndarray] | None,
    fisher_vectors: Dict[str, np.ndarray] | None,
    image_ids: Sequence[str],
    *,
    oracle: Callable[[str, str], int],
    keypoints: Dict[str, np.ndarray] | None = None,
    descriptors: Dict[str, np.ndarray] | None = None,
    proposal_mode: str = "calibrated",
    local_evidence: str = "inliers",
    local_mu: float = 0.5,
    shortlist_B: int = 200,
    mix_alpha: float = 0.9,
    calibrators: CountCalibrators | None = None,
    use_lightglue: bool = False,
    method: str = "disk",
    gv_matcher: str | None = None,
    n_vertices: int = 100,
    n_neighbors: int = 10,
    label_error_rate: float = 0.0,
    seed: int | None = None,
    return_stats: bool = False,
    q_eps: float = 1e-12,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, Any]]]:
    """Estimate population size with HITL Nested Importance Sampling (NIS).

    This follows the "Human-in-the-Loop Visual Re-ID for Population Size
    Estimation" estimator: sampling is driven by an approximate similarity
    (proposal), while the true edge indicator s(u,v) is provided by a human
    (``oracle``). The estimator remains unbiased for any proposal distributions
    with full support; proposal quality mainly affects variance / CI width.

    Key design choices:
      - No GV gate: every sampled pair queries the oracle.
      - Late fusion happens only in the proposal distribution.
      - Local evidence is computed only on a shortlist and mixed with a base
        proposal to keep full support (unbiasedness).
    """

    proposal_mode = str(proposal_mode).lower().strip()
    if proposal_mode not in {"calibrated", "power"}:
        raise ValueError("proposal_mode must be 'calibrated' or 'power'")

    local_evidence = str(local_evidence).lower().strip()
    if local_evidence not in {"inliers", "conf_matches"}:
        raise ValueError("local_evidence must be 'inliers' or 'conf_matches'")

    if not (0.0 <= float(mix_alpha) <= 1.0):
        raise ValueError("mix_alpha must be in [0, 1]")

    if shortlist_B <= 0:
        raise ValueError("shortlist_B must be > 0")

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

    need_local = True  # local evidence is part of the requested protocol
    have_local_inputs = bool(keypoints) and bool(descriptors)
    if need_local and not have_local_inputs:
        raise ValueError("Local evidence requires keypoints and descriptors dicts.")

    # Uniform Q(u) (valid for unbiasedness; can be upgraded later if desired).
    Q = np.full(n_images, 1.0 / float(n_images), dtype=np.float64)

    # Stats
    oracle_calls = 0
    unique_oracle_pairs: set[tuple[int, int]] = set()
    local_attempts = 0
    local_cache_hits = 0
    local_cache: Dict[tuple[int, int], tuple[int, int]] = {}
    # cache value: (n_inliers, n_conf_matches)

    contributions: list[float] = []

    for _ in range(n_vertices):
        u_idx = int(rng.choice(n_images, p=Q))
        u_id = image_ids[u_idx]

        # --- Base proposal: cheap similarities over all v != u ---
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

        # Exclude self-pair from proposals.
        p_base[u_idx] = 0.0

        base_weights = p_base + float(q_eps)
        base_weights[u_idx] = 0.0
        base_sum = float(np.sum(base_weights))
        if not np.isfinite(base_sum) or base_sum <= 0.0:
            base_weights = np.ones(n_images, dtype=np.float64)
            base_weights[u_idx] = 0.0
            base_sum = float(np.sum(base_weights))
        q_base = base_weights / base_sum

        # --- Shortlist: compute local evidence only for top-B by p_base ---
        B_eff = min(shortlist_B, n_images - 1)
        base_for_sort = p_base.copy()
        base_for_sort[u_idx] = float("-inf")
        shortlist = np.argsort(base_for_sort)[::-1][:B_eff]

        # Compute local evidence for shortlist pairs.
        shortlist_local_counts: Dict[int, int] = {}
        shortlist_local_inliers: Dict[int, int] = {}

        for v_idx in shortlist:
            v_idx = int(v_idx)
            if v_idx == u_idx:
                continue

            a, b = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
            cached = local_cache.get((a, b))
            if cached is not None:
                n_inliers, n_conf = cached
                local_cache_hits += 1
            else:
                local_attempts += 1
                n_inliers, n_conf = compute_local_evidence(
                    descriptors.get(u_id) if descriptors else None,
                    keypoints.get(u_id) if keypoints else None,
                    descriptors.get(image_ids[v_idx]) if descriptors else None,
                    keypoints.get(image_ids[v_idx]) if keypoints else None,
                    local_mu=local_mu,
                    use_lightglue=use_lightglue,
                    method=method,
                    gv_matcher=gv_matcher,
                )
                local_cache[(a, b)] = (int(n_inliers), int(n_conf))

            shortlist_local_inliers[v_idx] = int(n_inliers)
            if local_evidence == "inliers":
                shortlist_local_counts[v_idx] = int(n_inliers)
            else:
                shortlist_local_counts[v_idx] = int(n_conf)

        # Build q_short over shortlist only.
        short_ids = [int(v) for v in shortlist if int(v) != u_idx]
        if not short_ids:
            # Degenerate case: fallback to base-only sampling.
            short_ids = []
            q_short = np.empty((0,), dtype=np.float64)
            short_pos = {}
        else:
            weights_short = []
            for v_idx in short_ids:
                v_idx = int(v_idx)
                if proposal_mode == "calibrated":
                    p_parts = []
                    if p_global is not None:
                        p_parts.append(float(p_global[v_idx]))
                    if p_fisher is not None:
                        p_parts.append(float(p_fisher[v_idx]))

                    local_count = int(shortlist_local_counts.get(v_idx, 0))
                    p_local = None
                    if calibrators is not None and calibrators.local_cal is not None:
                        signal = float(np.log1p(max(0, local_count)))
                        p_local = float(calibrators.local_cal.predict_proba([signal])[0])
                    else:
                        # Fallback: map counts to a soft probability-like value.
                        p_local = float(1.0 - np.exp(-0.1 * float(max(0, local_count))))
                    p_parts.append(p_local)

                    p_fused = float(np.mean(p_parts)) if p_parts else 0.0
                    weights_short.append(p_fused + float(q_eps))
                else:
                    # Power-rule: boost base probability using local count.
                    local_count = int(shortlist_local_counts.get(v_idx, 0))
                    p_b = float(np.clip(p_base[v_idx], 0.0, 1.0))
                    d_l = float(np.clip(1.0 - p_b, 1e-12, 1.0))
                    power_score = float(max(0, local_count)) * float(-np.log(d_l))
                    weights_short.append(power_score + float(q_eps))

            weights_short_arr = np.asarray(weights_short, dtype=np.float64)
            if not np.isfinite(weights_short_arr).all():
                weights_short_arr = np.nan_to_num(weights_short_arr, nan=0.0, posinf=0.0, neginf=0.0)
            if float(weights_short_arr.sum()) <= 0.0:
                # Fallback to base weights within shortlist.
                weights_short_arr = np.asarray([float(p_base[v]) + float(q_eps) for v in short_ids], dtype=np.float64)
            q_short = weights_short_arr / float(weights_short_arr.sum())
            short_pos = {v_idx: i for i, v_idx in enumerate(short_ids)}

        # --- Sample neighbors, query oracle, compute importance-weighted degree ---
        d_hat_accum = 0.0

        for _ in range(n_neighbors):
            use_short = (len(short_ids) > 0) and (float(rng.random()) < float(mix_alpha))
            if use_short:
                j = int(rng.choice(len(short_ids), p=q_short))
                v_idx = int(short_ids[j])
                q_short_v = float(q_short[j])
            else:
                v_idx = int(rng.choice(n_images, p=q_base))
                if v_idx == u_idx:
                    # Extremely unlikely, but ensure v != u.
                    v_idx = int(rng.integers(0, n_images - 1))
                    if v_idx >= u_idx:
                        v_idx += 1
                q_short_v = 0.0
                pos = short_pos.get(v_idx)
                if pos is not None:
                    q_short_v = float(q_short[pos])

            q_u_v = float(mix_alpha) * q_short_v + float(1.0 - float(mix_alpha)) * float(q_base[v_idx])
            if q_u_v <= 0.0 or not np.isfinite(q_u_v):
                # Guard: shouldn't happen due to q_base support.
                q_u_v = float(max(float(q_eps), float(q_base[v_idx])))

            v_id = image_ids[v_idx]
            oracle_calls += 1
            a, b = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
            unique_oracle_pairs.add((a, b))
            y = int(oracle(u_id, v_id))
            if label_error_rate > 0.0 and float(rng.random()) < float(label_error_rate):
                y = 1 - y
            d_hat_accum += float(y) / q_u_v

        d_hat = d_hat_accum / float(n_neighbors)
        contrib = (1.0 / float(Q[u_idx])) * (1.0 / (1.0 + d_hat))
        contributions.append(float(contrib))

    contrib_arr = np.asarray(contributions, dtype=np.float64)
    estimate = float(np.mean(contrib_arr))
    stderr = float(np.std(contrib_arr, ddof=1) / np.sqrt(len(contrib_arr))) if len(contrib_arr) > 1 else 0.0

    if not return_stats:
        return estimate, stderr

    stats: Dict[str, Any] = {
        "n_images": int(n_images),
        "n_vertices": int(n_vertices),
        "n_neighbors": int(n_neighbors),
        "oracle_calls": int(oracle_calls),
        "unique_oracle_pairs": int(len(unique_oracle_pairs)),
        "local_attempts": int(local_attempts),
        "local_cache_hits": int(local_cache_hits),
        "proposal_mode": proposal_mode,
        "local_evidence": local_evidence,
        "local_mu": float(local_mu),
        "shortlist_B": int(shortlist_B),
        "mix_alpha": float(mix_alpha),
    }
    return estimate, stderr, stats

