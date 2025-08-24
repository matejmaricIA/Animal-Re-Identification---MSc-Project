"""Classification retrieval visualisations."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from . import collage, matching


def visualize_top_matches(
    query_img: np.ndarray,
    candidate_imgs: Sequence[np.ndarray],
    matches: Sequence[dict],
    gv_results: Sequence[dict] | None = None,
    top_k: int = 5,
) -> Tuple[np.ndarray, str]:
    """Show query and top-k candidates with drawn matches.

    Parameters
    ----------
    query_img : np.ndarray
        Query image in BGR format.
    candidate_imgs : Sequence[np.ndarray]
        Candidate images aligned with ``matches``.
    matches : Sequence[dict]
        Each element should contain ``query_kp``, ``train_kp``,
        ``query_desc``, ``train_desc`` and metadata such as ``score`` and
        ``n_inliers``.  Descriptor matching is performed internally.
    gv_results : Sequence[dict], optional
        Optional geometric verification results per candidate.
    top_k : int, optional
        Number of candidates to show.
    """
    visuals = []
    titles = []

    bf = None
    for idx, (img, info) in enumerate(zip(candidate_imgs[:top_k], matches[:top_k])):
        desc1 = info['query_desc']
        desc2 = info['train_desc']
        norm = cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2
        if bf is None or bf.normType != norm:
            bf = cv2.BFMatcher(norm, crossCheck=True)
        raw = bf.match(desc1, desc2)
        vis, _ = matching.draw_matches(query_img, info['query_kp'], img, info['train_kp'], raw)
        visuals.append(vis)
        title = f"{info.get('train_id', idx)}\nscore={info.get('score', 0):.2f}"\
                f" inliers={info.get('n_inliers', 0)}"
        titles.append(title)

    grid, _ = collage.make_grid(visuals, titles=titles, cols=1)
    caption = ", ".join(
        [
            f"{info.get('train_id', i)}: {info.get('score', 0):.2f} ({info.get('n_inliers', 0)} inliers)"
            for i, info in enumerate(matches[:top_k])
        ]
    )
    return grid, caption
