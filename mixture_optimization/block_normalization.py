"""Utilities for normalising descriptor blocks and fusing them."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np

DescriptorBlock = Dict[str, np.ndarray]


def zscore_block(
    block: DescriptorBlock,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> Tuple[DescriptorBlock, np.ndarray, np.ndarray]:
    """Apply z-score standardisation to a descriptor block.

    Parameters
    ----------
    block:
        Mapping of image identifiers to feature vectors.
    mean, std:
        Optional precomputed mean and standard deviation.  When not provided
        they are estimated from ``block``.

    Returns
    -------
    Tuple containing the normalised block along with the mean and standard
    deviation used.
    """

    if not block:
        # Nothing to do; create dummy mean/std to avoid type issues
        if mean is None:
            mean = np.array([], dtype=np.float32)
        if std is None:
            std = np.array([], dtype=np.float32)
        return block, mean, std

    ids = list(block.keys())
    mat = np.stack([block[i] for i in ids])
    if mean is None:
        mean = mat.mean(axis=0)
    if std is None:
        std = mat.std(axis=0) + 1e-6
    mat = (mat - mean) / std
    return dict(zip(ids, mat)), mean, std


def l2_normalize_block(block: DescriptorBlock) -> DescriptorBlock:
    """L2 normalise all vectors in ``block``."""

    normed: DescriptorBlock = {}
    for k, vec in block.items():
        n = np.linalg.norm(vec)
        if n > 0:
            normed[k] = vec / n
        else:
            normed[k] = vec
    return normed


def apply_zscore_and_l2_train_test(
    train_block: DescriptorBlock,
    test_block: DescriptorBlock,
    skip_zscore: bool = False,
) -> tuple[DescriptorBlock, DescriptorBlock]:
    """Standardise train/test blocks and apply L2 normalisation.

    When ``skip_zscore`` is ``True`` only L2 normalisation is applied.
    """

    if skip_zscore:
        return l2_normalize_block(train_block), l2_normalize_block(test_block)

    train_z, mean, std = zscore_block(train_block)
    test_z, _, _ = zscore_block(test_block, mean, std)
    return l2_normalize_block(train_z), l2_normalize_block(test_z)


def fuse_blocks_weighted_concat(
    blocks: Mapping[str, DescriptorBlock],
    weights: Mapping[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    """Fuse descriptor blocks by weighted concatenation with final L2 normalisation."""

    if weights is None:
        weights = {}

    dims: Dict[str, int] = {}
    for name, blk in blocks.items():
        first = next(iter(blk.values()), None)
        if first is None:
            dims[name] = 0
        else:
            dims[name] = int(first.shape[0])

    image_ids = set().union(*[blk.keys() for blk in blocks.values()]) if blocks else set()

    fused: Dict[str, np.ndarray] = {}
    for img_id in image_ids:
        parts = []
        for name, blk in blocks.items():
            vec = blk.get(img_id)
            if vec is None:
                dim = dims.get(name, 0)
                if dim == 0:
                    continue
                vec = np.zeros(dim, dtype=np.float32)
            parts.append(weights.get(name, 1.0) * vec)
        if not parts:
            continue
        fused_vec = np.concatenate(parts).astype(np.float32)
        n = np.linalg.norm(fused_vec)
        if n > 0:
            fused_vec = fused_vec / n
        fused[img_id] = fused_vec
    return fused