"""Utility functions for combining multiple descriptor types.

Each descriptor type is provided as a dictionary that maps image ids to a
numpy array.  The :func:`combine_descriptor_dicts` function normalises each
component, applies a user supplied weight and concatenates the weighted
vectors into a single representation per image.
"""

from __future__ import annotations

from typing import Dict, Mapping, Iterable

import numpy as np


DescriptorDict = Mapping[str, np.ndarray]


def combine_descriptor_dicts(
    descriptor_dicts: Mapping[str, DescriptorDict],
    weights: Mapping[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    """Combine multiple descriptor dictionaries.

    Parameters
    ----------
    descriptor_dicts:
        Mapping from a descriptor name (e.g. ``"fisher"`` or ``"color"``) to a
        dictionary of image ids and descriptor vectors.  All vectors belonging
        to a single descriptor type must have the same dimensionality.
    weights:
        Optional mapping that assigns a weight to each descriptor type.  If a
        weight is not provided for a descriptor its weight defaults to ``1``.

    Returns
    -------
    dict
        Dictionary mapping image ids to concatenated descriptor vectors.  For
        images missing a particular descriptor a zero vector of the appropriate
        length is inserted so that all combined vectors share the same
        dimensionality.
    """

    if weights is None:
        weights = {}

    # Determine dimensionality for each descriptor type using the first
    # available entry.
    dims: Dict[str, int] = {}
    for name, descs in descriptor_dicts.items():
        first_vec = next(iter(descs.values()), None)
        if first_vec is None:
            raise ValueError(f"Descriptor '{name}' contains no vectors")
        dims[name] = int(first_vec.shape[0])

    # Collect the union of all image ids
    image_ids: Iterable[str] = set().union(*[d.keys() for d in descriptor_dicts.values()])

    combined: Dict[str, np.ndarray] = {}
    for img_id in image_ids:
        parts = []
        for name, descs in descriptor_dicts.items():
            vec = descs.get(img_id)
            if vec is None:
                vec = np.zeros(dims[name], dtype=np.float32)
            else:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            parts.append(weights.get(name, 1.0) * vec)
        combined[img_id] = np.concatenate(parts).astype(np.float32)
    return combined