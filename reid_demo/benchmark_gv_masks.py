"""Foreground-only GV views over unchanged bbox local feature caches."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import pandas as pd

from reid_demo.benchmark import file_digest, fingerprint
from reid_demo.benchmark_masks import (
    MASKED_PREPROCESSING, MASK_KEYPOINT_POLICY, filter_foreground_features,
    load_foreground_mask,
)


def prepare_gv_masks(frame, metadata_path):
    """Attach aligned mask references to a separate copy; preserve retrieval inputs."""
    if frame.empty or set(frame.input_mode) != {"bbox"}:
        raise ValueError("--gv-mask-metadata requires bbox retrieval metadata.")
    path = Path(metadata_path)
    digest = file_digest(path)
    manifest = json.loads(path.with_name("manifest.json").read_text())
    if (manifest.get("prepared_metadata_sha256") != digest or
            manifest.get("masked_preprocessing") != MASKED_PREPROCESSING):
        raise ValueError("GV masks require checksummed metadata from --modes masked preparation.")
    masks = pd.read_csv(path, dtype={"image_id": str, "identity": str})
    provenance = {"identity", "source_sha256", "bbox", "encounter_id", "timestamp"}
    provenance.update(column for column in frame if column.startswith("split-"))
    required = provenance | {"image_id", "input_mode", "mask_path", "mask_sha256"}
    nonnullable = required - {"timestamp"}  # Missing capture dates are valid prepared metadata.
    if (not required.issubset(masks) or not (provenance | {"image_id"}).issubset(frame) or
            masks[list(nonnullable)].isna().any().any() or masks.image_id.duplicated().any() or
            frame.image_id.duplicated().any() or set(masks.input_mode) != {"masked"}):
        raise ValueError("GV mask metadata has missing fields, duplicate IDs or invalid input mode.")
    if set(masks.image_id) != set(frame.image_id):
        raise ValueError("GV masks and bbox metadata must cover the same prepared image IDs.")
    aligned = masks.set_index("image_id").loc[frame.image_id]
    for column in sorted(provenance):
        left, right = frame[column].fillna("").tolist(), aligned[column].fillna("").tolist()
        if column == "bbox":
            left, right = [json.loads(v) for v in left], [json.loads(v) for v in right]
        if left != right:
            raise ValueError(f"GV masks do not match bbox metadata: {column}.")
    result = frame.copy()
    for column in ("mask_path", "mask_sha256"):
        result[column] = aligned[column].to_numpy()
    spec = {
        "scope": "geometric_verification_only", "input_pixels": "bbox",
        "keypoint_policy": MASK_KEYPOINT_POLICY, "metadata_sha256": digest,
        # Full prepared collection: development and test use the same contract.
        "mask_dataset_sha256": fingerprint(sorted(zip(result.image_id, result.mask_sha256))),
        "implementation": {
            name: file_digest(Path(__file__).parent / name)
            for name in ("benchmark_gv_masks.py", "benchmark_masks.py")
        },
    }
    return result, spec


class ForegroundFeatures:
    """Bounded filtered view; never rewrite the shared Fisher/GV source features."""

    def __init__(self, local, frame):
        self.local = local
        self.rows = {row["image_id"]: row for row in frame.to_dict("records")}
        self.memory = OrderedDict()
        self.counts = {}
        # Validate every used mask and report retention even on a pair-cache replay.
        for image_id in self.rows:
            self.pair(image_id)

    def pair(self, image_id):
        if image_id in self.memory:
            self.memory.move_to_end(image_id)
            return self.memory[image_id]
        mask = load_foreground_mask(self.rows[image_id])
        descriptors, keypoints = self.local.pair(image_id)
        pair = filter_foreground_features(descriptors, keypoints, mask)
        self.counts[image_id] = (len(keypoints), len(pair[1]))
        self.memory[image_id] = pair
        if len(self.memory) > 64:
            self.memory.popitem(last=False)
        return pair

    @property
    def mask_counts(self):
        return {
            "images": len(self.counts),
            "raw_keypoints": sum(raw for raw, _ in self.counts.values()),
            "foreground_keypoints": sum(kept for _, kept in self.counts.values()),
            "empty_images": sum(kept == 0 for _, kept in self.counts.values()),
        }

    def close(self):
        self.memory.clear()
