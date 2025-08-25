"""Generate visualisations of randomly selected images.

This script samples a number of images that have features stored in the
``descriptors.h5`` files and creates keypoint and descriptor visualisations
for both the segmented and unsegmented versions of the images.  Only the top
keypoints/descriptors are shown to keep the images readable.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List

import cv2
import h5py
import pandas as pd

from constants import DATAFRAME_PATH, ROOT_DIR
from feature_aggregation import descriptor_dir
from feature_extraction import get_segmentation_tag
from visualization_suite import descriptors as vis_desc, io, keypoints, matching


def build_paths(dataset: str, method: str, seg_tag: str | None):
    """Resolve descriptor and image paths for ``dataset`` and ``method``.

    ``seg_tag`` matches the suffix used throughout the repository when saving
    features, e.g. ``segmented`` or ``unsegmented``.  If ``seg_tag`` is ``None``
    or empty, the default from :func:`feature_extraction.get_segmentation_tag`
    is used ("unsegmented").
    """

    if not seg_tag:
        seg_tag = get_segmentation_tag(False)

    base_dir = os.path.join(ROOT_DIR, "data", dataset)
    train_dir = descriptor_dir(base_dir, method, "train", seg_tag)
    test_dir = descriptor_dir(base_dir, method, "test", seg_tag)
    train_desc_h5 = os.path.join(train_dir, "descriptors.h5")
    test_desc_h5 = os.path.join(test_dir, "descriptors.h5")
    train_kp_h5 = os.path.join(train_dir, "keypoints.h5")
    test_kp_h5 = os.path.join(test_dir, "keypoints.h5")

    if seg_tag == "unsegmented":
        image_root = os.path.join(base_dir, "dataset")
    elif seg_tag == "segmented":
        image_root = os.path.join(base_dir, "segmented_dataset")
    else:
        image_root = os.path.join(base_dir, f"segmented_dataset_{seg_tag}")

    for path in (train_desc_h5, test_desc_h5, train_kp_h5, test_kp_h5):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    return train_desc_h5, test_desc_h5, train_kp_h5, test_kp_h5, image_root


def sample_image_ids(h5_paths: Dict[str, str], n: int) -> List[str]:
    """Return ``n`` random image ids present in all ``h5_paths``."""

    key_sets = []
    for path in h5_paths.values():
        with h5py.File(path, "r") as f:
            key_sets.append(set(f.keys()))

    common_ids = list(set.intersection(*key_sets)) if key_sets else []
    if not common_ids:
        raise ValueError("No common image ids found in provided HDF5 files")

    n = min(n, len(common_ids))
    return random.sample(common_ids, n)


def visualise_images(
    dataset: str,
    method: str,
    out_dir: str,
    num_images: int = 5,
    max_keypoints: int = 100,
) -> None:
    """Create visualisations for randomly chosen images.

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. ``ATRW``).
    method : str
        Feature extraction method.
    out_dir : str
        Directory where the visualisations will be written.
    num_images : int, optional
        Number of random images to draw.
    max_keypoints : int, optional
        Only the top ``max_keypoints`` keypoints/descriptors are displayed.
    """

    seg_tags = ["unsegmented", "segmented"]
    paths: Dict[str, Dict[str, str]] = {}
    for tag in seg_tags:
        train_desc_h5, test_desc_h5, train_kp_h5, test_kp_h5, image_root = build_paths(
            dataset, method, tag
        )
        paths[tag] = {
            "train_desc": train_desc_h5,
            "test_desc": test_desc_h5,
            "train_kp": train_kp_h5,
            "test_kp": test_kp_h5,
            "img_root": image_root,
        }

    os.makedirs(out_dir, exist_ok=True)

    # determine ids present in both segmented and unsegmented stores
    sample_ids = sample_image_ids(
        {t: p["test_desc"] for t, p in paths.items()}, num_images
    )

    df = pd.read_csv(DATAFRAME_PATH.format(dataset))
    id_to_identity = {
        str(row.image_id): row.identity for row in df.itertuples()
    }
    train_ids_by_identity: Dict[str, List[str]] = {}
    for row in df.itertuples():
        if getattr(row, "split", None) == "train":
            train_ids_by_identity.setdefault(str(row.identity), []).append(
                str(row.image_id)
            )

    for img_id in sample_ids:
        identity = id_to_identity.get(str(img_id))
        if identity is None:
            # skip if identity is unknown
            continue

        train_ids = train_ids_by_identity.get(str(identity))
        if not train_ids:
            continue
        train_img_id = random.choice(train_ids)

        for tag in seg_tags:
            tag_dir = os.path.join(out_dir, tag)
            os.makedirs(tag_dir, exist_ok=True)

            q_img_path = os.path.join(paths[tag]["img_root"], identity, f"{img_id}.jpg")
            t_img_path = os.path.join(paths[tag]["img_root"], identity, f"{train_img_id}.jpg")
            if not os.path.exists(q_img_path) or not os.path.exists(t_img_path):
                continue

            q_img = io.load_image(q_img_path)
            t_img = io.load_image(t_img_path)

            q_kp = io.load_keypoints_h5(paths[tag]["test_kp"], [img_id]).get(str(img_id))
            q_desc = io.load_descriptors_h5(paths[tag]["test_desc"], [img_id]).get(str(img_id))
            t_kp = io.load_keypoints_h5(paths[tag]["train_kp"], [train_img_id]).get(
                str(train_img_id)
            )
            t_desc = io.load_descriptors_h5(paths[tag]["train_desc"], [train_img_id]).get(
                str(train_img_id)
            )

            if (
                q_kp is None
                or q_desc is None
                or t_kp is None
                or t_desc is None
                or len(q_kp) == 0
                or len(q_desc) == 0
                or len(t_kp) == 0
                or len(t_desc) == 0
            ):
                continue

            if max_keypoints:
                if len(q_kp) > max_keypoints:
                    q_kp = q_kp[:max_keypoints]
                    q_desc = q_desc[:max_keypoints]
                if len(t_kp) > max_keypoints:
                    t_kp = t_kp[:max_keypoints]
                    t_desc = t_desc[:max_keypoints]

            # Save keypoints and descriptor visualisations for the query image
            kp_vis, _ = keypoints.draw_keypoints(q_img, q_kp)
            io.save_image(os.path.join(tag_dir, f"{img_id}_keypoints.png"), kp_vis)

            desc_vis, _ = vis_desc.visualize_descriptor(q_desc[0])
            io.save_image(os.path.join(tag_dir, f"{img_id}_descriptor.png"), desc_vis)

            # Match descriptors and save visualisation
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            matches = matcher.match(q_desc, t_desc)
            match_vis, _ = matching.draw_matches(q_img, q_kp, t_img, t_kp, matches)
            io.save_image(
                os.path.join(tag_dir, f"{img_id}_{train_img_id}_matches.png"), match_vis
            )

    print(f"Visualisations saved to '{out_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random visualisations")
    parser.add_argument("--dataset", default="ATRW", help="Dataset name")
    parser.add_argument("--method", default="disk", help="Feature extraction method")
    parser.add_argument("--out_dir", default="visualizations", help="Output directory")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to sample")
    parser.add_argument(
        "--max_keypoints", type=int, default=100, help="Maximum keypoints to display"
    )
    args = parser.parse_args()

    visualise_images(
        args.dataset, args.method, args.out_dir, args.num_images, args.max_keypoints
    )

