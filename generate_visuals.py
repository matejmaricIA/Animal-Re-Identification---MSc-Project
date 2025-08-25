from __future__ import annotations

import argparse
import os

import cv2
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


def main(dataset: str, method: str, seg_tag: str | None, query_id: str | None, out_dir: str) -> None:
    train_desc_h5, test_desc_h5, train_kp_h5, test_kp_h5, image_root = build_paths(
        dataset, method, seg_tag
    )

    df = pd.read_csv(DATAFRAME_PATH.format(dataset))
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    if query_id is None:
        query_id = str(test_df.iloc[0]['image_id'])

    # choose a training image with the same identity if possible
    row = test_df[test_df['image_id'].astype(str) == str(query_id)]
    if row.empty:
        raise ValueError(f"Query ID {query_id} not found in test split")
    query_identity = row['identity'].iloc[0]
    cand_df = train_df[train_df['identity'] == query_identity]
    if cand_df.empty:
        cand_row = train_df.iloc[0]
    else:
        cand_row = cand_df.iloc[0]
    candidate_id = str(cand_row['image_id'])
    candidate_identity = cand_row['identity']

    # load images from identity subfolders
    query_path = os.path.join(image_root, query_identity, f"{query_id}.jpg")
    cand_path = os.path.join(image_root, candidate_identity, f"{candidate_id}.jpg")
    if not os.path.exists(query_path):
        raise FileNotFoundError(query_path)
    if not os.path.exists(cand_path):
        raise FileNotFoundError(cand_path)
    query_img = io.load_image(query_path)
    cand_img = io.load_image(cand_path)

    # load keypoints and descriptors
    q_kp = io.load_keypoints_h5(test_kp_h5, [query_id]).get(query_id)
    q_desc = io.load_descriptors_h5(test_desc_h5, [query_id]).get(query_id)
    t_kp = io.load_keypoints_h5(train_kp_h5, [candidate_id]).get(candidate_id)
    t_desc = io.load_descriptors_h5(train_desc_h5, [candidate_id]).get(candidate_id)

    if q_kp is None or q_desc is None:
        raise ValueError(f"Features for query ID {query_id} not found in HDF5 stores")
    if t_kp is None or t_desc is None:
        raise ValueError(f"Features for candidate ID {candidate_id} not found in HDF5 stores")

    os.makedirs(out_dir, exist_ok=True)

    # visualise keypoints
    q_kp_vis, _ = keypoints.draw_keypoints(query_img, q_kp)
    io.save_image(os.path.join(out_dir, f"{query_id}_keypoints.png"), q_kp_vis)
    t_kp_vis, _ = keypoints.draw_keypoints(cand_img, t_kp)
    io.save_image(os.path.join(out_dir, f"{candidate_id}_keypoints.png"), t_kp_vis)

    # visualise first descriptor of the query image
    desc_vis, _ = vis_desc.visualize_descriptor(q_desc[0])
    io.save_image(os.path.join(out_dir, f"{query_id}_descriptor.png"), desc_vis)

    # match and visualise correspondences
    norm = cv2.NORM_HAMMING if q_desc.dtype == 'uint8' else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm, crossCheck=True)
    matches = matcher.match(q_desc, t_desc)
    match_vis, _ = matching.draw_matches(query_img, q_kp, cand_img, t_kp, matches)
    io.save_image(os.path.join(out_dir, f"{query_id}_matches.png"), match_vis)

    print(f"Visualisations saved to '{out_dir}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate example visualisations')
    parser.add_argument('--dataset', default='ATRW', help='Dataset name')
    parser.add_argument('--method', default='disk', help='Feature extraction method')
    parser.add_argument('--seg_tag', help='Segmentation tag (e.g. segmented or unsegmented)')
    parser.add_argument('--query_id', help='Test image identifier')
    parser.add_argument('--out_dir', default='visualizations', help='Output directory')
    args = parser.parse_args()

    main(args.dataset, args.method, args.seg_tag, args.query_id, args.out_dir)