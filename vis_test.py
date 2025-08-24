#!/usr/bin/env python3
"""
Example pipeline for feature‑based animal re‑identification with optional
visualisation.  Paths follow the layout produced by the project’s training
scripts (HDF5 for keypoints/descriptors and a single pickle for training
Fisher vectors).
"""
from __future__ import annotations

import pickle
import pandas as pd

from feature_aggregation import compute_fisher_vectors
from predict import classify_test_images_with_geometric_verification
from visualization_suite import io

from constants import (
    FISHER_VECTORS, PCA_PATH, GMM_PATH
)

# ---------------------------------------------------------------------
# 1. Configure dataset‑specific paths
# ---------------------------------------------------------------------
DATASET   = "ATRW"
METHOD    = "disk"
SEG_TAG   = "segmented"               # or "unsegmented"

BASE_DIR  = f"data/{DATASET}"
IMAGE_ROOT = f"{BASE_DIR}/segmented_dataset"  # or dataset/

TRAIN_KP_H5   = f"{BASE_DIR}/feature_descriptors_train_{METHOD}_{SEG_TAG}/keypoints.h5"
TRAIN_DESC_H5 = f"{BASE_DIR}/feature_descriptors_train_{METHOD}_{SEG_TAG}/descriptors.h5"
TEST_KP_H5    = f"{BASE_DIR}/feature_descriptors_test_{METHOD}_{SEG_TAG}/keypoints.h5"
TEST_DESC_H5  = f"{BASE_DIR}/feature_descriptors_test_{METHOD}_{SEG_TAG}/descriptors.h5"

def load_pickle(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)

# ---------------------------------------------------------------------
# 2. Run classification with optional retrieval visualisation
# ---------------------------------------------------------------------
def run_classification():
    df = pd.read_csv(f"{BASE_DIR}/processed_metadata.csv")
    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]
    train_labels = dict(zip(train_df["image_id"], train_df["identity"]))

    train_ids = train_df["image_id"].astype(str).tolist()
    test_ids  = test_df["image_id"].astype(str).tolist()

    # HDF5 → dictionaries
    train_kp   = io.load_keypoints_h5(TRAIN_KP_H5, train_ids)
    test_kp    = io.load_keypoints_h5(TEST_KP_H5, test_ids)
    train_desc = io.load_descriptors_h5(TRAIN_DESC_H5, train_ids)
    test_desc  = io.load_descriptors_h5(TEST_DESC_H5, test_ids)

    # Load training Fisher vectors and PCA/GMM models
    train_fv = load_pickle(FISHER_VECTORS.format(DATASET, METHOD, SEG_TAG))
    pca = load_pickle(PCA_PATH.format(DATASET, METHOD, SEG_TAG))
    gmm = load_pickle(GMM_PATH.format(DATASET, METHOD, SEG_TAG))

    # Compute test Fisher vectors on the fly
    test_fv = compute_fisher_vectors(test_desc, pca, gmm)

    preds = classify_test_images_with_geometric_verification(
        test_fv, train_fv,
        test_kp, train_kp,
        test_desc, train_desc,
        train_labels,
        visualize=True,
        image_root=IMAGE_ROOT,
        train_kp_h5=TRAIN_KP_H5,
        train_desc_h5=TRAIN_DESC_H5,
        test_kp_h5=TEST_KP_H5,
        test_desc_h5=TEST_DESC_H5,
        vis_output_dir=f"{BASE_DIR}/evaluations/vis",
    )
    print(preds)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_classification()
