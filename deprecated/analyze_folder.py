import argparse
import os
import shutil
import pickle

import numpy as np
from sklearn.cluster import DBSCAN

import preprocessing
from feature_extraction import extract_features
from feature_aggregation import (
    load_descriptors,
    compute_fisher_vectors,
    stack_all_descriptors,
    train_pca,
    train_gmm,
)
from constants import MODEL_PATH, TMP


def get_or_train_models(model_dir, descriptors):
    """Load PCA and GMM models from ``model_dir`` or train them using
    ``descriptors`` when files are missing."""

    pca_path = os.path.join(model_dir, "pca.pkl")
    gmm_path = os.path.join(model_dir, "gmm.pkl")

    if os.path.exists(pca_path) and os.path.exists(gmm_path):
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        with open(gmm_path, "rb") as f:
            gmm = pickle.load(f)
        return pca, gmm

    os.makedirs(model_dir, exist_ok=True)
    stacked = stack_all_descriptors(descriptors)
    pca = train_pca(stacked)
    gmm = train_gmm(pca.transform(stacked))

    with open(pca_path, "wb") as f:
        pickle.dump(pca, f)
    with open(gmm_path, "wb") as f:
        pickle.dump(gmm, f)

    return pca, gmm


def cluster_fisher_vectors(fv_dict, eps=0.3, min_samples=1):
    """Cluster Fisher vectors with DBSCAN using cosine distance."""
    if not fv_dict:
        return np.array([]), []
    image_ids = list(fv_dict.keys())
    features = np.stack(list(fv_dict.values()))
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(features)
    return clustering.labels_, image_ids


def copy_cluster_images(labels, image_ids, src_dir, out_dir):
    for label in set(labels):
        label_dir = os.path.join(out_dir, f"individual_{label}" if label != -1 else "noise")
        os.makedirs(label_dir, exist_ok=True)
        for img_id, lab in zip(image_ids, labels):
            if lab == label:
                src = os.path.join(src_dir, f"{img_id}.jpg")
                if os.path.exists(src):
                    shutil.copy(src, label_dir)


def main():
    parser = argparse.ArgumentParser(description="Cluster animals in a folder of images")
    parser.add_argument("folder", default = "/data/ds_test/", help="Path to folder containing images")
    parser.add_argument(
        "--model_dir",
        default="./data/ATRW/db",
        help="Directory containing pretrained PCA and GMM models",
    )
    parser.add_argument("--output_dir", default = "Output", help="Optional directory to copy clustered images")
    parser.add_argument("--eps", type=float, default=0.3, help="DBSCAN epsilon")
    parser.add_argument("--min_samples", type=int, default=1, help="DBSCAN min samples")
    parser.add_argument("--use_mantiuk", action="store_true")
    parser.add_argument("--remove_background", action="store_true")
    args = parser.parse_args()

    image_paths = [os.path.join(args.folder, p) for p in os.listdir(args.folder)]
    tmp_dir = preprocessing.preprocess_inference(image_paths, use_mantiuk=args.use_mantiuk, remove_background=args.remove_background)
    processed_paths = [os.path.join(tmp_dir, p) for p in os.listdir(tmp_dir)]

    extract_features(processed_paths, MODEL_PATH, TMP)
    descriptors = load_descriptors(os.path.join(TMP, "descriptors.h5"))

    pca, gmm = get_or_train_models(args.model_dir, descriptors)
    fisher_vectors = compute_fisher_vectors(descriptors, pca, gmm)

    labels, image_ids = cluster_fisher_vectors(fisher_vectors, eps=args.eps, min_samples=args.min_samples)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {num_clusters} unique individuals")

    if args.output_dir:
        copy_cluster_images(labels, image_ids, tmp_dir, args.output_dir)
        print(f"Clustered images copied to {args.output_dir}")


if __name__ == "__main__":
    main()