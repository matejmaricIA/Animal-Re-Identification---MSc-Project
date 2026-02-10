import os
import pandas as pd
import h5py
import numpy as np
import argparse
import sys
from typing import Callable
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from constants import (
    N_COMPONENTS_GMM,
    N_COMPONENTS_PCA,
    MAX_GMM_DESCRIPTORS,
    MAX_DESCRIPTORS_PER_IMAGE,
    MODEL_PATH,
    PCA_PATH,
    GMM_PATH,
    FISHER_VECTORS,
)
from feature_extraction import (
    extract_features,
    extract_features_keynet_hardnet_faster,
    extract_features_lightglue,
)
from utility_functions import load_stuff, save_stuff


def descriptor_dir(base_dir: str, method: str, split: str, seg_tag: str) -> str:
    """Build path to the descriptor directory for a given split and method."""
    return os.path.join(base_dir, f"feature_descriptors_{split}_{method}_{seg_tag}")


def feature_descriptor_dir(base_dir: str, method_name: str, split_name: str, seg_tag: str) -> str:
    """Resolve descriptor output dir for train/test/full split conventions."""
    if split_name in {"train", "test"}:
        return descriptor_dir(base_dir, method_name, split_name, seg_tag)
    if split_name == "full":
        return f"{base_dir}/feature_descriptors_{method_name}_{seg_tag}_full/"
    raise ValueError(f"Unsupported split_name: {split_name}")


def ensure_local_descriptors(image_items, method_name: str, out_dir: str) -> None:
    """Compute local descriptors/keypoints when they are not cached yet."""
    if os.path.isdir(out_dir):
        return
    if method_name == "disk":
        extract_features(image_items, MODEL_PATH, out_dir)
    elif method_name == "keynet_hardnet":
        extract_features_keynet_hardnet_faster(image_items, out_dir)
    elif method_name in {"lightglue", "aliked"}:
        extract_features_lightglue(image_items, out_dir, feature_type="aliked")
    elif method_name == "superpoint":
        extract_features_lightglue(image_items, out_dir, feature_type="superpoint")
    else:
        print(f"[ERROR] Unsupported feature method: {method_name}")
        sys.exit(1)


def load_or_train_fisher_vectors(
    *,
    ds_tag: str,
    method_name: str,
    cache_suffix: str,
    descriptors: dict | None = None,
    descriptors_loader: Callable[[], dict] | None = None,
):
    """Load cached Fisher vectors or train PCA/GMM and compute them."""
    pca_path = PCA_PATH.format(ds_tag, method_name, cache_suffix)
    gmm_path = GMM_PATH.format(ds_tag, method_name, cache_suffix)
    fv_path = FISHER_VECTORS.format(ds_tag, method_name, cache_suffix)
    if os.path.exists(pca_path) and os.path.exists(gmm_path) and os.path.exists(fv_path):
        return load_stuff(pca_path, gmm_path, fv_path)

    if descriptors is None:
        if descriptors_loader is None:
            raise ValueError("descriptors or descriptors_loader must be provided")
        descriptors = descriptors_loader()

    desc_stack = stack_all_descriptors(descriptors)
    pca = train_pca(desc_stack)
    gmm = train_gmm(pca.transform(desc_stack))
    fisher_vectors = compute_fisher_vectors(descriptors, pca, gmm)
    save_stuff(pca, gmm, fisher_vectors, (pca_path, gmm_path, fv_path))
    return pca, gmm, fisher_vectors


def load_descriptors(descriptors_file):
    data = {}
    with h5py.File(descriptors_file, 'r') as df:
        for key in df.keys():
            descriptors = np.array(df[key]).astype(np.float32)
            data[key] = descriptors
    print(f"Loaded dataset with {len(data)} images.")
    return data

def load_keypoints(keypoints_file):
    data = {}
    
    with h5py.File(keypoints_file, 'r') as f:
        for key in f.keys():
            keypoints = np.array(f[key])
            data[key] = keypoints
                
        print(f"Loaded keypoints for {len(data)} images from {keypoints_file}")
        return data

def stack_all_descriptors(descriptors, max_samples=MAX_GMM_DESCRIPTORS, per_image_max = MAX_DESCRIPTORS_PER_IMAGE):
    """Stack descriptors from all images.

    If ``max_samples`` is provided, a random subset of descriptors with at most
    ``max_samples`` rows will be returned. This reduces memory usage when
    training PCA and GMM on large datasets.
    """

    #arrays = [d for d in descriptors.values() if len(d) > 0]
    rng = np.random.default_rng(42)
    arrays = []
    for d in descriptors.values():
        if len(d) == 0:
            continue
        if per_image_max is not None and len(d) > per_image_max:
            idx = rng.choice(len(d), size=per_image_max, replace=False)
            arr = d[idx]
        else:
            arr = d
        arrays.append(arr.astype(np.float32))
    if not arrays:
        return np.empty((0, 0))

    if max_samples is None:
        return np.vstack(arrays)

    #rng = np.random.default_rng()

    #lengths = np.array([len(a) for a in arrays])
    #total = lengths.sum()

    # If the total number of descriptors is less than the desired sample,
    # fall back to using all descriptors.

    stacked = np.vstack(arrays)
    total = stacked.shape[0]
    #if total <= max_samples:
    #    print('Using all descriptors, total:', total)
    #    return np.vstack(arrays)
    if not max_samples or max_samples > total:
        print('Using all descriptors, total:', total)
        return stacked
    

    idx = rng.choice(total, size=max_samples, replace=False)
    stacked = stacked[idx]
    print(f"Number of total descriptors: {stacked.shape}")
    return stacked

def train_pca(stacked_descriptors, n_components = N_COMPONENTS_PCA):
    print("Training PCA...")
    stacked_descriptors = stacked_descriptors.astype(np.float32, copy=False)
    pca = PCA(n_components = n_components, whiten = True)
    pca.fit(stacked_descriptors)
    print("PCA training completed.")
    return pca

def train_gmm(reduced_stacked_descs, n_components = N_COMPONENTS_GMM):
    print("Training GMM...")
    reduced_stacked_descs = reduced_stacked_descs.astype(np.float32, copy=False)
    gmm = GaussianMixture(n_components = n_components, covariance_type = 'diag')
    gmm.fit(reduced_stacked_descs)
    return gmm

def compute_fisher_vector(reduced_stacked_descs, gmm):

    means = gmm.means_  # Shape: (K, D)
    covariances = gmm.covariances_  # Shape: (K, D)
    weights = gmm.weights_  # Shape: (K,)

    N, D = reduced_stacked_descs.shape
    K = len(weights)

    # Compute responsibilities
    responsibilities = gmm.predict_proba(reduced_stacked_descs)  # Shape: (N, K)

    # Initialize Fisher Vector components
    fisher_mean = np.zeros((K, D), dtype=np.float32)
    fisher_var = np.zeros((K, D), dtype=np.float32)

    # Compute mean and variance gradients
    for k in range(K):
        prob_k = responsibilities[:, k]  # Shape: (N,)
        diff = reduced_stacked_descs - means[k]  # Shape: (N, D)
        cov_k = covariances[k]

        # Looking back on how the implementation of fisher vectors should be done I think that this is wrong... Must investigate furhter.
        #fisher_mean[k] = np.sum(prob_k[:, np.newaxis] * diff / np.sqrt(covariances[k]), axis=0)
        #fisher_var[k] = np.sum(prob_k[:, np.newaxis] * (diff ** 2 - covariances[k]) / (2 * covariances[k] ** 1.5), axis=0)

        fisher_mean[k] = (1.0 / (N * np.sqrt(weights[k]))) * np.sum(
            prob_k[:, np.newaxis] * diff / np.sqrt(cov_k),
            axis=0,
        )
        # Old (non-standard) second-order term kept for reference:
        # fisher_var[k] = (1.0 / (N * np.sqrt(2 * weights[k]))) * np.sum(
        #     prob_k[:, np.newaxis] * (diff ** 2 - cov_k) / (2 * cov_k ** 1.5),
        #     axis=0,
        # )
        term = (diff * diff) / (cov_k + 1e-12) - 1.0
        fisher_var[k] = (1.0 / (N * np.sqrt(2 * weights[k]))) * np.sum(
            prob_k[:, np.newaxis] * term,
            axis=0,
        )

    # Flatten and concatenate mean and variance gradients
    fisher_vector = np.concatenate([fisher_mean.flatten(), fisher_var.flatten()])

    # Apply power normalization
    fisher_vector = np.sign(fisher_vector) * np.sqrt(np.abs(fisher_vector))

    # Apply L2 normalization
    fisher_vector /= np.linalg.norm(fisher_vector)

    return fisher_vector

def compute_fisher_vectors(image_descriptors, pca, gmm):
    fisher_vectors = {}

    fv_len = 2 * gmm.n_components * pca.n_components_ #testing

    for image_id, descriptors in image_descriptors.items():
        if descriptors.shape[0] == 0:
            print(f"Skipping image {image_id}: no descriptors found")
            fisher_vectors[image_id] = np.zeros(fv_len, dtype=np.float32)
            continue
        
        # Apply PCA
        reduced_descs = pca.transform(descriptors)

        # Computer fisher vector
        fisher_vector = compute_fisher_vector(reduced_descs, gmm)
        fisher_vectors[image_id] = fisher_vector

    return fisher_vectors


if __name__ == '__main__':
    print('...')
