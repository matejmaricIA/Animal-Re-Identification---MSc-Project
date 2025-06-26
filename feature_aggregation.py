import pandas as pd
import h5py
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from constants import N_COMPONENTS_GMM, N_COMPONENTS_PCA, MAX_GMM_DESCRIPTORS


def load_descriptors(descriptors_file):
    data = {}
    with h5py.File(descriptors_file, 'r') as df:
        for key in df.keys():
            descriptors = np.array(df[key])
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

def stack_all_descriptors(descriptors, max_samples=None):
    """Stack descriptors from all images.

    If ``max_samples`` is provided, a random subset of descriptors with at most
    ``max_samples`` rows will be returned. This reduces memory usage when
    training PCA and GMM on large datasets.
    """

    arrays = [d for d in descriptors.values() if len(d) > 0]
    if not arrays:
        return np.empty((0, 0))

    if max_samples is None:
        return np.vstack(arrays)

    rng = np.random.default_rng()

    lengths = np.array([len(a) for a in arrays])
    total = lengths.sum()

    # If the total number of descriptors is less than the desired sample,
    # fall back to using all descriptors.
    if total <= max_samples:
        print('Using all descriptors, total:', total)
        return np.vstack(arrays)

    samples = []
    for arr, L in zip(arrays, lengths):
        n = int(round(L / total * max_samples))
        n = min(n, len(arr))
        if n > 0:
            idx = rng.choice(len(arr), size=n, replace=False)
            samples.append(arr[idx])
    

    stacked = np.vstack(samples)
    if stacked.shape[0] > max_samples:
        idx = rng.choice(stacked.shape[0], size=max_samples, replace=False)
        stacked = stacked[idx]
    print(f"Number of total descriptors: {stacked.shape}")
    return stacked

def train_pca(stacked_descriptors, n_components = N_COMPONENTS_PCA):
    print("Training PCA...")
    pca = PCA(n_components = n_components, whiten = True)
    pca.fit(stacked_descriptors)
    print("PCA training completed.")
    return pca

def train_gmm(reduced_stacked_descs, n_components = N_COMPONENTS_GMM):
    print("Training GMM...")
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

        # Looking back on how the implementation of fisher vectors should be done I think that this is wrong... Must investigate furhter.
        #fisher_mean[k] = np.sum(prob_k[:, np.newaxis] * diff / np.sqrt(covariances[k]), axis=0)
        #fisher_var[k] = np.sum(prob_k[:, np.newaxis] * (diff ** 2 - covariances[k]) / (2 * covariances[k] ** 1.5), axis=0)

        fisher_mean[k] = (1.0 / (N * np.sqrt(weights[k]))) * np.sum(prob_k[:, np.newaxis] * diff / np.sqrt(covariances[k]), axis=0)
        fisher_var[k] = (1.0 / (N * np.sqrt(2 * weights[k]))) * np.sum(prob_k[:, np.newaxis] * (diff ** 2 - covariances[k]) / (2 * covariances[k] ** 1.5), axis=0)

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