# Calibration for geometric verification on the dataset being used.
import random, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from geometric_verification import (
    match_features_by_descriptors, geometric_verification_ransac,
    set_dataset_calibration                                   
)
from tqdm import tqdm
import time
import random
def _knn(vecs, query, k):
    """
    Return indices, distances of the k nearest neighbours of `query`
    inside `vecs`.  Tries FAISS first, then scikit-learn, then a NumPy
    fallback that allocates only (N, ) memory.
    """
    try:
        import faiss
        index = faiss.IndexFlatL2(vecs.shape[1])
        index.add(vecs.astype('float32', copy=False))
        d, i = index.search(query.astype('float32', copy=False), k)
        return i[0], d[0]
    except ModuleNotFoundError:
        pass

    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
        nn.fit(vecs)
        d, i = nn.kneighbors(query, return_distance=True)
        return i[0], d[0]
    except ModuleNotFoundError:
        pass

    # pure-NumPy fallback (O(ND) per query, but tiny memory)
    dist = np.linalg.norm(vecs - query, axis=1)
    idx  = np.argpartition(dist, k)[:k]
    return idx, dist[idx]

def _sample_pairs(fv, k_pos=5, k_neg=5, far_frac=0.8, echo=None):
    """
    Generator yielding (id1, id2, fd) tuples on the fly.
    If `echo` is a callable, it is invoked with the current image index.
    """
    img_ids = list(fv.keys())
    vecs    = np.stack([fv[i] for i in img_ids])   # (N, D)
    N       = len(img_ids)

    for idx, img_id in enumerate(img_ids):
        if echo and idx % 50 == 0:          # adjustable granularity
            echo(idx, N)

        query = vecs[idx : idx + 1]         # shape (1, D)

        # ---------- k_pos nearest neighbours ----------
        nn_idx, nn_dist = _knn(vecs, query, k_pos + 1)  # +1 to skip self
        for j_idx, d in zip(nn_idx[1:], nn_dist[1:]):
            if j_idx <= idx:                # keep pairs unique
                continue
            yield img_ids[idx], img_ids[j_idx], float(d)

        # ---------- k_neg far random neighbours ----------
        need_neg = k_neg
        attempts = 0
        while need_neg and attempts < 20 * k_neg:
            j_idx = np.random.randint(N)
            if j_idx == idx or j_idx < idx:
                attempts += 1
                continue
            d = np.linalg.norm(query - vecs[j_idx])
            if d < nn_dist.max():           # not far enough
                attempts += 1
                continue
            yield img_ids[idx], img_ids[j_idx], float(d)
            need_neg -= 1
            attempts += 1
        
def calibrate(dataset_tag, fisher_vecs,
              descriptors, keypoints,
              cache_dir="./calibration_cache",
              k_pos=5, k_neg=5, percentile=90, max_images=500, max_pairs = 10000):
    """
    One-shot, label-free calibration with live progress.

    • Shows a tqdm bar if tqdm is available.
    • Otherwise prints a status line every 50 images.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{dataset_tag}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            params = json.load(f)
        set_dataset_calibration(**params)
        print(f"[calibrate] loaded cached params from {cache_file}")
        return params

    img_count = len(fisher_vecs)
        img_ids_full = list(fisher_vecs.keys())
    if len(img_ids_full) > max_images:
        img_ids = random.sample(img_ids_full, max_images)
        print(f"[calibrate] down-sampled images: {len(img_ids_full)} → {max_images}")
        fisher_vecs = {k: fisher_vecs[k] for k in img_ids}
        descriptors  = {k: descriptors[k]  for k in img_ids}
        keypoints    = {k: keypoints[k]    for k in img_ids}
    else:
        img_ids = img_ids_full
    img_count = len(img_ids)
    est_pairs = img_count * (k_pos + k_neg) // 2   # rough upper bound

    rows, start = [], time.time()
    bar = tqdm(total=est_pairs, desc="[calibrate] pairs")

    for id1, id2, fd in _sample_pairs(fisher_vecs, k_pos, k_neg):
        desc1, desc2 = descriptors[id1], descriptors[id2]
        kp1, kp2     = keypoints[id1],    keypoints[id2]

        matches, mkp1, mkp2 = match_features_by_descriptors(
            desc1, desc2, kp1, kp2, ratio_threshold=0.95
        )
        n_inliers, _ = geometric_verification_ransac(
            mkp1, mkp2, inlier_threshold=8, min_matches=0
        )
        rows.append((fd, len(matches), n_inliers))

        if bar:
            bar.update(1)

    if bar:
        bar.close()

    df = pd.DataFrame(rows, columns=["fd", "matches", "inliers"])
    fd_min, fd_90 = df.fd.min(), np.percentile(df.fd, percentile)
    fd_scaled = (df.fd - fd_min) / (fd_90 - fd_min + 1e-9)
    I90 = np.percentile(df.inliers[fd_scaled < 0.5], percentile) or 1

    params = dict(fd_min=float(fd_min), fd_90=float(fd_90), I90=int(I90))
    with cache_file.open("w") as f:
        json.dump(params, f, indent=2)
    set_dataset_calibration(**params)

    print(f"[calibrate] done in {time.time() - start:.1f}s → {params}")
    return params