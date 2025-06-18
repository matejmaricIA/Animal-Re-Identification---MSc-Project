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
import faiss
from utils.distance_utils import fisher_distance

_FAISS_INDEX   = None

def _build_faiss_index(vecs):
    dim = vecs.shape[1]
    cpu = faiss.IndexFlatL2(dim)
    if faiss.get_num_gpus() > 0:
        gpu = faiss.index_cpu_to_all_gpus(cpu)
        print(f"[calibrate] FAISS GPU index built on {faiss.get_num_gpus()} GPU(s)")
        idx = gpu
    else:
        print("[calibrate] FAISS CPU index built")
        idx = cpu
    idx.add(vecs.astype("float32", copy=False))
    return idx

def _knn(vecs, query, k):
    """
    Return (indices, distances) of the k nearest neighbours of `query`.

    The first time it is called it will build whichever backend is
    available and cache it; subsequent calls are cheap.
    """
    global _FAISS_INDEX

    if _FAISS_INDEX is None:
        _FAISS_INDEX = _build_faiss_index(vecs)


    if _FAISS_INDEX is not None:
        D, I = _FAISS_INDEX.search(query.astype("float32", copy=False), k)
        return I[0], D[0]




def _sample_pairs(fv, k_pos=5, k_neg=5, far_frac=0.8, echo=None):
    img_ids = list(fv.keys())
    vecs    = np.stack([fv[i] for i in img_ids])          # (N, D)
    N       = len(img_ids)

    for idx, img_id in enumerate(img_ids):
        if echo and idx % 50 == 0:
            echo(idx, N)

        query = vecs[idx : idx + 1]

        nn_idx, _ = _knn(vecs, query, k_pos + 1)          # nearest
        for j_idx in nn_idx[1:]:
            if j_idx <= idx:
                continue
            fd = fisher_distance(query[0], vecs[j_idx])
            yield img_ids[idx], img_ids[j_idx], fd

        need_neg = k_neg
        attempts = 0
        while need_neg:
            j_idx = np.random.randint(N)
            if j_idx == idx or j_idx < idx:
                attempts += 1
                if attempts > 50:  # avoid infinite loop
                    print(f"[calibrate] WARNING: too many attempts for {img_id} (idx={idx})")
                    break
                continue
            fd = fisher_distance(query[0], vecs[j_idx])
            if fd < 0.5 and attempts < 20:
                attempts += 1# not really “far”
                continue
            yield img_ids[idx], img_ids[j_idx], fd
            need_neg -= 1
            attempts += 1
        
def calibrate(dataset_tag, fisher_vecs,
              descriptors, keypoints,
              cache_dir="./calibration_cache",
              k_pos=5, k_neg=5, percentile=90, max_images=2500, max_pairs = 10000):
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
    fd_05, fd_95 = np.percentile(df.fd, [5, 95])       # two-tail
    I90 = np.percentile(df.inliers[df.inliers > 0], 90) or 1
    params = dict(fd_min=float(fd_05), fd_90=float(fd_95), I90=int(I90))
    with cache_file.open("w") as f:
        json.dump(params, f, indent=2)
    set_dataset_calibration(**params)

    print(f"[calibrate] done in {time.time() - start:.1f}s → {params}")
    return params