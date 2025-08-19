"""Search for optimal descriptor weights on multiple datasets.

This module provides a command line interface that runs an Optuna search to
find weights for combining Fisher vectors and global embeddings. If any of
the required descriptor pickles are missing they are generated on-the-fly
before optimisation using the standard pipeline. Features are stored under
``data/<dataset>/`` using the following file names::

    fisher_vectors_{method}_{seg_tag}.pkl
    global_embeddings_{split}.pkl


Only the files that exist are loaded which allows experimentation with any
subset of descriptor types. Results are evaluated using the simple nearest
neighbour classifier from :mod:`predict`. When the ``--use-gv`` flag is
provided, top candidates are further re-ranked using geometric verification
based on local keypoints and descriptors.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterable, List, Mapping

import optuna
import numpy as np
import pandas as pd

from evaluate import evaluate_predictions
from predict import classify_test_images
from mixture_optimization.block_normalization import (
    apply_zscore_and_l2_train_test,
    fuse_blocks_weighted_concat,
)
from feature_aggregation import (
    load_descriptors,
    load_keypoints,
    stack_all_descriptors,
    train_pca,
    train_gmm,
    compute_fisher_vectors,
)
from geometric_verification import compute_geometric_similarity
from feature_extraction import (
    get_image_paths,
    extract_features,
    extract_features_keynet_hardnet_faster,
    extract_features_lightglue,
)
from global_embedding import extract_global_embeddings
from constants import (
    MODEL_PATH,
    PCA_PATH,
    GMM_PATH,
    FISHER_VECTORS,
    ENSEMBLE_WEIGHTS,
)

from utility_functions import combine_fisher_vectors

# List of datasets for convenience when running ``optimise_all``
DATASETS: List[str] = [
    "BelugaID",
    "ATRW",
    "CowDataset",
    "Giraffes",
    "HyenaID2022",
    "IPanda50",
    "roe_derr",
    "wild_boar",
    "SealID",
    "NyalaData",
    "StripeSpotter",
]


def _load_pickle(path: str) -> Dict[str, np.ndarray]:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_features(
    base_dir: str,
    split: str,
    embedding_model: str = "resnet50",
    method: str = "disk",
    seg_tag: str = "unsegmented",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load any available features for ``split`` from ``base_dir``.

    This function looks for pickle files with conventional names. Only files
    that exist are loaded, meaning the optimisation can run with arbitrary
    combinations of descriptors.
    """

    feature_dicts: Dict[str, Dict[str, np.ndarray]] = {}

    path_map = {
        "fisher": os.path.join(base_dir, f"fisher_vectors_{method}_{seg_tag}.pkl"),
        "global": os.path.join(base_dir, f"global_embeddings_{split}.pkl"),
    }

    for name, path in path_map.items():
        if os.path.exists(path):
            feature_dicts[name] = _load_pickle(path)
    return feature_dicts


def _standardize_generic(
    descriptor_dict: Dict[str, np.ndarray],
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Z-score standardisation for arbitrary descriptor dictionaries."""

    ids = list(descriptor_dict.keys())
    if not ids:
        return descriptor_dict, mean, std

    mat = np.stack([descriptor_dict[i] for i in ids])
    if mean is None:
        mean = mat.mean(axis=0)
    if std is None:
        std = mat.std(axis=0) + 1e-6
    mat = (mat - mean) / std
    return dict(zip(ids, mat)), mean, std

def suggest_weights(
    trial: optuna.Trial,
    descriptor_names: Iterable[str],
    fisher_min: float | None = None,
) -> Dict[str, float]:
    """Draw weights for the descriptors from an Optuna trial.

    Parameters
    ----------
    trial:
        The active Optuna trial used for suggesting weights.
    descriptor_names:
        Iterable with the names of descriptors to weight.
    fisher_min: float, optional
        If provided and the descriptor is ``"fisher"``, this value is used as
        the lower bound instead of ``1e-2``.
    """

    weights: Dict[str, float] = {}
    for name in descriptor_names:
        low = 1e-2
        if name == "fisher" and fisher_min is not None:
            low = max(low, fisher_min)
        weights[name] = trial.suggest_float(f"w_{name}", low, 3.0, log=True)
    return weights

def ensure_feature_files(
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str = "disk",
    embedding_model: str = "resnet50",
    remove_background: bool = False,
) -> None:
    """Create descriptor pickles on disk if they are missing.

    This covers Fisher vectors and global embeddings. Existing files are left
    untouched. When ``method`` is ``'ensamble'`` Fisher vectors are computed
    separately for DISK, KeyNet+HardNet and LightGlue features and combined
    using :data:`constants.ENSEMBLE_WEIGHTS`. All local feature descriptors
    are stored under method-specific folders, mirroring the behaviour in
    ``main.py``.
    """

    base_dir = os.path.join("./data", dataset)
    seg_tag = "segmented" if remove_background else "unsegmented"

    # Image paths for convenience
    train_paths = get_image_paths(train_df, remove_background)
    test_paths = get_image_paths(test_df, remove_background)

    # Global embeddings
    emb_tr_path = os.path.join(base_dir, f"global_embeddings_train_{embedding_model}.pkl")
    emb_te_path = os.path.join(base_dir, f"global_embeddings_test_{embedding_model}.pkl")
    if not (os.path.exists(emb_tr_path) and os.path.exists(emb_te_path)):
        train_map = dict(zip(train_df["image_id"].astype(str), train_paths))
        test_map = dict(zip(test_df["image_id"].astype(str), test_paths))
        emb_tr = extract_global_embeddings(train_map, model_name=embedding_model)
        emb_te = extract_global_embeddings(test_map, model_name=embedding_model)
        emb_tr, mean_e, std_e = _standardize_generic(emb_tr)
        emb_te, _, _ = _standardize_generic(emb_te, mean_e, std_e)
        with open(emb_tr_path, "wb") as f:
            pickle.dump(emb_tr, f)
        with open(emb_te_path, "wb") as f:
            pickle.dump(emb_te, f)

    # Fisher vectors (load if present; else train and save, per method)
    fv_tr_path = os.path.join(base_dir, f"fisher_vectors_{method}_{seg_tag}.pkl")
    fv_te_path = os.path.join(base_dir, f"fisher_vectors_{method}_{seg_tag}.pkl")

    if not (os.path.exists(fv_tr_path) and os.path.exists(fv_te_path)):
        methods = ["disk", "keynet_hardnet", "lightglue"] if method == "ensamble" else [method]
        fv_tr_list: List[Dict[str, np.ndarray]] = []
        fv_te_list: List[Dict[str, np.ndarray]] = []

        for m in methods:
            # Method-specific descriptor dirs
            desc_tr_dir = os.path.join(base_dir, f"feature_descriptors_train_{m}_{seg_tag}")
            desc_te_dir = os.path.join(base_dir, f"feature_descriptors_test_{m}_{seg_tag}")
            desc_tr_path = os.path.join(desc_tr_dir, "descriptors.h5")
            desc_te_path = os.path.join(desc_te_dir, "descriptors.h5")

            # Ensure local descriptors exist
            if not (os.path.exists(desc_tr_path) and os.path.exists(desc_te_path)):
                if m == "disk":
                    extract_features(train_paths, MODEL_PATH, desc_tr_dir)
                    extract_features(test_paths, MODEL_PATH, desc_te_dir)
                elif m == "keynet_hardnet":
                    extract_features_keynet_hardnet_faster(train_paths, desc_tr_dir)
                    extract_features_keynet_hardnet_faster(test_paths, desc_te_dir)
                elif m == "lightglue":
                    extract_features_lightglue(train_paths, desc_tr_dir)
                    extract_features_lightglue(test_paths, desc_te_dir)
                else:
                    raise ValueError(f"Unsupported method: {m}")

            # Load local descriptors
            train_desc_m = load_descriptors(desc_tr_path)
            test_desc_m  = load_descriptors(desc_te_path)

            # Paths for PCA/GMM and per-method Fisher vectors
            pca_path_m = PCA_PATH.format(dataset, m, seg_tag)
            gmm_path_m = GMM_PATH.format(dataset, m, seg_tag)
            fv_tr_path_m = os.path.join(base_dir, f"fisher_vectors_{m}_{seg_tag}.pkl")
            fv_te_path_m = os.path.join(base_dir, f"fisher_vectors_{m}_{seg_tag}.pkl")

            have_all = os.path.exists(pca_path_m) and os.path.exists(gmm_path_m) \
                    and os.path.exists(fv_tr_path_m) and os.path.exists(fv_te_path_m)

            if have_all:
                # Load PCA/GMM and per-method FVs
                with open(pca_path_m, "rb") as f:
                    pca_m = pickle.load(f)
                with open(gmm_path_m, "rb") as f:
                    gmm_m = pickle.load(f)
                with open(fv_tr_path_m, "rb") as f:
                    fv_tr_m = pickle.load(f)
                with open(fv_te_path_m, "rb") as f:
                    fv_te_m = pickle.load(f)
            else:
                # Train PCA/GMM on TRAIN descriptors and compute both TRAIN/TEST FVs
                stacked = stack_all_descriptors(train_desc_m)
                pca_m = train_pca(stacked)
                gmm_m = train_gmm(pca_m.transform(stacked))
                fv_tr_m = compute_fisher_vectors(train_desc_m, pca_m, gmm_m)
                fv_te_m = compute_fisher_vectors(test_desc_m,  pca_m, gmm_m)

                # Persist models and per-method FVs
                with open(pca_path_m, "wb") as f:
                    pickle.dump(pca_m, f)
                with open(gmm_path_m, "wb") as f:
                    pickle.dump(gmm_m, f)
                with open(fv_tr_path_m, "wb") as f:
                    pickle.dump(fv_tr_m, f)
                with open(fv_te_path_m, "wb") as f:
                    pickle.dump(fv_te_m, f)

            # Accumulate per-method FVs for optional ensemble
            fv_tr_list.append(fv_tr_m)
            fv_te_list.append(fv_te_m)

        # Build final Fisher vectors for the optimiser’s conventional filenames
        if method == "ensamble":
            fv_tr = combine_fisher_vectors(fv_tr_list, ENSEMBLE_WEIGHTS)
            fv_te = combine_fisher_vectors(fv_te_list, ENSEMBLE_WEIGHTS)
        else:
            fv_tr = fv_tr_list[0]
            fv_te = fv_te_list

        with open(fv_tr_path, "wb") as f:
            pickle.dump(fv_tr, f)
        with open(fv_te_path, "wb") as f:
            pickle.dump(fv_te, f)



def load_local_features(
    dataset: str,
    split: str,
    method: str = "disk",
    seg_tag: str = "unsegmented",
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load keypoints and local descriptors for geometric verification.

    When ``method`` is ``"ensamble"`` the DISK features are used for GV, so we
    fall back to loading ``feature_descriptors_*_disk``.
    """

    if method == "ensamble":
        method = "disk"

    base_dir = os.path.join(
        "./data", dataset, f"feature_descriptors_{split.lower()}_{method}_{seg_tag}"
    )
    desc_path = os.path.join(base_dir, "descriptors.h5")
    kp_path = os.path.join(base_dir, "keypoints.h5")

    if os.path.exists(desc_path) and os.path.exists(kp_path):
        return load_descriptors(desc_path), load_keypoints(kp_path)
    return {}, {}


def classify_with_gv(
    test_feats: Dict[str, np.ndarray],
    train_feats: Dict[str, np.ndarray],
    train_labels: Mapping[str, str],
    test_descs: Dict[str, np.ndarray],
    train_descs: Dict[str, np.ndarray],
    test_kp: Dict[str, np.ndarray],
    train_kp: Dict[str, np.ndarray],
    top_n: int = 5,
    gv_candidates: int = 5,
) -> Dict[str, Dict[str, object]]:
    """Nearest neighbour classification with optional geometric verification."""

    train_ids = list(train_feats.keys())
    train_vecs = np.stack([train_feats[i] for i in train_ids])
    train_vecs_norm = train_vecs / np.linalg.norm(train_vecs, axis=1, keepdims=True)
    train_label_arr = np.array([train_labels[i] for i in train_ids])

    predictions: Dict[str, Dict[str, object]] = {}
    for img_id, vec in test_feats.items():
        vec_norm = vec / np.linalg.norm(vec)
        sims = np.dot(train_vecs_norm, vec_norm)
        cand_idx = np.argsort(sims)[::-1][:gv_candidates]

        scores = []
        q_desc = test_descs.get(img_id)
        q_kp = test_kp.get(img_id)
        for idx in cand_idx:
            train_id = train_ids[idx]
            base_dist = 1.0 - sims[idx]
            final_dist = base_dist
            if q_desc is not None and q_kp is not None:
                db_desc = train_descs.get(train_id)
                db_kp = train_kp.get(train_id)
                if db_desc is not None and db_kp is not None:
                    final_dist, _ = compute_geometric_similarity(
                        q_desc, q_kp, db_desc, db_kp, base_dist
                    )
            scores.append((final_dist, train_label_arr[idx]))

        scores.sort(key=lambda x: x[0])
        top = [(1 - d, lbl) for d, lbl in scores[:top_n]]
        predictions[img_id] = {
            "predicted_class": scores[0][1],
            "top_n": top,
        }
    return predictions


def optimise_dataset(
    dataset: str,
    trials: int = 50,
    use_gv: bool = False,
    method: str = "disk",
    embedding_model: str = "resnet50",
    remove_background: bool = False,
) -> None:
    """Run a weight search for a single dataset.

    Parameters
    ----------
    dataset: str
        Name of the dataset.
    trials: int, optional
        Number of Optuna trials.
    use_gv: bool, optional
        If ``True`` geometric verification is applied using available
        keypoints and local descriptors.
    method: str, optional
        Feature extraction method to use when ensuring Fisher vectors. Use
        ``'ensamble'`` to combine DISK, KeyNetHardNet and LightGlue features.
    """

    base_dir = os.path.join("./data", dataset)
    seg_tag = "segmented" if remove_background else "unsegmented"

    # Load metadata for labels
    df = pd.read_csv(os.path.join(base_dir, "processed_metadata.csv"))
    df["image_id"] = df["image_id"].astype(str)
    train_df = df[df["split"].str.lower() != "test"].copy()
    test_df = df[df["split"].str.lower() == "test"].copy()
    train_labels = dict(zip(train_df["image_id"], train_df["identity"]))
    test_labels = dict(zip(test_df["image_id"], test_df["identity"]))

    # Ensure required descriptor files exist; compute them if necessary
    ensure_feature_files(
        dataset,
        train_df,
        test_df,
        method=method,
        embedding_model=embedding_model,
        remove_background=remove_background,
    )

    train_features = load_features(base_dir, "train", embedding_model=embedding_model, method=method, seg_tag=seg_tag)
    test_features = load_features(base_dir, "test", embedding_model=embedding_model, method=method, seg_tag=seg_tag)

    if use_gv:
        train_descs, train_kp = load_local_features(dataset, "train", method, seg_tag)
        test_descs, test_kp = load_local_features(dataset, "test", method, seg_tag)
    else:
        train_descs = test_descs = train_kp = test_kp = {}

    if not train_features:
        raise RuntimeError(f"No pre-computed features found for dataset '{dataset}'")

    descriptor_names = list(train_features.keys())

    # Normalise each descriptor block once per dataset
    normalized_train_blocks: Dict[str, Dict[str, np.ndarray]] = {}
    normalized_test_blocks: Dict[str, Dict[str, np.ndarray]] = {}
    for name in descriptor_names:
        tr_block = train_features[name]
        te_block = test_features.get(name, {})
        skip = name == "fisher"
        norm_tr, norm_te = apply_zscore_and_l2_train_test(
            tr_block, te_block, skip_zscore=skip
        )
        normalized_train_blocks[name] = norm_tr
        normalized_test_blocks[name] = norm_te

    def objective(trial: optuna.Trial) -> float:
        # Suggest weight for each descriptor; Fisher gets weight 1 by default
        #weights = {name: 1.0 for name in descriptor_names}
        #for name in descriptor_names:
            #if name == "fisher":
            #    continue
        #    weights[name] = trial.suggest_float(f"w_{name}", 0.0, 2.0)

        weights = suggest_weights(trial, descriptor_names, 2.0)
        combined_train = fuse_blocks_weighted_concat(normalized_train_blocks, weights)
        combined_test = fuse_blocks_weighted_concat(normalized_test_blocks, weights)

        if use_gv and train_descs and test_descs and train_kp and test_kp:
            preds = classify_with_gv(
                combined_test,
                combined_train,
                train_labels,
                test_descs,
                train_descs,
                test_kp,
                train_kp,
            )
        else:
            preds = classify_test_images(combined_test, combined_train, train_labels)
        metrics = evaluate_predictions(preds, test_labels)
        return float(metrics["accuracy"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    print(f"Best weights for {dataset}: {study.best_params}")


def optimise_all(
    datasets: Iterable[str] = DATASETS,
    trials: int = 50,
    use_gv: bool = False,
    method: str = "disk",
    embedding_model: str = "resnet50",
) -> None:
    for ds in datasets:
        optimise_dataset(
            ds,
            trials=trials,
            use_gv=use_gv,
            method=method,
            embedding_model=embedding_model,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimise descriptor weights")
    parser.add_argument("dataset", nargs="?", help="Name of the dataset to optimise")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--all", action="store_true", help="Optimise all known datasets")
    parser.add_argument("--use-gv", action="store_true", help="Enable geometric verification")
    parser.add_argument(
        "--method",
        type=str,
        default="disk",
        choices=["disk", "ensamble"],
        help="Feature extraction method for Fisher vectors",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="resnet50",
        choices=["resnet50", "megadescriptor-l-384"],
        help="Global embedding model to use",
    )

    args = parser.parse_args()

    if args.all:
        optimise_all(
            trials=args.trials,
            use_gv=args.use_gv,
            method=args.method,
            embedding_model=args.embedding_model,
        )
    elif args.dataset:
        optimise_dataset(
            args.dataset,
            trials=args.trials,
            use_gv=args.use_gv,
            method=args.method,
            embedding_model=args.embedding_model,
        )
    else:
        parser.print_help()