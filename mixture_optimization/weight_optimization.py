"""Search for optimal descriptor weights on multiple datasets.

This module provides a command line interface that runs an Optuna search to
find weights for combining Fisher vectors, colour descriptors, global
embeddings and shape descriptors. If any of the required descriptor pickles
are missing they are generated on-the-fly before optimisation using the
standard pipeline. Features are stored under ``data/<dataset>/`` using the
following file names::

    fisher_vectors_{split}.pkl
    color_descriptors_{split}.pkl
    global_embeddings_{split}.pkl
    shape_descriptors_{split}.pkl

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
from mixture_optimization.combine_descriptors import combine_descriptor_dicts
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
from color_descriptors import (
    compute_color_descriptors,
    standardize as standardize_colors,
    normalize_hsv,
)
from global_embedding import extract_global_embeddings
from shape_descriptors import (
    compute_shape_descriptors,
    standardize as standardize_shapes,
)
from constants import (
    SAVE_TRAIN_DESCRIPTORS_PATH,
    SAVE_TEST_DESCRIPTORS_PATH,
    SAVE_TRAIN_DESCRIPTORS_FOLDER,
    SAVE_TEST_DESCRIPTORS_FOLDER,
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


def load_features(base_dir: str, split: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load any available features for ``split`` from ``base_dir``.

    This function looks for pickle files with conventional names. Only files
    that exist are loaded, meaning the optimisation can run with arbitrary
    combinations of descriptors.
    """

    feature_dicts: Dict[str, Dict[str, np.ndarray]] = {}

    path_map = {
        "fisher": os.path.join(base_dir, f"fisher_vectors_{split}.pkl"),
        "color": os.path.join(base_dir, f"color_descriptors_{split}.pkl"),
        "global": os.path.join(base_dir, f"global_embeddings_{split}.pkl"),
        "shape": os.path.join(base_dir, f"shape_descriptors_{split}.pkl"),
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


def ensure_feature_files(
    dataset: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str = "disk",
) -> None:
    """Create descriptor pickles on disk if they are missing.

    This covers Fisher vectors, colour descriptors, global embeddings and
    shape descriptors. Existing files are left untouched. When ``method`` is
    ``'ensamble'`` Fisher vectors are computed separately for DISK,
    KeyNet+HardNet and LightGlue features and combined using
    :data:`constants.ENSEMBLE_WEIGHTS`."""

    base_dir = os.path.join("./data", dataset)

    # Image paths for convenience
    train_paths = get_image_paths(train_df)
    test_paths = get_image_paths(test_df)

    # ------------------------------------------------------------------
    # Colour descriptors
    color_tr_path = os.path.join(base_dir, "color_descriptors_train.pkl")
    color_te_path = os.path.join(base_dir, "color_descriptors_test.pkl")
    if not (os.path.exists(color_tr_path) and os.path.exists(color_te_path)):
        color_tr = compute_color_descriptors(train_paths)
        color_te = compute_color_descriptors(test_paths)
        color_tr, mean_c, std_c = standardize_colors(color_tr)
        color_te, _, _ = standardize_colors(color_te, mean_c, std_c)
        color_tr = normalize_hsv(color_tr)
        color_te = normalize_hsv(color_te)
        with open(color_tr_path, "wb") as f:
            pickle.dump(color_tr, f)
        with open(color_te_path, "wb") as f:
            pickle.dump(color_te, f)

    # ------------------------------------------------------------------
    # Global embeddings
    emb_tr_path = os.path.join(base_dir, "global_embeddings_train.pkl")
    emb_te_path = os.path.join(base_dir, "global_embeddings_test.pkl")
    if not (os.path.exists(emb_tr_path) and os.path.exists(emb_te_path)):
        train_map = dict(zip(train_df["image_id"].astype(str), train_paths))
        test_map = dict(zip(test_df["image_id"].astype(str), test_paths))
        emb_tr = extract_global_embeddings(train_map)
        emb_te = extract_global_embeddings(test_map)
        emb_tr, mean_e, std_e = _standardize_generic(emb_tr)
        emb_te, _, _ = _standardize_generic(emb_te, mean_e, std_e)
        with open(emb_tr_path, "wb") as f:
            pickle.dump(emb_tr, f)
        with open(emb_te_path, "wb") as f:
            pickle.dump(emb_te, f)

    # ------------------------------------------------------------------
    # Shape descriptors
    shape_tr_path = os.path.join(base_dir, "shape_descriptors_train.pkl")
    shape_te_path = os.path.join(base_dir, "shape_descriptors_test.pkl")
    if not (os.path.exists(shape_tr_path) and os.path.exists(shape_te_path)):
        shape_tr = compute_shape_descriptors(train_paths)
        shape_te = compute_shape_descriptors(test_paths)
        shape_tr, mean_s, std_s = standardize_shapes(shape_tr)
        shape_te, _, _ = standardize_shapes(shape_te, mean_s, std_s)
        with open(shape_tr_path, "wb") as f:
            pickle.dump(shape_tr, f)
        with open(shape_te_path, "wb") as f:
            pickle.dump(shape_te, f)

    # ------------------------------------------------------------------
    # Fisher vectors
    fv_tr_path = os.path.join(base_dir, "fisher_vectors_train.pkl")
    fv_te_path = os.path.join(base_dir, "fisher_vectors_test.pkl")
    if not (os.path.exists(fv_tr_path) and os.path.exists(fv_te_path)):
        if method == "ensamble":
            print('Using ensamble method')
            methods = ["disk", "keynet_hardnet", "lightglue"]
            fv_tr_list: List[Dict[str, np.ndarray]] = []
            fv_te_list: List[Dict[str, np.ndarray]] = []
            for m in methods:
                if m == "disk":
                    desc_tr_dir = SAVE_TRAIN_DESCRIPTORS_FOLDER.format(dataset)
                    desc_te_dir = SAVE_TEST_DESCRIPTORS_FOLDER.format(dataset)
                    desc_tr_path = SAVE_TRAIN_DESCRIPTORS_PATH.format(dataset)
                    desc_te_path = SAVE_TEST_DESCRIPTORS_PATH.format(dataset)
                    if not (os.path.exists(desc_tr_path) and os.path.exists(desc_te_path)):
                        extract_features(train_paths, MODEL_PATH, desc_tr_dir)
                        extract_features(test_paths, MODEL_PATH, desc_te_dir)
                else:
                    desc_tr_dir = os.path.join(base_dir, f"feature_descriptors_train_{m}")
                    desc_te_dir = os.path.join(base_dir, f"feature_descriptors_test_{m}")
                    desc_tr_path = os.path.join(desc_tr_dir, "descriptors.h5")
                    desc_te_path = os.path.join(desc_te_dir, "descriptors.h5")
                    if not (os.path.exists(desc_tr_path) and os.path.exists(desc_te_path)):
                        if m == "keynet_hardnet":
                            extract_features_keynet_hardnet_faster(train_paths, desc_tr_dir)
                            extract_features_keynet_hardnet_faster(test_paths, desc_te_dir)
                        elif m == "lightglue":
                            extract_features_lightglue(train_paths, desc_tr_dir)
                            extract_features_lightglue(test_paths, desc_te_dir)

                train_desc_m = load_descriptors(desc_tr_path)
                test_desc_m = load_descriptors(desc_te_path)
                stacked = stack_all_descriptors(train_desc_m)
                pca = train_pca(stacked)
                gmm = train_gmm(pca.transform(stacked))
                fv_tr_m = compute_fisher_vectors(train_desc_m, pca, gmm)
                fv_te_m = compute_fisher_vectors(test_desc_m, pca, gmm)
                with open(PCA_PATH.format(dataset, m), "wb") as f:
                    pickle.dump(pca, f)
                with open(GMM_PATH.format(dataset, m), "wb") as f:
                    pickle.dump(gmm, f)
                with open(FISHER_VECTORS.format(dataset, m), "wb") as f:
                    pickle.dump(fv_tr_m, f)
                fv_tr_list.append(fv_tr_m)
                fv_te_list.append(fv_te_m)

            fv_tr = combine_fisher_vectors(fv_tr_list, ENSEMBLE_WEIGHTS)
            fv_te = combine_fisher_vectors(fv_te_list, ENSEMBLE_WEIGHTS)
        else:
            desc_tr_path = SAVE_TRAIN_DESCRIPTORS_PATH.format(dataset)
            desc_te_path = SAVE_TEST_DESCRIPTORS_PATH.format(dataset)
            if not (os.path.exists(desc_tr_path) and os.path.exists(desc_te_path)):
                extract_features(train_paths, MODEL_PATH, SAVE_TRAIN_DESCRIPTORS_FOLDER.format(dataset))
                extract_features(test_paths, MODEL_PATH, SAVE_TEST_DESCRIPTORS_FOLDER.format(dataset))
            train_desc = load_descriptors(desc_tr_path)
            test_desc = load_descriptors(desc_te_path)
            stacked = stack_all_descriptors(train_desc)
            pca = train_pca(stacked)
            reduced = pca.transform(stacked)
            gmm = train_gmm(reduced)
            fv_tr = compute_fisher_vectors(train_desc, pca, gmm)
            fv_te = compute_fisher_vectors(test_desc, pca, gmm)
            with open(PCA_PATH.format(dataset, "auto"), "wb") as f:
                pickle.dump(pca, f)
            with open(GMM_PATH.format(dataset, "auto"), "wb") as f:
                pickle.dump(gmm, f)

        with open(fv_tr_path, "wb") as f:
            pickle.dump(fv_tr, f)
        with open(fv_te_path, "wb") as f:
            pickle.dump(fv_te, f)


def load_local_features(dataset: str, split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load keypoints and local descriptors for geometric verification."""

    if split.lower() == "train":
        desc_path = SAVE_TRAIN_DESCRIPTORS_PATH.format(dataset)
    else:
        desc_path = SAVE_TEST_DESCRIPTORS_PATH.format(dataset)
    kp_path = desc_path.replace("descriptors.h5", "keypoints.h5")

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

    # Load metadata for labels
    df = pd.read_csv(os.path.join(base_dir, "processed_metadata.csv"))
    df["image_id"] = df["image_id"].astype(str)
    train_df = df[df["split"].str.lower() != "test"].copy()
    test_df = df[df["split"].str.lower() == "test"].copy()
    train_labels = dict(zip(train_df["image_id"], train_df["identity"]))
    test_labels = dict(zip(test_df["image_id"], test_df["identity"]))

    # Ensure required descriptor files exist; compute them if necessary
    ensure_feature_files(dataset, train_df, test_df, method=method)

    train_features = load_features(base_dir, "train")
    test_features = load_features(base_dir, "test")

    if use_gv:
        train_descs, train_kp = load_local_features(dataset, "train")
        test_descs, test_kp = load_local_features(dataset, "test")
    else:
        train_descs = test_descs = train_kp = test_kp = {}

    if not train_features:
        raise RuntimeError(f"No pre-computed features found for dataset '{dataset}'")

    descriptor_names = list(train_features.keys())

    def objective(trial: optuna.Trial) -> float:
        # Suggest weight for each descriptor; Fisher gets weight 1 by default
        weights = {name: 1.0 for name in descriptor_names}
        for name in descriptor_names:
            #if name == "fisher":
            #    continue
            weights[name] = trial.suggest_float(f"w_{name}", 0.0, 2.0)

        combined_train = combine_descriptor_dicts(train_features, weights)
        combined_test = combine_descriptor_dicts(test_features, weights)

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
) -> None:
    for ds in datasets:
        optimise_dataset(ds, trials=trials, use_gv=use_gv, method=method)


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

    args = parser.parse_args()

    if args.all:
        optimise_all(trials=args.trials, use_gv=args.use_gv, method=args.method)
    elif args.dataset:
        optimise_dataset(args.dataset, trials=args.trials, use_gv=args.use_gv, method=args.method)
    else:
        parser.print_help()