"""Search for optimal descriptor weights on multiple datasets.

This module provides a command line interface that runs an Optuna search to
find weights for combining Fisher vectors, colour descriptors, global
embeddings and shape descriptors. The script assumes that pre-computed
features are stored under ``data/<dataset>/`` using the following file names::

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
from feature_aggregation import load_descriptors, load_keypoints
from geometric_verification import compute_geometric_similarity
from constants import SAVE_TRAIN_DESCRIPTORS_PATH, SAVE_TEST_DESCRIPTORS_PATH

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


def optimise_dataset(dataset: str, trials: int = 50, use_gv: bool = False) -> None:
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
    """

    base_dir = os.path.join("./data", dataset)

    # Load metadata for labels
    df = pd.read_csv(os.path.join(base_dir, "processed_metadata.csv"))
    df["image_id"] = df["image_id"].astype(str)
    train_df = df[df["split"].str.lower() != "test"].copy()
    test_df = df[df["split"].str.lower() == "test"].copy()
    train_labels = dict(zip(train_df["image_id"], train_df["identity"]))
    test_labels = dict(zip(test_df["image_id"], test_df["identity"]))

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
            if name == "fisher":
                continue
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


def optimise_all(datasets: Iterable[str] = DATASETS, trials: int = 50, use_gv: bool = False) -> None:
    for ds in datasets:
        optimise_dataset(ds, trials=trials, use_gv=use_gv)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimise descriptor weights")
    parser.add_argument("dataset", nargs="?", help="Name of the dataset to optimise")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--all", action="store_true", help="Optimise all known datasets")
    parser.add_argument("--use-gv", action="store_true", help="Enable geometric verification")

    args = parser.parse_args()

    if args.all:
        optimise_all(trials=args.trials, use_gv=args.use_gv)
    elif args.dataset:
        optimise_dataset(args.dataset, trials=args.trials, use_gv=args.use_gv)
    else:
        parser.print_help()
