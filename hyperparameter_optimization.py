import os
import pickle
import time
from typing import List
import pandas as pd
import optuna

from feature_aggregation import (
    load_descriptors,
    load_keypoints,
    stack_all_descriptors,
    train_pca,
    train_gmm,
    compute_fisher_vectors,
)
from predict import classify_test_images_with_geometric_verification
from evaluate import evaluate_predictions
from utility_functions import load_dataset, save_count_results
from constants import MAX_GMM_DESCRIPTORS, ENABLE_MULTISCALE, EVAL_RESULTS_XLSX, MAX_DESCRIPTORS_PER_IMAGE, WILD_DATASET_PATH
import numpy as np

def optimise_dataset(dataset: str, trials: int = 50):
    """Run Optuna hyperparameter search on a single dataset."""
    method = "disk"
    base_dir = f"./data/{dataset}"

    # Load metadata for statistics and labels
    #df = load_dataset(dataset)
    df = pd.read_csv(f'{base_dir}/processed_metadata.csv')
    df["image_id"] = df["image_id"].astype(str)
    train_df = df[df["split"].str.lower() != "test"].copy()
    test_df = df[df["split"].str.lower() == "test"].copy()

    train_labels = dict(zip(train_df["image_id"], train_df["identity"]))
    test_labels = dict(zip(test_df["image_id"], test_df["identity"]))

    # Feature descriptors and keypoints (assumed precomputed)
    train_desc_path = f"{base_dir}/feature_descriptors_train_{method}/descriptors.h5"
    test_desc_path = f"{base_dir}/feature_descriptors_test_{method}/descriptors.h5"
    train_kp_path = f"{base_dir}/feature_descriptors_train_{method}/keypoints.h5"
    test_kp_path = f"{base_dir}/feature_descriptors_test_{method}/keypoints.h5"

    train_descriptors = load_descriptors(train_desc_path)
    test_descriptors = load_descriptors(test_desc_path)
    train_keypoints = load_keypoints(train_kp_path)
    test_keypoints = load_keypoints(test_kp_path)
    #print(df)
    stacked_descriptors_train = stack_all_descriptors(
        train_descriptors, max_samples=MAX_GMM_DESCRIPTORS
    )

    def objective(trial: optuna.Trial):
            #n_pca = trial.suggest_int("n_pca", 128, 128, step=8)
            n_pca = 120
            n_gmm = 128
            #n_gmm = trial.suggest_int("n_gmm", 32, 512, step=32)
            min_inliers = trial.suggest_int("min_inliers", 4, 16, step=2)
            alpha = trial.suggest_float("alpha", 0.1, 0.9, step=0.05)
            geometric_candidates = trial.suggest_int("geometric_candidates", 5, 50, step=5)
            inlier_threshold = trial.suggest_float("inlier_threshold", 0.1, 0.9, step=0.1)

            # Paths including component counts
            pca_path = f"{base_dir}/pca_model_{method}_{n_pca}.pkl"
            gmm_path = f"{base_dir}/gmm_model_{method}_{n_pca}_{n_gmm}.pkl"
            fv_tr_path = f"{base_dir}/fisher_vectors_train_{method}_{n_pca}_{n_gmm}.pkl"
            fv_te_path = f"{base_dir}/fisher_vectors_test_{method}_{n_pca}_{n_gmm}.pkl"
            start_time = time.time()
            # PCA model
            if os.path.exists(pca_path):
                with open(pca_path, "rb") as f:
                    pca_model = pickle.load(f)
            else:
                pca_model = train_pca(stacked_descriptors_train, n_components=n_pca)
                with open(pca_path, "wb") as f:
                    pickle.dump(pca_model, f)

            # GMM model
            if os.path.exists(gmm_path):
                with open(gmm_path, "rb") as f:
                    gmm_model = pickle.load(f)
            else:
                reduced_train = pca_model.transform(stacked_descriptors_train)
                gmm_model = train_gmm(reduced_train, n_components=n_gmm)
                with open(gmm_path, "wb") as f:
                    pickle.dump(gmm_model, f)

            # Fisher vectors
            if os.path.exists(fv_tr_path):
                with open(fv_tr_path, "rb") as f:
                    fisher_vectors_train = pickle.load(f)
            else:
                fisher_vectors_train = compute_fisher_vectors(
                    train_descriptors, pca_model, gmm_model
                )
                with open(fv_tr_path, "wb") as f:
                    pickle.dump(fisher_vectors_train, f)

            if os.path.exists(fv_te_path):
                with open(fv_te_path, "rb") as f:
                    fisher_vectors_test = pickle.load(f)
            else:
                fisher_vectors_test = compute_fisher_vectors(
                    test_descriptors, pca_model, gmm_model
                )
                with open(fv_te_path, "wb") as f:
                    pickle.dump(fisher_vectors_test, f)
            
            predictions = classify_test_images_with_geometric_verification(
                fisher_vectors_test,
                fisher_vectors_train,
                test_keypoints,
                train_keypoints,
                test_descriptors,
                train_descriptors,
                train_labels,
                geometric_candidates=geometric_candidates,
                use_lightglue=True,
                alpha=alpha,
                min_inliers=min_inliers,
                inlier_threshold=inlier_threshold,
            )
            metrics = evaluate_predictions(predictions, test_labels)
            metrics["eval_runtime_sec"] = time.time() - start_time

            row = {
                "Dataset": dataset,
                "Training Examples": len(train_df.index),
                "Num Classes": train_df["identity"].nunique(),
                "Method": method,
                "Remove Background": False,
                "GMM Components": n_gmm,
                "PCA Components": n_pca,
                "Use GV": True,
                "Alpha (fv sim - gv)": alpha,
                "Geom. Candidates": geometric_candidates,
                "Min Inliers": min_inliers,
                "Inlier Threshold": inlier_threshold,
                "MAX GMM Descriptors (per image)": MAX_DESCRIPTORS_PER_IMAGE,
                "Multiscale Enabled": ENABLE_MULTISCALE,
                "Run Time (minutes)": round(metrics["eval_runtime_sec"] / 60, 2),
                "Accuracy": round(float(metrics["accuracy"]), 4),
                "Top-5 Accuracy": round(float(metrics["top_n_accuracy"]), 4),
                "F-1 Score": round(
                    float(metrics["classification_metrics"]["weighted avg"]["f1-score"]),
                    4,
                ),
            }
            save_count_results(row, EVAL_RESULTS_XLSX)
            return metrics["accuracy"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    print(f"Best parameters for {dataset}: {study.best_params}")

def optimise_all(datasets, trials=50):
    for ds in datasets:
        optimise_dataset(ds, trials=trials)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    args = parser.parse_args()
    DATASETS = [
        "ATRW",
    #    "CowDataset",
    #    "IPanda50",
    #    "NyalaData",
    #    "SealID",
    #    "BelugaID",
    #    "HyenaID2022",
    #    "StripeSpotter",
    #    "Giraffes",
    ]
    optimise_all(DATASETS, trials=args.trials)