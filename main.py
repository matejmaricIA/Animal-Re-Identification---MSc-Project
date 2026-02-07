import argparse
import hashlib
import json
from pathlib import Path
from wildlife_datasets import datasets, splits
from wildlife_datasets.datasets import WildlifeReID10k
from patches.elpephants_patch import PatchedELPephants
datasets.ELPephants = PatchedELPephants
from segmentation import segment_dataset, has_segmenter
import preprocessing
import sys
from train_late_fusion import train_calibrators_two_stage
from feature_extraction import (
    get_image_paths,
    extract_features,
    extract_features_keynet_hardnet,
    extract_features_keynet_hardnet_faster,
    extract_features_lightglue,
    get_segmentation_tag,
)
from feature_aggregation import (
    load_descriptors,
    stack_all_descriptors,
    train_pca,
    train_gmm,
    compute_fisher_vectors,
    load_keypoints,
    descriptor_dir,
)
from constants import *
import os
import pickle
from predict import classify_test_images_late_fusion
from evaluate import evaluate_predictions, save_evaluation_results
#from calibration import calibrate
import pandas as pd
import shutil
import cv2
#from visualize import visualize_results
import numpy as np
import torch
import time
from nested_importance_sampling import nested_importance_sampling
from nested_importance_sampling import CountCalibrators
from utility_functions import (
    load_dataset,
    save_stuff,
    load_stuff,
    save_count_results,
    combine_fisher_vectors,
    save_count_results_wrapper,
)

from global_embedding import extract_global_embeddings


if __name__ == '__main__':
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action = 'store_true')
    parser.add_argument('--count', action = 'store_true', help='Estimate the population size using Nested Importance Sampling')
    parser.add_argument(
        '--ds',
        nargs='+',
        help=(
            "Specify one or more datasets (e.g., ATRW BelugaID). "
            "Use 'full' to train on all datasets."
        ),
        default=['full'],
    )
    parser.add_argument('--save_eval', action='store_true', help='Save evaluation metrics during training', default = True)
    parser.add_argument('--use_mantiuk', action='store_true', help='Use Mantiuk tone mapping during preprocessing')
    parser.add_argument('--remove_background', action='store_true', help='Remove background during preprocessing')
    parser.add_argument('--version', type=str, default='1', help='Identifier for the current method version')
    parser.add_argument('--method', type = str, default = 'disk', help='Feature extraction method to use')
    parser.add_argument('--use_geometric_verification', action='store_true', help='Use geometric verification during prediction', default=False)
    parser.add_argument('--use_lightglue', action ='store_true', help='Use LightGlue for feature matching when performing geometric verification', default=True)
    parser.add_argument('--gv_matcher', type=str, choices=['ratio', 'lightglue', 'loftr'],
                        default=None, help='Matcher for geometric verification (ratio, lightglue, loftr)')
    parser.add_argument('--num_vertices', type=int, default=100, help='Vertices sampled in Nested-IS')
    parser.add_argument('--num_neighbors', type=int, default=10, help='Neighbors sampled per vertex in Nested-IS')
    parser.add_argument('--save_count', action='store_true', help='Save population estimation results to XLSX')
    parser.add_argument('--label_error_rate', type=float, default=0.0,
                        help='Fraction of pair labelings to flip during counting')
    parser.add_argument(
        '--count_proposal_mode',
        type=str,
        default='calibrated',
        choices=['calibrated', 'power'],
        help="HITL-NIS neighbor proposal: calibrated late fusion (WildFusion-style) or a 'power' boost rule.",
    )
    parser.add_argument(
        '--count_local_evidence',
        type=str,
        default='inliers',
        choices=['inliers', 'conf_matches'],
        help="Local evidence used in the proposal: RANSAC inliers or #confident descriptor matches.",
    )
    parser.add_argument(
        '--count_local_mu',
        type=float,
        default=0.5,
        help="Threshold µ for 'conf_matches' local evidence (descriptor cosine similarity).",
    )
    parser.add_argument(
        '--count_shortlist_B',
        type=int,
        default=300,
        help="Shortlist budget B for expensive local evidence evaluations per query image.",
    )
    parser.add_argument(
        '--count_mix_alpha',
        type=float,
        default=0.9,
        help="Mixture weight α between shortlist proposal and base proposal (keeps full support).",
    )
    parser.add_argument(
        '--count_cal_pairs',
        type=int,
        default=500,
        help="Number of GT-simulated calibration pairs for per-signal score calibration in count mode.",
    )
    parser.add_argument(
        '--count_cal_shortlist',
        type=int,
        default=300,
        help="Shortlist size used when sampling hard negatives for count calibration.",
    )
    parser.add_argument(
        '--count_cal_negs_per_query',
        type=int,
        default=100,
        help="Number of hard negatives per calibration query image.",
    )
    parser.add_argument(
        '--count_skip_calibration',
        action='store_true',
        help="Skip training/loading count calibrators; use raw similarities for proposals.",
    )
    parser.add_argument(
        '--count_force_recalibrate',
        action='store_true',
        help="Ignore cached count calibrators and retrain.",
    )
    parser.add_argument('--gv_method', type=str, default='RANSAC', choices=['RANSAC', 'MAGSAC'],
                        help='Geometric verification method to use (RANSAC or MAGSAC)')
    parser.add_argument('--use_global_embedding', action='store_true', help='Use global CNN')
    parser.add_argument('--use_fisher', action='store_true',
                        help='Use PCA/GMM/Fisher-vector features')

    parser.add_argument(
        '--embedding_model',
        type=str,
        default='megadescriptor-l-384',
        choices=[
            'resnet50',
            'megadescriptor-l-384',
        ],
        help='Model for global embeddings',
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible counting')
    parser.add_argument(
        '--use_md_baseline_split',
        action='store_true',
        help=(
            "Override splits for MD 'trained_on' datasets with the MegaDescriptor "
        ), # This is because MD was trained on some datasets so at least I want to use the same train/test split to avoid additional data leaks.
    )
    parser.add_argument('--calibration_method', type=str, default='isotonic_pchip', choices=['isotonic_pchip', 'logistic', 'isotonic'], help='Calibration method to use')
    parser.add_argument('--fusion_signals', type=str, nargs = '+', default=['global', 'fisher', 'gv'], choices=['global', 'fisher', 'gv'], help='Signals to fuse')
    parser.add_argument('--calib_size', type=int, default=200, help='Size of the calibration set')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    seg_tag = get_segmentation_tag(args.remove_background)
    args.split_type = 'closed'
    dataset_name = args.ds
    method = args.method
    #use_splitter = False
    
    # Set the geometric verification method
    #import constants
    gv_method = args.gv_method
    gv_matcher = args.gv_matcher
    if gv_matcher is None:
        gv_matcher = "lightglue" if args.use_lightglue else "ratio"
    gv_matcher = gv_matcher.lower()
    
    already_trained = False
    #split_type = 'closed_split'

    # create a configuration tag for saving evaluation results
    tag = (
        f"rmbkg_{args.remove_background}_tm_{args.use_mantiuk}_{method}"
        f"_PCA_{N_COMPONENTS_PCA}_GMM_{N_COMPONENTS_GMM}"
        f"_gv_{args.use_geometric_verification}_lg_{args.use_lightglue}"
        f"_v{args.version}"
    )
    
    def run_training_for_dataset(dataset_name: str, df_raw: pd.DataFrame) -> None:
        print(f"Training mode selected. Using dataset: {dataset_name}")
        print("training...")

        md_meta = MD_DATASET_SPLITS.get(str(dataset_name).strip().lower(), None)
        if md_meta is not None:
            print(
                "MD split metadata: "
                f"trained_on={md_meta['trained_on']}, "
                f"split_type={md_meta['split_type']}, "
                f"random_split={md_meta['random_split']}"
            )

        feature_method = method

        start_eval_time = time.time()
        use_md_baseline_split = (
            args.use_md_baseline_split and md_meta is not None and md_meta.get("trained_on")
        )

        # Pre‑processing (tone map / background removal)
        sub_dir = f"./data/{dataset_name}"
        os.makedirs(sub_dir, exist_ok=True)

        if df_raw is None:
            print("[ERROR] Missing dataset metadata (df_raw).")
            sys.exit(1)
        df_raw = df_raw.copy()
        df_raw["image_id"] = df_raw["image_id"].astype(str)
        csv_path = f"{sub_dir}/processed_metadata.csv"
        output_dir = (
            f"{sub_dir}/segmented_dataset" if args.remove_background else f"{sub_dir}/dataset"
        )

        # This whole segment is kinda deprecated and should be addressed.
        if not (os.path.exists(csv_path) and os.path.exists(output_dir)):
            if (
                args.remove_background
                and has_segmenter(dataset_name)
                and not os.path.exists(output_dir)
            ):
                df = segment_dataset(
                    df_raw.copy(),
                    f"{output_dir}/",
                    dataset_name,
                    use_mantiuk=args.use_mantiuk,
                )
            else:
                df = preprocessing.preprocess_dataset(
                    df_raw.copy(),
                    f"{output_dir}/",
                    dataset_name,
                    use_mantiuk=args.use_mantiuk,
                    remove_background=args.remove_background,
                )
            df.to_csv(csv_path, index=False)
        else:
            df = pd.read_csv(csv_path, dtype={"image_id": str})

        df["image_id"] = df["image_id"].astype(str)

        if use_md_baseline_split:
            repo_root = Path(__file__).resolve().parent
            baseline_root = repo_root / "test-scripts" / "wildlife-tools-data"
            metadata_root = baseline_root / "metadata" / "datasets"

            def _resolve_baseline_dir(root: Path, name: str) -> Path | None:
                direct = root / name
                if direct.exists():
                    return direct
                name_lower = str(name).lower()
                for child in root.iterdir():
                    if child.is_dir() and child.name.lower() == name_lower:
                        return child
                return None

            def _normalize_split_path(value: str) -> str:
                path = str(value).strip().replace("\\", "/")
                if path.startswith("./"):
                    path = path[2:]
                # WildlifeReID10k metadata paths are typically "images/<DATASET>/...".
                # MD baseline metadata paths are "...", without the prefix.
                if path.lower().startswith("images/"):
                    parts = path.split("/", 2)
                    if len(parts) >= 3:
                        path = parts[2]
                return path.lower()

            baseline_dir = _resolve_baseline_dir(metadata_root, str(dataset_name))
            if baseline_dir is None:
                print(f"[ERROR] MD baseline metadata not found for dataset: {dataset_name}")
                sys.exit(1)
            metadata_csv = baseline_dir / "metadata.csv"
            if not metadata_csv.exists():
                print(f"[ERROR] Missing MD baseline metadata CSV: {metadata_csv}")
                sys.exit(1)

            df_baseline = pd.read_csv(metadata_csv, dtype={"image_id": str, "identity": str})
            unnamed_cols = [c for c in df_baseline.columns if str(c).lower().startswith("unnamed")]
            if unnamed_cols:
                df_baseline = df_baseline.drop(columns=unnamed_cols)
            if "path" not in df_baseline.columns or "split" not in df_baseline.columns:
                print(f"[ERROR] MD baseline metadata missing required columns in: {metadata_csv}")
                sys.exit(1)

            baseline_split_map = dict(
                zip(
                    df_baseline["path"].astype(str).map(_normalize_split_path),
                    df_baseline["split"].astype(str),
                )
            )

            if "path" not in df.columns:
                print("[ERROR] Dataset metadata missing 'path' column; cannot apply MD baseline split.")
                sys.exit(1)

            key_series = df["path"].astype(str).map(_normalize_split_path)
            mapped_split = key_series.map(baseline_split_map)
            matched = int(mapped_split.notna().sum())
            total = int(len(df))
            if matched == 0:
                print(
                    f"[ERROR] Could not match any paths to MD baseline metadata for dataset: {dataset_name}."
                )
                sys.exit(1)
            if matched != total:
                print(
                    f"[WARN] MD baseline split matched {matched}/{total} rows; leaving {total - matched} unchanged."
                )

            existing_split = (
                df["split"].astype(str).str.lower()
                if "split" in df.columns
                else pd.Series(["train"] * len(df), index=df.index)
            )
            df["split_md_baseline"] = mapped_split.fillna(existing_split).astype(str).str.lower()
            if md_meta is not None:
                df["split_type"] = md_meta.get("split_type")
                df["trained_on"] = bool(md_meta.get("trained_on"))
                df["random_split"] = bool(md_meta.get("random_split"))

            df.to_csv(csv_path, index=False)

        if ("split" in df.columns) or (use_md_baseline_split and "split_md_baseline" in df.columns):
            print("Using predefined split in WildlifeReID10k.")
            split_col = "split"
            if use_md_baseline_split and "split_md_baseline" in df.columns:
                split_col = "split_md_baseline"
            train_mask = df[split_col].astype(str).str.lower() != "test"
            test_mask = df[split_col].astype(str).str.lower() == "test"
            df_train, df_test = df[train_mask], df[test_mask]

            if args.split_type == 'closed':
                print('Using closed set split.')
                # After loading the dataset splits
                train_identities = set(df_train["identity"].unique())
                test_identities = set(df_test["identity"].unique())

                # Filter test set to only include identities seen in training
                df_test = df_test[df_test["identity"].isin(train_identities)].copy()
            splits.analyze_split(df, df_train.index, df_test.index)
        else:
            print("[ERROR] Missing 'split' information in dataset metadata.")
            sys.exit(1)

        # Always use the local preprocessed folders (dataset/segmented_dataset), even for
        # MD baseline split datasets, to keep behavior consistent across datasets.
        train_image_items = get_image_paths(df_train, args.remove_background)
        test_image_items = get_image_paths(df_test, args.remove_background)
        train_paths_map = dict(zip(df_train["image_id"].astype(str), train_image_items))
        test_paths_map = dict(zip(df_test["image_id"].astype(str), test_image_items))

        image_paths = None
        if gv_matcher == "loftr":
            image_paths = dict(
                zip(df["image_id"].astype(str), get_image_paths(df, args.remove_background))
            )

        ds_tag = dataset_name
        base_dir = f"./data/{ds_tag}"
        os.makedirs(base_dir, exist_ok=True)

        # ── Feature extraction ──
        train_dict, test_dict = {}, {}
        train_keypoints, test_keypoints = {}, {}
        fv_tr, fv_te = {}, {}
        if args.use_fisher:
            if feature_method == 'ensamble':
                methods = ['disk', 'superpoint', 'aliked']
                for m in methods:
                    dir_tr = descriptor_dir(base_dir, m, 'train', seg_tag)
                    dir_te = descriptor_dir(base_dir, m, 'test', seg_tag)
                    if not os.path.isdir(dir_tr):
                        if m == 'disk':
                            extract_features(train_image_items, MODEL_PATH, dir_tr)
                            extract_features(test_image_items, MODEL_PATH, dir_te)
                        elif m == 'superpoint':
                            extract_features_lightglue(train_image_items, dir_tr, feature_type="superpoint")
                            extract_features_lightglue(test_image_items, dir_te, feature_type="superpoint")
                        elif m == 'aliked':
                            extract_features_lightglue(train_image_items, dir_tr, feature_type="aliked")
                            extract_features_lightglue(test_image_items, dir_te, feature_type="aliked")

                fv_tr_list = []
                fv_te_list = []
                for m in methods:
                    dir_tr = descriptor_dir(base_dir, m, 'train', seg_tag)
                    dir_te = descriptor_dir(base_dir, m, 'test', seg_tag)
                    train_dict_m = load_descriptors(os.path.join(dir_tr, 'descriptors.h5'))
                    test_dict_m = load_descriptors(os.path.join(dir_te, 'descriptors.h5'))
                    if m == 'disk':
                        train_dict = train_dict_m
                        test_dict = test_dict_m
                        train_keypoints = load_keypoints(os.path.join(dir_tr, 'keypoints.h5'))
                        test_keypoints = load_keypoints(os.path.join(dir_te, 'keypoints.h5'))
                    desc_tr_m = stack_all_descriptors(train_dict_m)
                    pca_path_m = PCA_PATH.format(ds_tag, m, seg_tag)
                    gmm_path_m = GMM_PATH.format(ds_tag, m, seg_tag)
                    fv_path_m  = FISHER_VECTORS.format(ds_tag, m, seg_tag)
                    if os.path.exists(pca_path_m) and os.path.exists(gmm_path_m) and os.path.exists(fv_path_m):
                        pca_m, gmm_m, fv_tr_m = load_stuff(pca_path_m, gmm_path_m, fv_path_m)
                        fv_te_m = compute_fisher_vectors(test_dict_m, pca_m, gmm_m)
                    else:
                        pca_m = train_pca(desc_tr_m)
                        gmm_m = train_gmm(pca_m.transform(desc_tr_m))
                        fv_tr_m = compute_fisher_vectors(train_dict_m, pca_m, gmm_m)
                        fv_te_m = compute_fisher_vectors(test_dict_m, pca_m, gmm_m)
                        save_stuff(pca_m, gmm_m, fv_tr_m, (pca_path_m, gmm_path_m, fv_path_m))
                    fv_tr_list.append(fv_tr_m)
                    fv_te_list.append(fv_te_m)

                fv_tr = combine_fisher_vectors(fv_tr_list, ENSEMBLE_WEIGHTS)
                fv_te = combine_fisher_vectors(fv_te_list, ENSEMBLE_WEIGHTS)

            else:
                dir_tr = descriptor_dir(base_dir, feature_method, 'train', seg_tag)
                dir_te = descriptor_dir(base_dir, feature_method, 'test', seg_tag)
                if not os.path.isdir(dir_tr):
                    if feature_method == 'disk':
                        extract_features(train_image_items, MODEL_PATH, dir_tr)
                        extract_features(test_image_items, MODEL_PATH, dir_te)
                    elif feature_method == 'keynet_hardnet':
                        extract_features_keynet_hardnet_faster(train_image_items, dir_tr)
                        extract_features_keynet_hardnet_faster(test_image_items, dir_te)
                    elif feature_method in {"lightglue", "aliked"}:
                        extract_features_lightglue(train_image_items, dir_tr, feature_type="aliked")
                        extract_features_lightglue(test_image_items, dir_te, feature_type="aliked")
                    elif feature_method == "superpoint":
                        extract_features_lightglue(train_image_items, dir_tr, feature_type="superpoint")
                        extract_features_lightglue(test_image_items, dir_te, feature_type="superpoint")

                train_dict = load_descriptors(os.path.join(dir_tr, 'descriptors.h5'))
                test_dict = load_descriptors(os.path.join(dir_te, 'descriptors.h5'))
                train_keypoints = load_keypoints(os.path.join(dir_tr, 'keypoints.h5'))
                test_keypoints = load_keypoints(os.path.join(dir_te, 'keypoints.h5'))

                desc_tr = stack_all_descriptors(train_dict)
                pca_path = PCA_PATH.format(ds_tag, feature_method, seg_tag)
                gmm_path = GMM_PATH.format(ds_tag, feature_method, seg_tag)
                fv_path  = FISHER_VECTORS.format(ds_tag, feature_method, seg_tag)
                if os.path.exists(pca_path) and os.path.exists(gmm_path) and os.path.exists(fv_path):
                    print("Using already trained PCA and GMM models.")
                    pca, gmm, fv_tr = load_stuff(pca_path, gmm_path, fv_path)
                    fv_te = compute_fisher_vectors(test_dict, pca, gmm)
                    print("Fisher vectors computed for test set.")
                else:
                    pca = train_pca(desc_tr)
                    gmm = train_gmm(pca.transform(desc_tr))
                    fv_tr = compute_fisher_vectors(train_dict, pca, gmm)
                    fv_te = compute_fisher_vectors(test_dict, pca, gmm)
                    print("Fisher vectors computed for training and test sets.")
                    save_stuff(pca, gmm, fv_tr, (pca_path, gmm_path, fv_path))

        if args.use_global_embedding:
            print("Extracting global embeddings...")
            train_paths = train_paths_map
            test_paths = test_paths_map
            emb_tr_path = f"{base_dir}/global_embeddings_train_{args.embedding_model}_{seg_tag}.pkl"
            emb_te_path = f"{base_dir}/global_embeddings_test_{args.embedding_model}_{seg_tag}.pkl"
            if os.path.exists(emb_tr_path) and os.path.exists(emb_te_path):
                with open(emb_tr_path, "rb") as f:
                    emb_tr = pickle.load(f)
                with open(emb_te_path, "rb") as f:
                    emb_te = pickle.load(f)
            else:
                emb_tr = extract_global_embeddings(train_paths, model_name=args.embedding_model)
                emb_te = extract_global_embeddings(test_paths, model_name=args.embedding_model)
                with open(emb_tr_path, "wb") as f:
                    pickle.dump(emb_tr, f)
                with open(emb_te_path, "wb") as f:
                    pickle.dump(emb_te, f)
        else:
            emb_tr, emb_te = {}, {}

        train_labels = dict(zip(df_train["image_id"], df_train["identity"]))
        test_labels = dict(zip(df_test["image_id"], df_test["identity"]))
        calibrators = train_calibrators_two_stage(
            train_labels=train_labels,
            global_emb=emb_tr,
            fisher_vectors=fv_tr,
            keypoints=train_keypoints,
            descriptors=train_dict,
            cal_size=args.calib_size,
            calibration_method=args.calibration_method,
            use_lightglue=args.use_lightglue,
            method='disk',
            fusion_signals=args.fusion_signals,
            gv_matcher=gv_matcher,
        )
        preds = classify_test_images_late_fusion(
            test_global_emb=emb_te,
            train_global_emb=emb_tr,
            test_fisher=fv_te,
            train_fisher=fv_tr,
            test_keypoints=test_keypoints,
            train_keypoints=train_keypoints,
            test_descriptors=test_dict,
            train_descriptors=train_dict,
            train_labels=train_labels,
            calibrators=calibrators,
            top_n=5,
            shortlist_size=UNION_CANDIDATES,
            use_lightglue=args.use_lightglue,
            gv_matcher=gv_matcher,
            image_paths=image_paths,
            fusion_signals=args.fusion_signals,
            test_labels=test_labels,
            debug=args.debug,
            dataset_name=dataset_name,
            calibration_method=args.calibration_method,
            calib_size=args.calib_size,
        )

        metrics = evaluate_predictions(preds, test_labels)

        if md_meta is not None:
            metrics["md_trained_on"] = md_meta["trained_on"]
            metrics["md_split_type"] = md_meta["split_type"]
            metrics["md_random_split"] = md_meta["random_split"]

        if use_cuda:
            torch.cuda.synchronize()

        runtime_sec = time.time() - start_eval_time
        metrics["eval_runtime_sec"] = runtime_sec

        if args.save_eval:
            save_evaluation_results(metrics, ds_tag, tag=tag)
            row = {
                "Dataset": dataset_name,
                "Training Examples": len(df_train.index),
                "Num Classes": df_train['identity'].nunique(),
                "Method": args.method if args.use_fisher else "N/A",
                "Use Fisher": args.use_fisher,
                #"Remove Background": args.remove_background,
                "Use Global Embedding": args.use_global_embedding,
                "Embedding Model": args.embedding_model if args.use_global_embedding else "None",
                #"GMM Components": N_COMPONENTS_GMM,
                #"PCA Components": N_COMPONENTS_PCA,
                "Use GV": args.use_geometric_verification,
                #"GV Method": gv_method if args.use_geometric_verification else "N/A",
                #"Alpha (fv sim - gv)": ALPHA,
                "Geom. Candidates": GEOMETRIC_CANDIDATES,
                #"Min Inliers": MIN_INLIERS,
                #"Inlier Threshold": INLIER_THRESHOLD,
                #"MAX GMM Descriptors (per image)": MAX_DESCRIPTORS_PER_IMAGE,
                "Calibration Method": args.calibration_method,
                "Calibration Size": args.calib_size,
                "Run Time (minutes)": round((float(metrics["eval_runtime_sec"]) / 60), 2),
                "Accuracy": round(float(metrics["accuracy"]), 4),
                "Top-5 Accuracy": round(float(metrics["top_n_accuracy"]), 4),
                "F-1 Score": round(float(metrics["classification_metrics"]["weighted avg"]["f1-score"]), 4)
            }
            if md_meta is not None:
                row["MD Trained On"] = md_meta["trained_on"]
                row["MD Split Type"] = md_meta["split_type"]
                row["MD Random Split"] = md_meta["random_split"]
            save_count_results_wrapper(row, EVAL_RESULTS_XLSX)

        _input = 'no'
        if _input.strip().lower() == "yes" and args.use_fisher:
            extract_features(get_image_paths(df, args.remove_background), MODEL_PATH, f"{base_dir}/db/")
            db_dict = load_descriptors(f"{base_dir}/db/descriptors.h5")
            desc = stack_all_descriptors(db_dict)
            pca_db = train_pca(desc)
            gmm_db = train_gmm(pca_db.transform(desc))
            fv_db = compute_fisher_vectors(db_dict, pca_db, gmm_db)
            save_stuff(pca_db, gmm_db, fv_db,
                (f"{base_dir}/db/pca.pkl",
                f"{base_dir}/db/gmm.pkl",
                f"{base_dir}/db/fisher_vectors.pkl"))
            print("Database saved.")

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if args.train and not dataset_name:
        print("Please specify the dataset to train on using the --ds argument.")
        sys.exit(0)

    if args.train:
        requested = [str(name).strip() for name in dataset_name if str(name).strip()]
        if not requested:
            print("Please specify the dataset to train on using the --ds argument.")
            sys.exit(0)

        if any(name.lower() == "full" for name in requested):
            df_all = load_dataset("full")
            if "dataset" not in df_all.columns:
                print("Expected 'dataset' column in all_datasets.csv for full mode.")
                sys.exit(1)
            for sub_name, sub_df in df_all.groupby("dataset", sort=True):
                run_training_for_dataset(sub_name, sub_df)
        else:
            for name in requested:
                df_raw = load_dataset(name)
                run_training_for_dataset(name, df_raw)
        sys.exit(0)

    if args.count:
        if not dataset_name or len(dataset_name) != 1:
            print("[ERROR] --count requires exactly one dataset: use `--ds <DATASET>`.")
            sys.exit(1)

        dataset_name = str(dataset_name[0]).strip()
        ds_tag = dataset_name.lower()
        base_dir = f"./data/{ds_tag}"
        os.makedirs(base_dir, exist_ok=True)

        print(f"[COUNT] Dataset: {dataset_name} (dir: {base_dir})")
        print(f"[COUNT] Proposal: {args.count_proposal_mode} | Local evidence: {args.count_local_evidence}")

        if gv_matcher == "loftr":
            print(
                "[ERROR] Count mode local evidence currently does not support LoFTR "
                "(it requires image inputs in the local-evidence path). "
                "Use `--gv_matcher ratio` or `--gv_matcher lightglue`."
            )
            sys.exit(1)

        start_time = time.time()

        df_raw = load_dataset(dataset_name)
        df_raw["image_id"] = df_raw["image_id"].astype(str)

        csv_path = f"{base_dir}/processed_metadata.csv"
        output_dir = (
            f"{base_dir}/segmented_dataset" if args.remove_background else f"{base_dir}/dataset"
        )
        print(f"[COUNT] Output directory: {output_dir}")

        df = None
        missing_cols = []
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, dtype={"image_id": str})
            required_cols = (
                ["processed_path_segmented"]
                if args.remove_background
                else ["processed_path"]
            )
            missing_cols = [
                col
                for col in required_cols
                if col not in df.columns or df[col].isna().all()
            ]

        if missing_cols or not os.path.exists(csv_path) or not os.path.exists(output_dir):
            if (
                args.remove_background
                and has_segmenter(dataset_name)
                and not os.path.exists(output_dir)
            ):
                print("[COUNT] Segmenting dataset...")
                df = segment_dataset(
                    df_raw.copy(),
                    f"{output_dir}/",
                    dataset_name,
                    use_mantiuk=args.use_mantiuk,
                )
            else:
                df = preprocessing.preprocess_dataset(
                    df_raw.copy(),
                    f"{output_dir}/",
                    dataset_name,
                    use_mantiuk=args.use_mantiuk,
                    remove_background=args.remove_background,
                )
            df.to_csv(csv_path, index=False)

        # --- Local features (always required for local evidence) ---
        local_feature_method = method if method != "ensamble" else "disk"
        feat_dir_local = f"{base_dir}/feature_descriptors_{local_feature_method}_{seg_tag}_full/"
        if not os.path.isdir(feat_dir_local):
            if local_feature_method == "disk":
                extract_features(get_image_paths(df, args.remove_background), MODEL_PATH, feat_dir_local)
            elif local_feature_method == "keynet_hardnet":
                extract_features_keynet_hardnet_faster(get_image_paths(df, args.remove_background), feat_dir_local)
            elif local_feature_method in {"lightglue", "aliked"}:
                extract_features_lightglue(get_image_paths(df, args.remove_background), feat_dir_local, feature_type="aliked")
            elif local_feature_method == "superpoint":
                extract_features_lightglue(get_image_paths(df, args.remove_background), feat_dir_local, feature_type="superpoint")
            else:
                print(f"[ERROR] Unsupported local feature method for counting: {local_feature_method}")
                sys.exit(1)

        descriptors = load_descriptors(os.path.join(feat_dir_local, "descriptors.h5"))
        keypoints = load_keypoints(os.path.join(feat_dir_local, "keypoints.h5"))

        # --- Fisher vectors (optional signal) ---
        fisher_vectors = None
        if args.use_fisher:
            if method == "ensamble":
                methods = ["disk", "superpoint", "aliked"]
                fv_list = []
                for m in methods:
                    feat_dir = f"{base_dir}/feature_descriptors_{m}_{seg_tag}_full/"
                    if not os.path.isdir(feat_dir):
                        if m == "disk":
                            extract_features(get_image_paths(df, args.remove_background), MODEL_PATH, feat_dir)
                        elif m == "superpoint":
                            extract_features_lightglue(get_image_paths(df, args.remove_background), feat_dir, feature_type="superpoint")
                        elif m == "aliked":
                            extract_features_lightglue(get_image_paths(df, args.remove_background), feat_dir, feature_type="aliked")

                    desc = load_descriptors(os.path.join(feat_dir, "descriptors.h5"))
                    pca_path_m = PCA_PATH.format(ds_tag, m, f"{seg_tag}_full")
                    gmm_path_m = GMM_PATH.format(ds_tag, m, f"{seg_tag}_full")
                    fv_path_m = FISHER_VECTORS.format(ds_tag, m, f"{seg_tag}_full")
                    if os.path.exists(fv_path_m):
                        pca_m, gmm_m, fv_m = load_stuff(pca_path_m, gmm_path_m, fv_path_m)
                    else:
                        desc_stack = stack_all_descriptors(desc)
                        pca_m = train_pca(desc_stack)
                        gmm_m = train_gmm(pca_m.transform(desc_stack))
                        fv_m = compute_fisher_vectors(desc, pca_m, gmm_m)
                        save_stuff(pca_m, gmm_m, fv_m, (pca_path_m, gmm_path_m, fv_path_m))
                    fv_list.append(fv_m)
                fisher_vectors = combine_fisher_vectors(fv_list, ENSEMBLE_WEIGHTS)
            else:
                pca_path = PCA_PATH.format(ds_tag, method, f"{seg_tag}_full")
                gmm_path = GMM_PATH.format(ds_tag, method, f"{seg_tag}_full")
                fv_path = FISHER_VECTORS.format(ds_tag, method, f"{seg_tag}_full")
                if os.path.exists(fv_path):
                    _, _, fisher_vectors = load_stuff(pca_path, gmm_path, fv_path)
                else:
                    desc_stack = stack_all_descriptors(descriptors)
                    pca = train_pca(desc_stack)
                    gmm = train_gmm(pca.transform(desc_stack))
                    fisher_vectors = compute_fisher_vectors(descriptors, pca, gmm)
                    save_stuff(pca, gmm, fisher_vectors, (pca_path, gmm_path, fv_path))

        # --- Global embeddings (optional signal) ---
        global_embeddings = None
        if args.use_global_embedding:
            print("[COUNT] Loading/computing global embeddings...")
            image_paths = dict(zip(df["image_id"].astype(str), get_image_paths(df, args.remove_background)))
            emb_path = f"{base_dir}/global_embeddings_count_{args.embedding_model}_{seg_tag}_full.pkl"
            if os.path.exists(emb_path):
                with open(emb_path, "rb") as f:
                    global_embeddings = pickle.load(f)
            else:
                global_embeddings = extract_global_embeddings(image_paths, model_name=args.embedding_model)
                with open(emb_path, "wb") as f:
                    pickle.dump(global_embeddings, f)

        if global_embeddings is None and fisher_vectors is None:
            print(
                "[ERROR] Count mode requires at least one cheap base signal. "
                "Enable `--use_global_embedding` and/or `--use_fisher`."
            )
            sys.exit(1)

        # --- Oracle: for now, use GT identity labels to simulate human vetting ---
        if "identity" not in df.columns:
            print("[ERROR] Counting currently requires ground-truth `identity` labels in metadata.")
            sys.exit(1)

        df["image_id"] = df["image_id"].astype(str)
        df["identity"] = df["identity"].astype(str)
        labels = dict(zip(df["image_id"], df["identity"]))

        def oracle(u_id: str, v_id: str) -> int:
            return int(labels.get(u_id) == labels.get(v_id) and u_id != v_id)

        # --- Calibration (WildFusion-style per-signal score → P(match)) ---
        calibrators_bundle = None
        cal_info = {}
        if args.count_proposal_mode == "calibrated" and not args.count_skip_calibration:
            from calibration import train_count_calibrators_gt

            cal_key = {
                "ds_tag": ds_tag,
                "seg_tag": seg_tag,
                "use_global_embedding": bool(args.use_global_embedding),
                "embedding_model": args.embedding_model if args.use_global_embedding else None,
                "use_fisher": bool(args.use_fisher),
                "fisher_method": method if args.use_fisher else None,
                "local_feature_method": local_feature_method,
                "local_evidence": args.count_local_evidence,
                "local_mu": float(args.count_local_mu),
                "count_cal_pairs": int(args.count_cal_pairs),
                "count_cal_shortlist": int(args.count_cal_shortlist),
                "count_cal_negs_per_query": int(args.count_cal_negs_per_query),
                "calibration_method": args.calibration_method,
                "gv_matcher": gv_matcher,
                "use_lightglue": bool(args.use_lightglue),
            }
            cal_key_json = json.dumps(cal_key, sort_keys=True, separators=(",", ":"))
            cal_hash = hashlib.sha1(cal_key_json.encode("utf-8")).hexdigest()[:12]
            cal_cache_path = f"{base_dir}/count_calibrators_{cal_hash}.pkl"

            cal_loaded = False
            cal_dict = None
            if os.path.exists(cal_cache_path) and not args.count_force_recalibrate:
                try:
                    with open(cal_cache_path, "rb") as f:
                        payload = pickle.load(f)
                    if payload.get("key") == cal_key:
                        cal_dict = payload.get("calibrators", None)
                        cal_info = payload.get("info", {}) or {}
                        cal_loaded = isinstance(cal_dict, dict) and bool(cal_dict)
                except Exception:
                    cal_loaded = False

            if not cal_loaded:
                print("[COUNT] Training count calibrators (GT simulation)...")
                cal_dict, cal_info = train_count_calibrators_gt(
                    train_labels=labels,
                    image_ids=df["image_id"].astype(str).tolist(),
                    global_emb=global_embeddings,
                    fisher_vectors=fisher_vectors,
                    keypoints=keypoints,
                    descriptors=descriptors,
                    local_evidence=args.count_local_evidence,
                    local_mu=float(args.count_local_mu),
                    target_pairs=int(args.count_cal_pairs),
                    shortlist_size=int(args.count_cal_shortlist),
                    n_negatives_per_query=int(args.count_cal_negs_per_query),
                    calibration_method=args.calibration_method,
                    use_lightglue=args.use_lightglue,
                    method=local_feature_method,
                    gv_matcher=gv_matcher,
                    seed=args.seed if args.seed is not None else 42,
                )
                with open(cal_cache_path, "wb") as f:
                    pickle.dump({"key": cal_key, "calibrators": cal_dict, "info": cal_info}, f)
                print(f"[COUNT] Saved count calibrators to: {cal_cache_path}")
            else:
                print(f"[COUNT] Loaded cached count calibrators: {cal_cache_path}")

            calibrators_bundle = CountCalibrators(
                global_cal=cal_dict.get("global") if isinstance(cal_dict, dict) else None,
                fisher_cal=cal_dict.get("fisher") if isinstance(cal_dict, dict) else None,
                local_cal=cal_dict.get("local") if isinstance(cal_dict, dict) else None,
            )

        # --- Estimate with HITL-NIS (no GV gate; local computed on shortlist only) ---
        image_ids = df["image_id"].astype(str).tolist()

        estimate, se, stats = nested_importance_sampling(
            global_embeddings,
            fisher_vectors,
            image_ids,
            oracle=oracle,
            keypoints=keypoints,
            descriptors=descriptors,
            proposal_mode=args.count_proposal_mode,
            local_evidence=args.count_local_evidence,
            local_mu=float(args.count_local_mu),
            shortlist_B=int(args.count_shortlist_B),
            mix_alpha=float(args.count_mix_alpha),
            calibrators=calibrators_bundle,
            use_lightglue=args.use_lightglue,
            method=local_feature_method,
            gv_matcher=gv_matcher,
            n_vertices=args.num_vertices,
            n_neighbors=args.num_neighbors,
            label_error_rate=args.label_error_rate,
            seed=args.seed,
            return_stats=True,
        )

        runtime = time.time() - start_time
        ci_low = float(estimate - 1.96 * se)
        ci_high = float(estimate + 1.96 * se)

        print(f"[COUNT] Estimated individuals: {estimate:.2f} ± {se:.2f} (95% CI [{ci_low:.2f}, {ci_high:.2f}])")
        print(stats)

        if args.save_count:
            row = {
                "Dataset": dataset_name,
                "Num Images": int(len(image_ids)),
                "Ground Truth": int(df['identity'].nunique()),
                "Use Global Embedding": bool(args.use_global_embedding),
                "Embedding Model": args.embedding_model if args.use_global_embedding else "None",
                "Use Fisher": bool(args.use_fisher),
                "Fisher Method": method if args.use_fisher else "None",
                "Local Feature Method": local_feature_method,
                "GV Matcher": gv_matcher,
                "Remove Background": bool(args.remove_background),
                "Proposal Mode": args.count_proposal_mode,
                "Local Evidence": args.count_local_evidence,
                "Local µ": float(args.count_local_mu),
                "Shortlist B": int(args.count_shortlist_B),
                "Mix α": float(args.count_mix_alpha),
                "Calibration Method": args.calibration_method,
                "Cal Pairs": int(args.count_cal_pairs),
                "Cal Shortlist": int(args.count_cal_shortlist),
                "Cal Negs/Query": int(args.count_cal_negs_per_query),
                "Skip Calibration": bool(args.count_skip_calibration),
                "Num Vertices": int(args.num_vertices),
                "Num Neighbors": int(args.num_neighbors),
                "Label Error Rate": float(args.label_error_rate),
                "Oracle Calls": int(stats.get("oracle_calls", 0)),
                "Unique Oracle Pairs": int(stats.get("unique_oracle_pairs", 0)),
                "Local Attempts": int(stats.get("local_attempts", 0)),
                "Local Cache Hits": int(stats.get("local_cache_hits", 0)),
                "Result": round(float(estimate), 2),
                "Std Error": round(float(se), 4),
                "CI Low (95%)": round(float(ci_low), 2),
                "CI High (95%)": round(float(ci_high), 2),
                "Runtime (minutes)": round(float(runtime) / 60, 2),
            }
            if cal_info:
                row["Cal Pos"] = int(cal_info.get("pos", 0))
                row["Cal Neg"] = int(cal_info.get("neg", 0))
                row["Cal Pos Rate"] = float(cal_info.get("pos_rate", 0.0))
            save_count_results_wrapper(row)

        sys.exit(0)
